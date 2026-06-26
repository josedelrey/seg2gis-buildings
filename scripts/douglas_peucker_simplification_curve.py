"""
Plot how Douglas-Peucker simplification affects vector information retention.

For each epsilon ratio, the script simplifies predicted building contours,
rasterizes the simplified polygons back to the image grid, and measures how
well they preserve the post-processed raster mask. It also reports the IoU
against GT after simplification and the retained vertex count.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path  # noqa: E402
from dataset import collect_image_mask_pairs, describe_image_ids, image_id_list  # noqa: E402
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled  # noqa: E402
from metrics import confusion_from_masks, metrics_from_confusion  # noqa: E402
from vectorize import mask_to_contours, simplify_contours  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_EPSILON_RATIOS = (
    "0,0.00005,0.0001,0.0002,0.0003,0.0005,0.00075,0.001,"
    "0.0015,0.002,0.003,0.004,0.006,0.008,0.01,0.015,"
    "0.02,0.03,0.05"
)

CSV_HEADER = [
    "run_name",
    "split",
    "image_ids",
    "n_images",
    "threshold",
    "min_area",
    "open_kernel_size",
    "epsilon_ratio",
    "mask_preservation_iou",
    "mask_preservation_dice",
    "mask_preservation_precision",
    "mask_preservation_recall",
    "gt_iou_after_simplification",
    "gt_dice_after_simplification",
    "gt_precision_after_simplification",
    "gt_recall_after_simplification",
    "retained_area_ratio",
    "n_polygons",
    "mean_vertices_per_polygon",
    "median_vertices_per_polygon",
    "total_vertices",
    "original_total_vertices",
    "vertex_retention_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Douglas-Peucker simplification curve figure.",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--open_kernel_size", type=int, default=None)
    parser.add_argument(
        "--epsilon_ratios",
        type=str,
        default=DEFAULT_EPSILON_RATIOS,
        help="Comma-separated Douglas-Peucker epsilon ratios to evaluate.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results/cache/postprocess_ablation",
        help="Base directory for cached probability maps.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default=None,
        help="Optional PDF figure path.",
    )
    parser.add_argument(
        "--pdf_only",
        action="store_true",
        help="Write only the PDF figure and skip the PNG preview.",
    )
    parser.add_argument(
        "--no_disk_cache",
        action="store_true",
        help="Disable reading/writing cached probability maps.",
    )
    return parser.parse_args()


def require_config_value(config, *keys):
    value = get_config_value(config, *keys)
    if value is None:
        raise ValueError(f"Missing required config value: {'.'.join(keys)}")
    return value


def parse_float_list(value, option_name):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one value.")
    return [float(item) for item in items]


def safe_path_part(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unnamed"


def resolve_prediction_cache_dir(base_dir, run_name, split, tile_size, stride):
    cache_name = (
        f"{safe_path_part(run_name)}_{safe_path_part(split)}"
        f"_tile{int(tile_size)}_stride{int(stride)}"
    )
    return REPO_ROOT / base_dir / cache_name


def cache_file_for_image(cache_dir, image_path):
    return cache_dir / f"{safe_path_part(Path(image_path).stem)}.npz"


def load_cached_prediction(cache_path):
    with np.load(cache_path) as data:
        prob_map = data["prob_map"].astype(np.float32, copy=False)
        target_mask = data["target_mask"].astype(bool, copy=False)

    if prob_map.shape != target_mask.shape:
        raise RuntimeError(
            f"Cached prediction/target shape mismatch in {cache_path}: "
            f"{prob_map.shape} vs {target_mask.shape}"
        )

    return prob_map, target_mask


def save_cached_prediction(cache_path, image_path, mask_path, prob_map, target_mask):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_path=str(image_path),
        mask_path=str(mask_path),
        prob_map=prob_map.astype(np.float32, copy=False),
        target_mask=target_mask.astype(np.uint8, copy=False),
    )


def collect_cached_images(
    model_loader,
    image_dir,
    mask_dir,
    image_ids,
    tile_size,
    stride,
    disk_cache_dir,
):
    pairs = collect_image_mask_pairs(image_dir, mask_dir, image_ids)
    cached_images = []
    model = None
    cache_hits = 0
    cache_misses = 0

    for image_path, mask_path in tqdm(pairs, desc="Loading prediction cache"):
        cache_path = None
        if disk_cache_dir is not None:
            cache_path = cache_file_for_image(disk_cache_dir, image_path)
            if cache_path.exists():
                prob_map, target_mask = load_cached_prediction(cache_path)
                cache_hits += 1
                cached_images.append((image_path, prob_map, target_mask))
                continue

        if model is None:
            model = model_loader()
            model.eval()

        image_rgb = load_rgb_image(image_path)
        target_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if target_mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        target_mask = target_mask > 127
        prob_map = predict_full_image_tiled(
            model=model,
            image_rgb=image_rgb,
            tile_size=tile_size,
            stride=stride,
            device=DEVICE,
        )

        if cache_path is not None:
            save_cached_prediction(
                cache_path=cache_path,
                image_path=image_path,
                mask_path=mask_path,
                prob_map=prob_map,
                target_mask=target_mask,
            )

        cache_misses += 1
        cached_images.append((image_path, prob_map, target_mask))

    if disk_cache_dir is not None:
        print(
            "Prediction cache:",
            f"{cache_hits} hit(s), {cache_misses} miss(es)",
            f"at {disk_cache_dir}",
        )

    return cached_images


def remove_small_components_fast(mask, min_area):
    if min_area is None or min_area <= 0:
        return (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


def postprocess_mask_fast(mask, min_area, open_kernel_size):
    cleaned = remove_small_components_fast(mask, min_area)

    if open_kernel_size is not None and open_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel_size, open_kernel_size),
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    cleaned = remove_small_components_fast(cleaned, min_area)
    return (cleaned > 0).astype(np.uint8)


def rasterize_contours(contours, shape):
    rasterized = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(rasterized, contours, -1, color=1, thickness=-1)
    return rasterized.astype(bool)


def vertex_counts(contours):
    return [int(len(contour)) for contour in contours]


def summarize_vertices(counts):
    if not counts:
        return 0.0, 0.0, 0

    return float(np.mean(counts)), float(np.median(counts)), int(sum(counts))


def compute_curve_rows(
    image_records,
    epsilon_ratios,
    run_name,
    split,
    image_ids,
    threshold,
    min_area,
    open_kernel_size,
):
    rows = []
    original_total_vertices = sum(
        record["original_total_vertices"]
        for record in image_records
    )
    post_area = sum(record["post_area"] for record in image_records)

    for epsilon_ratio in tqdm(epsilon_ratios, desc="Evaluating epsilon ratios"):
        preserve_tp = preserve_fp = preserve_fn = preserve_tn = 0
        gt_tp = gt_fp = gt_fn = gt_tn = 0
        retained_area = 0
        total_polygons = 0
        all_vertex_counts = []

        for record in image_records:
            if abs(epsilon_ratio) <= 1e-12:
                simplified = record["contours"]
                simplified_mask = record["post_mask"]
            else:
                simplified = simplify_contours(
                    record["contours"],
                    epsilon_ratio=epsilon_ratio,
                )
                simplified_mask = rasterize_contours(
                    simplified,
                    shape=record["post_mask"].shape,
                )

            tp, fp, fn, tn = confusion_from_masks(
                simplified_mask,
                record["post_mask"],
            )
            preserve_tp += tp
            preserve_fp += fp
            preserve_fn += fn
            preserve_tn += tn

            tp, fp, fn, tn = confusion_from_masks(
                simplified_mask,
                record["target_mask"],
            )
            gt_tp += tp
            gt_fp += fp
            gt_fn += fn
            gt_tn += tn

            retained_area += int(simplified_mask.sum())
            total_polygons += len(simplified)
            all_vertex_counts.extend(vertex_counts(simplified))

        preserve_metrics = metrics_from_confusion(
            preserve_tp,
            preserve_fp,
            preserve_fn,
            preserve_tn,
        )
        gt_metrics = metrics_from_confusion(gt_tp, gt_fp, gt_fn, gt_tn)
        mean_vertices, median_vertices, total_vertices = summarize_vertices(
            all_vertex_counts,
        )

        rows.append({
            "run_name": run_name,
            "split": split,
            "image_ids": describe_image_ids(image_ids),
            "n_images": len(image_records),
            "threshold": round(float(threshold), 2),
            "min_area": int(min_area),
            "open_kernel_size": int(open_kernel_size),
            "epsilon_ratio": round(float(epsilon_ratio), 6),
            "mask_preservation_iou": round(preserve_metrics["iou_building"], 6),
            "mask_preservation_dice": round(preserve_metrics["dice_f1"], 6),
            "mask_preservation_precision": round(preserve_metrics["precision"], 6),
            "mask_preservation_recall": round(preserve_metrics["recall"], 6),
            "gt_iou_after_simplification": round(gt_metrics["iou_building"], 6),
            "gt_dice_after_simplification": round(gt_metrics["dice_f1"], 6),
            "gt_precision_after_simplification": round(gt_metrics["precision"], 6),
            "gt_recall_after_simplification": round(gt_metrics["recall"], 6),
            "retained_area_ratio": round(retained_area / post_area, 6),
            "n_polygons": int(total_polygons),
            "mean_vertices_per_polygon": round(mean_vertices, 6),
            "median_vertices_per_polygon": round(median_vertices, 6),
            "total_vertices": int(total_vertices),
            "original_total_vertices": int(original_total_vertices),
            "vertex_retention_ratio": round(
                total_vertices / original_total_vertices,
                6,
            ),
        })

    return rows


def write_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size, bold=False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_center(draw, xy, text, font, fill):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((x - width / 2, y - height / 2), text, font=font, fill=fill)


def draw_rotated_text(image, xy, text, font, fill, angle=45):
    text_bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
        (0, 0),
        text,
        font=font,
    )
    text_width = text_bbox[2] - text_bbox[0] + 8
    text_height = text_bbox[3] - text_bbox[1] + 8
    text_image = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4, 4), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = xy
    image.alpha_composite(rotated, (int(x - rotated.width / 2), int(y)))


def plot_line(draw, points, color, width=5, radius=7):
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )


def plot_curve(rows, out_png=None, out_pdf=None):
    epsilons = np.array([float(row["epsilon_ratio"]) for row in rows])
    plot_eps = np.where(epsilons == 0, 1e-5, epsilons)
    preservation_iou = [float(row["mask_preservation_iou"]) for row in rows]
    gt_iou = [float(row["gt_iou_after_simplification"]) for row in rows]
    vertex_retention = [float(row["vertex_retention_ratio"]) for row in rows]
    mean_vertices = [float(row["mean_vertices_per_polygon"]) for row in rows]

    if out_png is None and out_pdf is None:
        raise ValueError("At least one of out_png or out_pdf must be provided.")

    primary_output = out_png if out_png is not None else out_pdf
    output_dir = os.path.dirname(primary_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    scale = 3
    base_width, base_height = 2600, 1740
    width, height = base_width * scale, base_height * scale
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    def s(value):
        return int(round(value * scale))

    def sp(point):
        x, y = point
        return (s(x), s(y))

    def sb(box):
        x1, y1, x2, y2 = box
        return (s(x1), s(y1), s(x2), s(y2))

    def sw(value):
        return max(1, s(value))

    title_font = load_font(s(58), bold=True)
    label_font = load_font(s(38), bold=True)
    tick_font = load_font(s(30))
    legend_font = load_font(s(32))

    blue = (31, 119, 180, 255)
    green = (44, 160, 44, 255)
    red = (214, 39, 40, 255)
    purple = (148, 103, 189, 255)
    dark = (55, 55, 55, 255)
    grid = (220, 220, 220, 255)
    axis = (70, 70, 70, 255)

    draw_text_center(
        draw,
        sp((base_width / 2, 66)),
        "Effect of Douglas-Peucker simplification on building polygons",
        title_font,
        dark,
    )

    top = (300, 240, base_width - 240, 815)
    bottom = (300, 1085, base_width - 240, 1485)
    x_min = float(np.log10(plot_eps.min()))
    x_max = float(np.log10(plot_eps.max()))
    top_values = preservation_iou + gt_iou
    top_min = max(0.0, np.floor((min(top_values) - 0.03) * 20) / 20)
    top_max = min(1.0, np.ceil((max(top_values) + 0.02) * 20) / 20)
    if top_max - top_min < 0.2:
        top_min = max(0.0, top_max - 0.2)
    mean_max = max(mean_vertices) if mean_vertices else 1.0
    mean_axis_max = max(10.0, np.ceil(mean_max / 10.0) * 10.0)

    def x_to_px(value):
        log_value = float(np.log10(value))
        return top[0] + (log_value - x_min) / (x_max - x_min) * (top[2] - top[0])

    def y_to_px(value, panel, y_min, y_max):
        return panel[3] - (value - y_min) / (y_max - y_min) * (panel[3] - panel[1])

    def format_axis_tick(value):
        if abs(value - round(value)) < 1e-9:
            return f"{value:.0f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def format_epsilon_label(value):
        if value == 0:
            return "0"
        if value < 0.001:
            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
        return f"{value:g}"

    def draw_text(position, text, font, fill):
        draw.text(sp(position), text, font=font, fill=fill)

    def text_width(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0]) / scale

    def draw_panel(panel, y_ticks, y_min, y_max, left_label):
        draw.rectangle(sb(panel), outline=axis, width=sw(1.5))
        for tick in y_ticks:
            y = y_to_px(tick, panel, y_min, y_max)
            draw.line((s(panel[0]), s(y), s(panel[2]), s(y)), fill=grid, width=sw(1))
            draw_text((panel[0] - 130, y - 19), format_axis_tick(tick), tick_font, dark)
        draw_text((panel[0], panel[1] - 56), left_label, label_font, dark)

    def nice_ticks(y_min, y_max, step):
        values = []
        start = np.ceil(y_min / step) * step
        value = start
        while value <= y_max + 1e-9:
            values.append(round(float(value), 6))
            value += step
        return values

    draw_panel(
        top,
        nice_ticks(top_min, top_max, 0.05),
        top_min,
        top_max,
        "IoU",
    )
    draw_panel(bottom, [0.0, 0.25, 0.5, 0.75, 1.0], 0.0, 1.0, "Vertex retention ratio")

    right_ticks = list(np.linspace(0, mean_axis_max, 5))
    draw_text(
        (bottom[2] - text_width("Mean vertices per polygon", label_font), bottom[1] - 56),
        "Mean vertices per polygon",
        label_font,
        dark,
    )
    for tick in right_ticks:
        y = y_to_px(tick, bottom, 0.0, mean_axis_max)
        draw.line((s(bottom[2]), s(y), s(bottom[2] + 10), s(y)), fill=axis, width=sw(1.5))
        draw_text((bottom[2] + 18, y - 19), f"{tick:.0f}", tick_font, dark)

    for eps in plot_eps:
        x = x_to_px(eps)
        draw.line((s(x), s(top[1]), s(x), s(top[3])), fill=(238, 238, 238, 255), width=sw(0.7))
        draw.line((s(x), s(bottom[1]), s(x), s(bottom[3])), fill=(238, 238, 238, 255), width=sw(0.7))

    selected_x = x_to_px(0.002)
    dash_y = top[1]
    while dash_y < bottom[3]:
        draw.line((s(selected_x), s(dash_y), s(selected_x), s(dash_y + 16)), fill=dark, width=sw(1.5))
        dash_y += 34

    top_preservation = [
        sp((x_to_px(x), y_to_px(y, top, top_min, top_max)))
        for x, y in zip(plot_eps, preservation_iou)
    ]
    top_gt = [
        sp((x_to_px(x), y_to_px(y, top, top_min, top_max)))
        for x, y in zip(plot_eps, gt_iou)
    ]
    bottom_retention = [
        sp((x_to_px(x), y_to_px(y, bottom, 0.0, 1.0)))
        for x, y in zip(plot_eps, vertex_retention)
    ]
    bottom_mean_vertices = [
        sp((x_to_px(x), y_to_px(y, bottom, 0.0, mean_axis_max)))
        for x, y in zip(plot_eps, mean_vertices)
    ]

    plot_line(draw, top_preservation, blue, width=sw(2.3), radius=sw(4.5))
    plot_line(draw, top_gt, green, width=sw(2.0), radius=sw(4.0))
    plot_line(draw, bottom_retention, red, width=sw(2.3), radius=sw(4.5))
    plot_line(draw, bottom_mean_vertices, purple, width=sw(2.0), radius=sw(4.0))

    def draw_legend(items, y):
        x = top[0]
        for color, text in items:
            draw.line((s(x), s(y + 14), s(x + 46), s(y + 14)), fill=color, width=sw(3))
            draw_text((x + 58, y - 4), text, legend_font, dark)
            x += 58 + text_width(text, legend_font) + 70

    draw_legend(
        [
            (blue, "Mask preservation IoU"),
            (green, "GT IoU after simplification"),
            (dark, "Selected epsilon 0.002"),
        ],
        160,
    )
    draw_legend(
        [
            (red, "Vertex retention ratio"),
            (purple, "Mean vertices per polygon"),
        ],
        1000,
    )

    labeled_eps = {
        0.0,
        0.00005,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.004,
        0.008,
        0.015,
        0.03,
        0.05,
    }
    tick_labels = [format_epsilon_label(value) for value in epsilons]
    labeled_index = 0
    for eps, label in zip(plot_eps, tick_labels):
        original_eps = 0.0 if label == "0" else float(label.replace("e", "e"))
        if not any(abs(original_eps - item) < 1e-12 for item in labeled_eps):
            continue
        x = x_to_px(eps)
        draw.line((s(x), s(bottom[3]), s(x), s(bottom[3] + 12)), fill=axis, width=sw(1.3))
        label_width = text_width(label, tick_font)
        label_x = min(max(x, bottom[0] + label_width / 2), bottom[2] - label_width / 2)
        label_y = bottom[3] + 36 + (labeled_index % 2) * 52
        draw_text_center(draw, sp((label_x, label_y)), label, tick_font, dark)
        labeled_index += 1

    draw_text_center(
        draw,
        sp((base_width / 2, bottom[3] + 170)),
        "Douglas-Peucker epsilon ratio",
        label_font,
        dark,
    )

    rgb_image = image.convert("RGB").resize(
        (base_width, base_height),
        resample=Image.Resampling.LANCZOS,
    )
    if out_png is not None:
        rgb_image.save(out_png, "PNG")
    if out_pdf is not None:
        pdf_dir = os.path.dirname(out_pdf)
        if pdf_dir:
            os.makedirs(pdf_dir, exist_ok=True)
        try:
            rgb_image.save(out_pdf, "PDF", resolution=300.0)
        except PermissionError:
            fallback_pdf = str(Path(out_pdf).with_name(
                f"{Path(out_pdf).stem}_updated{Path(out_pdf).suffix}"
            ))
            rgb_image.save(fallback_pdf, "PDF", resolution=300.0)
            print(
                f"Could not overwrite locked PDF {out_pdf}; "
                f"wrote {fallback_pdf} instead."
            )


def print_summary(rows):
    print()
    print("Douglas-Peucker simplification curve")
    print(
        f"{'Epsilon':>9} {'Mask IoU':>9} {'GT IoU':>8} "
        f"{'VertRet':>8} {'MeanVtx':>8} {'Polygons':>8}"
    )
    for row in rows:
        print(
            f"{row['epsilon_ratio']:>9.4f} "
            f"{row['mask_preservation_iou']:>9.4f} "
            f"{row['gt_iou_after_simplification']:>8.4f} "
            f"{row['vertex_retention_ratio']:>8.4f} "
            f"{row['mean_vertices_per_polygon']:>8.2f} "
            f"{row['n_polygons']:>8}"
        )


def main():
    args = parse_args()
    epsilon_ratios = parse_float_list(args.epsilon_ratios, "--epsilon_ratios")
    config = load_config(args.config)

    run_name = require_config_value(config, "training", "run_name")
    architecture = require_config_value(config, "model", "architecture")
    encoder = require_config_value(config, "model", "encoder")
    model_dir = require_config_value(config, "model", "model_dir")
    model_path = resolve_model_path(model_dir, run_name)
    image_ids = image_id_list(
        require_config_value(config, "protocol", f"{args.split}_image_ids")
    )
    image_dir = require_config_value(config, "data", "raw_train_image_dir")
    mask_dir = require_config_value(config, "data", "raw_train_mask_dir")
    tile_size = require_config_value(config, "evaluation", "tile_size")
    stride = require_config_value(config, "evaluation", "stride")
    threshold = (
        args.threshold
        if args.threshold is not None
        else require_config_value(config, "evaluation", "threshold")
    )
    min_area = (
        args.min_area
        if args.min_area is not None
        else require_config_value(config, "evaluation", "min_area")
    )
    open_kernel_size = (
        args.open_kernel_size
        if args.open_kernel_size is not None
        else require_config_value(config, "evaluation", "open_kernel_size")
    )

    suffix = (
        f"{args.split}_thr{int(round(threshold * 100)):03d}_"
        f"area{int(min_area):04d}_open{int(open_kernel_size)}"
    )
    out_csv = args.out_csv or (
        f"results/tables/douglas_peucker_simplification_curve_{suffix}.csv"
    )
    out_png = None
    if not args.pdf_only:
        out_png = args.out_png or (
            f"results/figures/douglas_peucker_simplification_curve_{suffix}.png"
        )
    out_pdf = args.out_pdf or (
        f"results/figures/douglas_peucker_simplification_curve_{suffix}.pdf"
    )

    disk_cache_dir = None
    if not args.no_disk_cache:
        disk_cache_dir = resolve_prediction_cache_dir(
            base_dir=args.cache_dir,
            run_name=run_name,
            split=args.split,
            tile_size=tile_size,
            stride=stride,
        )

    print("Device:", DEVICE)
    print("Experiment:", run_name)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Model path:", model_path)
    print("Split:", args.split)
    print("Image ids:", describe_image_ids(image_ids))
    print("Threshold:", threshold)
    print("Min area:", min_area)
    print("Open kernel size:", open_kernel_size)
    print("Epsilon ratios:", ", ".join(f"{item:g}" for item in epsilon_ratios))
    print("Prediction cache dir:", disk_cache_dir or "disabled")

    def load_configured_model():
        return load_model(
            model_path=model_path,
            architecture=architecture,
            encoder=encoder,
            device=DEVICE,
        )

    cached_images = collect_cached_images(
        model_loader=load_configured_model,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=image_ids,
        tile_size=tile_size,
        stride=stride,
        disk_cache_dir=disk_cache_dir,
    )

    image_records = []
    for _, prob_map, target_mask in tqdm(cached_images, desc="Preparing contours"):
        raw_mask = prob_map >= threshold
        post_mask = postprocess_mask_fast(
            raw_mask.astype(np.uint8),
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        )
        contours = mask_to_contours(post_mask, min_area=0)
        _, _, original_total_vertices = summarize_vertices(vertex_counts(contours))
        image_records.append({
            "post_mask": post_mask.astype(bool),
            "target_mask": target_mask,
            "contours": contours,
            "post_area": int(post_mask.sum()),
            "original_total_vertices": original_total_vertices,
        })

    rows = compute_curve_rows(
        image_records=image_records,
        epsilon_ratios=epsilon_ratios,
        run_name=run_name,
        split=args.split,
        image_ids=image_ids,
        threshold=threshold,
        min_area=min_area,
        open_kernel_size=open_kernel_size,
    )
    write_csv(out_csv, rows)
    plot_curve(rows, out_png=out_png, out_pdf=out_pdf)
    print_summary(rows)

    print()
    print("Curve CSV:", out_csv)
    if out_png is not None:
        print("Curve PNG:", out_png)
    print("Curve PDF:", out_pdf)


if __name__ == "__main__":
    main()
