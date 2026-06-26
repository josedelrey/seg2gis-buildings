"""
Generate a post-processing ablation table for the final segmentation pipeline.

This script is intended for validation-set ablation of thresholding, connected
component filtering, and morphological opening. Keep `--split val` as the
default for thesis tuning; use `--split test` only for final reporting, not for
choosing post-processing settings.
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
from shapely.geometry import Polygon
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path  # noqa: E402
from dataset import collect_image_mask_pairs, describe_image_ids, image_id_list  # noqa: E402
from evaluate import finalize_accumulator, new_metric_accumulator, update_accumulator  # noqa: E402
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled  # noqa: E402
from postprocess import postprocess_mask  # noqa: E402
from vectorize import mask_to_contours, simplify_contours  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_THRESHOLDS = "0.43,0.45,0.47,0.50,0.53"
DEFAULT_MIN_AREAS = "0,100,500,1000"
DEFAULT_OPEN_KERNEL_SIZES = "0,3,5"

FULL_CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "protocol",
    "split",
    "image_ids",
    "n_images",
    "tile_size",
    "stride",
    "postprocess_name",
    "threshold",
    "min_area",
    "open_kernel_size",
    "epsilon_ratio",
    "iou_building",
    "dice_f1",
    "precision",
    "recall",
    "accuracy",
    "boundary_f1_2px",
    "boundary_f1_5px",
    "n_polygons",
    "invalid_polygons",
    "invalid_polygon_ratio",
    "mean_vertices",
    "median_vertices",
    "total_polygon_area_px",
    "tp",
    "fp",
    "fn",
    "tn",
]

SUMMARY_CSV_HEADER = ["selection_reason", "balanced_score"] + FULL_CSV_HEADER

NAMED_CONFIGS = {
    (0.50, 0, 0): "raw",
    (0.50, 500, 0): "area_filter",
    (0.50, 500, 5): "selected",
    (0.43, 500, 5): "threshold_tuned",
}

NAMED_REASONS = {
    "raw": "named_raw",
    "area_filter": "named_area_filter",
    "selected": "named_selected",
    "threshold_tuned": "named_threshold_tuned",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a validation/test post-processing ablation table for a "
            "trained full-image building segmentation model."
        ),
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/tables/postprocess_ablation_validation.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/tables/postprocess_ablation_validation_summary.csv",
    )
    parser.add_argument("--epsilon_ratio", type=float, default=0.002)
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--min_areas", type=str, default=DEFAULT_MIN_AREAS)
    parser.add_argument(
        "--open_kernel_sizes",
        type=str,
        default=DEFAULT_OPEN_KERNEL_SIZES,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results/cache/postprocess_ablation",
        help=(
            "Directory for reusable full-image probability-map caches. A "
            "run/split/tile/stride subdirectory is created automatically."
        ),
    )
    parser.add_argument(
        "--no_disk_cache",
        action="store_true",
        help="Disable reading and writing cached full-image probability maps.",
    )
    return parser.parse_args()


def require_config_value(config, *keys):
    value = get_config_value(config, *keys)

    if value is None:
        dotted_key = ".".join(keys)
        raise ValueError(f"Missing required config value: {dotted_key}")

    return value


def parse_float_list(value, option_name):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one float.")

    return [round(float(item), 2) for item in items]


def parse_int_list(value, option_name):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one integer.")

    return [int(item) for item in items]


def get_postprocess_name(threshold, min_area, open_kernel_size):
    key = (round(float(threshold), 2), int(min_area), int(open_kernel_size))
    if key in NAMED_CONFIGS:
        return NAMED_CONFIGS[key]

    threshold_label = int(round(float(threshold) * 100))
    return f"thr{threshold_label:03d}_area{int(min_area):04d}_open{int(open_kernel_size)}"


def get_split_image_ids(config, split):
    return image_id_list(
        require_config_value(config, "protocol", f"{split}_image_ids")
    )


def get_full_image_dirs(config, split):
    # The INRIA public holdout protocol keeps labels for image ids 1-36 under
    # the training imagery directory; the official unlabeled test directory is
    # not useful for metric computation.
    image_dir = require_config_value(config, "data", "raw_train_image_dir")
    mask_dir = require_config_value(config, "data", "raw_train_mask_dir")

    if split == "test":
        return image_dir, mask_dir

    return image_dir, mask_dir


def safe_path_part(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unnamed"


def resolve_prediction_cache_dir(base_dir, run_name, split, tile_size, stride):
    if base_dir is None:
        return None

    cache_name = (
        f"{safe_path_part(run_name)}_{safe_path_part(split)}"
        f"_tile{int(tile_size)}_stride{int(stride)}"
    )
    return REPO_ROOT / base_dir / cache_name


def cache_file_for_image(cache_dir, image_path):
    image_name = safe_path_part(Path(image_path).stem)
    return cache_dir / f"{image_name}.npz"


def load_cached_prediction(cache_path, image_path, mask_path):
    with np.load(cache_path) as data:
        prob_map = data["prob_map"].astype(np.float32, copy=False)
        target_mask = data["target_mask"].astype(bool, copy=False)

    if prob_map.shape != target_mask.shape:
        raise RuntimeError(
            f"Cached prediction/target shape mismatch for {image_path}: "
            f"{prob_map.shape} vs {target_mask.shape}"
        )

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "prob_map": prob_map,
        "target_mask": target_mask,
        "cache_status": "hit",
    }


def save_cached_prediction(cache_path, image_path, mask_path, prob_map, target_mask):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_path=str(image_path),
        mask_path=str(mask_path),
        prob_map=prob_map.astype(np.float32, copy=False),
        target_mask=target_mask.astype(np.uint8, copy=False),
    )


def cache_full_image_predictions(
    model_loader,
    image_dir,
    mask_dir,
    image_ids,
    tile_size,
    stride,
    disk_cache_dir=None,
):
    pairs = collect_image_mask_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=image_ids,
    )
    cached_images = []
    model = None
    cache_hits = 0
    cache_misses = 0

    for image_path, mask_path in tqdm(pairs, desc="Caching full-image predictions"):
        cache_path = None
        if disk_cache_dir is not None:
            cache_path = cache_file_for_image(disk_cache_dir, image_path)
            if cache_path.exists():
                cached_images.append(
                    load_cached_prediction(cache_path, image_path, mask_path)
                )
                cache_hits += 1
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

        if prob_map.shape != target_mask.shape:
            raise RuntimeError(
                f"Prediction/target shape mismatch for {image_path}: "
                f"{prob_map.shape} vs {target_mask.shape}"
            )

        if cache_path is not None:
            save_cached_prediction(
                cache_path=cache_path,
                image_path=image_path,
                mask_path=mask_path,
                prob_map=prob_map,
                target_mask=target_mask,
            )

        cached_images.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "prob_map": prob_map,
            "target_mask": target_mask,
            "cache_status": "miss",
        })
        cache_misses += 1

    if disk_cache_dir is not None:
        print(
            "Prediction cache:",
            f"{cache_hits} hit(s), {cache_misses} miss(es)",
            f"at {disk_cache_dir}",
        )

    return cached_images


def contour_to_polygon(contour):
    points = contour[:, 0, :]
    if len(points) < 3:
        return None

    coords = [(float(point[0]), float(point[1])) for point in points]
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return Polygon(coords)


def vector_stats_for_mask(mask, epsilon_ratio):
    contours = mask_to_contours(mask, min_area=0)
    polygons = simplify_contours(contours, epsilon_ratio=epsilon_ratio)

    n_polygons = len(polygons)
    invalid_polygons = 0
    vertex_counts = []
    total_area = 0.0

    for contour in polygons:
        vertex_counts.append(int(len(contour)))
        total_area += float(cv2.contourArea(contour))

        polygon = contour_to_polygon(contour)
        if polygon is None or not polygon.is_valid:
            invalid_polygons += 1

    if vertex_counts:
        mean_vertices = float(np.mean(vertex_counts))
        median_vertices = float(np.median(vertex_counts))
    else:
        mean_vertices = 0.0
        median_vertices = 0.0

    invalid_polygon_ratio = (
        float(invalid_polygons / n_polygons)
        if n_polygons > 0
        else 0.0
    )

    return {
        "n_polygons": int(n_polygons),
        "invalid_polygons": int(invalid_polygons),
        "invalid_polygon_ratio": invalid_polygon_ratio,
        "mean_vertices": mean_vertices,
        "median_vertices": median_vertices,
        "total_polygon_area_px": total_area,
        "vertex_counts": vertex_counts,
    }


def evaluate_combination(
    cached_images,
    threshold,
    min_area,
    open_kernel_size,
    epsilon_ratio,
    desc=None,
):
    accumulator = new_metric_accumulator()
    vector_totals = {
        "n_polygons": 0,
        "invalid_polygons": 0,
        "total_polygon_area_px": 0.0,
    }
    vertex_counts = []

    image_iterator = tqdm(cached_images, desc=desc, unit="image")

    for cached_image in image_iterator:
        pred_mask = cached_image["prob_map"] >= threshold
        pred_mask = postprocess_mask(
            pred_mask.astype(np.uint8),
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        ).astype(bool)

        update_accumulator(
            accumulator=accumulator,
            pred_mask=pred_mask,
            target_mask=cached_image["target_mask"],
        )

        stats = vector_stats_for_mask(
            mask=pred_mask.astype(np.uint8),
            epsilon_ratio=epsilon_ratio,
        )
        vector_totals["n_polygons"] += stats["n_polygons"]
        vector_totals["invalid_polygons"] += stats["invalid_polygons"]
        vector_totals["total_polygon_area_px"] += stats["total_polygon_area_px"]
        vertex_counts.extend(stats["vertex_counts"])

    metrics = finalize_accumulator(accumulator, "ALL")
    n_polygons = vector_totals["n_polygons"]
    invalid_polygons = vector_totals["invalid_polygons"]

    if vertex_counts:
        mean_vertices = float(np.mean(vertex_counts))
        median_vertices = float(np.median(vertex_counts))
    else:
        mean_vertices = 0.0
        median_vertices = 0.0

    invalid_polygon_ratio = (
        float(invalid_polygons / n_polygons)
        if n_polygons > 0
        else 0.0
    )

    metrics.update({
        "n_polygons": int(n_polygons),
        "invalid_polygons": int(invalid_polygons),
        "invalid_polygon_ratio": invalid_polygon_ratio,
        "mean_vertices": mean_vertices,
        "median_vertices": median_vertices,
        "total_polygon_area_px": vector_totals["total_polygon_area_px"],
    })
    return metrics


def round_row(row):
    rounded = dict(row)
    for key in (
        "threshold",
        "epsilon_ratio",
        "iou_building",
        "dice_f1",
        "precision",
        "recall",
        "accuracy",
        "boundary_f1_2px",
        "boundary_f1_5px",
        "invalid_polygon_ratio",
        "mean_vertices",
        "median_vertices",
        "total_polygon_area_px",
    ):
        rounded[key] = round(float(rounded[key]), 6)

    return rounded


def write_csv(path, rows, fieldnames):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def balanced_score(row):
    return (
        float(row["dice_f1"])
        + 0.5 * float(row["boundary_f1_2px"])
        - 0.00001 * int(row["n_polygons"])
    )


def build_summary_rows(rows):
    selected = {}

    def add_reason(row, reason):
        key = (
            row["threshold"],
            row["min_area"],
            row["open_kernel_size"],
        )
        if key not in selected:
            selected[key] = {
                "row": row,
                "reasons": [],
            }
        if reason not in selected[key]["reasons"]:
            selected[key]["reasons"].append(reason)

    for row in rows:
        reason = NAMED_REASONS.get(row["postprocess_name"])
        if reason is not None:
            add_reason(row, reason)

    add_reason(max(rows, key=lambda row: row["dice_f1"]), "best_dice")
    add_reason(max(rows, key=lambda row: row["iou_building"]), "best_iou")
    add_reason(
        max(rows, key=lambda row: row["boundary_f1_2px"]),
        "best_boundary_f1_2px",
    )
    add_reason(max(rows, key=balanced_score), "best_balanced_score")

    summary_rows = []
    for item in selected.values():
        row = dict(item["row"])
        row["selection_reason"] = ";".join(item["reasons"])
        row["balanced_score"] = round(balanced_score(item["row"]), 6)
        summary_rows.append(row)

    return sorted(
        summary_rows,
        key=lambda row: (
            "named_selected" not in row["selection_reason"],
            row["postprocess_name"],
        ),
    )


def print_terminal_table(rows):
    display_rows = build_summary_rows(rows)
    columns = [
        ("postprocess_name", 24),
        ("threshold", 9),
        ("min_area", 8),
        ("open_kernel_size", 16),
        ("iou_building", 12),
        ("dice_f1", 8),
        ("boundary_f1_2px", 16),
        ("boundary_f1_5px", 16),
        ("n_polygons", 10),
        ("invalid_polygon_ratio", 22),
        ("mean_vertices", 14),
    ]

    print()
    print("Post-processing ablation summary")
    print(" ".join(name[:width].ljust(width) for name, width in columns))
    print(" ".join("-" * width for _, width in columns))

    for row in display_rows:
        values = []
        for name, width in columns:
            value = row[name]
            if isinstance(value, float):
                if name == "threshold":
                    text = f"{value:.2f}"
                else:
                    text = f"{value:.4f}"
            else:
                text = str(value)
            values.append(text[:width].ljust(width))
        print(" ".join(values))


def main():
    args = parse_args()
    thresholds = parse_float_list(args.thresholds, "--thresholds")
    min_areas = parse_int_list(args.min_areas, "--min_areas")
    open_kernel_sizes = parse_int_list(
        args.open_kernel_sizes,
        "--open_kernel_sizes",
    )

    config = load_config(args.config)
    run_name = require_config_value(config, "training", "run_name")
    architecture = require_config_value(config, "model", "architecture")
    encoder = require_config_value(config, "model", "encoder")
    model_dir = require_config_value(config, "model", "model_dir")
    model_path = resolve_model_path(model_dir, run_name)
    protocol = require_config_value(config, "protocol", "name")
    image_ids = get_split_image_ids(config, args.split)
    image_dir, mask_dir = get_full_image_dirs(config, args.split)
    tile_size = require_config_value(config, "evaluation", "tile_size")
    stride = require_config_value(config, "evaluation", "stride")

    print("Device:", DEVICE)
    print("Experiment:", run_name)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Model path:", model_path)
    print("Protocol:", protocol)
    print("Split:", args.split)
    print("Image ids:", describe_image_ids(image_ids))
    print("Full-image source images:", image_dir)
    print("Full-image source masks:", mask_dir)
    print("Tile size:", tile_size)
    print("Stride:", stride)
    print("Thresholds:", ", ".join(f"{threshold:.2f}" for threshold in thresholds))
    print("Min areas:", ", ".join(str(item) for item in min_areas))
    print("Open kernel sizes:", ", ".join(str(item) for item in open_kernel_sizes))
    disk_cache_dir = None
    if not args.no_disk_cache:
        disk_cache_dir = resolve_prediction_cache_dir(
            base_dir=args.cache_dir,
            run_name=run_name,
            split=args.split,
            tile_size=tile_size,
            stride=stride,
        )

    print("Epsilon ratio:", args.epsilon_ratio)
    if disk_cache_dir is not None:
        print("Prediction cache dir:", disk_cache_dir)
    else:
        print("Prediction cache dir: disabled")

    def load_configured_model():
        return load_model(
            model_path=model_path,
            architecture=architecture,
            encoder=encoder,
            device=DEVICE,
        )

    cached_images = cache_full_image_predictions(
        model_loader=load_configured_model,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=image_ids,
        tile_size=tile_size,
        stride=stride,
        disk_cache_dir=disk_cache_dir,
    )

    rows = []
    combinations = [
        (threshold, min_area, open_kernel_size)
        for threshold in thresholds
        for min_area in min_areas
        for open_kernel_size in open_kernel_sizes
    ]

    for index, (threshold, min_area, open_kernel_size) in enumerate(
        combinations,
        start=1,
    ):
        postprocess_name = get_postprocess_name(
            threshold=threshold,
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        )
        config_desc = (
            f"{index}/{len(combinations)} {postprocess_name} "
            f"thr={threshold:.2f} area={min_area} open={open_kernel_size}"
        )
        print(f"Evaluating config {config_desc}")
        metrics = evaluate_combination(
            cached_images=cached_images,
            threshold=threshold,
            min_area=min_area,
            open_kernel_size=open_kernel_size,
            epsilon_ratio=args.epsilon_ratio,
            desc=config_desc,
        )
        row = {
            "run_name": run_name,
            "architecture": architecture,
            "encoder": encoder,
            "protocol": protocol,
            "split": args.split,
            "image_ids": describe_image_ids(image_ids),
            "n_images": len(cached_images),
            "tile_size": tile_size,
            "stride": stride,
            "postprocess_name": postprocess_name,
            "threshold": threshold,
            "min_area": min_area,
            "open_kernel_size": open_kernel_size,
            "epsilon_ratio": args.epsilon_ratio,
            "iou_building": metrics["iou_building"],
            "dice_f1": metrics["dice_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "boundary_f1_2px": metrics["boundary_f1_2px"],
            "boundary_f1_5px": metrics["boundary_f1_5px"],
            "n_polygons": metrics["n_polygons"],
            "invalid_polygons": metrics["invalid_polygons"],
            "invalid_polygon_ratio": metrics["invalid_polygon_ratio"],
            "mean_vertices": metrics["mean_vertices"],
            "median_vertices": metrics["median_vertices"],
            "total_polygon_area_px": metrics["total_polygon_area_px"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        }
        rows.append(round_row(row))

    summary_rows = build_summary_rows(rows)
    write_csv(args.out_csv, rows, FULL_CSV_HEADER)
    write_csv(args.summary_csv, summary_rows, SUMMARY_CSV_HEADER)
    print_terminal_table(rows)

    print()
    print("Full ablation CSV:", args.out_csv)
    print("Summary CSV:", args.summary_csv)


if __name__ == "__main__":
    main()
