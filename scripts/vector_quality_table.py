"""
Generate basic vector-quality metrics for full-image building predictions.

The table complements raster segmentation metrics with GIS-oriented measures:
polygon counts, GT component counts, invalid polygon ratio, vertex counts,
predicted-vs-GT area, boundary F1 before/after post-processing, and IoU after
rasterizing simplified polygons back to the image grid.
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
from dataset import (  # noqa: E402
    INRIA_PUBLIC_CITIES,
    collect_image_mask_pairs,
    describe_image_ids,
    image_id_list,
    parse_inria_name,
)
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled  # noqa: E402
from metrics import (  # noqa: E402
    boundary_metrics_multi,
    confusion_from_masks,
    metrics_from_confusion,
)
from vectorize import mask_to_contours, simplify_contours  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "protocol",
    "split",
    "city",
    "image_ids",
    "n_images",
    "threshold",
    "tile_size",
    "stride",
    "min_area",
    "open_kernel_size",
    "epsilon_ratio",
    "pred_polygons",
    "gt_components",
    "polygon_to_gt_component_ratio",
    "invalid_polygons",
    "invalid_polygon_ratio",
    "mean_vertices",
    "median_vertices",
    "pred_area_px",
    "gt_area_px",
    "pred_to_gt_area_ratio",
    "boundary_f1_2px_raw",
    "boundary_f1_2px_post",
    "boundary_f1_2px_delta",
    "boundary_f1_5px_raw",
    "boundary_f1_5px_post",
    "boundary_f1_5px_delta",
    "polygon_raster_iou",
    "polygon_raster_dice",
    "polygon_raster_precision",
    "polygon_raster_recall",
    "polygon_raster_accuracy",
    "polygon_raster_tp",
    "polygon_raster_fp",
    "polygon_raster_fn",
    "polygon_raster_tn",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate basic vector-quality metrics for INRIA full images.",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--open_kernel_size", type=int, default=None)
    parser.add_argument("--epsilon_ratio", type=float, default=0.002)
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
        help="Output CSV path. Defaults to results/tables/vector_quality_<split>.csv.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional one-row ALL summary CSV path.",
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


def count_components(mask):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    return int(max(0, num_labels - 1))


def contour_to_polygon(contour):
    points = contour[:, 0, :]
    if len(points) < 3:
        return None

    coords = [(float(point[0]), float(point[1])) for point in points]
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return Polygon(coords)


def polygon_stats(mask, epsilon_ratio):
    contours = mask_to_contours(mask, min_area=0)
    polygons = simplify_contours(contours, epsilon_ratio=epsilon_ratio)
    vertex_counts = []
    invalid_polygons = 0
    polygon_area_px = 0.0
    rasterized = np.zeros(mask.shape, dtype=np.uint8)

    if polygons:
        cv2.drawContours(rasterized, polygons, -1, color=1, thickness=-1)

    for contour in polygons:
        vertex_counts.append(int(len(contour)))
        polygon_area_px += float(cv2.contourArea(contour))
        polygon = contour_to_polygon(contour)
        if polygon is None or not polygon.is_valid:
            invalid_polygons += 1

    return {
        "n_polygons": int(len(polygons)),
        "invalid_polygons": int(invalid_polygons),
        "vertex_counts": vertex_counts,
        "polygon_area_px": polygon_area_px,
        "rasterized": rasterized.astype(bool),
    }


def new_accumulator():
    return {
        "n_images": 0,
        "pred_polygons": 0,
        "gt_components": 0,
        "invalid_polygons": 0,
        "vertex_counts": [],
        "pred_area_px": 0,
        "gt_area_px": 0,
        "boundary_f1_2px_raw_sum": 0.0,
        "boundary_f1_2px_post_sum": 0.0,
        "boundary_f1_5px_raw_sum": 0.0,
        "boundary_f1_5px_post_sum": 0.0,
        "polygon_raster_tp": 0,
        "polygon_raster_fp": 0,
        "polygon_raster_fn": 0,
        "polygon_raster_tn": 0,
    }


def build_image_metrics(raw_mask, post_mask, target_mask, epsilon_ratio):
    raw_boundary = boundary_metrics_multi(raw_mask, target_mask, tolerances=(2, 5))
    post_boundary = boundary_metrics_multi(post_mask, target_mask, tolerances=(2, 5))
    vector_stats = polygon_stats(post_mask, epsilon_ratio=epsilon_ratio)
    tp, fp, fn, tn = confusion_from_masks(
        vector_stats["rasterized"],
        target_mask,
    )

    return {
        "pred_polygons": vector_stats["n_polygons"],
        "gt_components": count_components(target_mask),
        "invalid_polygons": vector_stats["invalid_polygons"],
        "vertex_counts": vector_stats["vertex_counts"],
        "pred_area_px": int(post_mask.sum()),
        "gt_area_px": int(target_mask.sum()),
        "boundary_f1_2px_raw": raw_boundary["boundary_f1_2px"],
        "boundary_f1_2px_post": post_boundary["boundary_f1_2px"],
        "boundary_f1_5px_raw": raw_boundary["boundary_f1_5px"],
        "boundary_f1_5px_post": post_boundary["boundary_f1_5px"],
        "polygon_raster_tp": tp,
        "polygon_raster_fp": fp,
        "polygon_raster_fn": fn,
        "polygon_raster_tn": tn,
    }


def update_accumulator(accumulator, image_metrics):
    accumulator["n_images"] += 1
    accumulator["pred_polygons"] += image_metrics["pred_polygons"]
    accumulator["gt_components"] += image_metrics["gt_components"]
    accumulator["invalid_polygons"] += image_metrics["invalid_polygons"]
    accumulator["vertex_counts"].extend(image_metrics["vertex_counts"])
    accumulator["pred_area_px"] += image_metrics["pred_area_px"]
    accumulator["gt_area_px"] += image_metrics["gt_area_px"]
    accumulator["boundary_f1_2px_raw_sum"] += image_metrics["boundary_f1_2px_raw"]
    accumulator["boundary_f1_2px_post_sum"] += image_metrics["boundary_f1_2px_post"]
    accumulator["boundary_f1_5px_raw_sum"] += image_metrics["boundary_f1_5px_raw"]
    accumulator["boundary_f1_5px_post_sum"] += image_metrics["boundary_f1_5px_post"]
    accumulator["polygon_raster_tp"] += image_metrics["polygon_raster_tp"]
    accumulator["polygon_raster_fp"] += image_metrics["polygon_raster_fp"]
    accumulator["polygon_raster_fn"] += image_metrics["polygon_raster_fn"]
    accumulator["polygon_raster_tn"] += image_metrics["polygon_raster_tn"]


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def finalize_accumulator(
    accumulator,
    run_name,
    architecture,
    encoder,
    protocol,
    split,
    city,
    image_ids,
    threshold,
    tile_size,
    stride,
    min_area,
    open_kernel_size,
    epsilon_ratio,
):
    n_images = accumulator["n_images"]
    if n_images == 0:
        raise RuntimeError(f"Cannot finalize empty city group: {city}")

    vertex_counts = accumulator["vertex_counts"]
    if vertex_counts:
        mean_vertices = float(np.mean(vertex_counts))
        median_vertices = float(np.median(vertex_counts))
    else:
        mean_vertices = 0.0
        median_vertices = 0.0

    polygon_metrics = metrics_from_confusion(
        accumulator["polygon_raster_tp"],
        accumulator["polygon_raster_fp"],
        accumulator["polygon_raster_fn"],
        accumulator["polygon_raster_tn"],
    )
    boundary_f1_2px_raw = accumulator["boundary_f1_2px_raw_sum"] / n_images
    boundary_f1_2px_post = accumulator["boundary_f1_2px_post_sum"] / n_images
    boundary_f1_5px_raw = accumulator["boundary_f1_5px_raw_sum"] / n_images
    boundary_f1_5px_post = accumulator["boundary_f1_5px_post_sum"] / n_images

    row = {
        "run_name": run_name,
        "architecture": architecture,
        "encoder": encoder,
        "protocol": protocol,
        "split": split,
        "city": city,
        "image_ids": describe_image_ids(image_ids),
        "n_images": n_images,
        "threshold": round(float(threshold), 2),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "min_area": int(min_area),
        "open_kernel_size": int(open_kernel_size),
        "epsilon_ratio": round(float(epsilon_ratio), 6),
        "pred_polygons": int(accumulator["pred_polygons"]),
        "gt_components": int(accumulator["gt_components"]),
        "polygon_to_gt_component_ratio": round(
            safe_ratio(accumulator["pred_polygons"], accumulator["gt_components"]),
            6,
        ),
        "invalid_polygons": int(accumulator["invalid_polygons"]),
        "invalid_polygon_ratio": round(
            safe_ratio(accumulator["invalid_polygons"], accumulator["pred_polygons"]),
            6,
        ),
        "mean_vertices": round(mean_vertices, 6),
        "median_vertices": round(median_vertices, 6),
        "pred_area_px": int(accumulator["pred_area_px"]),
        "gt_area_px": int(accumulator["gt_area_px"]),
        "pred_to_gt_area_ratio": round(
            safe_ratio(accumulator["pred_area_px"], accumulator["gt_area_px"]),
            6,
        ),
        "boundary_f1_2px_raw": round(boundary_f1_2px_raw, 6),
        "boundary_f1_2px_post": round(boundary_f1_2px_post, 6),
        "boundary_f1_2px_delta": round(
            boundary_f1_2px_post - boundary_f1_2px_raw,
            6,
        ),
        "boundary_f1_5px_raw": round(boundary_f1_5px_raw, 6),
        "boundary_f1_5px_post": round(boundary_f1_5px_post, 6),
        "boundary_f1_5px_delta": round(
            boundary_f1_5px_post - boundary_f1_5px_raw,
            6,
        ),
        "polygon_raster_iou": round(polygon_metrics["iou_building"], 6),
        "polygon_raster_dice": round(polygon_metrics["dice_f1"], 6),
        "polygon_raster_precision": round(polygon_metrics["precision"], 6),
        "polygon_raster_recall": round(polygon_metrics["recall"], 6),
        "polygon_raster_accuracy": round(polygon_metrics["accuracy"], 6),
        "polygon_raster_tp": int(polygon_metrics["tp"]),
        "polygon_raster_fp": int(polygon_metrics["fp"]),
        "polygon_raster_fn": int(polygon_metrics["fn"]),
        "polygon_raster_tn": int(polygon_metrics["tn"]),
    }

    return row


def print_summary(rows):
    print()
    print("Vector quality summary")
    print(
        f"{'City':<10} {'PredPoly':>9} {'GTComp':>8} {'Inv%':>7} "
        f"{'VtxMean':>8} {'AreaRatio':>9} {'BF1@2 raw':>10} "
        f"{'BF1@2 post':>11} {'PolyIoU':>8}"
    )
    for row in rows:
        print(
            f"{row['city']:<10} "
            f"{row['pred_polygons']:>9} "
            f"{row['gt_components']:>8} "
            f"{100 * row['invalid_polygon_ratio']:>6.2f}% "
            f"{row['mean_vertices']:>8.2f} "
            f"{row['pred_to_gt_area_ratio']:>9.3f} "
            f"{row['boundary_f1_2px_raw']:>10.4f} "
            f"{row['boundary_f1_2px_post']:>11.4f} "
            f"{row['polygon_raster_iou']:>8.4f}"
        )


def write_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)

    run_name = require_config_value(config, "training", "run_name")
    architecture = require_config_value(config, "model", "architecture")
    encoder = require_config_value(config, "model", "encoder")
    model_dir = require_config_value(config, "model", "model_dir")
    model_path = resolve_model_path(model_dir, run_name)
    protocol = require_config_value(config, "protocol", "name")
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
    out_csv = args.out_csv or (
        f"results/tables/vector_quality_{args.split}_"
        f"thr{int(round(threshold * 100)):03d}_area{int(min_area):04d}_"
        f"open{int(open_kernel_size)}.csv"
    )
    summary_csv = args.summary_csv or (
        f"results/tables/vector_quality_{args.split}_"
        f"thr{int(round(threshold * 100)):03d}_area{int(min_area):04d}_"
        f"open{int(open_kernel_size)}_summary.csv"
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
    print("Protocol:", protocol)
    print("Split:", args.split)
    print("Image ids:", describe_image_ids(image_ids))
    print("Threshold:", threshold)
    print("Min area:", min_area)
    print("Open kernel size:", open_kernel_size)
    print("Epsilon ratio:", args.epsilon_ratio)
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

    city_accumulators = {
        city: new_accumulator()
        for city in INRIA_PUBLIC_CITIES
    }
    all_accumulator = new_accumulator()

    for image_path, prob_map, target_mask in tqdm(
        cached_images,
        desc="Computing vector metrics",
    ):
        city, _ = parse_inria_name(image_path)
        raw_mask = prob_map >= threshold
        post_mask = postprocess_mask_fast(
            raw_mask.astype(np.uint8),
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        )
        image_metrics = build_image_metrics(
            raw_mask=raw_mask,
            post_mask=post_mask,
            target_mask=target_mask,
            epsilon_ratio=args.epsilon_ratio,
        )
        update_accumulator(city_accumulators[city], image_metrics)
        update_accumulator(all_accumulator, image_metrics)

    rows = [
        finalize_accumulator(
            city_accumulators[city],
            run_name=run_name,
            architecture=architecture,
            encoder=encoder,
            protocol=protocol,
            split=args.split,
            city=city,
            image_ids=image_ids,
            threshold=threshold,
            tile_size=tile_size,
            stride=stride,
            min_area=min_area,
            open_kernel_size=open_kernel_size,
            epsilon_ratio=args.epsilon_ratio,
        )
        for city in INRIA_PUBLIC_CITIES
    ]
    rows.append(
        finalize_accumulator(
            all_accumulator,
            run_name=run_name,
            architecture=architecture,
            encoder=encoder,
            protocol=protocol,
            split=args.split,
            city="ALL",
            image_ids=image_ids,
            threshold=threshold,
            tile_size=tile_size,
            stride=stride,
            min_area=min_area,
            open_kernel_size=open_kernel_size,
            epsilon_ratio=args.epsilon_ratio,
        )
    )

    write_csv(out_csv, rows)
    write_csv(summary_csv, [rows[-1]])
    print_summary(rows)
    print()
    print("Vector quality CSV:", out_csv)
    print("Vector quality summary CSV:", summary_csv)


if __name__ == "__main__":
    main()
