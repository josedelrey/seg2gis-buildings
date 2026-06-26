"""
Compute component-level instance AP for full-image building predictions.

The INRIA labels are semantic masks, not official instance polygons. This script
therefore derives pseudo-instances from connected components in the GT and
post-processed prediction masks. Report the metric as component-level AP.
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IOU_THRESHOLDS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"

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
    "score_name",
    "n_gt_instances",
    "n_pred_instances",
    "ap50",
    "ap75",
    "map_50_95",
    "ar50",
    "ar75",
    "precision50",
    "precision75",
    "tp50",
    "fp50",
    "fn50",
    "tp75",
    "fp75",
    "fn75",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate component-level instance AP metrics.",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--open_kernel_size", type=int, default=None)
    parser.add_argument(
        "--iou_thresholds",
        type=str,
        default=DEFAULT_IOU_THRESHOLDS,
        help="Comma-separated IoU thresholds for mAP computation.",
    )
    parser.add_argument(
        "--score",
        choices=["mean_prob", "max_prob", "p95_prob"],
        default="mean_prob",
        help="Confidence score assigned to each predicted component.",
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
    return [round(float(item), 2) for item in items]


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

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
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


def component_scores(prob_map, labels, n_labels, score_name):
    scores = np.zeros(n_labels, dtype=np.float64)
    if n_labels <= 1:
        return scores

    flat_labels = labels.ravel()
    flat_prob = prob_map.ravel()
    areas = np.bincount(flat_labels, minlength=n_labels).astype(np.float64)

    if score_name == "mean_prob":
        sums = np.bincount(flat_labels, weights=flat_prob, minlength=n_labels)
        nonzero = areas > 0
        scores[nonzero] = sums[nonzero] / areas[nonzero]
        return scores

    for label in range(1, n_labels):
        values = flat_prob[flat_labels == label]
        if values.size == 0:
            continue
        if score_name == "max_prob":
            scores[label] = float(values.max())
        elif score_name == "p95_prob":
            scores[label] = float(np.percentile(values, 95))
        else:
            raise ValueError(f"Unsupported score: {score_name}")

    return scores


def extract_predictions_for_image(
    image_key,
    prob_map,
    target_mask,
    threshold,
    min_area,
    open_kernel_size,
    score_name,
):
    pred_mask = postprocess_mask_fast(
        (prob_map >= threshold).astype(np.uint8),
        min_area=min_area,
        open_kernel_size=open_kernel_size,
    )
    n_pred, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(
        pred_mask.astype(np.uint8),
        connectivity=8,
    )
    n_gt, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(
        target_mask.astype(np.uint8),
        connectivity=8,
    )
    gt_areas = gt_stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    pred_areas = pred_stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    scores = component_scores(
        prob_map=prob_map,
        labels=pred_labels,
        n_labels=n_pred,
        score_name=score_name,
    )

    predictions = []
    for pred_label in range(1, n_pred):
        pred_pixels = pred_labels == pred_label
        overlapping_gt = gt_labels[pred_pixels]
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if overlapping_gt.size:
            overlap_counts = np.bincount(overlapping_gt, minlength=n_gt)
            gt_ids = np.flatnonzero(overlap_counts)
        else:
            overlap_counts = np.zeros(n_gt, dtype=np.int64)
            gt_ids = np.array([], dtype=np.int64)

        overlaps = []
        for gt_label in gt_ids:
            intersection = float(overlap_counts[gt_label])
            union = pred_areas[pred_label] + gt_areas[gt_label] - intersection
            if union <= 0:
                continue
            overlaps.append((int(gt_label), float(intersection / union)))
        overlaps.sort(key=lambda item: item[1], reverse=True)

        predictions.append({
            "image_key": image_key,
            "score": float(scores[pred_label]),
            "overlaps": overlaps,
        })

    return predictions, int(n_gt - 1)


def average_precision(recalls, precisions):
    if recalls.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap)


def evaluate_at_iou(predictions, n_gt, iou_threshold):
    if n_gt == 0:
        return {
            "ap": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "tp": 0,
            "fp": len(predictions),
            "fn": 0,
        }

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True,
    )
    matched_gt = set()
    tp_values = []
    fp_values = []

    for prediction in sorted_predictions:
        match_key = None
        for gt_label, iou in prediction["overlaps"]:
            candidate_key = (prediction["image_key"], gt_label)
            if iou < iou_threshold:
                break
            if candidate_key not in matched_gt:
                match_key = candidate_key
                break

        if match_key is None:
            tp_values.append(0)
            fp_values.append(1)
        else:
            matched_gt.add(match_key)
            tp_values.append(1)
            fp_values.append(0)

    if not sorted_predictions:
        return {
            "ap": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
        }

    tp_cum = np.cumsum(tp_values)
    fp_cum = np.cumsum(fp_values)
    recalls = tp_cum / float(n_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    tp = int(tp_cum[-1])
    fp = int(fp_cum[-1])
    fn = int(n_gt - tp)

    return {
        "ap": average_precision(recalls, precisions),
        "recall": float(recalls[-1]),
        "precision": float(precisions[-1]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize_group(
    group,
    iou_thresholds,
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
    score_name,
):
    metrics_by_threshold = {
        iou_threshold: evaluate_at_iou(
            group["predictions"],
            group["n_gt"],
            iou_threshold,
        )
        for iou_threshold in iou_thresholds
    }
    ap50 = metrics_by_threshold[0.50]["ap"]
    ap75 = metrics_by_threshold[0.75]["ap"]
    ar50 = metrics_by_threshold[0.50]["recall"]
    ar75 = metrics_by_threshold[0.75]["recall"]
    precision50 = metrics_by_threshold[0.50]["precision"]
    precision75 = metrics_by_threshold[0.75]["precision"]
    map_50_95 = float(np.mean([
        metrics_by_threshold[iou_threshold]["ap"]
        for iou_threshold in iou_thresholds
    ]))
    metrics50 = metrics_by_threshold[0.50]
    metrics75 = metrics_by_threshold[0.75]

    return {
        "run_name": run_name,
        "architecture": architecture,
        "encoder": encoder,
        "protocol": protocol,
        "split": split,
        "city": city,
        "image_ids": describe_image_ids(image_ids),
        "n_images": group["n_images"],
        "threshold": round(float(threshold), 2),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "min_area": int(min_area),
        "open_kernel_size": int(open_kernel_size),
        "score_name": score_name,
        "n_gt_instances": int(group["n_gt"]),
        "n_pred_instances": int(len(group["predictions"])),
        "ap50": round(ap50, 6),
        "ap75": round(ap75, 6),
        "map_50_95": round(map_50_95, 6),
        "ar50": round(ar50, 6),
        "ar75": round(ar75, 6),
        "precision50": round(precision50, 6),
        "precision75": round(precision75, 6),
        "tp50": metrics50["tp"],
        "fp50": metrics50["fp"],
        "fn50": metrics50["fn"],
        "tp75": metrics75["tp"],
        "fp75": metrics75["fp"],
        "fn75": metrics75["fn"],
    }


def new_group():
    return {
        "n_images": 0,
        "n_gt": 0,
        "predictions": [],
    }


def add_image_to_group(group, predictions, n_gt):
    group["n_images"] += 1
    group["n_gt"] += n_gt
    group["predictions"].extend(predictions)


def write_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print()
    print("Component-level instance AP summary")
    print(
        f"{'City':<10} {'GT':>7} {'Pred':>7} {'AP50':>7} "
        f"{'AP75':>7} {'mAP':>7} {'AR50':>7} {'Prec50':>8}"
    )
    for row in rows:
        print(
            f"{row['city']:<10} "
            f"{row['n_gt_instances']:>7} "
            f"{row['n_pred_instances']:>7} "
            f"{row['ap50']:>7.4f} "
            f"{row['ap75']:>7.4f} "
            f"{row['map_50_95']:>7.4f} "
            f"{row['ar50']:>7.4f} "
            f"{row['precision50']:>8.4f}"
        )


def main():
    args = parse_args()
    iou_thresholds = parse_float_list(args.iou_thresholds, "--iou_thresholds")
    required_thresholds = {0.50, 0.75}
    missing_thresholds = required_thresholds - set(iou_thresholds)
    if missing_thresholds:
        raise ValueError(
            "--iou_thresholds must include 0.50 and 0.75 for AP50/AP75 reporting."
        )

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
        f"results/tables/instance_ap_{args.split}_"
        f"thr{int(round(threshold * 100)):03d}_area{int(min_area):04d}_"
        f"open{int(open_kernel_size)}.csv"
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
    print("Score:", args.score)
    print("IoU thresholds:", ", ".join(f"{item:.2f}" for item in iou_thresholds))
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

    groups = {city: new_group() for city in INRIA_PUBLIC_CITIES}
    groups["ALL"] = new_group()

    for image_path, prob_map, target_mask in tqdm(
        cached_images,
        desc="Computing component AP inputs",
    ):
        city, image_id = parse_inria_name(image_path)
        image_key = f"{city}{image_id}"
        predictions, n_gt = extract_predictions_for_image(
            image_key=image_key,
            prob_map=prob_map,
            target_mask=target_mask,
            threshold=threshold,
            min_area=min_area,
            open_kernel_size=open_kernel_size,
            score_name=args.score,
        )
        add_image_to_group(groups[city], predictions, n_gt)
        add_image_to_group(groups["ALL"], predictions, n_gt)

    rows = [
        summarize_group(
            groups[city],
            iou_thresholds=iou_thresholds,
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
            score_name=args.score,
        )
        for city in INRIA_PUBLIC_CITIES
    ]
    rows.append(
        summarize_group(
            groups["ALL"],
            iou_thresholds=iou_thresholds,
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
            score_name=args.score,
        )
    )

    write_csv(out_csv, rows)
    print_summary(rows)
    print()
    print("Instance AP CSV:", out_csv)


if __name__ == "__main__":
    main()
