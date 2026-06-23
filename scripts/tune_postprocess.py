import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path  # noqa: E402
from dataset import collect_image_mask_pairs, describe_image_ids, image_id_list  # noqa: E402
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled  # noqa: E402
from metrics import confusion_from_masks, metrics_from_confusion  # noqa: E402
from postprocess import postprocess_mask  # noqa: E402


try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is required to load the segmentation model.") from exc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "protocol",
    "split",
    "image_ids",
    "n_images",
    "tile_size",
    "stride",
    "selection_metric",
    "threshold",
    "min_area",
    "open_kernel_size",
    "iou_building",
    "dice_f1",
    "precision",
    "recall",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
    "is_best",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune postprocessing on the INRIA validation split only.",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out_csv", type=str, default="results/tables/postprocess_tuning_exploratory.csv")
    parser.add_argument("--thresholds", type=str, default="0.30:0.80:0.01")
    parser.add_argument("--min_areas", type=str, default="0,64,128,256,500,1000")
    parser.add_argument("--open_kernel_sizes", type=str, default="0,3,5")
    parser.add_argument(
        "--metric",
        type=str,
        default="iou_building",
        choices=["iou_building", "dice_f1"],
    )
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def require_config_value(config, *keys):
    value = get_config_value(config, *keys)

    if value is None:
        dotted_key = ".".join(keys)
        raise ValueError(f"Missing required config value: {dotted_key}")

    return value


def parse_thresholds(value):
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("--thresholds must use start:end:step format.")

    start, end, step = (float(part) for part in parts)
    if step <= 0:
        raise ValueError("--thresholds step must be greater than 0.")
    if end < start:
        raise ValueError("--thresholds end must be greater than or equal to start.")

    thresholds = []
    current = start
    while current <= end + (step / 2):
        thresholds.append(round(float(current), 2))
        current += step

    return thresholds


def parse_int_list(value, option_name):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one integer.")

    return [int(item) for item in items]


def cache_validation_predictions(
    model,
    pairs,
    tile_size,
    stride,
):
    cached_images = []

    for image_path, mask_path in tqdm(pairs, desc="Caching validation predictions"):
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

        cached_images.append({
            "image_path": image_path,
            "prob_map": prob_map,
            "target_mask": target_mask,
        })

    return cached_images


def evaluate_postprocess_combo(
    cached_images,
    threshold,
    min_area,
    open_kernel_size,
):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for cached_image in cached_images:
        pred_mask = cached_image["prob_map"] >= threshold
        pred_mask = postprocess_mask(
            pred_mask.astype(np.uint8),
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        ).astype(bool)
        target_mask = cached_image["target_mask"].astype(bool)

        tp, fp, fn, tn = confusion_from_masks(pred_mask, target_mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    return metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)


def best_sort_key(row, selection_metric):
    return (
        row[selection_metric],
        row["dice_f1"],
        row["iou_building"],
        -row["min_area"],
        -row["open_kernel_size"],
        -abs(row["threshold"] - 0.50),
    )


def write_results_csv(out_csv, rows):
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_row_for_csv(row):
    formatted = row.copy()

    for key in (
        "threshold",
        "iou_building",
        "dice_f1",
        "precision",
        "recall",
        "accuracy",
    ):
        formatted[key] = round(formatted[key], 4)

    formatted["is_best"] = "true" if formatted["is_best"] else "false"
    return formatted


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
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
    val_image_ids = image_id_list(
        require_config_value(config, "protocol", "val_image_ids")
    )
    image_dir = require_config_value(config, "data", "raw_train_image_dir")
    mask_dir = require_config_value(config, "data", "raw_train_mask_dir")
    tile_size = require_config_value(config, "evaluation", "tile_size")
    stride = require_config_value(config, "evaluation", "stride")

    pairs = collect_image_mask_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=val_image_ids,
    )
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max_images must be greater than 0 when provided.")
        pairs = pairs[:args.max_images]

    print("Device:", DEVICE)
    print("Experiment:", run_name)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Model path:", model_path)
    print("INRIA protocol:", protocol)
    print("Split: val")
    print("Protocol val image ids:", describe_image_ids(val_image_ids))
    print("Validation images used:", len(pairs))
    print("Full-image source images:", image_dir)
    print("Full-image source masks:", mask_dir)
    print("Tile size:", tile_size)
    print("Stride:", stride)
    print("Thresholds:", f"{thresholds[0]:.2f}..{thresholds[-1]:.2f}", f"({len(thresholds)})")
    print("Min areas:", ",".join(str(item) for item in min_areas))
    print("Open kernel sizes:", ",".join(str(item) for item in open_kernel_sizes))
    print("Selection metric:", args.metric)

    model = load_model(
        model_path=model_path,
        architecture=architecture,
        encoder=encoder,
        device=DEVICE,
    )

    cached_images = cache_validation_predictions(
        model=model,
        pairs=pairs,
        tile_size=tile_size,
        stride=stride,
    )

    rows = []
    total_combinations = len(thresholds) * len(min_areas) * len(open_kernel_sizes)
    combinations = (
        (threshold, min_area, open_kernel_size)
        for threshold in thresholds
        for min_area in min_areas
        for open_kernel_size in open_kernel_sizes
    )

    for threshold, min_area, open_kernel_size in tqdm(
        combinations,
        total=total_combinations,
        desc="Tuning postprocess",
    ):
        metrics = evaluate_postprocess_combo(
            cached_images=cached_images,
            threshold=threshold,
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        )
        rows.append({
            "run_name": run_name,
            "architecture": architecture,
            "encoder": encoder,
            "protocol": protocol,
            "split": "val",
            "image_ids": describe_image_ids(val_image_ids),
            "n_images": len(cached_images),
            "tile_size": tile_size,
            "stride": stride,
            "selection_metric": args.metric,
            "threshold": threshold,
            "min_area": min_area,
            "open_kernel_size": open_kernel_size,
            "iou_building": metrics["iou_building"],
            "dice_f1": metrics["dice_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "is_best": False,
        })

    best_row = max(rows, key=lambda row: best_sort_key(row, args.metric))
    best_row["is_best"] = True
    write_results_csv(
        args.out_csv,
        [format_row_for_csv(row) for row in rows],
    )

    suggested_config = {
        "evaluation": {
            "threshold": best_row["threshold"],
            "min_area": best_row["min_area"],
            "open_kernel_size": best_row["open_kernel_size"],
        }
    }

    print("\nBest validation postprocessing:")
    print(f"  metric: {args.metric}")
    print(f"  threshold: {best_row['threshold']:.2f}")
    print(f"  min_area: {best_row['min_area']}")
    print(f"  open_kernel_size: {best_row['open_kernel_size']}")
    print(f"  IoU: {best_row['iou_building']:.4f}")
    print(f"  Dice/F1: {best_row['dice_f1']:.4f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall: {best_row['recall']:.4f}")
    print(f"  Accuracy: {best_row['accuracy']:.4f}")
    print("\nSuggested config update:")
    print(json.dumps(suggested_config, indent=2))
    print(f"\nResults CSV:\n{args.out_csv}")


if __name__ == "__main__":
    main()
