import argparse
import csv
import cv2
import numpy as np
import os
import torch
from tqdm import tqdm

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path
from dataset import (
    INRIA_PUBLIC_CITIES,
    collect_image_mask_pairs,
    describe_image_ids,
    image_id_list,
    parse_inria_name,
)
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled
from metrics import (
    boundary_metrics_multi,
    confusion_from_masks,
    metrics_from_confusion,
)
from postprocess import postprocess_mask


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_PATHS = {
    "val": "results/val_full_image_results.csv",
    "test": "outputs/test_results.csv",
}
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
    "iou_building",
    "dice_f1",
    "precision",
    "recall",
    "accuracy",
    "boundary_f1_2px",
    "boundary_iou_2px",
    "boundary_precision_2px",
    "boundary_recall_2px",
    "boundary_f1_5px",
    "boundary_iou_5px",
    "boundary_precision_5px",
    "boundary_recall_5px",
    "tp",
    "fp",
    "fn",
    "tn",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    return parser.parse_args()


def require_config_value(config, *keys):
    value = get_config_value(config, *keys)

    if value is None:
        dotted_key = ".".join(keys)
        raise ValueError(f"Missing required config value: {dotted_key}")

    return value


def new_metric_accumulator():
    return {
        "n_images": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "boundary_metric_sums": {
            "boundary_f1_2px": 0.0,
            "boundary_iou_2px": 0.0,
            "boundary_precision_2px": 0.0,
            "boundary_recall_2px": 0.0,
            "boundary_f1_5px": 0.0,
            "boundary_iou_5px": 0.0,
            "boundary_precision_5px": 0.0,
            "boundary_recall_5px": 0.0,
        },
    }


def update_accumulator(accumulator, pred_mask, target_mask):
    tp, fp, fn, tn = confusion_from_masks(pred_mask, target_mask)
    accumulator["tp"] += tp
    accumulator["fp"] += fp
    accumulator["fn"] += fn
    accumulator["tn"] += tn

    image_boundary_metrics = boundary_metrics_multi(
        pred_mask,
        target_mask,
        tolerances=(2, 5),
    )
    for key, value in image_boundary_metrics.items():
        accumulator["boundary_metric_sums"][key] += value

    accumulator["n_images"] += 1


def finalize_accumulator(accumulator, group_name):
    n_images = accumulator["n_images"]
    if n_images == 0:
        raise RuntimeError(
            f"Cannot finalize metrics for empty image group: {group_name}"
        )

    metrics = metrics_from_confusion(
        accumulator["tp"],
        accumulator["fp"],
        accumulator["fn"],
        accumulator["tn"],
    )
    metrics["n_images"] = n_images

    # Keep standard metrics pixel-aggregated, but average boundary metrics per
    # image so large or building-dense images do not dominate boundary quality.
    for key, value in accumulator["boundary_metric_sums"].items():
        metrics[key] = float(value / n_images)

    return metrics


def evaluate_full_images(
    model,
    image_dir,
    mask_dir,
    image_ids,
    threshold,
    tile_size,
    stride,
    min_area,
    open_kernel_size,
):
    pairs = collect_image_mask_pairs(image_dir, mask_dir, image_ids)

    city_accumulators = {
        city: new_metric_accumulator()
        for city in INRIA_PUBLIC_CITIES
    }
    all_accumulator = new_metric_accumulator()

    model.eval()

    for image_path, mask_path in tqdm(pairs, desc="INRIA full-image evaluation"):
        city, _ = parse_inria_name(image_path)
        if city not in city_accumulators:
            raise RuntimeError(f"Unexpected INRIA city '{city}' in {image_path}")

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
        pred_mask = prob_map >= threshold
        pred_mask = postprocess_mask(
            pred_mask.astype(np.uint8),
            min_area=min_area,
            open_kernel_size=open_kernel_size,
        ).astype(bool)

        if pred_mask.shape != target_mask.shape:
            raise RuntimeError(
                f"Prediction/target shape mismatch for {image_path}: "
                f"{pred_mask.shape} vs {target_mask.shape}"
            )

        update_accumulator(city_accumulators[city], pred_mask, target_mask)
        update_accumulator(all_accumulator, pred_mask, target_mask)

    metrics_by_city = {
        city: finalize_accumulator(city_accumulators[city], city)
        for city in INRIA_PUBLIC_CITIES
    }
    metrics_by_city["ALL"] = finalize_accumulator(all_accumulator, "ALL")

    return metrics_by_city


def log_evaluation(
    log_path,
    run_name,
    architecture,
    encoder,
    protocol,
    split,
    image_ids,
    threshold,
    tile_size,
    stride,
    min_area,
    open_kernel_size,
    metrics_by_city,
):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_exists = os.path.exists(log_path)

    if file_exists:
        with open(log_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != CSV_HEADER:
            raise ValueError(
                f"CSV header mismatch in {log_path}.\n"
                f"Expected: {CSV_HEADER}\n"
                f"Found:    {existing_header}\n"
                "Delete or rename the old results CSV because the evaluation "
                "schema changed."
            )

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        for city in list(INRIA_PUBLIC_CITIES) + ["ALL"]:
            metrics = metrics_by_city[city]
            writer.writerow([
                run_name,
                architecture,
                encoder,
                protocol,
                split,
                city,
                describe_image_ids(image_ids),
                metrics["n_images"],
                round(threshold, 2),
                tile_size,
                stride,
                min_area,
                open_kernel_size,
                round(metrics["iou_building"], 4),
                round(metrics["dice_f1"], 4),
                round(metrics["precision"], 4),
                round(metrics["recall"], 4),
                round(metrics["accuracy"], 4),
                round(metrics["boundary_f1_2px"], 4),
                round(metrics["boundary_iou_2px"], 4),
                round(metrics["boundary_precision_2px"], 4),
                round(metrics["boundary_recall_2px"], 4),
                round(metrics["boundary_f1_5px"], 4),
                round(metrics["boundary_iou_5px"], 4),
                round(metrics["boundary_precision_5px"], 4),
                round(metrics["boundary_recall_5px"], 4),
                metrics["tp"],
                metrics["fp"],
                metrics["fn"],
                metrics["tn"],
            ])


def print_evaluation_summary(
    split,
    threshold,
    tile_size,
    stride,
    min_area,
    open_kernel_size,
    metrics_by_city,
):
    print(f"\nINRIA {split} full-image evaluation")
    print(
        f"threshold: {threshold:.2f} | "
        f"tile_size: {tile_size} | "
        f"stride: {stride} | "
        f"min_area: {min_area} | "
        f"open_kernel_size: {open_kernel_size}"
    )
    print()
    print(
        f"{'City':<10} {'Images':>6} {'IoU':>7} {'Dice':>7} "
        f"{'Prec':>7} {'Rec':>7} {'Acc':>7} {'BF1@2':>7} {'BF1@5':>7}"
    )

    for city in list(INRIA_PUBLIC_CITIES) + ["ALL"]:
        metrics = metrics_by_city[city]
        print(
            f"{city:<10} "
            f"{metrics['n_images']:>6} "
            f"{metrics['iou_building']:>7.4f} "
            f"{metrics['dice_f1']:>7.4f} "
            f"{metrics['precision']:>7.4f} "
            f"{metrics['recall']:>7.4f} "
            f"{metrics['accuracy']:>7.4f} "
            f"{metrics['boundary_f1_2px']:>7.4f} "
            f"{metrics['boundary_f1_5px']:>7.4f}"
        )


def main():
    args = parse_args()

    config = load_config(args.config)

    run_name = require_config_value(config, "training", "run_name")
    architecture = require_config_value(config, "model", "architecture")
    encoder = require_config_value(config, "model", "encoder")
    model_dir = require_config_value(config, "model", "model_dir")
    model_path = resolve_model_path(model_dir, run_name)

    protocol = require_config_value(config, "protocol", "name")
    image_ids = image_id_list(require_config_value(
        config,
        "protocol",
        f"{args.split}_image_ids",
    ))

    image_dir = require_config_value(config, "data", "raw_train_image_dir")
    mask_dir = require_config_value(config, "data", "raw_train_mask_dir")
    threshold = require_config_value(config, "evaluation", "threshold")
    tile_size = require_config_value(config, "evaluation", "tile_size")
    stride = require_config_value(config, "evaluation", "stride")
    min_area = require_config_value(config, "evaluation", "min_area")
    open_kernel_size = require_config_value(config, "evaluation", "open_kernel_size")

    print("Device:", DEVICE)
    print("Experiment:", run_name)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Model path:", model_path)
    print("INRIA protocol:", protocol)
    print("Split:", args.split)
    print("Protocol image ids:", describe_image_ids(image_ids))
    print("Full-image source images:", image_dir)
    print("Full-image source masks:", mask_dir)
    print("Threshold:", threshold)
    print("Tile size:", tile_size)
    print("Stride:", stride)
    print("Min area:", min_area)
    print("Open kernel size:", open_kernel_size)

    model = load_model(
        model_path=model_path,
        architecture=architecture,
        encoder=encoder,
        device=DEVICE,
    )

    metrics_by_city = evaluate_full_images(
        model=model,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=image_ids,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
        min_area=min_area,
        open_kernel_size=open_kernel_size,
    )

    log_path = RESULT_PATHS[args.split]
    log_evaluation(
        log_path=log_path,
        run_name=run_name,
        architecture=architecture,
        encoder=encoder,
        protocol=protocol,
        split=args.split,
        image_ids=image_ids,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
        min_area=min_area,
        open_kernel_size=open_kernel_size,
        metrics_by_city=metrics_by_city,
    )

    print_evaluation_summary(
        split=args.split,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
        min_area=min_area,
        open_kernel_size=open_kernel_size,
        metrics_by_city=metrics_by_city,
    )
    print("Results CSV:", log_path)


if __name__ == "__main__":
    main()
