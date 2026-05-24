import argparse
import csv
import cv2
import numpy as np
import os
import torch
from tqdm import tqdm

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path
from dataset import collect_image_mask_pairs, describe_image_ids, image_id_list
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled
from metrics import confusion_from_masks, metrics_from_confusion
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

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    model.eval()

    for image_path, mask_path in tqdm(pairs, desc="INRIA full-image evaluation"):
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

        tp, fp, fn, tn = confusion_from_masks(pred_mask, target_mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    metrics = metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)
    metrics["n_images"] = len(pairs)
    metrics["threshold"] = float(threshold)

    return metrics


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
    metrics,
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
                f"Found:    {existing_header}"
            )

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow([
            run_name,
            architecture,
            encoder,
            protocol,
            split,
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
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
            metrics["tn"],
        ])


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

    metrics = evaluate_full_images(
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
        metrics=metrics,
    )

    print(
        f"\nINRIA {args.split} full-image evaluation | "
        f"threshold: {metrics['threshold']:.2f} | "
        f"images: {metrics['n_images']} | "
        f"IoU_building: {metrics['iou_building']:.4f} | "
        f"Dice/F1: {metrics['dice_f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"TP: {metrics['tp']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']} | "
        f"TN: {metrics['tn']}"
    )
    print("Results CSV:", log_path)


if __name__ == "__main__":
    main()
