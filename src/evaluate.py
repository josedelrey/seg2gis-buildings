import cv2
import numpy as np
import sys
import torch
from tqdm import tqdm

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path
from dataset import collect_image_mask_pairs, describe_image_ids, image_id_list
from gis_utils import load_model, load_rgb_image, predict_full_image_tiled
from metrics import confusion_from_masks, metrics_from_confusion
from postprocess import postprocess_mask


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def main():
    if len(sys.argv) > 1:
        raise SystemExit(
            "evaluate.py does not accept CLI arguments; edit configs/default.json."
        )

    config = load_config(DEFAULT_CONFIG_PATH)

    run_name = require_config_value(config, "training", "run_name")
    architecture = require_config_value(config, "model", "architecture")
    encoder = require_config_value(config, "model", "encoder")
    model_dir = require_config_value(config, "model", "model_dir")
    model_path = resolve_model_path(model_dir, run_name)

    protocol = require_config_value(config, "protocol", "name")
    test_image_ids = image_id_list(require_config_value(config, "protocol", "test_image_ids"))

    raw_test_image_dir = require_config_value(config, "evaluation", "raw_test_image_dir")
    raw_test_mask_dir = require_config_value(config, "evaluation", "raw_test_mask_dir")
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
    print("Protocol test image ids:", describe_image_ids(test_image_ids))
    print("Full-image test images:", raw_test_image_dir)
    print("Full-image test masks:", raw_test_mask_dir)
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
        image_dir=raw_test_image_dir,
        mask_dir=raw_test_mask_dir,
        image_ids=test_image_ids,
        threshold=threshold,
        tile_size=tile_size,
        stride=stride,
        min_area=min_area,
        open_kernel_size=open_kernel_size,
    )

    print(
        "\nINRIA held-out full-image evaluation | "
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


if __name__ == "__main__":
    main()
