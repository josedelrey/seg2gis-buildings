import os
import csv
import argparse
import random
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config, resolve_model_path
from dataset import BuildingDataset
from gis_utils import load_rgb_image, predict_full_image_tiled
from inria_split import (
    INRIA_TEST_IMAGE_IDS,
    INRIA_TRAIN_IMAGE_IDS,
    INRIA_VAL_IMAGE_IDS,
    collect_image_mask_pairs,
    describe_image_ids,
    image_id_list,
)
from models import build_model
from postprocess import postprocess_mask


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "augmentation_type",
    "protocol",
    "train_image_ids",
    "val_image_ids",
    "test_image_ids",
    "train_tiles",
    "val_tiles",
    "test_images",
    "test_min_area",
    "test_open_kernel_size",
    "epochs",
    "batch_size",
    "lr",
    "best_epoch",
    "val_loss",
    "val_dice_threshold_05",
    "val_iou_threshold_05",
    "val_precision_threshold_05",
    "val_recall_threshold_05",
    "val_accuracy_threshold_05",
    "best_threshold",
    "val_best_dice",
    "val_best_iou",
    "val_best_precision",
    "val_best_recall",
    "val_best_accuracy",
    "test_threshold",
    "test_iou_building",
    "test_dice_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
    "test_tp",
    "test_fp",
    "test_fn",
    "test_tn",
    "notes",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLD_VALUES = tuple(round(float(t), 2) for t in np.arange(0.30, 0.801, 0.01))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument(
        "--augmentation_type",
        type=str,
        default=None,
        choices=["noaug", "geomaug", "mildaug", "strongaug"],
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_image_dir", type=str, default=None)
    parser.add_argument("--train_mask_dir", type=str, default=None)
    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument("--val_mask_dir", type=str, default=None)
    parser.add_argument("--raw_test_image_dir", type=str, default=None)
    parser.add_argument("--raw_test_mask_dir", type=str, default=None)
    parser.add_argument("--train_image_ids", type=str, default=None)
    parser.add_argument("--val_image_ids", type=str, default=None)
    parser.add_argument("--test_image_ids", type=str, default=None)
    parser.add_argument("--eval_tile_size", type=int, default=None)
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--eval_min_area", type=int, default=None)
    parser.add_argument("--eval_open_kernel_size", type=int, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--experiment_log_path", type=str, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    if minutes:
        return f"{minutes}m {seconds:02d}s"

    return f"{seconds}s"


def format_eta(seconds_from_now):
    finish_time = datetime.now() + timedelta(seconds=seconds_from_now)
    return finish_time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_values_above_thresholds(values, thresholds):
    if values.numel() == 0:
        return torch.zeros(
            thresholds.numel(),
            dtype=torch.float64,
            device=thresholds.device,
        )

    values = values.detach().float().flatten()
    bucket_indices = torch.bucketize(values, thresholds, right=False)
    counts_by_bucket = torch.bincount(
        bucket_indices,
        minlength=thresholds.numel() + 1,
    ).to(torch.float64)

    return torch.flip(
        torch.cumsum(torch.flip(counts_by_bucket[1:], dims=[0]), dim=0),
        dims=[0],
    )


def finish_transform():
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def get_train_transform(augmentation_type):
    if augmentation_type == "noaug":
        return A.Compose([
            *finish_transform(),
        ])

    if augmentation_type == "geomaug":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.25),
            *finish_transform(),
        ])

    if augmentation_type == "mildaug":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.25),

            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.30,
            ),

            A.HueSaturationValue(
                hue_shift_limit=3,
                sat_shift_limit=8,
                val_shift_limit=5,
                p=0.20,
            ),

            *finish_transform(),
        ])

    if augmentation_type == "strongaug":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.25),

            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.35,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.40,
            ),

            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=12,
                val_shift_limit=8,
                p=0.30,
            ),

            A.GaussNoise(
                std_range=(0.02, 0.08),
                p=0.15,
            ),

            A.MotionBlur(
                blur_limit=3,
                p=0.10,
            ),

            *finish_transform(),
        ])

    raise ValueError(f"Unsupported augmentation_type: {augmentation_type}")


def get_val_transform():
    return A.Compose([
        *finish_transform(),
    ])


def evaluate(model, loader, loss_fn, threshold_values=THRESHOLD_VALUES, desc="Val"):
    model.eval()

    val_loss = 0.0
    thresholds = torch.tensor(
        threshold_values,
        dtype=torch.float32,
        device=DEVICE,
    )
    pred_counts = torch.zeros(
        thresholds.numel(),
        dtype=torch.float64,
        device=DEVICE,
    )
    true_positive_counts = torch.zeros_like(pred_counts)
    target_count = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
    total_count = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
    threshold_05_index = threshold_values.index(0.5)

    val_pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for imgs, masks in val_pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(DEVICE == "cuda"),
            ):
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            probs = torch.sigmoid(logits).float()
            masks_bool = masks > 0.5

            batch_pred_counts = count_values_above_thresholds(probs, thresholds)
            batch_true_positive_counts = count_values_above_thresholds(
                probs[masks_bool],
                thresholds,
            )

            pred_counts += batch_pred_counts
            true_positive_counts += batch_true_positive_counts
            target_count += masks_bool.sum(dtype=torch.float64)
            total_count += masks_bool.numel()

            batch_dice_05 = (
                2 * batch_true_positive_counts[threshold_05_index] + 1e-7
            ) / (
                batch_pred_counts[threshold_05_index] + masks_bool.sum() + 1e-7
            )

            val_loss += loss.item()

            val_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_dice_05": f"{batch_dice_05.item():.4f}",
            })

    val_loss /= len(loader)

    dice_scores = (2 * true_positive_counts + 1e-7) / (
        pred_counts + target_count + 1e-7
    )
    iou_scores = (true_positive_counts + 1e-7) / (
        pred_counts + target_count - true_positive_counts + 1e-7
    )
    precision_scores = (true_positive_counts + 1e-7) / (
        pred_counts + 1e-7
    )
    recall_scores = (true_positive_counts + 1e-7) / (
        target_count + 1e-7
    )
    accuracy_scores = (
        total_count - pred_counts - target_count + (2 * true_positive_counts)
    ) / (total_count + 1e-7)
    best_index = int(torch.argmax(dice_scores).item())

    return {
        "loss": val_loss,
        "dice_threshold_05": float(dice_scores[threshold_05_index].item()),
        "iou_threshold_05": float(iou_scores[threshold_05_index].item()),
        "precision_threshold_05": float(precision_scores[threshold_05_index].item()),
        "recall_threshold_05": float(recall_scores[threshold_05_index].item()),
        "accuracy_threshold_05": float(accuracy_scores[threshold_05_index].item()),
        "best_threshold": threshold_values[best_index],
        "best_threshold_val_dice": float(dice_scores[best_index].item()),
        "best_threshold_val_iou": float(iou_scores[best_index].item()),
        "best_threshold_val_precision": float(precision_scores[best_index].item()),
        "best_threshold_val_recall": float(recall_scores[best_index].item()),
        "best_threshold_val_accuracy": float(accuracy_scores[best_index].item()),
    }


def find_best_threshold(model, loader, loss_fn):
    print(
        "\nTuning threshold on validation set "
        f"({THRESHOLD_VALUES[0]:.2f}..{THRESHOLD_VALUES[-1]:.2f}, step 0.01)..."
    )
    metrics = evaluate(model, loader, loss_fn, desc="Threshold sweep")
    print(
        f"Best threshold: {metrics['best_threshold']:.2f} | "
        f"Val Dice: {metrics['best_threshold_val_dice']:.4f} | "
        f"Val IoU: {metrics['best_threshold_val_iou']:.4f} | "
        f"Val Precision: {metrics['best_threshold_val_precision']:.4f} | "
        f"Val Recall: {metrics['best_threshold_val_recall']:.4f} | "
        f"Val Accuracy: {metrics['best_threshold_val_accuracy']:.4f}"
    )
    return metrics


def confusion_from_masks(pred_mask, target_mask):
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())

    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn, eps=1e-7):
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "iou_building": float((tp + eps) / (tp + fp + fn + eps)),
        "dice_f1": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "accuracy": float((tp + tn + eps) / (tp + fp + fn + tn + eps)),
    }


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

    for image_path, mask_path in tqdm(pairs, desc="INRIA(155) full-image test"):
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


def log_experiment(
    log_path,
    run_name,
    architecture,
    encoder,
    augmentation_type,
    protocol,
    train_image_ids,
    val_image_ids,
    test_image_ids,
    train_tiles,
    val_tiles,
    test_images,
    test_min_area,
    test_open_kernel_size,
    epochs,
    batch_size,
    lr,
    best_epoch,
    val_metrics,
    test_metrics,
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
                "Start a new experiment CSV or update the header before logging this experiment."
            )

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow([
            run_name,
            architecture,
            encoder,
            augmentation_type,
            protocol,
            describe_image_ids(train_image_ids),
            describe_image_ids(val_image_ids),
            describe_image_ids(test_image_ids),
            train_tiles,
            val_tiles,
            test_images,
            test_min_area,
            test_open_kernel_size,
            epochs,
            batch_size,
            lr,
            best_epoch,
            round(val_metrics["loss"], 4),
            round(val_metrics["dice_threshold_05"], 4),
            round(val_metrics["iou_threshold_05"], 4),
            round(val_metrics["precision_threshold_05"], 4),
            round(val_metrics["recall_threshold_05"], 4),
            round(val_metrics["accuracy_threshold_05"], 4),
            round(val_metrics["best_threshold"], 2),
            round(val_metrics["best_threshold_val_dice"], 4),
            round(val_metrics["best_threshold_val_iou"], 4),
            round(val_metrics["best_threshold_val_precision"], 4),
            round(val_metrics["best_threshold_val_recall"], 4),
            round(val_metrics["best_threshold_val_accuracy"], 4),
            round(test_metrics["threshold"], 2),
            round(test_metrics["iou_building"], 4),
            round(test_metrics["dice_f1"], 4),
            round(test_metrics["precision"], 4),
            round(test_metrics["recall"], 4),
            round(test_metrics["accuracy"], 4),
            test_metrics["tp"],
            test_metrics["fp"],
            test_metrics["fn"],
            test_metrics["tn"],
            "",
        ])


def main():
    args = parse_args()
    config = load_config(args.config)

    run_name = select_value(
        args.run_name,
        config,
        "training",
        "run_name",
        default="experiment",
    )
    architecture = select_value(
        args.architecture,
        config,
        "model",
        "architecture",
        default="unet",
    )
    encoder = select_value(
        args.encoder,
        config,
        "model",
        "encoder",
        default="efficientnet-b3",
    )
    batch_size = select_value(args.batch_size, config, "training", "batch_size", default=8)
    epochs = select_value(args.epochs, config, "training", "epochs", default=10)
    lr = select_value(args.lr, config, "training", "lr", default=1e-4)
    augmentation_type = select_value(
        args.augmentation_type,
        config,
        "training",
        "augmentation_type",
        default="noaug",
    )
    seed = select_value(args.seed, config, "training", "seed", default=42)

    train_image_dir = select_value(
        args.train_image_dir,
        config,
        "data",
        "train_image_dir",
    )
    train_mask_dir = select_value(
        args.train_mask_dir,
        config,
        "data",
        "train_mask_dir",
    )
    val_image_dir = select_value(
        args.val_image_dir,
        config,
        "data",
        "val_image_dir",
    )
    val_mask_dir = select_value(
        args.val_mask_dir,
        config,
        "data",
        "val_mask_dir",
    )
    raw_test_image_dir = select_value(
        args.raw_test_image_dir,
        config,
        "evaluation",
        "raw_test_image_dir",
        default=get_config_value(config, "data", "raw_train_image_dir"),
    )
    raw_test_mask_dir = select_value(
        args.raw_test_mask_dir,
        config,
        "evaluation",
        "raw_test_mask_dir",
        default=get_config_value(config, "data", "raw_train_mask_dir"),
    )
    train_image_ids = image_id_list(select_value(
        args.train_image_ids,
        config,
        "evaluation",
        "train_image_ids",
        default=INRIA_TRAIN_IMAGE_IDS,
    ))
    val_image_ids = image_id_list(select_value(
        args.val_image_ids,
        config,
        "evaluation",
        "val_image_ids",
        default=INRIA_VAL_IMAGE_IDS,
    ))
    test_image_ids = image_id_list(select_value(
        args.test_image_ids,
        config,
        "evaluation",
        "test_image_ids",
        default=INRIA_TEST_IMAGE_IDS,
    ))
    eval_tile_size = select_value(
        args.eval_tile_size,
        config,
        "evaluation",
        "tile_size",
        default=get_config_value(config, "inference", "tile_size", default=256),
    )
    eval_stride = select_value(
        args.eval_stride,
        config,
        "evaluation",
        "stride",
        default=get_config_value(config, "inference", "stride", default=128),
    )
    eval_min_area = select_value(
        args.eval_min_area,
        config,
        "evaluation",
        "min_area",
        default=get_config_value(config, "inference", "min_area", default=0),
    )
    eval_open_kernel_size = select_value(
        args.eval_open_kernel_size,
        config,
        "evaluation",
        "open_kernel_size",
        default=get_config_value(config, "inference", "open_kernel_size", default=0),
    )
    model_dir = select_value(args.model_dir, config, "model", "model_dir", default="models")
    log_path = select_value(
        args.experiment_log_path,
        config,
        "training",
        "experiment_log_path",
    )

    if train_image_dir is None:
        raise ValueError("No training image directory provided. Set data.train_image_dir or pass --train_image_dir.")

    if train_mask_dir is None:
        raise ValueError("No training mask directory provided. Set data.train_mask_dir or pass --train_mask_dir.")

    if val_image_dir is None:
        raise ValueError("No validation image directory provided. Set data.val_image_dir or pass --val_image_dir.")

    if val_mask_dir is None:
        raise ValueError("No validation mask directory provided. Set data.val_mask_dir or pass --val_mask_dir.")

    if raw_test_image_dir is None:
        raise ValueError("No raw test image directory provided. Set evaluation.raw_test_image_dir or pass --raw_test_image_dir.")

    if raw_test_mask_dir is None:
        raise ValueError("No raw test mask directory provided. Set evaluation.raw_test_mask_dir or pass --raw_test_mask_dir.")

    if log_path is None:
        raise ValueError("No experiment log path provided. Set training.experiment_log_path or pass --experiment_log_path.")

    model_path = resolve_model_path(model_dir, run_name)

    os.makedirs(model_dir, exist_ok=True)

    set_seed(seed)

    print("Device:", DEVICE)
    print("Experiment:", run_name)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Augmentation type:", augmentation_type)
    print("Train images:", train_image_dir)
    print("Validation images:", val_image_dir)
    print("INRIA protocol:", "public_155_internal_val")
    print("Protocol train image ids:", describe_image_ids(train_image_ids))
    print("Protocol val image ids:", describe_image_ids(val_image_ids))
    print("Protocol test image ids:", describe_image_ids(test_image_ids))
    print("Full-image test images:", raw_test_image_dir)
    print("Full-image test masks:", raw_test_mask_dir)
    print("Full-image test tile size:", eval_tile_size)
    print("Full-image test stride:", eval_stride)
    print("Full-image test min area:", eval_min_area)
    print("Full-image test open kernel size:", eval_open_kernel_size)
    print("Model path:", model_path)
    print("Experiment log:", log_path)
    print("Mixed precision:", DEVICE == "cuda")

    train_dataset = BuildingDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_train_transform(augmentation_type),
    )

    val_dataset = BuildingDataset(
        val_image_dir,
        val_mask_dir,
        transform=get_val_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    model = build_model(
        architecture,
        encoder,
        encoder_weights="imagenet",
    ).to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    def loss_fn(logits, masks):
        return dice_loss(logits, masks) + bce_loss(logits, masks)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE == "cuda"),
    )

    best_checkpoint_dice = 0.0
    best_val_dice_threshold_05 = 0.0
    best_val_iou_threshold_05 = 0.0
    best_val_accuracy_threshold_05 = 0.0
    best_threshold = 0.5
    best_threshold_val_iou = 0.0
    best_threshold_val_accuracy = 0.0
    best_epoch = 0
    experiment_start_time = time.monotonic()
    epoch_durations = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.monotonic()
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc="Train", leave=False)

        for imgs, masks in train_pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(DEVICE == "cuda"),
            ):
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            train_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{train_loss / (train_pbar.n + 1):.4f}",
            })

        train_loss /= len(train_loader)

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            desc="Val",
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice @0.5: {val_metrics['dice_threshold_05']:.4f} | "
            f"Val IoU @0.5: {val_metrics['iou_threshold_05']:.4f} | "
            f"Val Precision @0.5: {val_metrics['precision_threshold_05']:.4f} | "
            f"Val Recall @0.5: {val_metrics['recall_threshold_05']:.4f} | "
            f"Val Accuracy @0.5: {val_metrics['accuracy_threshold_05']:.4f} | "
            f"Best threshold: {val_metrics['best_threshold']:.2f} | "
            f"Best Dice: {val_metrics['best_threshold_val_dice']:.4f} | "
            f"Best IoU: {val_metrics['best_threshold_val_iou']:.4f} | "
            f"Best Precision: {val_metrics['best_threshold_val_precision']:.4f} | "
            f"Best Recall: {val_metrics['best_threshold_val_recall']:.4f} | "
            f"Best Accuracy: {val_metrics['best_threshold_val_accuracy']:.4f}"
        )

        epoch_duration = time.monotonic() - epoch_start_time
        epoch_durations.append(epoch_duration)

        elapsed = time.monotonic() - experiment_start_time
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = epochs - epoch
        estimated_remaining = avg_epoch_duration * remaining_epochs
        estimated_total = elapsed + estimated_remaining

        print(
            "Time estimate | "
            f"epoch: {format_duration(epoch_duration)} | "
            f"elapsed: {format_duration(elapsed)} | "
            f"remaining: {format_duration(estimated_remaining)} | "
            f"estimated total: {format_duration(estimated_total)} | "
            f"ETA: {format_eta(estimated_remaining)}"
        )

        if val_metrics["best_threshold_val_dice"] > best_checkpoint_dice:
            best_checkpoint_dice = val_metrics["best_threshold_val_dice"]
            best_val_dice_threshold_05 = val_metrics["dice_threshold_05"]
            best_val_iou_threshold_05 = val_metrics["iou_threshold_05"]
            best_val_accuracy_threshold_05 = val_metrics["accuracy_threshold_05"]
            best_threshold = val_metrics["best_threshold"]
            best_threshold_val_iou = val_metrics["best_threshold_val_iou"]
            best_threshold_val_accuracy = val_metrics["best_threshold_val_accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("Saved best model by tuned validation Dice.")

        scheduler.step()

    print(f"\nBest checkpoint Val Dice: {best_checkpoint_dice:.4f}")
    print(f"Best checkpoint Val Dice @0.5: {best_val_dice_threshold_05:.4f}")
    print(f"Best checkpoint Val IoU @0.5: {best_val_iou_threshold_05:.4f}")
    print(f"Best checkpoint Val Accuracy @0.5: {best_val_accuracy_threshold_05:.4f}")
    print(f"Best checkpoint threshold: {best_threshold:.2f}")
    print(f"Best checkpoint Val IoU: {best_threshold_val_iou:.4f}")
    print(f"Best checkpoint Val Accuracy: {best_threshold_val_accuracy:.4f}")
    print(f"Best Epoch: {best_epoch}")

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    final_metrics = find_best_threshold(
        model,
        val_loader,
        loss_fn,
    )

    print(
        f"\nBest checkpoint retest | "
        f"Dice @0.5: {final_metrics['dice_threshold_05']:.4f} | "
        f"IoU @0.5: {final_metrics['iou_threshold_05']:.4f} | "
        f"Precision @0.5: {final_metrics['precision_threshold_05']:.4f} | "
        f"Recall @0.5: {final_metrics['recall_threshold_05']:.4f} | "
        f"Accuracy @0.5: {final_metrics['accuracy_threshold_05']:.4f} | "
        f"Best threshold: {final_metrics['best_threshold']:.2f} | "
        f"Best Dice: {final_metrics['best_threshold_val_dice']:.4f} | "
        f"Best IoU: {final_metrics['best_threshold_val_iou']:.4f} | "
        f"Best Precision: {final_metrics['best_threshold_val_precision']:.4f} | "
        f"Best Recall: {final_metrics['best_threshold_val_recall']:.4f} | "
        f"Best Accuracy: {final_metrics['best_threshold_val_accuracy']:.4f}"
    )

    test_metrics = evaluate_full_images(
        model=model,
        image_dir=raw_test_image_dir,
        mask_dir=raw_test_mask_dir,
        image_ids=test_image_ids,
        threshold=final_metrics["best_threshold"],
        tile_size=eval_tile_size,
        stride=eval_stride,
        min_area=eval_min_area,
        open_kernel_size=eval_open_kernel_size,
    )

    print(
        "\nINRIA(155) held-out full-image test | "
        f"threshold: {test_metrics['threshold']:.2f} | "
        f"IoU_building: {test_metrics['iou_building']:.4f} | "
        f"Dice/F1: {test_metrics['dice_f1']:.4f} | "
        f"Precision: {test_metrics['precision']:.4f} | "
        f"Recall: {test_metrics['recall']:.4f} | "
        f"Accuracy: {test_metrics['accuracy']:.4f} | "
        f"TP: {test_metrics['tp']} | "
        f"FP: {test_metrics['fp']} | "
        f"FN: {test_metrics['fn']} | "
        f"TN: {test_metrics['tn']}"
    )

    log_experiment(
        log_path=log_path,
        run_name=run_name,
        architecture=architecture,
        encoder=encoder,
        augmentation_type=augmentation_type,
        protocol="inria155_public_holdout_internal_val",
        train_image_ids=train_image_ids,
        val_image_ids=val_image_ids,
        test_image_ids=test_image_ids,
        train_tiles=len(train_dataset),
        val_tiles=len(val_dataset),
        test_images=test_metrics["n_images"],
        test_min_area=eval_min_area,
        test_open_kernel_size=eval_open_kernel_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        best_epoch=best_epoch,
        val_metrics=final_metrics,
        test_metrics=test_metrics,
    )


if __name__ == "__main__":
    main()
