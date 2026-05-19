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
from models import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "augmentation_type",
    "epochs",
    "batch_size",
    "lr",
    "best_epoch",
    "best_val_dice_threshold_05",
    "best_val_iou_threshold_05",
    "best_val_loss",
    "best_threshold",
    "best_threshold_val_dice",
    "best_threshold_val_iou",
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
    best_index = int(torch.argmax(dice_scores).item())

    return {
        "loss": val_loss,
        "dice_threshold_05": float(dice_scores[threshold_05_index].item()),
        "iou_threshold_05": float(iou_scores[threshold_05_index].item()),
        "best_threshold": threshold_values[best_index],
        "best_threshold_val_dice": float(dice_scores[best_index].item()),
        "best_threshold_val_iou": float(iou_scores[best_index].item()),
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
        f"Val IoU: {metrics['best_threshold_val_iou']:.4f}"
    )
    return metrics


def ensure_experiment_log_header(log_path):
    with open(log_path, "r", newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
        existing_rows = list(reader)

    if existing_header is None:
        existing_header = []

    if existing_header == CSV_HEADER:
        return

    unknown_columns = [
        column for column in existing_header or [] if column not in CSV_HEADER
    ]
    if unknown_columns:
        raise ValueError(
            f"CSV header mismatch in {log_path}.\n"
            f"Expected columns: {CSV_HEADER}\n"
            f"Found:            {existing_header}\n"
            f"Unknown columns:  {unknown_columns}\n"
            "Delete the file or update the header before logging this experiment."
        )

    header_index = {column: idx for idx, column in enumerate(existing_header)}
    migrated_rows = []

    for row in existing_rows:
        migrated_rows.append([
            row[header_index[column]]
            if column in header_index and header_index[column] < len(row)
            else ""
            for column in CSV_HEADER
        ])

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(migrated_rows)


def log_experiment(
    log_path,
    run_name,
    architecture,
    encoder,
    augmentation_type,
    epochs,
    batch_size,
    lr,
    best_epoch,
    best_val_dice_threshold_05,
    best_val_iou_threshold_05,
    best_val_loss,
    best_threshold,
    best_threshold_val_dice,
    best_threshold_val_iou,
):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_exists = os.path.exists(log_path)

    if file_exists:
        ensure_experiment_log_header(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow([
            run_name,
            architecture,
            encoder,
            augmentation_type,
            epochs,
            batch_size,
            lr,
            best_epoch,
            round(best_val_dice_threshold_05, 4),
            round(best_val_iou_threshold_05, 4),
            round(best_val_loss, 4) if best_val_loss is not None else None,
            round(best_threshold, 2),
            round(best_threshold_val_dice, 4),
            round(best_threshold_val_iou, 4),
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE == "cuda"),
    )

    best_checkpoint_dice = 0.0
    best_val_dice_threshold_05 = 0.0
    best_val_iou_threshold_05 = 0.0
    best_threshold = 0.5
    best_threshold_val_iou = 0.0
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

        scheduler.step(val_metrics["best_threshold_val_dice"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice @0.5: {val_metrics['dice_threshold_05']:.4f} | "
            f"Val IoU @0.5: {val_metrics['iou_threshold_05']:.4f} | "
            f"Best threshold: {val_metrics['best_threshold']:.2f} | "
            f"Best Dice: {val_metrics['best_threshold_val_dice']:.4f} | "
            f"Best IoU: {val_metrics['best_threshold_val_iou']:.4f}"
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
            best_threshold = val_metrics["best_threshold"]
            best_threshold_val_iou = val_metrics["best_threshold_val_iou"]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("Saved best model by tuned validation Dice.")

    print(f"\nBest checkpoint Val Dice: {best_checkpoint_dice:.4f}")
    print(f"Best checkpoint Val Dice @0.5: {best_val_dice_threshold_05:.4f}")
    print(f"Best checkpoint Val IoU @0.5: {best_val_iou_threshold_05:.4f}")
    print(f"Best checkpoint threshold: {best_threshold:.2f}")
    print(f"Best checkpoint Val IoU: {best_threshold_val_iou:.4f}")
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
        f"Best threshold: {final_metrics['best_threshold']:.2f} | "
        f"Best Dice: {final_metrics['best_threshold_val_dice']:.4f} | "
        f"Best IoU: {final_metrics['best_threshold_val_iou']:.4f}"
    )

    log_experiment(
        log_path=log_path,
        run_name=run_name,
        architecture=architecture,
        encoder=encoder,
        augmentation_type=augmentation_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        best_epoch=best_epoch,
        best_val_dice_threshold_05=final_metrics["dice_threshold_05"],
        best_val_iou_threshold_05=final_metrics["iou_threshold_05"],
        best_val_loss=final_metrics["loss"],
        best_threshold=final_metrics["best_threshold"],
        best_threshold_val_dice=final_metrics["best_threshold_val_dice"],
        best_threshold_val_iou=final_metrics["best_threshold_val_iou"],
    )


if __name__ == "__main__":
    main()
