import os
import csv
import argparse
import json
import random
import subprocess
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from config import (
    DEFAULT_CONFIG_PATH,
    get_config_value,
    load_config,
    resolve_model_metadata_path,
    resolve_model_path,
)
from dataset import (
    BuildingDataset,
    describe_image_ids,
    image_id_list,
)
from models import build_model
from transforms import get_train_transform, get_val_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "augmentation",
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

THRESHOLD_VALUES = tuple(round(float(t), 2) for t in np.arange(0.30, 0.801, 0.01))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def require_config_value(config, *keys):
    value = get_config_value(config, *keys)

    if value is None:
        dotted_key = ".".join(keys)
        raise ValueError(f"Missing required config value: {dotted_key}")

    return value


def require_bool_config(config, *keys):
    value = require_config_value(config, *keys)

    if not isinstance(value, bool):
        dotted_key = ".".join(keys)
        raise TypeError(f"{dotted_key} must be a JSON boolean true/false value.")

    return value


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


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def to_json_serializable(value):
    if isinstance(value, dict):
        return {
            str(key): to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_json_serializable(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

    return value


def save_model_metadata(
    metadata_path,
    run_name,
    architecture,
    encoder,
    encoder_weights,
    protocol,
    train_image_ids,
    val_image_ids,
    test_image_ids,
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    train_tiles,
    val_tiles,
    augmentation,
    epochs,
    batch_size,
    lr,
    optimizer,
    weight_decay,
    scheduler,
    scheduler_t_max,
    scheduler_eta_min,
    loss,
    seed,
    device,
    mixed_precision,
    best_epoch,
    val_metrics,
    checkpoint_path,
    config_path,
):
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)

    metadata = {
        "run_name": run_name,
        "architecture": architecture,
        "encoder": encoder,
        "encoder_weights": encoder_weights,
        "protocol": protocol,
        "train_image_ids": train_image_ids,
        "val_image_ids": val_image_ids,
        "test_image_ids": test_image_ids,
        "train_image_dir": train_image_dir,
        "train_mask_dir": train_mask_dir,
        "val_image_dir": val_image_dir,
        "val_mask_dir": val_mask_dir,
        "train_tiles": train_tiles,
        "val_tiles": val_tiles,
        "augmentation": augmentation,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "scheduler_t_max": scheduler_t_max,
        "scheduler_eta_min": scheduler_eta_min,
        "loss": loss,
        "seed": seed,
        "device": device,
        "mixed_precision": mixed_precision,
        "best_epoch": best_epoch,
        "val_loss": val_metrics["loss"],
        "best_threshold": val_metrics["best_threshold"],
        "best_val_dice": val_metrics["best_threshold_val_dice"],
        "best_val_iou": val_metrics["best_threshold_val_iou"],
        "best_val_precision": val_metrics["best_threshold_val_precision"],
        "best_val_recall": val_metrics["best_threshold_val_recall"],
        "best_val_accuracy": val_metrics["best_threshold_val_accuracy"],
        "val_dice_threshold_05": val_metrics["dice_threshold_05"],
        "val_iou_threshold_05": val_metrics["iou_threshold_05"],
        "val_precision_threshold_05": val_metrics["precision_threshold_05"],
        "val_recall_threshold_05": val_metrics["recall_threshold_05"],
        "val_accuracy_threshold_05": val_metrics["accuracy_threshold_05"],
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "config_path": config_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(metadata), f, indent=2)
        f.write("\n")

    print(f"Saved model metadata: {metadata_path}")


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


def log_experiment(
    log_path,
    run_name,
    architecture,
    encoder,
    augmentation,
    protocol,
    train_image_ids,
    val_image_ids,
    test_image_ids,
    train_tiles,
    val_tiles,
    epochs,
    batch_size,
    lr,
    best_epoch,
    val_metrics,
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
            str(augmentation).lower(),
            protocol,
            describe_image_ids(train_image_ids),
            describe_image_ids(val_image_ids),
            describe_image_ids(test_image_ids),
            train_tiles,
            val_tiles,
            "",
            "",
            "",
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
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ])


def main():
    args = parse_args()
    config = load_config(args.config)

    run_name = require_config_value(config, "training", "run_name")
    architecture = require_config_value(config, "model", "architecture")
    encoder = require_config_value(config, "model", "encoder")
    batch_size = require_config_value(config, "training", "batch_size")
    epochs = require_config_value(config, "training", "epochs")
    lr = require_config_value(config, "training", "lr")
    augmentation = require_bool_config(config, "training", "augmentation")
    seed = require_config_value(config, "training", "seed")

    protocol = require_config_value(config, "protocol", "name")
    train_image_dir = require_config_value(config, "data", "train_image_dir")
    train_mask_dir = require_config_value(config, "data", "train_mask_dir")
    val_image_dir = require_config_value(config, "data", "val_image_dir")
    val_mask_dir = require_config_value(config, "data", "val_mask_dir")
    train_image_ids = image_id_list(require_config_value(config, "protocol", "train_image_ids"))
    val_image_ids = image_id_list(require_config_value(config, "protocol", "val_image_ids"))
    test_image_ids = image_id_list(require_config_value(config, "protocol", "test_image_ids"))
    model_dir = require_config_value(config, "model", "model_dir")
    log_path = require_config_value(config, "training", "experiment_log_path")

    model_path = resolve_model_path(model_dir, run_name)
    metadata_path = resolve_model_metadata_path(model_dir, run_name)

    os.makedirs(model_dir, exist_ok=True)

    set_seed(seed)

    print("Device:", DEVICE)
    print("Experiment:", run_name)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Augmentation:", augmentation)
    print("Train images:", train_image_dir)
    print("Validation images:", val_image_dir)
    print("INRIA protocol:", protocol)
    print("Protocol train image ids:", describe_image_ids(train_image_ids))
    print("Protocol val image ids:", describe_image_ids(val_image_ids))
    print("Protocol test image ids:", describe_image_ids(test_image_ids))
    print("Model path:", model_path)
    print("Metadata path:", metadata_path)
    print("Experiment log:", log_path)
    print("Mixed precision:", DEVICE == "cuda")

    train_dataset = BuildingDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_train_transform(augmentation),
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

    encoder_weights = "imagenet"
    optimizer_weight_decay = 1e-4
    scheduler_eta_min = 1e-6
    loss_name = "DiceLoss + BCEWithLogitsLoss"

    model = build_model(
        architecture,
        encoder,
        encoder_weights=encoder_weights,
    ).to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    def loss_fn(logits, masks):
        return dice_loss(logits, masks) + bce_loss(logits, masks)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=optimizer_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=scheduler_eta_min,
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
            save_model_metadata(
                metadata_path=metadata_path,
                run_name=run_name,
                architecture=architecture,
                encoder=encoder,
                encoder_weights=encoder_weights,
                protocol=protocol,
                train_image_ids=train_image_ids,
                val_image_ids=val_image_ids,
                test_image_ids=test_image_ids,
                train_image_dir=train_image_dir,
                train_mask_dir=train_mask_dir,
                val_image_dir=val_image_dir,
                val_mask_dir=val_mask_dir,
                train_tiles=len(train_dataset),
                val_tiles=len(val_dataset),
                augmentation=augmentation,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer.__class__.__name__,
                weight_decay=optimizer_weight_decay,
                scheduler=scheduler.__class__.__name__,
                scheduler_t_max=epochs,
                scheduler_eta_min=scheduler_eta_min,
                loss=loss_name,
                seed=seed,
                device=DEVICE,
                mixed_precision=(DEVICE == "cuda"),
                best_epoch=best_epoch,
                val_metrics=val_metrics,
                checkpoint_path=model_path,
                config_path=args.config,
            )
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

    save_model_metadata(
        metadata_path=metadata_path,
        run_name=run_name,
        architecture=architecture,
        encoder=encoder,
        encoder_weights=encoder_weights,
        protocol=protocol,
        train_image_ids=train_image_ids,
        val_image_ids=val_image_ids,
        test_image_ids=test_image_ids,
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        train_tiles=len(train_dataset),
        val_tiles=len(val_dataset),
        augmentation=augmentation,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer.__class__.__name__,
        weight_decay=optimizer_weight_decay,
        scheduler=scheduler.__class__.__name__,
        scheduler_t_max=epochs,
        scheduler_eta_min=scheduler_eta_min,
        loss=loss_name,
        seed=seed,
        device=DEVICE,
        mixed_precision=(DEVICE == "cuda"),
        best_epoch=best_epoch,
        val_metrics=final_metrics,
        checkpoint_path=model_path,
        config_path=args.config,
    )

    log_experiment(
        log_path=log_path,
        run_name=run_name,
        architecture=architecture,
        encoder=encoder,
        augmentation=augmentation,
        protocol=protocol,
        train_image_ids=train_image_ids,
        val_image_ids=val_image_ids,
        test_image_ids=test_image_ids,
        train_tiles=len(train_dataset),
        val_tiles=len(val_dataset),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        best_epoch=best_epoch,
        val_metrics=final_metrics,
    )


if __name__ == "__main__":
    main()
