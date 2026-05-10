import os
import csv
import argparse
import random

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
    "best_val_loss",
    "best_threshold",
    "best_threshold_val_dice",
    "notes",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def dice_score(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()

    return (2 * intersection + 1e-7) / (total + 1e-7)


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


def evaluate(model, loader, loss_fn, threshold=0.5, desc="Val"):
    model.eval()

    val_loss = 0.0
    val_dice = 0.0

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

            dice = dice_score(logits, masks, threshold=threshold)

            val_loss += loss.item()
            val_dice += dice.item()

            val_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_dice": f"{dice.item():.4f}",
            })

    val_loss /= len(loader)
    val_dice /= len(loader)

    return val_loss, val_dice


def find_best_threshold(model, loader, loss_fn):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    best_threshold = 0.5
    best_dice = 0.0

    print("\nTuning threshold on validation set...")

    for threshold in thresholds:
        _, val_dice = evaluate(
            model,
            loader,
            loss_fn,
            threshold=threshold,
            desc=f"Threshold {threshold:.2f}",
        )

        print(f"Threshold {threshold:.2f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_threshold = threshold

    return best_threshold, best_dice


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
    best_val_loss,
    best_threshold,
    best_threshold_val_dice,
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
                "Delete the file or update the header before logging this experiment."
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
            epochs,
            batch_size,
            lr,
            best_epoch,
            round(best_val_dice_threshold_05, 4),
            round(best_val_loss, 4) if best_val_loss is not None else None,
            round(best_threshold, 2),
            round(best_threshold_val_dice, 4),
            "",
        ])


def build_model(architecture, encoder):
    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    if architecture == "fpn":
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    if architecture == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")


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
        default="data/tiles_256/train/images",
    )
    train_mask_dir = select_value(
        args.train_mask_dir,
        config,
        "data",
        "train_mask_dir",
        default="data/tiles_256/train/masks",
    )
    val_image_dir = select_value(
        args.val_image_dir,
        config,
        "data",
        "val_image_dir",
        default="data/tiles_256/val/images",
    )
    val_mask_dir = select_value(
        args.val_mask_dir,
        config,
        "data",
        "val_mask_dir",
        default="data/tiles_256/val/masks",
    )
    model_dir = select_value(args.model_dir, config, "model", "model_dir", default="models")
    log_path = select_value(
        args.experiment_log_path,
        config,
        "training",
        "experiment_log_path",
        default="outputs/experiments.csv",
    )

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

    model = build_model(architecture, encoder).to(DEVICE)

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

    best_val_dice = 0.0
    best_val_loss = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
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

        val_loss, val_dice = evaluate(
            model,
            val_loader,
            loss_fn,
            threshold=0.5,
            desc="Val",
        )

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice @0.5: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")

    print(f"\nBest Val Dice @0.5: {best_val_dice:.4f}")
    print(f"Best Epoch: {best_epoch}")

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    best_threshold, best_threshold_val_dice = find_best_threshold(
        model,
        val_loader,
        loss_fn,
    )

    print(
        f"\nBest threshold: {best_threshold:.2f} | "
        f"Val Dice: {best_threshold_val_dice:.4f}"
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
        best_val_dice_threshold_05=best_val_dice,
        best_val_loss=best_val_loss,
        best_threshold=best_threshold,
        best_threshold_val_dice=best_threshold_val_dice,
    )


if __name__ == "__main__":
    main()
