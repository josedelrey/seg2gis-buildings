import os
import csv

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
from tqdm import tqdm

from dataset import BuildingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "data/tiles_256/train/images"
TRAIN_MASK_DIR = "data/tiles_256/train/masks"
VAL_IMG_DIR = "data/tiles_256/val/images"
VAL_MASK_DIR = "data/tiles_256/val/masks"

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
USE_AUGMENTATION = True

RUN_NAME = "unet_r34_e5"
MODEL_PATH = f"models/{RUN_NAME}.pth"
LOG_PATH = "experiments/experiments.csv"


def dice_score(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()

    return (2 * intersection + 1e-7) / (total + 1e-7)


def get_train_transform():
    if not USE_AUGMENTATION:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.0,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),

        A.ElasticTransform(
            alpha=120,
            sigma=6,
            p=0.3,
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),

        A.RandomGamma(p=0.3),
    ])


def main():
    print("Device:", DEVICE)
    print("Augmentation:", USE_AUGMENTATION)

    train_transform = get_train_transform()

    train_dataset = BuildingDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        transform=train_transform,
    )

    val_dataset = BuildingDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_dice = 0.0
    best_val_loss = None
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc="Train", leave=False)

        for imgs, masks in train_pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # update progress bar
            train_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{train_loss / (train_pbar.n + 1):.4f}"
            })

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        val_pbar = tqdm(val_loader, desc="Val", leave=False)

        with torch.no_grad():
            for imgs, masks in val_pbar:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(imgs)
                loss = loss_fn(logits, masks)

                dice = dice_score(logits, masks)

                val_loss += loss.item()
                val_dice += dice.item()

                val_pbar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "batch_dice": f"{dice.item():.4f}"
                })

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # ---------------- SUMMARY ----------------
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # ---------------- SAVE ----------------
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

    # ---------------- EXPERIMENT LOG ----------------
    os.makedirs("experiments", exist_ok=True)

    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "run_name",
                "model",
                "tile_size",
                "augmentation",
                "epochs",
                "batch_size",
                "lr",
                "loss",
                "best_epoch",
                "best_val_dice",
                "best_val_loss",
                "model_path",
                "notes",
            ])

        writer.writerow([
            RUN_NAME,
            "UNet-ResNet34",
            256,
            USE_AUGMENTATION,
            EPOCHS,
            BATCH_SIZE,
            LR,
            "Dice",
            best_epoch,
            round(best_val_dice, 4),
            round(best_val_loss, 4),
            MODEL_PATH,
            "",
        ])


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()