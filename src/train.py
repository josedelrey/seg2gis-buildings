import os
import csv
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm

from dataset import BuildingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "data/tiles_256/train/images"
TRAIN_MASK_DIR = "data/tiles_256/train/masks"
VAL_IMG_DIR = "data/tiles_256/val/images"
VAL_MASK_DIR = "data/tiles_256/val/masks"

LOG_PATH = "outputs/experiments.csv"

CSV_HEADER = [
    "run_name",
    "architecture",
    "encoder",
    "augmentation",
    "epochs",
    "batch_size",
    "lr",
    "best_epoch",
    "best_val_dice",
    "best_val_loss",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="unet")
    parser.add_argument("--encoder", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_augmentation", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_score(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()

    return (2 * intersection + 1e-7) / (total + 1e-7)


def get_train_transform(use_augmentation):
    if not use_augmentation:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=0,
            p=0.5,
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3,
        ),

        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.15),
    ])


def log_experiment(
    run_name,
    architecture,
    encoder,
    use_augmentation,
    epochs,
    batch_size,
    lr,
    best_epoch,
    best_val_dice,
    best_val_loss,
):
    os.makedirs("outputs", exist_ok=True)

    file_exists = os.path.exists(LOG_PATH)

    if file_exists:
        with open(LOG_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != CSV_HEADER:
            raise ValueError(
                f"CSV header mismatch in {LOG_PATH}.\n"
                f"Expected: {CSV_HEADER}\n"
                f"Found:    {existing_header}\n"
                "Delete the file or update the header before logging this experiment."
            )

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow([
            run_name,
            architecture,
            encoder,
            use_augmentation,
            epochs,
            batch_size,
            lr,
            best_epoch,
            round(best_val_dice, 4),
            round(best_val_loss, 4) if best_val_loss is not None else None,
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

    elif architecture == "fpn":
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    elif architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    elif architecture == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def main():
    args = parse_args()

    run_name = args.run_name
    architecture = args.architecture
    encoder = args.encoder
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    use_augmentation = args.use_augmentation
    seed = args.seed

    model_path = f"models/{run_name}.pth"

    os.makedirs("models", exist_ok=True)

    set_seed(seed)

    print("Device:", DEVICE)
    print("Experiment:", run_name)

    train_transform = get_train_transform(use_augmentation)

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
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

    best_val_dice = 0.0
    best_val_loss = None
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

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

            train_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{train_loss / (train_pbar.n + 1):.4f}",
            })

        train_loss /= len(train_loader)

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
                    "batch_dice": f"{dice.item():.4f}",
                })

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")

    log_experiment(
        run_name=run_name,
        architecture=architecture,
        encoder=encoder,
        use_augmentation=use_augmentation,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        best_epoch=best_epoch,
        best_val_dice=best_val_dice,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()