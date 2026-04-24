import os
import random

import cv2
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import BuildingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_IMG_DIR = "data/tiles_256/val/images"
VAL_MASK_DIR = "data/tiles_256/val/masks"

MODEL_PATH = "models/unet_resnet34_best.pth"
OUT_DIR = "outputs/predictions"

NUM_SAMPLES = 8
THRESHOLD = 0.5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = BuildingDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    indices = random.sample(range(len(dataset)), NUM_SAMPLES)

    for out_idx, idx in enumerate(indices):
        img_tensor, mask_tensor = dataset[idx]

        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_input)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_mask = prob > THRESHOLD

        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        gt_mask = mask_tensor[0].cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))

        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray", interpolation="nearest")
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="gray", interpolation="nearest")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(img)
        axes[3].imshow(pred_mask, cmap="Reds", alpha=0.35, interpolation="nearest")
        axes[3].set_title("Overlay")
        axes[3].axis("off")

        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"prediction_{out_idx:02d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()