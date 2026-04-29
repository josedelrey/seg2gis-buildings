import os
import argparse
import random

import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import BuildingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_IMG_DIR = "data/tiles_256/val/images"
VAL_MASK_DIR = "data/tiles_256/val/masks"

OUT_DIR = "outputs/predictions"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)

    # Minimum number of positive pixels required in the GT mask.
    # Use 1 to reject only fully black masks.
    # Use e.g. 100 or 500 to reject almost-empty masks too.
    parser.add_argument("--min_mask_pixels", type=int, default=1)

    return parser.parse_args()


def build_model(architecture, encoder):
    architecture = architecture.lower()

    common_args = {
        "encoder_name": encoder,
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 1,
    }

    if architecture == "unet":
        return smp.Unet(**common_args)

    elif architecture == "fpn":
        return smp.FPN(**common_args)

    elif architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common_args)

    elif architecture == "pspnet":
        return smp.PSPNet(**common_args)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def get_valid_indices(dataset, min_mask_pixels):
    valid_indices = []

    for i in range(len(dataset)):
        _, mask_tensor = dataset[i]

        positive_pixels = torch.count_nonzero(mask_tensor > 0).item()

        if positive_pixels >= min_mask_pixels:
            valid_indices.append(i)

    return valid_indices


def main():
    args = parse_args()

    if args.min_mask_pixels < 1:
        raise ValueError("--min_mask_pixels must be at least 1.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    run_name = args.run_name
    architecture = args.architecture
    encoder = args.encoder
    num_samples = args.num_samples
    threshold = args.threshold

    model_path = f"models/{run_name}.pth"

    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = BuildingDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    model = build_model(architecture, encoder).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("Device:", DEVICE)
    print("Model:", model_path)
    print("Architecture:", architecture)
    print("Encoder:", encoder)
    print("Prediction threshold:", threshold)
    print("Minimum GT mask pixels:", args.min_mask_pixels)

    valid_indices = get_valid_indices(dataset, args.min_mask_pixels)

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No validation samples found with at least "
            f"{args.min_mask_pixels} positive mask pixels."
        )

    print(f"Found {len(valid_indices)} valid samples.")

    num_samples = min(num_samples, len(valid_indices))
    indices = random.sample(valid_indices, num_samples)

    for out_idx, idx in enumerate(indices):
        img_tensor, mask_tensor = dataset[idx]

        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(img_input)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_mask = prob > threshold

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

        out_path = os.path.join(
            OUT_DIR,
            f"{run_name}_prediction_{out_idx:02d}_sample_{idx}.png"
        )

        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()