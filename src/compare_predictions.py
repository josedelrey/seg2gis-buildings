import os
import csv
import argparse
import random
import math
import numpy as np
import cv2

import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import BuildingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_IMG_DIR = "data/tiles_256/val/images"
VAL_MASK_DIR = "data/tiles_256/val/masks"

MODELS_DIR = "models"
EXPERIMENTS_CSV = "outputs/experiments.csv"
OUT_DIR = "outputs/model_comparisons"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--plots_per_row", type=int, default=4)

    # Minimum number of positive pixels required in the GT mask.
    # Use 1 to reject only fully black masks.
    # Use e.g. 100 or 500 to reject almost-empty masks too.
    parser.add_argument("--min_mask_pixels", type=int, default=1)

    parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
    parser.add_argument("--experiments_csv", type=str, default=EXPERIMENTS_CSV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)

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


def load_experiments(experiments_csv, models_dir):
    experiments = []

    with open(experiments_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            run_name = row["run_name"]
            model_path = os.path.join(models_dir, f"{run_name}.pth")

            if not os.path.exists(model_path):
                print(f"Skipping {run_name}: model file not found.")
                continue

            experiments.append({
                "run_name": run_name,
                "architecture": row["architecture"],
                "encoder": row["encoder"],
                "model_path": model_path,
            })

    return experiments


def load_models(experiments):
    loaded_models = []

    for exp in experiments:
        print(f"Loading {exp['run_name']}...")

        model = build_model(exp["architecture"], exp["encoder"]).to(DEVICE)
        model.load_state_dict(torch.load(exp["model_path"], map_location=DEVICE))
        model.eval()

        loaded_models.append({
            "run_name": exp["run_name"],
            "architecture": exp["architecture"],
            "encoder": exp["encoder"],
            "model": model,
        })

    return loaded_models


def find_non_empty_mask_indices(dataset, min_mask_pixels):
    """
    Finds validation samples whose ground-truth mask is not empty.

    Fast path:
        If the dataset exposes mask paths, read masks directly with cv2.

    Fallback:
        If the dataset does not expose mask paths, use dataset[i].
        This is slower because it loads the image too and applies dataset processing.
    """

    valid_indices = []

    mask_paths = None

    for attr_name in ["mask_paths", "masks", "mask_files"]:
        if hasattr(dataset, attr_name):
            mask_paths = getattr(dataset, attr_name)
            break

    if mask_paths is not None:
        print("Checking masks using fast path: direct cv2 mask loading.")

        for i, mask_path in enumerate(mask_paths):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: could not read mask: {mask_path}")
                continue

            positive_pixels = np.count_nonzero(mask > 0)

            if positive_pixels >= min_mask_pixels:
                valid_indices.append(i)

    else:
        print("Checking masks using fallback path: dataset[i].")
        print("This is slower because it loads the image and mask through the dataset.")

        for i in range(len(dataset)):
            _, mask_tensor = dataset[i]

            positive_pixels = torch.count_nonzero(mask_tensor > 0).item()

            if positive_pixels >= min_mask_pixels:
                valid_indices.append(i)

    return valid_indices


def main():
    args = parse_args()

    if args.plots_per_row < 1:
        raise ValueError("--plots_per_row must be at least 1.")

    if args.min_mask_pixels < 1:
        raise ValueError("--min_mask_pixels must be at least 1.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)

    dataset = BuildingDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    experiments = load_experiments(args.experiments_csv, args.models_dir)

    if len(experiments) == 0:
        raise RuntimeError("No valid models found.")

    models = load_models(experiments)

    print("Device:", DEVICE)
    print(f"Loaded {len(models)} models.")
    print(f"Prediction threshold: {args.threshold}")
    print(f"Plots per row: {args.plots_per_row}")
    print(f"Minimum GT mask pixels: {args.min_mask_pixels}")

    valid_indices = find_non_empty_mask_indices(
        dataset=dataset,
        min_mask_pixels=args.min_mask_pixels,
    )

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No validation samples found with at least "
            f"{args.min_mask_pixels} positive mask pixels."
        )

    print(f"Found {len(valid_indices)} valid samples.")

    num_samples = min(args.num_samples, len(valid_indices))
    indices = random.sample(valid_indices, num_samples)

    for out_idx, idx in enumerate(indices):
        img_tensor, mask_tensor = dataset[idx]

        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        gt_mask = mask_tensor[0].cpu().numpy()

        predictions = []

        with torch.inference_mode():
            for item in models:
                logits = item["model"](img_input)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                pred_mask = prob > args.threshold

                predictions.append({
                    "run_name": item["run_name"],
                    "mask": pred_mask,
                })

        plots = [
            {
                "title": "Input image",
                "kind": "image",
                "data": img,
            },
            {
                "title": "Ground truth",
                "kind": "mask",
                "data": gt_mask,
            },
        ]

        for pred in predictions:
            plots.append({
                "title": pred["run_name"],
                "kind": "mask",
                "data": pred["mask"],
            })

        n_plots = len(plots)
        n_cols = min(args.plots_per_row, n_plots)
        n_rows = math.ceil(n_plots / args.plots_per_row)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
        )

        axes = np.array(axes).ravel()

        for ax, item in zip(axes, plots):
            if item["kind"] == "image":
                ax.imshow(item["data"])
            else:
                ax.imshow(item["data"], cmap="gray", interpolation="nearest")

            ax.set_title(item["title"], fontsize=10)
            ax.axis("off")

        for ax in axes[len(plots):]:
            ax.axis("off")

        plt.tight_layout()

        out_path = os.path.join(
            args.out_dir,
            f"comparison_{out_idx:02d}_sample_{idx}.png"
        )

        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()