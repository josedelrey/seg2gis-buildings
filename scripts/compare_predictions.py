import os
import csv
import argparse
import random
import math
import sys

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("src"))

from dataset import BuildingDataset
from config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from gis_utils import denormalize_image
from models import build_model
from transforms import get_val_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--plots_per_row", type=int, default=None)

    parser.add_argument("--min_mask_pixels", type=int, default=None)

    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument("--val_mask_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--experiments_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def load_experiments(experiments_csv, models_dir):
    experiments = []

    with open(experiments_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            run_name = row["run_name"]
            model_path = os.path.join(models_dir, f"{run_name}.pth")

            if not os.path.exists(model_path):
                print(f"Skipping {run_name}: model file not found at {model_path}")
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
        model.load_state_dict(
            torch.load(
                exp["model_path"],
                weights_only=True,
                map_location=DEVICE,
            )
        )
        model.eval()

        loaded_models.append({
            "run_name": exp["run_name"],
            "architecture": exp["architecture"],
            "encoder": exp["encoder"],
            "model": model,
        })

    return loaded_models


def find_non_empty_mask_indices(dataset, min_mask_pixels):
    valid_indices = []

    mask_paths = None

    for attr_name in ["mask_paths", "masks", "mask_files"]:
        if hasattr(dataset, attr_name):
            mask_paths = getattr(dataset, attr_name)
            break

    if mask_paths is not None:
        print("Checking masks...")

        for i, mask_path in enumerate(mask_paths):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: could not read mask: {mask_path}")
                continue

            positive_pixels = np.count_nonzero(mask > 127)

            if positive_pixels >= min_mask_pixels:
                valid_indices.append(i)

    else:
        print("Checking masks using fallback path: dataset[i].")

        for i in range(len(dataset)):
            _, mask_tensor = dataset[i]

            positive_pixels = torch.count_nonzero(mask_tensor > 0).item()

            if positive_pixels >= min_mask_pixels:
                valid_indices.append(i)

    return valid_indices


def main():
    args = parse_args()
    config = load_config(args.config)

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
    args.models_dir = select_value(
        args.models_dir,
        config,
        "analysis",
        "models_dir",
        default="models",
    )
    args.experiments_csv = select_value(
        args.experiments_csv,
        config,
        "analysis",
        "experiments_csv",
    )
    args.out_dir = select_value(
        args.out_dir,
        config,
        "analysis",
        "comparison_out_dir",
    )
    args.num_samples = select_value(
        args.num_samples,
        config,
        "analysis",
        "num_comparison_samples",
        default=8,
    )
    args.threshold = select_value(
        args.threshold,
        config,
        "analysis",
        "threshold",
        default=0.5,
    )
    args.seed = select_value(args.seed, config, "analysis", "seed", default=42)
    args.plots_per_row = select_value(
        args.plots_per_row,
        config,
        "analysis",
        "plots_per_row",
        default=4,
    )
    args.min_mask_pixels = select_value(
        args.min_mask_pixels,
        config,
        "analysis",
        "min_mask_pixels",
        default=1,
    )

    if val_image_dir is None:
        raise ValueError("No validation image directory provided. Set data.val_image_dir or pass --val_image_dir.")

    if val_mask_dir is None:
        raise ValueError("No validation mask directory provided. Set data.val_mask_dir or pass --val_mask_dir.")

    if args.experiments_csv is None:
        raise ValueError("No experiments CSV provided. Set analysis.experiments_csv or pass --experiments_csv.")

    if args.out_dir is None:
        raise ValueError("No output directory provided. Set analysis.comparison_out_dir or pass --out_dir.")

    if args.plots_per_row < 1:
        raise ValueError("--plots_per_row must be at least 1.")

    if args.min_mask_pixels < 1:
        raise ValueError("--min_mask_pixels must be at least 1.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)

    dataset = BuildingDataset(
        val_image_dir,
        val_mask_dir,
        transform=get_val_transform(),
    )

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
        img = denormalize_image(img_tensor)
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
            figsize=(4.2 * n_cols, 4.2 * n_rows),
        )

        axes = np.array(axes).ravel()

        for ax, item in zip(axes, plots):
            if item["kind"] == "image":
                ax.imshow(item["data"])
            else:
                ax.imshow(item["data"], cmap="gray", interpolation="nearest")

            ax.set_title(
                item["title"],
                fontsize=18,
                pad=12,
            )
            ax.axis("off")

        for ax in axes[len(plots):]:
            ax.axis("off")

        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.95,
            bottom=0.03,
            wspace=0.12,
            hspace=0.22,
        )

        out_path = os.path.join(
            args.out_dir,
            f"comparison_{out_idx:02d}_sample_{idx}.png",
        )

        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
