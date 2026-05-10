import os
import csv
import argparse
import random
import sys

import cv2
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath("src"))

from dataset import BuildingDataset
from config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from gis_utils import denormalize_image
from models import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)

    parser.add_argument("--all_models", action="store_true")

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--num_grids",
        type=int,
        default=None,
        help="Number of different random prediction grids to generate.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_mask_pixels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

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


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_valid_indices(dataset, min_mask_pixels):
    valid_indices = []

    print("Checking validation masks...")

    for i, mask_path in enumerate(dataset.mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: could not read mask: {mask_path}")
            continue

        positive_pixels = (mask > 127).sum()

        if positive_pixels >= min_mask_pixels:
            valid_indices.append(i)

    return valid_indices


def load_experiments_from_csv(experiments_csv, models_dir):
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


def get_experiments(args):
    if args.all_models:
        experiments = load_experiments_from_csv(
            args.experiments_csv,
            args.models_dir,
        )

        if len(experiments) == 0:
            raise RuntimeError("No valid models found from experiments CSV.")

        return experiments

    if args.run_name is None or args.architecture is None or args.encoder is None:
        raise ValueError(
            "For single-model mode, you must provide "
            "--run_name, --architecture and --encoder. "
            "Alternatively, use --all_models."
        )

    model_path = os.path.join(args.models_dir, f"{args.run_name}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return [
        {
            "run_name": args.run_name,
            "architecture": args.architecture,
            "encoder": args.encoder,
            "model_path": model_path,
        }
    ]


def save_prediction_visualization(
    model,
    run_name,
    dataset,
    indices,
    threshold,
    out_dir,
    grid_idx,
):
    run_out_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_out_dir, exist_ok=True)

    n_rows = len(indices)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 3.2 * n_rows),
    )

    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    column_titles = [
        "Input image",
        "Ground truth",
        "Prediction",
        "Overlay",
    ]

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=12)

    for row_idx, idx in enumerate(indices):
        img_tensor, mask_tensor = dataset[idx]

        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(img_input)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_mask = prob > threshold
        img = denormalize_image(img_tensor)
        gt_mask = mask_tensor[0].cpu().numpy()

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_mask, cmap="gray", interpolation="nearest")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred_mask, cmap="gray", interpolation="nearest")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(img)
        axes[row_idx, 3].imshow(
            pred_mask,
            cmap="Reds",
            alpha=0.35,
            interpolation="nearest",
        )
        axes[row_idx, 3].axis("off")

    plt.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.96,
        bottom=0.02,
        wspace=0.04,
        hspace=0.08,
    )

    sample_tag = "_".join(str(idx) for idx in indices)

    out_path = os.path.join(
        run_out_dir,
        f"{run_name}_grid_{grid_idx:02d}_samples_{sample_tag}.png",
    )

    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved {out_path}")


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
        "prediction_out_dir",
    )
    args.num_samples = select_value(
        args.num_samples,
        config,
        "analysis",
        "num_samples",
        default=3,
    )
    args.num_grids = select_value(
        args.num_grids,
        config,
        "analysis",
        "num_prediction_grids",
        default=1,
    )
    args.threshold = select_value(
        args.threshold,
        config,
        "analysis",
        "threshold",
        default=0.5,
    )
    args.min_mask_pixels = select_value(
        args.min_mask_pixels,
        config,
        "analysis",
        "min_mask_pixels",
        default=100,
    )
    args.seed = select_value(args.seed, config, "analysis", "seed", default=42)

    if val_image_dir is None:
        raise ValueError("No validation image directory provided. Set data.val_image_dir or pass --val_image_dir.")

    if val_mask_dir is None:
        raise ValueError("No validation mask directory provided. Set data.val_mask_dir or pass --val_mask_dir.")

    if args.experiments_csv is None:
        raise ValueError("No experiments CSV provided. Set analysis.experiments_csv or pass --experiments_csv.")

    if args.out_dir is None:
        raise ValueError("No output directory provided. Set analysis.prediction_out_dir or pass --out_dir.")

    if args.num_grids < 1:
        raise ValueError("--num_grids must be at least 1.")

    if args.min_mask_pixels < 1:
        raise ValueError("--min_mask_pixels must be at least 1.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = BuildingDataset(
        val_image_dir,
        val_mask_dir,
        transform=get_val_transform(),
    )

    print("Device:", DEVICE)
    print("Prediction threshold:", args.threshold)
    print("Minimum GT mask pixels:", args.min_mask_pixels)
    print("Number of grids:", args.num_grids)

    valid_indices = get_valid_indices(dataset, args.min_mask_pixels)

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No validation samples found with at least "
            f"{args.min_mask_pixels} positive mask pixels."
        )

    print(f"Found {len(valid_indices)} valid samples.")

    num_samples = min(args.num_samples, len(valid_indices))

    experiments = get_experiments(args)

    print(f"Models to process: {len(experiments)}")

    for exp in experiments:
        run_name = exp["run_name"]
        architecture = exp["architecture"]
        encoder = exp["encoder"]
        model_path = exp["model_path"]

        print()
        print("=" * 80)
        print("Run:", run_name)
        print("Model:", model_path)
        print("Architecture:", architecture)
        print("Encoder:", encoder)

        model = build_model(architecture, encoder).to(DEVICE)
        model.load_state_dict(
            torch.load(
                model_path,
                weights_only=True,
                map_location=DEVICE,
            )
        )
        model.eval()

        for grid_idx in range(args.num_grids):
            indices = random.sample(valid_indices, num_samples)

            print()
            print(f"Grid {grid_idx + 1}/{args.num_grids}")
            print(f"Using {len(indices)} samples:")
            print(indices)

            save_prediction_visualization(
                model=model,
                run_name=run_name,
                dataset=dataset,
                indices=indices,
                threshold=args.threshold,
                out_dir=args.out_dir,
                grid_idx=grid_idx,
            )

        del model

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
