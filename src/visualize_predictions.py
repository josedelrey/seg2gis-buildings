import os
import csv
import argparse
import random

import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import BuildingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_IMG_DIR = "data/tiles_256/val/images"
VAL_MASK_DIR = "data/tiles_256/val/masks"

MODELS_DIR = "models"
EXPERIMENTS_CSV = "outputs/experiments.csv"
OUT_DIR = "outputs/predictions"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)

    parser.add_argument("--all_models", action="store_true")

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_mask_pixels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

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


def get_valid_indices(dataset, min_mask_pixels):
    valid_indices = []

    for i in range(len(dataset)):
        _, mask_tensor = dataset[i]
        positive_pixels = torch.count_nonzero(mask_tensor > 0).item()

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
                print(f"Skipping {run_name}: model file not found.")
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
):
    run_out_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_out_dir, exist_ok=True)

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
            run_out_dir,
            f"{run_name}_prediction_{out_idx:02d}_sample_{idx}.png"
        )

        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved {out_path}")


def main():
    args = parse_args()

    if args.min_mask_pixels < 1:
        raise ValueError("--min_mask_pixels must be at least 1.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = BuildingDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    print("Device:", DEVICE)
    print("Prediction threshold:", args.threshold)
    print("Minimum GT mask pixels:", args.min_mask_pixels)

    valid_indices = get_valid_indices(dataset, args.min_mask_pixels)

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No validation samples found with at least "
            f"{args.min_mask_pixels} positive mask pixels."
        )

    print(f"Found {len(valid_indices)} valid samples.")

    num_samples = min(args.num_samples, len(valid_indices))
    indices = random.sample(valid_indices, num_samples)

    print(f"Using {len(indices)} samples:")
    print(indices)

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
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        save_prediction_visualization(
            model=model,
            run_name=run_name,
            dataset=dataset,
            indices=indices,
            threshold=args.threshold,
            out_dir=args.out_dir,
        )

        del model

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print()
    print("Done.")


if __name__ == "__main__":
    main()