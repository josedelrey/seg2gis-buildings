import os
import argparse

import torch

from gis_utils import (
    load_model,
    load_rgb_image,
    predict_full_image_tiled,
    threshold_probability_map,
    save_probability_map,
    save_probability_png,
    save_mask_png,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUT_DIR = "outputs/full_predictions"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)

    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_name", type=str, default=None)

    return parser.parse_args()


def get_output_name(image_path, output_name):
    if output_name is not None:
        return output_name

    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    return name


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    output_name = get_output_name(args.image_path, args.output_name)

    prob_npy_path = os.path.join(args.out_dir, f"{output_name}_prob.npy")
    prob_png_path = os.path.join(args.out_dir, f"{output_name}_prob.png")
    mask_png_path = os.path.join(args.out_dir, f"{output_name}_mask.png")

    print("Device:", DEVICE)
    print("Image:", args.image_path)
    print("Model:", args.model_path)
    print("Architecture:", args.architecture)
    print("Encoder:", args.encoder)
    print("Threshold:", args.threshold)
    print("Tile size:", args.tile_size)
    print("Stride:", args.stride)

    image_rgb = load_rgb_image(args.image_path)

    print("Input image shape:", image_rgb.shape)

    model = load_model(
        model_path=args.model_path,
        architecture=args.architecture,
        encoder=args.encoder,
        device=DEVICE,
    )

    prob_map = predict_full_image_tiled(
        model=model,
        image_rgb=image_rgb,
        tile_size=args.tile_size,
        stride=args.stride,
        device=DEVICE,
    )

    mask = threshold_probability_map(
        prob_map=prob_map,
        threshold=args.threshold,
    )

    save_probability_map(prob_map, prob_npy_path)
    save_probability_png(prob_map, prob_png_path)
    save_mask_png(mask, mask_png_path)

    print()
    print("Saved probability map:", prob_npy_path)
    print("Saved probability preview:", prob_png_path)
    print("Saved binary mask:", mask_png_path)
    print("Done.")


if __name__ == "__main__":
    main()