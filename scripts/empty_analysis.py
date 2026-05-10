import argparse
import os
import sys

import cv2
from glob import glob
from tqdm import tqdm

sys.path.append(os.path.abspath("src"))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mask_dir", type=str, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def main():
    args = parse_args()
    config = load_config(args.config)

    mask_dir = select_value(
        args.mask_dir,
        config,
        "data",
        "train_mask_dir",
    )

    if mask_dir is None:
        raise ValueError("No mask directory provided. Set data.train_mask_dir or pass --mask_dir.")

    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    if len(mask_paths) == 0:
        raise RuntimeError(f"No mask tiles found in {mask_dir}")

    empty = 0
    non_empty = 0
    building_pixels = 0
    total_pixels = 0

    for p in tqdm(mask_paths):
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Could not read mask tile: {p}")

        binary = mask > 127

        if binary.sum() == 0:
            empty += 1
        else:
            non_empty += 1

        building_pixels += binary.sum()
        total_pixels += binary.size

    print("Mask dir:", mask_dir)
    print("Total masks:", len(mask_paths))
    print("Empty masks:", empty)
    print("Non-empty masks:", non_empty)
    print("Empty %:", empty / len(mask_paths) * 100)
    print("Building pixel %:", building_pixels / total_pixels * 100)


if __name__ == "__main__":
    main()
