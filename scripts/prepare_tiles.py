import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath("src"))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from dataset import (
    collect_image_mask_pairs,
    describe_image_ids,
    image_id_list,
)


# Suppress OpenCV GeoTIFF metadata warnings.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--test_image_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train_image_ids", type=str, default=None)
    parser.add_argument("--val_image_ids", type=str, default=None)
    parser.add_argument("--test_image_ids", type=str, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def require_value(value, key_name):
    if value is None:
        raise ValueError(f"Missing required config value: {key_name}")

    return value


def reset_output_dir(out_dir):
    if out_dir is None:
        raise ValueError("Tile output directory cannot be None.")

    if not str(out_dir).strip() or str(out_dir).strip() == ".":
        raise ValueError("Unsafe tile output directory. Refusing to delete it.")

    output_path = Path(out_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    filesystem_root = Path(output_path.anchor).resolve()

    if output_path == repo_root:
        raise ValueError("Unsafe tile output directory: cannot delete repository root.")

    if output_path == filesystem_root:
        raise ValueError("Unsafe tile output directory: cannot delete filesystem root.")

    if output_path.exists():
        print(f"Removing existing tile output directory: {output_path}")
        shutil.rmtree(output_path)

    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "masks").mkdir(parents=True, exist_ok=True)

    (output_path / "public_test" / "images").mkdir(parents=True, exist_ok=True)


def get_test_files(test_image_dir):
    test_images = sorted(glob(os.path.join(test_image_dir, "*.tif")))

    if len(test_images) == 0:
        print(f"Warning: no test images found in: {test_image_dir}")

    return test_images


def tile_image_and_mask(img, mask, tile_size, stride):
    tiles = []
    h, w = img.shape[:2]

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            img_tile = img[y:y + tile_size, x:x + tile_size]
            mask_tile = mask[y:y + tile_size, x:x + tile_size]

            if img_tile.shape[:2] == (tile_size, tile_size):
                tiles.append((img_tile, mask_tile, y, x))

    return tiles


def tile_image_only(img, tile_size, stride):
    tiles = []
    h, w = img.shape[:2]

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            img_tile = img[y:y + tile_size, x:x + tile_size]

            if img_tile.shape[:2] == (tile_size, tile_size):
                tiles.append((img_tile, y, x))

    return tiles


def process_labeled_split(split, pairs, out_dir, tile_size, stride):
    split_data = {
        split: pairs,
    }

    for split, split_pairs in split_data.items():
        print(f"Processing {split} set...")

        total_tiles = 0

        for img_path, mask_path in tqdm(
            split_pairs,
            desc=f"{split}",
            unit="img",
        ):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise RuntimeError(f"Could not read image: {img_path}")

            if mask is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")

            mask = (mask > 127).astype(np.uint8) * 255

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            tiles = tile_image_and_mask(img, mask, tile_size, stride)

            for tile_img, tile_mask, y, x in tiles:
                name = f"{base_name}_y{y:04d}_x{x:04d}.png"

                out_img_path = os.path.join(out_dir, split, "images", name)
                out_mask_path = os.path.join(out_dir, split, "masks", name)

                cv2.imwrite(out_img_path, tile_img)
                cv2.imwrite(out_mask_path, tile_mask)

            total_tiles += len(tiles)

        print(f"{split} done. Saved {total_tiles} tiles.")


def process_public_unlabeled_test(test_image_dir, out_dir, tile_size, stride):
    test_images = get_test_files(test_image_dir)

    print("Processing test set...")

    total_tiles = 0

    for img_path in tqdm(test_images, desc="test", unit="img"):
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Could not read test image: {img_path}")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        tiles = tile_image_only(img, tile_size, stride)

        for tile_img, y, x in tiles:
            name = f"{base_name}_y{y:04d}_x{x:04d}.png"
            out_img_path = os.path.join(out_dir, "public_test", "images", name)
            cv2.imwrite(out_img_path, tile_img)

        total_tiles += len(tiles)

    print(f"test done. Saved {total_tiles} tiles.")


def main():
    args = parse_args()
    config = load_config(args.config)

    image_dir = select_value(
        args.image_dir,
        config,
        "data",
        "raw_train_image_dir",
    )
    mask_dir = select_value(
        args.mask_dir,
        config,
        "data",
        "raw_train_mask_dir",
    )
    test_image_dir = select_value(
        args.test_image_dir,
        config,
        "data",
        "raw_test_image_dir",
    )
    out_dir = select_value(args.out_dir, config, "data", "tile_dir")
    protocol = select_value(
        None,
        config,
        "protocol",
        "name",
    )
    tile_size = select_value(args.tile_size, config, "tiling", "tile_size", default=256)
    stride = select_value(args.stride, config, "tiling", "stride", default=256)
    train_image_ids = image_id_list(select_value(
        args.train_image_ids,
        config,
        "protocol",
        "train_image_ids",
    ))
    val_image_ids = image_id_list(select_value(
        args.val_image_ids,
        config,
        "protocol",
        "val_image_ids",
    ))
    test_image_ids = image_id_list(select_value(
        args.test_image_ids,
        config,
        "protocol",
        "test_image_ids",
    ))

    protocol = require_value(protocol, "protocol.name")
    train_image_ids = require_value(train_image_ids, "protocol.train_image_ids")
    val_image_ids = require_value(val_image_ids, "protocol.val_image_ids")
    test_image_ids = require_value(test_image_ids, "protocol.test_image_ids")

    if image_dir is None:
        raise ValueError("No raw training image directory provided. Set data.raw_train_image_dir or pass --image_dir.")

    if mask_dir is None:
        raise ValueError("No raw training mask directory provided. Set data.raw_train_mask_dir or pass --mask_dir.")

    if out_dir is None:
        raise ValueError("No tile output directory provided. Set data.tile_dir or pass --out_dir.")

    print("Image dir:", image_dir)
    print("Mask dir:", mask_dir)
    print("Public unlabeled test image dir:", test_image_dir)
    print("Output dir:", out_dir)
    print("Tile size:", tile_size)
    print("Stride:", stride)
    print("Protocol:", protocol)
    print("Train image ids:", describe_image_ids(train_image_ids))
    print("Validation image ids:", describe_image_ids(val_image_ids))
    print("Test image ids:", describe_image_ids(test_image_ids))

    reset_output_dir(out_dir)
    process_labeled_split(
        "train",
        collect_image_mask_pairs(image_dir, mask_dir, train_image_ids),
        out_dir,
        tile_size,
        stride,
    )
    process_labeled_split(
        "val",
        collect_image_mask_pairs(image_dir, mask_dir, val_image_ids),
        out_dir,
        tile_size,
        stride,
    )
    process_labeled_split(
        "test",
        collect_image_mask_pairs(image_dir, mask_dir, test_image_ids),
        out_dir,
        tile_size,
        stride,
    )

    if test_image_dir is not None:
        process_public_unlabeled_test(test_image_dir, out_dir, tile_size, stride)

    print("All tiling completed.")


if __name__ == "__main__":
    main()
