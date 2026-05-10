import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath("src"))

from config import DEFAULT_CONFIG_PATH, get_config_value, load_config


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
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def make_dirs(out_dir):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "masks"), exist_ok=True)

    os.makedirs(os.path.join(out_dir, "test", "images"), exist_ok=True)


def get_train_files(image_dir, mask_dir):
    images = sorted(glob(os.path.join(image_dir, "*.tif")))
    masks = sorted(glob(os.path.join(mask_dir, "*.tif")))

    if len(images) == 0:
        raise RuntimeError(f"No training images found in: {image_dir}")

    if len(masks) == 0:
        raise RuntimeError(f"No masks found in: {mask_dir}")

    if len(images) != len(masks):
        raise RuntimeError(
            f"Number of images and masks does not match: "
            f"{len(images)} images vs {len(masks)} masks"
        )

    return images, masks


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


def process_train_val(image_dir, mask_dir, out_dir, tile_size, stride, val_split, seed):
    images, masks = get_train_files(image_dir, mask_dir)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=val_split,
        random_state=seed,
    )

    split_data = {
        "train": (train_imgs, train_masks),
        "val": (val_imgs, val_masks),
    }

    for split, (img_list, mask_list) in split_data.items():
        print(f"Processing {split} set...")

        total_tiles = 0

        for img_path, mask_path in tqdm(
            list(zip(img_list, mask_list)),
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


def process_test(test_image_dir, out_dir, tile_size, stride):
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
            out_img_path = os.path.join(out_dir, "test", "images", name)
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
        default="data/AerialImageDataset/train/images",
    )
    mask_dir = select_value(
        args.mask_dir,
        config,
        "data",
        "raw_train_mask_dir",
        default="data/AerialImageDataset/train/gt",
    )
    test_image_dir = select_value(
        args.test_image_dir,
        config,
        "data",
        "raw_test_image_dir",
        default="data/AerialImageDataset/test/images",
    )
    out_dir = select_value(args.out_dir, config, "data", "tile_dir", default="data/tiles_256")
    tile_size = select_value(args.tile_size, config, "tiling", "tile_size", default=256)
    stride = select_value(args.stride, config, "tiling", "stride", default=256)
    val_split = select_value(args.val_split, config, "tiling", "val_split", default=0.2)
    seed = select_value(args.seed, config, "tiling", "seed", default=42)

    print("Image dir:", image_dir)
    print("Mask dir:", mask_dir)
    print("Test image dir:", test_image_dir)
    print("Output dir:", out_dir)
    print("Tile size:", tile_size)
    print("Stride:", stride)
    print("Validation split:", val_split)
    print("Seed:", seed)

    make_dirs(out_dir)
    process_train_val(image_dir, mask_dir, out_dir, tile_size, stride, val_split, seed)
    process_test(test_image_dir, out_dir, tile_size, stride)
    print("All tiling completed.")


if __name__ == "__main__":
    main()
