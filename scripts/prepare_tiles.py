import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Suppress OpenCV GeoTIFF metadata warnings
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except:
    pass

# -----------------------
# CONFIG
# -----------------------
IMG_DIR = "data/AerialImageDataset/train/images"
MASK_DIR = "data/AerialImageDataset/train/gt"
TEST_IMG_DIR = "data/AerialImageDataset/test/images"

OUT_DIR = "data/tiles_256"

TILE_SIZE = 256
STRIDE = 256  # no overlap

VAL_SPLIT = 0.2
SEED = 42

# -----------------------
# CREATE OUTPUT FOLDERS
# -----------------------
def make_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "masks"), exist_ok=True)

    os.makedirs(os.path.join(OUT_DIR, "test", "images"), exist_ok=True)

# -----------------------
# GET FILE LISTS
# -----------------------
def get_train_files():
    images = sorted(glob(os.path.join(IMG_DIR, "*.tif")))
    masks = sorted(glob(os.path.join(MASK_DIR, "*.tif")))

    if len(images) == 0:
        raise RuntimeError(f"No training images found in: {IMG_DIR}")

    if len(masks) == 0:
        raise RuntimeError(f"No masks found in: {MASK_DIR}")

    if len(images) != len(masks):
        raise RuntimeError(
            f"Number of images and masks does not match: "
            f"{len(images)} images vs {len(masks)} masks"
        )

    return images, masks


def get_test_files():
    test_images = sorted(glob(os.path.join(TEST_IMG_DIR, "*.tif")))

    if len(test_images) == 0:
        print(f"Warning: no test images found in: {TEST_IMG_DIR}")

    return test_images

# -----------------------
# TILE FUNCTIONS
# -----------------------
def tile_image_and_mask(img, mask, tile_size=256, stride=256):
    tiles = []
    h, w = img.shape[:2]

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            img_tile = img[y:y + tile_size, x:x + tile_size]
            mask_tile = mask[y:y + tile_size, x:x + tile_size]

            if img_tile.shape[:2] == (tile_size, tile_size):
                tiles.append((img_tile, mask_tile, y, x))

    return tiles


def tile_image_only(img, tile_size=256, stride=256):
    tiles = []
    h, w = img.shape[:2]

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            img_tile = img[y:y + tile_size, x:x + tile_size]

            if img_tile.shape[:2] == (tile_size, tile_size):
                tiles.append((img_tile, y, x))

    return tiles

# -----------------------
# PROCESS TRAIN / VAL
# -----------------------
def process_train_val():
    images, masks = get_train_files()

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=VAL_SPLIT,
        random_state=SEED
    )

    split_data = {
        "train": (train_imgs, train_masks),
        "val": (val_imgs, val_masks)
    }

    for split, (img_list, mask_list) in split_data.items():
        print(f"Processing {split} set...")

        total_tiles = 0

        for img_path, mask_path in tqdm(
            list(zip(img_list, mask_list)),
            desc=f"{split}",
            unit="img"
        ):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise RuntimeError(f"Could not read image: {img_path}")

            if mask is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")

            mask = (mask > 127).astype(np.uint8) * 255

            base_name = os.path.splitext(os.path.basename(img_path))[0]

            tiles = tile_image_and_mask(img, mask, TILE_SIZE, STRIDE)

            for tile_img, tile_mask, y, x in tiles:
                name = f"{base_name}_y{y:04d}_x{x:04d}.png"

                out_img_path = os.path.join(OUT_DIR, split, "images", name)
                out_mask_path = os.path.join(OUT_DIR, split, "masks", name)

                cv2.imwrite(out_img_path, tile_img)
                cv2.imwrite(out_mask_path, tile_mask)

            total_tiles += len(tiles)

        print(f"{split} done. Saved {total_tiles} tiles.")

# -----------------------
# PROCESS TEST
# -----------------------
def process_test():
    test_images = get_test_files()

    print("Processing test set...")

    total_tiles = 0

    for img_path in tqdm(
        test_images,
        desc="test",
        unit="img"
    ):
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Could not read test image: {img_path}")

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        tiles = tile_image_only(img, TILE_SIZE, STRIDE)

        for tile_img, y, x in tiles:
            name = f"{base_name}_y{y:04d}_x{x:04d}.png"
            out_img_path = os.path.join(OUT_DIR, "test", "images", name)
            cv2.imwrite(out_img_path, tile_img)

        total_tiles += len(tiles)

    print(f"test done. Saved {total_tiles} tiles.")

# -----------------------
# MAIN
# -----------------------
def main():
    make_dirs()
    process_train_val()
    process_test()
    print("All tiling completed.")


if __name__ == "__main__":
    main()