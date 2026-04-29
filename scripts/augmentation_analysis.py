import os
import sys
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("src"))

from dataset import BuildingDataset
from train import get_train_transform


IMG_DIR = "data/tiles_256/train/images"
MASK_DIR = "data/tiles_256/train/masks"

NUM_SAMPLES = 4
NUM_AUGS = 4


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--augmentation_type",
        type=str,
        default="mildaug",
        choices=["noaug", "geomaug", "mildaug", "strongaug"],
    )

    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num_augs", type=int, default=NUM_AUGS)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def tensor_img_to_uint8(img):
    # CHW -> HWC
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # If image is in [0, 1], convert to [0, 255].
    # If already [0, 255], just clip.
    if img_np.max() <= 1.0:
        img_np = img_np * 255.0

    return img_np.clip(0, 255).astype(np.uint8)


def tensor_mask_to_np(mask):
    mask_np = mask.squeeze().cpu().numpy()

    # Make sure mask is display-friendly.
    if mask_np.max() > 1:
        mask_np = mask_np / 255.0

    return mask_np.astype(np.float32)


def plot_sample(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    transform = get_train_transform(args.augmentation_type)

    if transform is None:
        print("augmentation_type=noaug selected. Showing repeated originals.")

    dataset = BuildingDataset(
        IMG_DIR,
        MASK_DIR,
        transform=None,
    )

    indices = random.sample(range(len(dataset)), args.num_samples)

    for idx in indices:
        img, mask = dataset[idx]

        img_np = tensor_img_to_uint8(img)
        mask_np = tensor_mask_to_np(mask)

        fig, axes = plt.subplots(
            1,
            args.num_augs + 1,
            figsize=(4 * (args.num_augs + 1), 4),
        )

        plt.sca(axes[0])
        plot_sample(img_np, title=f"Original idx={idx}")

        for i in range(args.num_augs):
            if transform is None:
                aug_img = img_np.copy()
                aug_mask = mask_np.copy()
            else:
                augmented = transform(image=img_np, mask=mask_np)
                aug_img = augmented["image"]
                aug_mask = augmented["mask"]

            plt.sca(axes[i + 1])
            plot_sample(aug_img, title=f"{args.augmentation_type} {i + 1}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()