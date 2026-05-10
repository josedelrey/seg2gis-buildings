import os
import sys
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath("src"))

from dataset import BuildingDataset
from config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from train import IMAGENET_MEAN, IMAGENET_STD, get_train_transform


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)

    parser.add_argument(
        "--augmentation_type",
        type=str,
        default=None,
        choices=["noaug", "geomaug", "mildaug", "strongaug"],
    )

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_augs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def select_value(cli_value, config, *keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_config_value(config, *keys, default=default)


def tensor_img_to_uint8(img):
    # CHW -> HWC
    img_np = img.permute(1, 2, 0).cpu().numpy()

    if img_np.min() < 0.0:
        mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD).reshape(1, 1, 3)
        img_np = img_np * std + mean

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


def to_display_image(img):
    if isinstance(img, torch.Tensor):
        return tensor_img_to_uint8(img)

    return img.clip(0, 255).astype(np.uint8)


def main():
    args = parse_args()
    config = load_config(args.config)

    image_dir = select_value(
        args.image_dir,
        config,
        "data",
        "train_image_dir",
    )
    mask_dir = select_value(
        args.mask_dir,
        config,
        "data",
        "train_mask_dir",
    )
    augmentation_type = select_value(
        args.augmentation_type,
        config,
        "analysis",
        "augmentation_type",
        default="mildaug",
    )
    num_samples_arg = select_value(
        args.num_samples,
        config,
        "analysis",
        "num_samples",
        default=4,
    )
    num_augs = select_value(
        args.num_augs,
        config,
        "analysis",
        "num_augs",
        default=4,
    )
    seed = select_value(args.seed, config, "analysis", "seed", default=42)

    if image_dir is None:
        raise ValueError("No image directory provided. Set data.train_image_dir or pass --image_dir.")

    if mask_dir is None:
        raise ValueError("No mask directory provided. Set data.train_mask_dir or pass --mask_dir.")

    random.seed(seed)
    np.random.seed(seed)

    transform = get_train_transform(augmentation_type)

    if transform is None:
        print("augmentation_type=noaug selected. Showing repeated originals.")

    dataset = BuildingDataset(
        image_dir,
        mask_dir,
        transform=None,
    )

    num_samples = min(num_samples_arg, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

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
                aug_img = to_display_image(augmented["image"])

            plt.sca(axes[i + 1])
            plot_sample(aug_img, title=f"{augmentation_type} {i + 1}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
