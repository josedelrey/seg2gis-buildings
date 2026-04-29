import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("src"))

from dataset import BuildingDataset
from train import get_train_transform


VAL_IMG_DIR = "data/tiles_256/train/images"
VAL_MASK_DIR = "data/tiles_256/train/masks"

NUM_SAMPLES = 4        # how many different images
NUM_AUGS = 4           # augmentations per image


def denormalize(img):
    # If you normalize in dataset, undo here.
    # Otherwise just return as is.
    return img


def plot_sample(img, mask, title=""):
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4, cmap="Reds")
    plt.title(title)
    plt.axis("off")


def main():
    transform = get_train_transform(use_augmentation=True)

    dataset = BuildingDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        transform=None  # IMPORTANT: we apply aug manually
    )

    indices = random.sample(range(len(dataset)), NUM_SAMPLES)

    for idx in indices:
        img, mask = dataset[idx]

        # CHW → HWC
        img_np = img.permute(1, 2, 0).numpy()

        # If your dataset outputs [0,1], convert to uint8
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        mask_np = mask.squeeze().numpy()

        fig, axes = plt.subplots(1, NUM_AUGS + 1, figsize=(15, 4))

        # Original
        plt.sca(axes[0])
        plot_sample(img_np, mask_np, title="Original")

        # Augmented versions
        for i in range(NUM_AUGS):
            augmented = transform(image=img_np, mask=mask_np)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            plt.sca(axes[i + 1])
            plot_sample(aug_img, aug_mask, title=f"Aug {i+1}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()