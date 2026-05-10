import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image tiles found in: {image_dir}")

        if len(self.mask_paths) == 0:
            raise RuntimeError(f"No mask tiles found in: {mask_dir}")

        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                f"Image/mask count mismatch: "
                f"{len(self.image_paths)} images vs {len(self.mask_paths)} masks"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])

        if img is None:
            raise RuntimeError(f"Could not read image tile: {self.image_paths[idx]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Could not read mask tile: {self.mask_paths[idx]}")

        mask = (mask > 127).astype("float32")

        if self.transform is None:
            img = np.transpose(img, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)
            return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

        augmented = self.transform(image=img, mask=mask)

        img = augmented["image"]  # [3, H, W]
        mask = augmented["mask"].unsqueeze(0) # [1, H, W]

        return img.float(), mask.float()
