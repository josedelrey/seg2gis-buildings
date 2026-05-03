import os
from glob import glob

import cv2
import torch
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        augmented = self.transform(image=img, mask=mask)

        img = augmented["image"]              # [3, H, W]
        mask = augmented["mask"].unsqueeze(0) # [1, H, W]

        return img.float(), mask.float()