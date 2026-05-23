import os
import re
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


INRIA_PUBLIC_CITIES = ("austin", "chicago", "kitsap", "tyrol-w", "vienna")
_IMAGE_NAME_RE = re.compile(r"^(?P<city>.+?)(?P<image_id>\d+)$")


def parse_inria_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _IMAGE_NAME_RE.match(stem)

    if match is None:
        raise ValueError(f"Could not parse INRIA image name: {path}")

    city = match.group("city").lower()
    image_id = int(match.group("image_id"))

    return city, image_id


def image_id_list(values):
    if values is None:
        return None

    if isinstance(values, str):
        return [int(item.strip()) for item in values.split(",") if item.strip()]

    return [int(item) for item in values]


def describe_image_ids(values):
    values = image_id_list(values)
    if values is None:
        return ""

    return ",".join(str(item) for item in values)


def collect_image_mask_pairs(image_dir, mask_dir, image_ids, cities=None):
    image_ids = set(image_id_list(image_ids))
    cities = set(city.lower() for city in (cities or INRIA_PUBLIC_CITIES))

    images_by_stem = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob(os.path.join(image_dir, "*.tif"))
    }
    masks_by_stem = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob(os.path.join(mask_dir, "*.tif"))
    }

    pairs = []

    for stem, image_path in sorted(images_by_stem.items()):
        city, image_id = parse_inria_name(stem)

        if city not in cities or image_id not in image_ids:
            continue

        mask_path = masks_by_stem.get(stem)

        if mask_path is None:
            raise RuntimeError(f"Missing mask for image: {image_path}")

        pairs.append((image_path, mask_path))

    expected = len(cities) * len(image_ids)

    if len(pairs) != expected:
        raise RuntimeError(
            f"Expected {expected} INRIA image/mask pairs for cities "
            f"{sorted(cities)} and image ids {sorted(image_ids)}, found {len(pairs)}."
        )

    return pairs


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
