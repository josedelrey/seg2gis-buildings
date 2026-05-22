import os
import re
from glob import glob


INRIA_PUBLIC_CITIES = ("austin", "chicago", "kitsap", "tyrol-w", "vienna")
INRIA_TRAIN_IMAGE_IDS = tuple(range(11, 37))
INRIA_VAL_IMAGE_IDS = tuple(range(6, 11))
INRIA_TEST_IMAGE_IDS = tuple(range(1, 6))
INRIA_FINAL_TRAIN_IMAGE_IDS = tuple(range(6, 37))

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
        if values == "inria155_train":
            return list(INRIA_TRAIN_IMAGE_IDS)
        if values == "inria155_val":
            return list(INRIA_VAL_IMAGE_IDS)
        if values == "inria155_test":
            return list(INRIA_TEST_IMAGE_IDS)
        if values == "inria155_final_train":
            return list(INRIA_FINAL_TRAIN_IMAGE_IDS)

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
