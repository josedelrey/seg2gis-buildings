import cv2
import numpy as np


def remove_small_components(mask, min_area=64):
    """
    Remove connected foreground components smaller than min_area.

    Args:
        mask: uint8 binary mask with values 0/1.
        min_area: minimum component area in pixels.

    Returns:
        cleaned uint8 mask with values 0/1.
    """
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned


def morphological_open(mask, kernel_size=3, iterations=1):
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
    )

    return opened.astype(np.uint8)


def postprocess_mask(
    mask,
    min_area=64,
    open_kernel_size=3,
):
    """
    Basic building-mask post-processing pipeline.

    Args:
        mask: uint8 binary mask with values 0/1.

    Returns:
        cleaned uint8 mask with values 0/1.
    """
    cleaned = mask.astype(np.uint8)

    cleaned = remove_small_components(
        cleaned,
        min_area=min_area,
    )

    cleaned = morphological_open(
        cleaned,
        kernel_size=open_kernel_size,
    )

    cleaned = remove_small_components(
        cleaned,
        min_area=min_area,
    )

    return cleaned.astype(np.uint8)