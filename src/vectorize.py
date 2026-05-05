import cv2
import numpy as np


def mask_to_contours(mask, min_area=64):
    """
    Extract external contours from a binary mask.

    Args:
        mask: uint8 binary mask with values 0/1.
        min_area: minimum contour area in pixels.

    Returns:
        List of OpenCV contours.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            filtered.append(contour)

    return filtered


def simplify_contours(contours, epsilon_ratio=0.01):
    """
    Simplify contours using Douglas-Peucker approximation.
    """
    simplified = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_ratio * perimeter

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )

        if len(polygon) >= 3:
            simplified.append(polygon)

    return simplified


def draw_polygons_on_image(image_rgb, polygons, color=(255, 0, 0), thickness=2):
    """
    Draw polygon outlines on an RGB image.
    """
    overlay = image_rgb.copy()

    cv2.drawContours(
        overlay,
        polygons,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )

    return overlay