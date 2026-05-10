import numpy as np

from vectorize import mask_to_contours, simplify_contours


def test_polygon_extraction_finds_and_simplifies_building_region():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:14, 5:16] = 1

    contours = mask_to_contours(mask, min_area=10)
    polygons = simplify_contours(contours, epsilon_ratio=0.01)

    assert len(contours) == 1
    assert len(polygons) == 1
    assert len(polygons[0]) >= 4
