import numpy as np

from postprocess import postprocess_mask


def test_postprocess_removes_small_components_and_keeps_large_region():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1, 1] = 1
    mask[4:9, 4:9] = 1

    cleaned = postprocess_mask(mask, min_area=4, open_kernel_size=1)

    assert cleaned[1, 1] == 0
    assert cleaned[6, 6] == 1
    assert cleaned.dtype == np.uint8
