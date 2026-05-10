import cv2
import numpy as np

from dataset import BuildingDataset


def test_building_dataset_loads_png_tiles_without_transform(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    image_bgr[:, :, 0] = 10
    image_bgr[:, :, 1] = 20
    image_bgr[:, :, 2] = 30

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    cv2.imwrite(str(image_dir / "tile.png"), image_bgr)
    cv2.imwrite(str(mask_dir / "tile.png"), mask)

    dataset = BuildingDataset(str(image_dir), str(mask_dir), transform=None)

    image_tensor, mask_tensor = dataset[0]

    assert len(dataset) == 1
    assert tuple(image_tensor.shape) == (3, 8, 8)
    assert tuple(mask_tensor.shape) == (1, 8, 8)
    assert image_tensor[0, 0, 0].item() == 30
    assert image_tensor[1, 0, 0].item() == 20
    assert image_tensor[2, 0, 0].item() == 10
    assert mask_tensor.max().item() == 1.0
    assert mask_tensor.min().item() == 0.0
