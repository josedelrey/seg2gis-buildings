import numpy as np
import torch

from gis_utils import predict_full_image_tiled


class ConstantLogitModel(torch.nn.Module):
    def forward(self, x):
        batch_size, _, height, width = x.shape
        return torch.zeros(
            (batch_size, 1, height, width),
            dtype=x.dtype,
            device=x.device,
        )


def test_tiled_inference_preserves_original_image_shape():
    image = np.zeros((37, 45, 3), dtype=np.uint8)
    model = ConstantLogitModel()

    prob_map = predict_full_image_tiled(
        model=model,
        image_rgb=image,
        tile_size=16,
        stride=8,
        device="cpu",
    )

    assert prob_map.shape == image.shape[:2]
    assert prob_map.dtype == np.float32
    assert np.allclose(prob_map, 0.5)
