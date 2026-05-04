import math
import os

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(architecture, encoder):
    architecture = architecture.lower()

    common_args = {
        "encoder_name": encoder,
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 1,
    }

    if architecture == "unet":
        return smp.Unet(**common_args)

    if architecture == "fpn":
        return smp.FPN(**common_args)

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common_args)

    if architecture == "pspnet":
        return smp.PSPNet(**common_args)

    raise ValueError(f"Unsupported architecture: {architecture}")


def load_model(model_path, architecture, encoder, device):
    model = build_model(architecture, encoder).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def get_inference_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_rgb_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pad_image_to_tile_grid(image, tile_size, stride):
    """
    Pads image so tiled inference covers the full image.
    Padding is done with reflection to reduce border artifacts.
    """

    h, w = image.shape[:2]

    if h <= tile_size:
        padded_h = tile_size
    else:
        n_steps_h = math.ceil((h - tile_size) / stride)
        padded_h = n_steps_h * stride + tile_size

    if w <= tile_size:
        padded_w = tile_size
    else:
        n_steps_w = math.ceil((w - tile_size) / stride)
        padded_w = n_steps_w * stride + tile_size

    pad_h = padded_h - h
    pad_w = padded_w - w

    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded, h, w


def predict_tile(model, tile_rgb, transform, device):
    transformed = transform(image=tile_rgb)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return prob.astype(np.float32)


def predict_full_image_tiled(
    model,
    image_rgb,
    tile_size,
    stride,
    device,
):
    """
    Runs overlapping tiled inference and averages probabilities
    in overlapping areas.

    Returns:
        full_prob_map: float32 array of shape H x W in [0, 1]
    """

    if stride > tile_size:
        raise ValueError("stride should be <= tile_size for complete coverage.")

    transform = get_inference_transform()

    padded_image, original_h, original_w = pad_image_to_tile_grid(
        image=image_rgb,
        tile_size=tile_size,
        stride=stride,
    )

    padded_h, padded_w = padded_image.shape[:2]

    prob_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    y_positions = range(0, padded_h - tile_size + 1, stride)
    x_positions = range(0, padded_w - tile_size + 1, stride)

    for y in y_positions:
        for x in x_positions:
            tile = padded_image[y:y + tile_size, x:x + tile_size]

            prob = predict_tile(
                model=model,
                tile_rgb=tile,
                transform=transform,
                device=device,
            )

            prob_sum[y:y + tile_size, x:x + tile_size] += prob
            count_map[y:y + tile_size, x:x + tile_size] += 1.0

    if np.any(count_map == 0):
        raise RuntimeError("Some pixels were not covered by tiled inference.")

    full_prob = prob_sum / count_map

    full_prob = full_prob[:original_h, :original_w]

    return full_prob


def threshold_probability_map(prob_map, threshold):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    return (prob_map >= threshold).astype(np.uint8)


def save_probability_map(prob_map, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, prob_map)


def save_mask_png(mask, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(out_path, mask_uint8)


def save_probability_png(prob_map, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    prob_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, prob_uint8)