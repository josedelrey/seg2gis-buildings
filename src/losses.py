import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp


SUPPORTED_LOSSES = ("dice_bce", "dice_boundary_bce")


def make_boundary_mask(masks, boundary_width):
    if boundary_width <= 0:
        return torch.zeros_like(masks)

    kernel_size = 2 * boundary_width + 1
    padding = boundary_width

    masks = (masks > 0.5).float()

    dilated = F.max_pool2d(
        masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    eroded = 1.0 - F.max_pool2d(
        1.0 - masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)

    return boundary


def boundary_weighted_bce_with_logits(
    logits,
    masks,
    boundary_weight,
    boundary_width,
):
    with torch.no_grad():
        boundary_mask = make_boundary_mask(masks, boundary_width)
        weights = 1.0 + boundary_weight * boundary_mask

    bce = F.binary_cross_entropy_with_logits(
        logits,
        masks,
        reduction="none",
    )

    return (bce * weights).sum() / weights.sum().clamp_min(1e-7)


def _normalized_loss_config(loss_config):
    if loss_config is None:
        loss_config = {}

    name = loss_config.get("name", "dice_bce")
    dice_weight = float(loss_config.get("dice_weight", 1.0))
    bce_weight = float(loss_config.get("bce_weight", 1.0))
    boundary_weight = float(loss_config.get("boundary_weight", 0.0))
    boundary_width = int(loss_config.get("boundary_width", 3))

    if name not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported loss name: {name}. "
            f"Expected one of: {', '.join(SUPPORTED_LOSSES)}"
        )

    if dice_weight < 0:
        raise ValueError("loss.dice_weight must be >= 0.")

    if bce_weight < 0:
        raise ValueError("loss.bce_weight must be >= 0.")

    if boundary_weight < 0:
        raise ValueError("loss.boundary_weight must be >= 0.")

    if boundary_width < 0:
        raise ValueError("loss.boundary_width must be >= 0.")

    if dice_weight == 0 and bce_weight == 0:
        raise ValueError(
            "At least one of loss.dice_weight or loss.bce_weight must be > 0."
        )

    return {
        "name": name,
        "dice_weight": dice_weight,
        "bce_weight": bce_weight,
        "boundary_weight": boundary_weight,
        "boundary_width": boundary_width,
    }


def build_loss_fn(loss_config):
    config = _normalized_loss_config(loss_config)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    if config["name"] == "dice_bce":
        def loss_fn(logits, masks):
            return (
                config["dice_weight"] * dice_loss(logits, masks)
                + config["bce_weight"] * bce_loss(logits, masks)
            )

        return loss_fn

    if config["name"] == "dice_boundary_bce":
        def loss_fn(logits, masks):
            return (
                config["dice_weight"] * dice_loss(logits, masks)
                + config["bce_weight"] * boundary_weighted_bce_with_logits(
                    logits=logits,
                    masks=masks,
                    boundary_weight=config["boundary_weight"],
                    boundary_width=config["boundary_width"],
                )
            )

        return loss_fn

    raise ValueError(f"Unsupported loss name: {config['name']}")


def describe_loss(loss_config):
    config = _normalized_loss_config(loss_config)

    if config["name"] == "dice_bce":
        return (
            "dice_bce("
            f"dice_weight={config['dice_weight']},"
            f"bce_weight={config['bce_weight']}"
            ")"
        )

    if config["name"] == "dice_boundary_bce":
        return (
            "dice_boundary_bce("
            f"dice_weight={config['dice_weight']},"
            f"bce_weight={config['bce_weight']},"
            f"boundary_weight={config['boundary_weight']},"
            f"boundary_width={config['boundary_width']}"
            ")"
        )

    raise ValueError(f"Unsupported loss name: {config['name']}")
