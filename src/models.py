import segmentation_models_pytorch as smp


def build_model(architecture, encoder, encoder_weights=None):
    architecture = architecture.lower()

    common_args = {
        "encoder_name": encoder,
        "encoder_weights": encoder_weights,
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
