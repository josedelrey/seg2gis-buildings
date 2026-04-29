import subprocess


experiments = [
    # UNet
    {"run_name": "unet_r34_256_noaug_e20", "architecture": "unet", "encoder": "resnet34", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "unet_r50_256_noaug_e20", "architecture": "unet", "encoder": "resnet50", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "unet_effb0_256_noaug_e20", "architecture": "unet", "encoder": "efficientnet-b0", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "unet_effb3_256_noaug_e20", "architecture": "unet", "encoder": "efficientnet-b3", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "unet_mobilenetv2_256_noaug_e20", "architecture": "unet", "encoder": "mobilenet_v2", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},

    # FPN
    {"run_name": "fpn_r34_256_noaug_e20", "architecture": "fpn", "encoder": "resnet34", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "fpn_r50_256_noaug_e20", "architecture": "fpn", "encoder": "resnet50", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "fpn_effb0_256_noaug_e20", "architecture": "fpn", "encoder": "efficientnet-b0", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "fpn_effb3_256_noaug_e20", "architecture": "fpn", "encoder": "efficientnet-b3", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "fpn_mobilenetv2_256_noaug_e20", "architecture": "fpn", "encoder": "mobilenet_v2", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},

    # DeepLabV3+
    {"run_name": "deeplabv3p_r50_256_noaug_e20", "architecture": "deeplabv3plus", "encoder": "resnet50", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "deeplabv3p_effb3_256_noaug_e20", "architecture": "deeplabv3plus", "encoder": "efficientnet-b3", "batch_size": 8, "epochs": 20, "lr": 1e-4, "augmentation_type": "noaug"},
]


for exp in experiments:
    command = [
        "python",
        "src/train.py",
        "--run_name", exp["run_name"],
        "--architecture", exp["architecture"],
        "--encoder", exp["encoder"],
        "--batch_size", str(exp["batch_size"]),
        "--epochs", str(exp["epochs"]),
        "--lr", str(exp["lr"]),
        "--augmentation_type", exp["augmentation_type"],
    ]

    if exp["use_augmentation"]:
        command.append("--use_augmentation")

    print("\n" + "=" * 80)
    print("Running experiment:", exp["run_name"])
    print("=" * 80)

    subprocess.run(command, check=True)