import subprocess


experiments = [
    # Longer training, no augmentation: check if current best keeps improving
    {
        "run_name": "unet_effb3_256_noaug_e30",
        "architecture": "unet",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "fpn_effb3_256_noaug_e30",
        "architecture": "fpn",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "deeplabv3p_effb3_256_noaug_e30",
        "architecture": "deeplabv3plus",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "noaug",
    },

    # Same best models with geometric augmentation
    {
        "run_name": "unet_effb3_256_geomaug_e30",
        "architecture": "unet",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "geomaug",
    },
    {
        "run_name": "fpn_effb3_256_geomaug_e30",
        "architecture": "fpn",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "geomaug",
    },
    {
        "run_name": "deeplabv3p_effb3_256_geomaug_e30",
        "architecture": "deeplabv3plus",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "geomaug",
    },

    # Mild color augmentation: useful for aerial imagery generalization
    {
        "run_name": "unet_effb3_256_mildaug_e30",
        "architecture": "unet",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "mildaug",
    },
    {
        "run_name": "fpn_effb3_256_mildaug_e30",
        "architecture": "fpn",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "mildaug",
    },
    {
        "run_name": "deeplabv3p_effb3_256_mildaug_e30",
        "architecture": "deeplabv3plus",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "mildaug",
    },

    # Strong augmentation: stress-test robustness
    {
        "run_name": "unet_effb3_256_strongaug_e30",
        "architecture": "unet",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "strongaug",
    },
    {
        "run_name": "fpn_effb3_256_strongaug_e30",
        "architecture": "fpn",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "strongaug",
    },
    {
        "run_name": "deeplabv3p_effb3_256_strongaug_e30",
        "architecture": "deeplabv3plus",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "augmentation_type": "strongaug",
    },
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

    print("\n" + "=" * 80)
    print("Running experiment:", exp["run_name"])
    print("=" * 80)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {exp['run_name']} with return code {e.returncode}")
        print("Continuing with next experiment...")