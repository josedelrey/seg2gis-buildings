import subprocess


experiments = [
    {
        "run_name": "unet_r34_256_strongaug_e30",
        "architecture": "unet",
        "encoder": "resnet34",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "use_augmentation": True,
    },
    {
        "run_name": "unet_r50_256_strongaug_e30",
        "architecture": "unet",
        "encoder": "resnet50",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "use_augmentation": True,
    },
    {
        "run_name": "unet_effb3_256_strongaug_e30",
        "architecture": "unet",
        "encoder": "efficientnet-b3",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "use_augmentation": True,
    },
    {
        "run_name": "unet_effb4_256_strongaug_e30",
        "architecture": "unet",
        "encoder": "efficientnet-b4",
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "use_augmentation": True,
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
    ]

    if exp["use_augmentation"]:
        command.append("--use_augmentation")

    print("\n" + "=" * 80)
    print("Running experiment:", exp["run_name"])
    print("=" * 80)

    subprocess.run(command, check=True)