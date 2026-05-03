import subprocess


experiments = [
    # DeepLabV3+
    {"run_name": "deeplabv3p_effb0_256_noaug_e10", "architecture": "deeplabv3plus", "encoder": "efficientnet-b0", "batch_size": 8, "epochs": 10, "lr": 1e-4, "augmentation_type": "noaug"},
    {"run_name": "deeplabv3p_mobilenetv2_256_noaug_e10", "architecture": "deeplabv3plus", "encoder": "mobilenet_v2", "batch_size": 8, "epochs": 10, "lr": 1e-4, "augmentation_type": "noaug"},
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