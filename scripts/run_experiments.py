import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_EXPERIMENT_CONFIG = "configs/experiments_phase2_augmentation.yaml"

REQUIRED_EXPERIMENT_FIELDS = [
    "run_name",
    "architecture",
    "encoder",
    "augmentation_type",
]

OPTIONAL_TRAIN_ARGS = [
    "batch_size",
    "epochs",
    "lr",
    "seed",
    "train_image_dir",
    "train_mask_dir",
    "val_image_dir",
    "val_mask_dir",
    "model_dir",
    "experiment_log_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a batch of training experiments from a YAML file.",
    )
    parser.add_argument(
        "--experiments_config",
        type=str,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="YAML file describing experiment defaults and experiment runs.",
    )
    parser.add_argument(
        "--project_config",
        type=str,
        default=None,
        help="Optional JSON project config passed through to src/train.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop after the first failed experiment.",
    )

    return parser.parse_args()


def load_experiment_config(path):
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping.")

    experiments = config.get("experiments")

    if not isinstance(experiments, list) or len(experiments) == 0:
        raise ValueError("Experiment config must contain a non-empty experiments list.")

    return config


def merge_defaults(defaults, experiment):
    merged = dict(defaults)
    merged.update(experiment)
    return merged


def validate_experiment(exp):
    missing = [
        field
        for field in REQUIRED_EXPERIMENT_FIELDS
        if exp.get(field) in (None, "")
    ]

    if missing:
        raise ValueError(
            f"Experiment is missing required fields {missing}: {exp}"
        )


def build_command(exp, project_config):
    command = [
        sys.executable,
        "src/train.py",
    ]

    if project_config is not None:
        command.extend(["--config", project_config])

    for field in REQUIRED_EXPERIMENT_FIELDS + OPTIONAL_TRAIN_ARGS:
        value = exp.get(field)

        if value is None:
            continue

        command.extend([f"--{field}", str(value)])

    return command


def print_command(command):
    print(" ".join(command))


def main():
    args = parse_args()
    config = load_experiment_config(args.experiments_config)

    defaults = config.get("defaults", {})
    if defaults is None:
        defaults = {}

    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping when provided.")

    experiments = [
        merge_defaults(defaults, exp)
        for exp in config["experiments"]
    ]

    print("Experiment config:", args.experiments_config)
    print("Experiments:", len(experiments))
    print("Dry run:", args.dry_run)
    print("Fail fast:", args.fail_fast)

    failures = []

    for idx, exp in enumerate(experiments, start=1):
        validate_experiment(exp)
        command = build_command(exp, args.project_config)

        print()
        print("=" * 80)
        print(f"Experiment {idx}/{len(experiments)}: {exp['run_name']}")
        print("=" * 80)
        print_command(command)

        if args.dry_run:
            continue

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            failures.append((exp["run_name"], e.returncode))
            print(f"FAILED: {exp['run_name']} with return code {e.returncode}")

            if args.fail_fast:
                break

            print("Continuing with next experiment...")

    if failures:
        print()
        print("Failed experiments:")

        for run_name, return_code in failures:
            print(f"- {run_name}: return code {return_code}")

        raise SystemExit(1)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
