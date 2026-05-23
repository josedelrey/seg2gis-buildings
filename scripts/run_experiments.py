import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml


DEFAULT_EXPERIMENT_CONFIG = "configs/experiments_phase2_augmentation.yaml"
DEFAULT_PROJECT_CONFIG = "configs/default.json"

REQUIRED_EXPERIMENT_FIELDS = ("run_name", "architecture", "encoder", "augmentation")

EXPERIMENT_FIELD_PATHS = {
    "run_name": ("training", "run_name"),
    "protocol": ("protocol", "name"),
    "architecture": ("model", "architecture"),
    "encoder": ("model", "encoder"),
    "augmentation": ("training", "augmentation"),
    "batch_size": ("training", "batch_size"),
    "epochs": ("training", "epochs"),
    "lr": ("training", "lr"),
    "seed": ("training", "seed"),
    "train_image_dir": ("data", "train_image_dir"),
    "train_mask_dir": ("data", "train_mask_dir"),
    "val_image_dir": ("data", "val_image_dir"),
    "val_mask_dir": ("data", "val_mask_dir"),
    "raw_test_image_dir": ("evaluation", "raw_test_image_dir"),
    "raw_test_mask_dir": ("evaluation", "raw_test_mask_dir"),
    "train_image_ids": ("protocol", "train_image_ids"),
    "val_image_ids": ("protocol", "val_image_ids"),
    "test_image_ids": ("protocol", "test_image_ids"),
    "eval_threshold": ("evaluation", "threshold"),
    "eval_tile_size": ("evaluation", "tile_size"),
    "eval_stride": ("evaluation", "stride"),
    "eval_min_area": ("evaluation", "min_area"),
    "eval_open_kernel_size": ("evaluation", "open_kernel_size"),
    "model_dir": ("model", "model_dir"),
    "experiment_log_path": ("training", "experiment_log_path"),
}


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
        default=DEFAULT_PROJECT_CONFIG,
        help="Base JSON project config to merge with each experiment.",
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


def load_project_config(path):
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Project config must be a JSON object.")

    return config


def merge_defaults(defaults, experiment):
    merged = dict(defaults)
    merged.update(experiment)
    return merged


def set_nested_value(config, key_path, value):
    section = config

    for key in key_path[:-1]:
        if key not in section or not isinstance(section[key], dict):
            section[key] = {}
        section = section[key]

    section[key_path[-1]] = value


def build_training_config(base_config, exp):
    train_config = copy.deepcopy(base_config)

    for field, value in exp.items():
        if field not in EXPERIMENT_FIELD_PATHS:
            raise ValueError(f"Unsupported experiment field '{field}' in {exp}")

        set_nested_value(train_config, EXPERIMENT_FIELD_PATHS[field], value)

    return train_config


def write_training_config(config, config_dir, run_name):
    safe_run_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in run_name
    )
    config_path = Path(config_dir) / f"{safe_run_name}.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return str(config_path)


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


def build_command(config_path):
    return [sys.executable, "src/train.py", "--config", config_path]


def print_command(command):
    print(" ".join(command))


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    if minutes:
        return f"{minutes}m {seconds:02d}s"

    return f"{seconds}s"


def format_eta(seconds_from_now):
    finish_time = datetime.now() + timedelta(seconds=seconds_from_now)
    return finish_time.strftime("%Y-%m-%d %H:%M:%S")


def print_batch_estimate(batch_start_time, completed_durations, remaining_count):
    elapsed = time.monotonic() - batch_start_time

    if not completed_durations:
        print(
            "Batch estimate: waiting for the first completed experiment "
            "to estimate total runtime."
        )
        return

    avg_experiment_duration = sum(completed_durations) / len(completed_durations)
    estimated_remaining = avg_experiment_duration * remaining_count
    estimated_total = elapsed + estimated_remaining

    print(
        "Batch estimate | "
        f"elapsed: {format_duration(elapsed)} | "
        f"avg experiment: {format_duration(avg_experiment_duration)} | "
        f"remaining: {format_duration(estimated_remaining)} | "
        f"estimated total: {format_duration(estimated_total)} | "
        f"ETA: {format_eta(estimated_remaining)}"
    )


def main():
    args = parse_args()
    config = load_experiment_config(args.experiments_config)
    base_config = load_project_config(args.project_config)

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
    print("Project config:", args.project_config)
    print("Experiments:", len(experiments))
    print("Dry run:", args.dry_run)
    print("Fail fast:", args.fail_fast)

    failures = []
    batch_start_time = time.monotonic()
    completed_durations = []

    with tempfile.TemporaryDirectory(prefix="seg2gis_experiment_configs_") as config_dir:
        for idx, exp in enumerate(experiments, start=1):
            validate_experiment(exp)
            train_config = build_training_config(base_config, exp)
            config_path = write_training_config(
                config=train_config,
                config_dir=config_dir,
                run_name=exp["run_name"],
            )
            command = build_command(config_path)

            print()
            print("=" * 80)
            print(f"Experiment {idx}/{len(experiments)}: {exp['run_name']}")
            print("=" * 80)
            print("Generated config:", config_path)
            print_command(command)

            if args.dry_run:
                continue

            print_batch_estimate(
                batch_start_time=batch_start_time,
                completed_durations=completed_durations,
                remaining_count=len(experiments) - idx + 1,
            )

            experiment_start_time = time.monotonic()

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                experiment_duration = time.monotonic() - experiment_start_time
                completed_durations.append(experiment_duration)
                failures.append((exp["run_name"], e.returncode))
                print(
                    f"FAILED: {exp['run_name']} with return code {e.returncode} "
                    f"after {format_duration(experiment_duration)}"
                )

                if args.fail_fast:
                    break

                print("Continuing with next experiment...")
            else:
                experiment_duration = time.monotonic() - experiment_start_time
                completed_durations.append(experiment_duration)
                print(
                    f"Finished {exp['run_name']} in "
                    f"{format_duration(experiment_duration)}"
                )

            remaining_after_current = len(experiments) - idx
            if remaining_after_current:
                print_batch_estimate(
                    batch_start_time=batch_start_time,
                    completed_durations=completed_durations,
                    remaining_count=remaining_after_current,
                )

    if failures:
        print()
        print("Failed experiments:")

        for run_name, return_code in failures:
            print(f"- {run_name}: return code {return_code}")

        total_duration = time.monotonic() - batch_start_time
        print(f"Batch elapsed time: {format_duration(total_duration)}")

        raise SystemExit(1)

    print()
    total_duration = time.monotonic() - batch_start_time
    print(f"Done. Total batch time: {format_duration(total_duration)}")


if __name__ == "__main__":
    main()
