import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
    batch_start_time = time.monotonic()
    completed_durations = []

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
