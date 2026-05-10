import json
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs/default.json"


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config_value(config, *keys, default=None):
    value = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def resolve_model_path(model_dir, run_name):
    return str(Path(model_dir) / f"{run_name}.pth")
