"""Load strict application configuration from YAML."""

from pathlib import Path

import yaml

from mad_driving.config.models import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(payload)
