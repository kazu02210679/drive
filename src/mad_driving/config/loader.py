"""Load strict application configuration from YAML."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mad_driving.config.models import AppConfig


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML file and require a root mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"Configuration mapping keys must be strings: {path}")
    return deepcopy(dict(payload))


def _merge_mapping(
    base: Mapping[str, Any], overlay: Mapping[str, Any], *, path: str
) -> dict[str, Any]:
    """Recursively merge YAML mappings without allowing shape replacement."""

    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        key_path = f"{path}.{key}" if path else key
        if key not in merged:
            merged[key] = deepcopy(value)
            continue
        current = merged[key]
        current_is_mapping = isinstance(current, Mapping)
        value_is_mapping = isinstance(value, Mapping)
        if current_is_mapping and value_is_mapping:
            merged[key] = _merge_mapping(current, value, path=key_path)
        elif current_is_mapping != value_is_mapping:
            raise ValueError(f"mapping conflict at {key_path}")
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, *overlays: str | Path) -> AppConfig:
    """Load a strict base configuration and ordered recursive overlays."""

    payload = _load_yaml_mapping(Path(path))
    for overlay in overlays:
        payload = _merge_mapping(payload, _load_yaml_mapping(Path(overlay)), path="")
    return AppConfig.model_validate(payload)
