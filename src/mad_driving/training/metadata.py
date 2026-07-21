"""Versioned research-contract metadata for PPO training artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import yaml
from gymnasium import spaces

from mad_driving.config.models import AppConfig

RESEARCH_CONTRACT_VERSION: Final = 2
OBSERVATION_SCHEMA_VERSION: Final = 1
OBSERVATION_SHAPE: Final = (24,)
ACTION_SCHEMA_VERSION: Final = 1
ACTION_ORDER: Final = ("KEEP", "SLOW", "PREPARE_STOP", "STOP")
_ALLOWED_CONFIG_DIFFS: Final = frozenset(
    {
        "training.checkpoint_interval_steps",
        "training.eval_interval_steps",
        "training.run_root",
        "training.seed",
        "training.smoke_timesteps",
        "training.total_timesteps",
    }
)


@dataclass(frozen=True)
class ResumeMetadata:
    """Immutable provenance for one continuation run."""

    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_run_dir: str | None
    parent_config: dict[str, Any]
    config_diff: dict[str, Any]
    start_num_timesteps: int


@dataclass(frozen=True)
class RunMetadata:
    """Complete versioned identity of one fresh or continuation run."""

    resolved_config: dict[str, Any]
    resume: ResumeMetadata | None = None
    research_contract_version: int = RESEARCH_CONTRACT_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    observation_shape: tuple[int, ...] = OBSERVATION_SHAPE
    action_schema_version: int = ACTION_SCHEMA_VERSION
    action_count: int = len(ACTION_ORDER)
    action_order: tuple[str, ...] = ACTION_ORDER


@dataclass(frozen=True)
class ResumeSource:
    """Validated read-only source data resolved before destination writes."""

    checkpoint: Path
    run_dir: Path
    metadata: RunMetadata
    resolved_config: dict[str, Any]
    checkpoint_sha256: str
    config_diff: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file without modifying it."""

    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Resume metadata {name} must be an integer")
    return int(value)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Resume metadata {name} must be a non-empty string")
    return value


def _require_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Resume metadata {name} must be a JSON object")
    return cast(dict[str, Any], value)


def _parse_resume_metadata(value: object) -> ResumeMetadata | None:
    if value is None:
        return None
    payload = _require_dict(value, "resume")
    required = {
        "parent_checkpoint_path",
        "parent_checkpoint_sha256",
        "parent_run_dir",
        "parent_config",
        "config_diff",
        "start_num_timesteps",
    }
    if set(payload) != required:
        raise ValueError("Resume metadata resume fields are malformed")
    parent_run_dir_value = payload["parent_run_dir"]
    if parent_run_dir_value is not None and not isinstance(parent_run_dir_value, str):
        raise ValueError("Resume metadata parent_run_dir must be a string or null")
    digest = _require_string(payload["parent_checkpoint_sha256"], "checkpoint digest")
    if len(digest) != 64 or digest != digest.lower() or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("Resume metadata checkpoint digest must be lowercase SHA-256")
    start_num_timesteps = _require_int(payload["start_num_timesteps"], "start_num_timesteps")
    if start_num_timesteps < 0:
        raise ValueError("Resume metadata start_num_timesteps must be non-negative")
    return ResumeMetadata(
        parent_checkpoint_path=_require_string(
            payload["parent_checkpoint_path"], "parent_checkpoint_path"
        ),
        parent_checkpoint_sha256=digest,
        parent_run_dir=parent_run_dir_value,
        parent_config=_require_dict(payload["parent_config"], "parent_config"),
        config_diff=_require_dict(payload["config_diff"], "config_diff"),
        start_num_timesteps=start_num_timesteps,
    )


def _parse_run_metadata(payload: object) -> RunMetadata:
    values = _require_dict(payload, "root")
    required = {
        "research_contract_version",
        "observation_schema_version",
        "observation_shape",
        "action_schema_version",
        "action_count",
        "action_order",
        "resolved_config",
        "resume",
    }
    if set(values) != required:
        raise ValueError("Resume metadata fields are malformed")
    observation_shape_value = values["observation_shape"]
    action_order_value = values["action_order"]
    if not isinstance(observation_shape_value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in observation_shape_value
    ):
        raise ValueError("Resume metadata observation_shape is malformed")
    if not isinstance(action_order_value, list) or not all(
        isinstance(item, str) for item in action_order_value
    ):
        raise ValueError("Resume metadata action_order is malformed")
    return RunMetadata(
        resolved_config=_require_dict(values["resolved_config"], "resolved_config"),
        resume=_parse_resume_metadata(values["resume"]),
        research_contract_version=_require_int(
            values["research_contract_version"], "research_contract_version"
        ),
        observation_schema_version=_require_int(
            values["observation_schema_version"], "observation_schema_version"
        ),
        observation_shape=tuple(observation_shape_value),
        action_schema_version=_require_int(
            values["action_schema_version"], "action_schema_version"
        ),
        action_count=_require_int(values["action_count"], "action_count"),
        action_order=tuple(action_order_value),
    )


def _validate_metadata_contract(metadata: RunMetadata) -> None:
    if metadata.research_contract_version != RESEARCH_CONTRACT_VERSION:
        raise ValueError(
            "Resume research contract version mismatch: "
            f"expected {RESEARCH_CONTRACT_VERSION}, got {metadata.research_contract_version}"
        )
    if (
        metadata.observation_schema_version != OBSERVATION_SCHEMA_VERSION
        or metadata.observation_shape != OBSERVATION_SHAPE
    ):
        raise ValueError("Resume observation schema mismatch")
    if (
        metadata.action_schema_version != ACTION_SCHEMA_VERSION
        or metadata.action_count != len(ACTION_ORDER)
        or metadata.action_order != ACTION_ORDER
    ):
        raise ValueError("Resume action schema mismatch")


def _load_run_metadata(path: Path) -> RunMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume metadata is malformed: {path}") from exc
    metadata = _parse_run_metadata(payload)
    _validate_metadata_contract(metadata)
    return metadata


def _flatten_config(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("Resolved config contains a non-string key")
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_config(value[key], path))
        return flattened
    return {prefix: value}


def _allowed_config_diff(
    parent: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, Any]:
    parent_values = _flatten_config(parent)
    current_values = _flatten_config(current)
    changed = sorted(
        path
        for path in parent_values.keys() | current_values.keys()
        if parent_values.get(path) != current_values.get(path)
    )
    incompatible = [path for path in changed if path not in _ALLOWED_CONFIG_DIFFS]
    if incompatible:
        raise ValueError(f"Resume config mismatch: {', '.join(incompatible)}")
    return {
        path: {"parent": parent_values.get(path), "current": current_values.get(path)}
        for path in changed
    }


def resolve_resume_source(checkpoint: str | Path, current_config: AppConfig) -> ResumeSource:
    """Resolve and validate one read-only parent run before creating a destination."""

    checkpoint_path = Path(checkpoint).resolve(strict=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    run_dir: Path | None = None
    for candidate in (checkpoint_path.parent, *checkpoint_path.parents[1:]):
        if (candidate / "run_metadata.json").is_file():
            run_dir = candidate
            break
    if run_dir is None:
        raise ValueError(f"Resume source metadata not found for checkpoint: {checkpoint_path}")
    metadata = _load_run_metadata(run_dir / "run_metadata.json")
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise ValueError(f"Resume source resolved config not found: {config_path}")
    try:
        parent_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Resume source resolved config is malformed: {config_path}") from exc
    parent_config = _require_dict(parent_payload, "resolved config")
    if parent_config != metadata.resolved_config:
        raise ValueError("Resume metadata resolved config does not match config_resolved.yaml")
    try:
        AppConfig.model_validate(parent_config)
    except Exception as exc:
        raise ValueError(f"Resume source resolved config is malformed: {config_path}") from exc
    current_payload = current_config.model_dump(mode="json")
    config_diff = _allowed_config_diff(parent_config, current_payload)
    return ResumeSource(
        checkpoint=checkpoint_path,
        run_dir=run_dir.resolve(strict=True),
        metadata=metadata,
        resolved_config=parent_config,
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_diff=config_diff,
    )


def _require_equal(name: str, actual: object, expected: int | float) -> None:
    if isinstance(actual, bool) or not isinstance(actual, Real):
        raise ValueError(f"Resume {name} mismatch: expected numeric {expected}, got {actual!r}")
    numeric = float(actual)
    if not math.isfinite(numeric) or numeric != float(expected):
        raise ValueError(f"Resume {name} mismatch: expected {expected}, got {actual!r}")


def _require_constant_schedule(
    name: str,
    schedule: object,
    expected: float,
) -> None:
    if not callable(schedule):
        raise ValueError(f"Resume {name} mismatch: expected a constant numeric schedule")
    for progress_remaining in (0.0, 0.5, 1.0):
        try:
            value = schedule(progress_remaining)
        except Exception as exc:
            raise ValueError(f"Resume {name} schedule could not be evaluated") from exc
        _require_equal(name, value, expected)


def write_run_metadata(metadata: RunMetadata, destination: Path) -> None:
    """Atomically replace metadata through a cleaned sibling temporary file."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    primary_error: BaseException | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(
                dataclasses.asdict(metadata),
                output,
                ensure_ascii=False,
                indent=2,
            )
            output.write("\n")
        os.replace(temporary, destination)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Metadata temporary cleanup also failed: {cleanup_error}")


def validate_resume_contract(model: object, config: AppConfig, metadata: RunMetadata) -> None:
    """Validate a loaded checkpoint against the current research contract."""

    _validate_metadata_contract(metadata)
    policy_class = getattr(model, "policy_class", None)
    expected_policy_class = {"MlpPolicy": "ActorCriticPolicy"}[config.training.policy]
    if getattr(policy_class, "__name__", None) != expected_policy_class:
        raise ValueError(
            "Resume policy mismatch: "
            f"expected {config.training.policy}, got {getattr(policy_class, '__name__', None)!r}"
        )
    _require_equal(
        "learning_rate",
        getattr(model, "learning_rate", None),
        config.training.learning_rate,
    )
    _require_constant_schedule(
        "learning_rate",
        getattr(model, "lr_schedule", None),
        config.training.learning_rate,
    )
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ):
        _require_equal(name, getattr(model, name, None), getattr(config.training, name))
    _require_constant_schedule(
        "clip_range",
        getattr(model, "clip_range", None),
        config.training.clip_range,
    )

    observation_space = getattr(model, "observation_space", None)
    if (
        not isinstance(observation_space, spaces.Box)
        or observation_space.shape != OBSERVATION_SHAPE
        or observation_space.dtype != np.dtype(np.float32)
        or not np.all(observation_space.low == -1.0)
        or not np.all(observation_space.high == 1.0)
    ):
        raise ValueError("Resume observation space mismatch")
    action_space = getattr(model, "action_space", None)
    if (
        not isinstance(action_space, spaces.Discrete)
        or action_space.n != len(ACTION_ORDER)
        or action_space.start != 0
    ):
        raise ValueError("Resume action space mismatch")


__all__ = [
    "ResumeMetadata",
    "RunMetadata",
    "sha256_file",
    "validate_resume_contract",
]
