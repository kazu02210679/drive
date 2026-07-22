"""Versioned research-contract metadata for PPO training artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3.common.utils import ConstantSchedule, FloatSchedule

from mad_driving.config.models import AppConfig
from mad_driving.config.parsing import load_unique_yaml
from mad_driving.methods import get_method_profile
from mad_driving.training.curriculum import (
    CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION,
    CURRICULUM_STATE_FILENAME,
    CurriculumController,
    CurriculumState,
    checkpoint_curriculum_sidecar_path,
    read_checkpoint_curriculum_artifact,
    read_curriculum_state_artifact,
    read_stable_artifact_bytes,
)

RESEARCH_CONTRACT_VERSION: Final = 7
OBSERVATION_SCHEMA_VERSION: Final = 1
OBSERVATION_SHAPE: Final = (24,)
OBSERVATION_DTYPE: Final = "float32"
ACTION_SCHEMA_VERSION: Final = 1
ACTION_ORDER: Final = ("KEEP", "SLOW", "PREPARE_STOP", "STOP")
EPISODE_SEED_ARTIFACT_SCHEMA_VERSION: Final = 4
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


class FrozenJsonObject(Mapping[str, Any]):
    """A detached, deterministic, recursively immutable JSON object."""

    __slots__ = ("_items",)
    _items: tuple[tuple[str, Any], ...]

    def __init__(self, items: Sequence[tuple[str, Any]]) -> None:
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("FrozenJsonObject is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("FrozenJsonObject is immutable")

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return _json_equal(self, other)

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenJsonObject:
        del memo
        return self


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
            return False
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    return bool(left == right)


def _freeze_json(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        entries: list[tuple[str, Any]] = []
        for key in value:
            if not isinstance(key, str):
                raise ValueError(f"{name} must contain only string keys")
            entries.append((key, _freeze_json(value[key], f"{name}.{key}")))
        return FrozenJsonObject(sorted(entries, key=lambda item: item[0]))
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    raise ValueError(f"{name} contains a value that is not JSON safe: {value!r}")


def _thaw_json(value: object, name: str) -> object:
    """Validate and convert an immutable JSON value to encoder-native containers."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError(f"{name} must contain only string keys")
            result[key] = _thaw_json(value[key], f"{name}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_thaw_json(item, f"{name}[]") for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    raise ValueError(f"{name} contains a value that is not JSON safe: {value!r}")


def _validated_integral(value: object, name: str, *, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-bool integer")
    result = int(value)
    if expected is not None and result != expected:
        raise ValueError(f"{name} must equal {expected}")
    return result


def _provenance_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty provenance string")
    return value


@dataclass(frozen=True)
class ResumeMetadata:
    """Immutable provenance for one continuation run."""

    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_run_dir: str | None
    parent_config: Mapping[str, Any]
    config_diff: Mapping[str, Any]
    start_num_timesteps: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_checkpoint_path",
            _provenance_string(self.parent_checkpoint_path, "parent_checkpoint_path"),
        )
        digest = self.parent_checkpoint_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("parent_checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.parent_run_dir is not None:
            object.__setattr__(
                self,
                "parent_run_dir",
                _provenance_string(self.parent_run_dir, "parent_run_dir"),
            )
        if not isinstance(self.parent_config, Mapping):
            raise ValueError("parent_config must be a required JSON mapping")
        if not isinstance(self.config_diff, Mapping):
            raise ValueError("config_diff must be a required JSON mapping")
        object.__setattr__(
            self,
            "parent_config",
            _freeze_json(self.parent_config, "parent_config"),
        )
        object.__setattr__(
            self,
            "config_diff",
            _freeze_json(self.config_diff, "config_diff"),
        )
        timesteps = _validated_integral(self.start_num_timesteps, "start_num_timesteps")
        if timesteps < 0:
            raise ValueError("start_num_timesteps must be non-negative")
        object.__setattr__(self, "start_num_timesteps", timesteps)


@dataclass(frozen=True)
class MethodProfileSnapshot:
    """Immutable runtime composition identity recorded with every run."""

    method_id: str
    policy_kind: str
    specialist_ids: tuple[str, ...]
    critic_enabled: bool
    shield_mode: str

    @classmethod
    def from_method_id(cls, method_id: str) -> MethodProfileSnapshot:
        try:
            profile = get_method_profile(cast(Any, method_id))
        except (KeyError, TypeError) as error:
            raise ValueError("method_profile.method_id is unknown") from error
        return cls(
            method_id=profile.method_id,
            policy_kind=profile.policy_kind,
            specialist_ids=profile.specialist_ids,
            critic_enabled=profile.critic_enabled,
            shield_mode=profile.default_shield_mode,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_profile.method_id must be a non-empty string")
        if not isinstance(self.policy_kind, str) or not self.policy_kind:
            raise ValueError("method_profile.policy_kind must be a non-empty string")
        if (
            not isinstance(self.specialist_ids, list | tuple)
            or not all(isinstance(agent_id, str) and agent_id for agent_id in self.specialist_ids)
        ):
            raise ValueError("method_profile.specialist_ids must be non-empty strings")
        if not isinstance(self.critic_enabled, bool):
            raise ValueError("method_profile.critic_enabled must be boolean")
        if not isinstance(self.shield_mode, str) or not self.shield_mode:
            raise ValueError("method_profile.shield_mode must be a non-empty string")
        try:
            profile = get_method_profile(cast(Any, self.method_id))
        except (KeyError, TypeError) as error:
            raise ValueError("method_profile.method_id is unknown") from error
        expected = (
            profile.method_id,
            profile.policy_kind,
            profile.specialist_ids,
            profile.critic_enabled,
            profile.default_shield_mode,
        )
        actual = (
            self.method_id,
            self.policy_kind,
            tuple(self.specialist_ids),
            self.critic_enabled,
            self.shield_mode,
        )
        if actual != expected:
            raise ValueError("method_profile must equal the central method profile")
        object.__setattr__(self, "specialist_ids", profile.specialist_ids)


@dataclass(frozen=True)
class RunMetadata:
    """Complete versioned identity of one fresh or continuation run."""

    resolved_config: Mapping[str, Any]
    curriculum_state: Mapping[str, Any]
    method_profile: MethodProfileSnapshot = MethodProfileSnapshot.from_method_id("proposed")
    resume: ResumeMetadata | None = None
    research_contract_version: int = RESEARCH_CONTRACT_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    observation_shape: tuple[int, ...] = OBSERVATION_SHAPE
    observation_dtype: str = OBSERVATION_DTYPE
    action_schema_version: int = ACTION_SCHEMA_VERSION
    action_count: int = len(ACTION_ORDER)
    action_order: tuple[str, ...] = ACTION_ORDER
    checkpoint_curriculum_artifacts: tuple[Mapping[str, Any], ...] = ()
    episode_seed_artifacts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "research_contract_version",
            _validated_integral(
                self.research_contract_version,
                "research_contract_version",
                expected=RESEARCH_CONTRACT_VERSION,
            ),
        )
        object.__setattr__(
            self,
            "observation_schema_version",
            _validated_integral(
                self.observation_schema_version,
                "observation_schema_version",
                expected=OBSERVATION_SCHEMA_VERSION,
            ),
        )
        if not isinstance(self.observation_shape, list | tuple) or any(
            isinstance(item, bool) or not isinstance(item, Integral)
            for item in self.observation_shape
        ):
            raise ValueError(f"observation_shape must equal {OBSERVATION_SHAPE}")
        normalized_shape = tuple(int(item) for item in self.observation_shape)
        if normalized_shape != OBSERVATION_SHAPE:
            raise ValueError(f"observation_shape must equal {OBSERVATION_SHAPE}")
        object.__setattr__(self, "observation_shape", OBSERVATION_SHAPE)
        if self.observation_dtype != OBSERVATION_DTYPE:
            raise ValueError(f"observation_dtype must equal {OBSERVATION_DTYPE!r}")
        object.__setattr__(
            self,
            "action_schema_version",
            _validated_integral(
                self.action_schema_version,
                "action_schema_version",
                expected=ACTION_SCHEMA_VERSION,
            ),
        )
        object.__setattr__(
            self,
            "action_count",
            _validated_integral(
                self.action_count,
                "action_count",
                expected=len(ACTION_ORDER),
            ),
        )
        if (
            not isinstance(self.action_order, list | tuple)
            or tuple(self.action_order) != ACTION_ORDER
        ):
            raise ValueError(f"action_order must equal {ACTION_ORDER}")
        object.__setattr__(self, "action_order", ACTION_ORDER)
        if not isinstance(self.resolved_config, Mapping):
            raise ValueError("resolved_config must be a required JSON mapping")
        object.__setattr__(
            self,
            "resolved_config",
            _freeze_json(self.resolved_config, "resolved_config"),
        )
        if not isinstance(self.method_profile, MethodProfileSnapshot):
            raise ValueError("method_profile must be a MethodProfileSnapshot")
        expected_profile = _method_profile_from_resolved_config(self.resolved_config)
        if expected_profile is not None and self.method_profile != expected_profile:
            raise ValueError("method_profile must match resolved_config.method.id")
        if self.resume is not None and not isinstance(self.resume, ResumeMetadata):
            raise ValueError("resume must be ResumeMetadata or null")
        object.__setattr__(
            self,
            "curriculum_state",
            _validated_curriculum_state_artifact(self.curriculum_state),
        )
        object.__setattr__(
            self,
            "checkpoint_curriculum_artifacts",
            _validated_checkpoint_curriculum_artifacts(self.checkpoint_curriculum_artifacts),
        )
        object.__setattr__(
            self,
            "episode_seed_artifacts",
            _validated_episode_seed_artifacts(self.episode_seed_artifacts),
        )


def _method_profile_from_resolved_config(
    resolved_config: Mapping[str, Any],
) -> MethodProfileSnapshot | None:
    if "method" not in resolved_config:
        return None
    method = resolved_config["method"]
    if not isinstance(method, Mapping) or set(method) != {"id"}:
        raise ValueError("resolved_config.method is malformed")
    method_id = method["id"]
    if not isinstance(method_id, str):
        raise ValueError("resolved_config.method.id must be a string")
    try:
        return MethodProfileSnapshot.from_method_id(method_id)
    except ValueError as error:
        raise ValueError("resolved_config.method.id is malformed") from error


def _validated_curriculum_state_artifact(
    value: object,
) -> Mapping[str, Any]:
    required = {
        "path",
        "sha256",
        "level",
        "consecutive_passes",
        "evaluations",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("curriculum_state fields are malformed")
    if value["path"] != CURRICULUM_STATE_FILENAME:
        raise ValueError(f"curriculum_state.path must equal {CURRICULUM_STATE_FILENAME!r}")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("curriculum_state.sha256 must be a lowercase SHA-256 digest")
    try:
        state = CurriculumState(
            level=value["level"],
            consecutive_passes=value["consecutive_passes"],
            evaluations=value["evaluations"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("curriculum_state values are malformed") from error
    return cast(
        Mapping[str, Any],
        _freeze_json(
            {
                "path": CURRICULUM_STATE_FILENAME,
                "sha256": digest,
                "level": state.level,
                "consecutive_passes": state.consecutive_passes,
                "evaluations": state.evaluations,
            },
            "curriculum_state",
        ),
    )


def _validated_episode_seed_artifacts(
    value: object,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("episode_seed_artifacts must be a list or tuple")
    required = {
        "file_identity",
        "path",
        "record_count",
        "role",
        "schema_version",
        "sha256",
        "worker_index",
    }
    validated: list[Mapping[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    paths: set[str] = set()
    for index, raw_summary in enumerate(value):
        name = f"episode_seed_artifacts[{index}]"
        if not isinstance(raw_summary, Mapping) or set(raw_summary) != required:
            raise ValueError(f"{name} fields are malformed")
        role = raw_summary["role"]
        if role not in ("train", "validation", "test"):
            raise ValueError(f"{name}.role is malformed")
        worker_index = _validated_integral(raw_summary["worker_index"], f"{name}.worker_index")
        if worker_index < 0:
            raise ValueError(f"{name}.worker_index must be non-negative")
        record_count = _validated_integral(raw_summary["record_count"], f"{name}.record_count")
        if record_count < 0:
            raise ValueError(f"{name}.record_count must be non-negative")
        schema_version = _validated_integral(
            raw_summary["schema_version"],
            f"{name}.schema_version",
            expected=EPISODE_SEED_ARTIFACT_SCHEMA_VERSION,
        )
        raw_file_identity = raw_summary["file_identity"]
        if not isinstance(raw_file_identity, Mapping) or set(raw_file_identity) != {
            "device",
            "inode",
        }:
            raise ValueError(f"{name}.file_identity is malformed")
        device = _validated_integral(
            raw_file_identity["device"],
            f"{name}.file_identity.device",
        )
        inode = _validated_integral(
            raw_file_identity["inode"],
            f"{name}.file_identity.inode",
        )
        if device < 0 or inode <= 0:
            raise ValueError(f"{name}.file_identity is not verifiable")
        path = raw_summary["path"]
        expected_path = f"episode_seeds/{role}-worker-{worker_index:03d}.jsonl"
        if not isinstance(path, str) or path != expected_path:
            raise ValueError(f"{name}.path must equal {expected_path!r}")
        digest = raw_summary["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
        identity = (str(role), worker_index)
        if identity in identities or path in paths:
            raise ValueError("episode_seed_artifacts must not contain duplicates")
        identities.add(identity)
        paths.add(path)
        frozen = _freeze_json(
            {
                "file_identity": {"device": device, "inode": inode},
                "path": path,
                "record_count": record_count,
                "role": role,
                "schema_version": schema_version,
                "sha256": digest,
                "worker_index": worker_index,
            },
            name,
        )
        validated.append(cast(Mapping[str, Any], frozen))
    return tuple(validated)


def _validated_checkpoint_curriculum_artifacts(
    value: object,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("checkpoint_curriculum_artifacts must be a list or tuple")
    required = {
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "state_path",
        "state_sha256",
        "level",
        "consecutive_passes",
        "evaluations",
    }
    validated: list[Mapping[str, Any]] = []
    checkpoint_paths: set[str] = set()
    state_paths: set[str] = set()
    for index, raw_artifact in enumerate(value):
        name = f"checkpoint_curriculum_artifacts[{index}]"
        if not isinstance(raw_artifact, Mapping) or set(raw_artifact) != required:
            raise ValueError(f"{name} fields are malformed")
        schema_version = _validated_integral(
            raw_artifact["schema_version"],
            f"{name}.schema_version",
            expected=CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION,
        )
        checkpoint_path = raw_artifact["checkpoint_path"]
        if (
            not isinstance(checkpoint_path, str)
            or not checkpoint_path.startswith("checkpoints/")
            or not checkpoint_path.endswith(".zip")
            or Path(checkpoint_path).as_posix() != checkpoint_path
            or ".." in Path(checkpoint_path).parts
        ):
            raise ValueError(f"{name}.checkpoint_path is malformed")
        state_path = raw_artifact["state_path"]
        expected_state_path = f"{checkpoint_path}.curriculum.yaml"
        if state_path != expected_state_path:
            raise ValueError(f"{name}.state_path must equal {expected_state_path!r}")
        if checkpoint_path in checkpoint_paths or state_path in state_paths:
            raise ValueError("checkpoint_curriculum_artifacts must not contain duplicates")
        checkpoint_paths.add(checkpoint_path)
        state_paths.add(state_path)
        digests: dict[str, str] = {}
        for field in ("checkpoint_sha256", "state_sha256"):
            digest = raw_artifact[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name}.{field} must be a lowercase SHA-256 digest")
            digests[field] = digest
        try:
            state = CurriculumState(
                level=raw_artifact["level"],
                consecutive_passes=raw_artifact["consecutive_passes"],
                evaluations=raw_artifact["evaluations"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} state values are malformed") from error
        validated.append(
            cast(
                Mapping[str, Any],
                _freeze_json(
                    {
                        "schema_version": schema_version,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": digests["checkpoint_sha256"],
                        "state_path": state_path,
                        "state_sha256": digests["state_sha256"],
                        "level": state.level,
                        "consecutive_passes": state.consecutive_passes,
                        "evaluations": state.evaluations,
                    },
                    name,
                ),
            )
        )
    return tuple(validated)


@dataclass(frozen=True)
class ResumeSource:
    """Validated read-only source data resolved before destination writes."""

    checkpoint: Path
    run_dir: Path
    metadata: RunMetadata
    resolved_config: dict[str, Any]
    checkpoint_bytes: bytes
    checkpoint_sha256: str
    config_diff: dict[str, Any]
    curriculum_state: CurriculumState


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file without modifying it."""

    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def curriculum_state_artifact(
    path: str | Path,
    state: CurriculumState,
) -> Mapping[str, Any]:
    """Return the validated metadata identity of one curriculum state file."""

    state_path = Path(path)
    if state_path.name != CURRICULUM_STATE_FILENAME:
        raise ValueError(f"curriculum state path must be named {CURRICULUM_STATE_FILENAME}")
    return _validated_curriculum_state_artifact(
        {
            "path": CURRICULUM_STATE_FILENAME,
            "sha256": sha256_file(state_path),
            "level": state.level,
            "consecutive_passes": state.consecutive_passes,
            "evaluations": state.evaluations,
        }
    )


def checkpoint_curriculum_artifact_inventory(
    checkpoints_dir: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Inventory every checkpoint and its exact adjacent curriculum sidecar."""

    directory = Path(checkpoints_dir)
    if directory.name != "checkpoints" or not directory.is_dir():
        raise ValueError(f"Checkpoint directory is malformed: {directory}")
    checkpoints = sorted(directory.glob("*.zip"), key=lambda path: path.name)
    if not checkpoints:
        raise ValueError(f"No checkpoints were published in: {directory}")
    expected_sidecars = {checkpoint_curriculum_sidecar_path(path) for path in checkpoints}
    actual_sidecars = set(directory.glob("*.zip.curriculum.yaml"))
    if actual_sidecars != expected_sidecars:
        raise ValueError("Checkpoint curriculum sidecars do not exactly match checkpoints")
    artifacts: list[Mapping[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_digest = sha256_file(checkpoint)
        sidecar = checkpoint_curriculum_sidecar_path(checkpoint)
        state, state_digest = read_checkpoint_curriculum_artifact(
            sidecar,
            expected_checkpoint_sha256=checkpoint_digest,
        )
        relative_checkpoint = f"checkpoints/{checkpoint.name}"
        artifacts.append(
            {
                "schema_version": CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION,
                "checkpoint_path": relative_checkpoint,
                "checkpoint_sha256": checkpoint_digest,
                "state_path": f"{relative_checkpoint}.curriculum.yaml",
                "state_sha256": state_digest,
                "level": state.level,
                "consecutive_passes": state.consecutive_passes,
                "evaluations": state.evaluations,
            }
        )
    return _validated_checkpoint_curriculum_artifacts(artifacts)


def _require_int(value: object, name: str) -> int:
    return _validated_integral(value, f"Resume metadata {name}")


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
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
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
        "observation_dtype",
        "action_schema_version",
        "action_count",
        "action_order",
        "method_profile",
        "resolved_config",
        "resume",
        "curriculum_state",
        "checkpoint_curriculum_artifacts",
    }
    optional = {"episode_seed_artifacts"}
    if not required <= set(values) or set(values) - required - optional:
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
    method_profile_value = _require_dict(values["method_profile"], "method_profile")
    if set(method_profile_value) != {
        "method_id",
        "policy_kind",
        "specialist_ids",
        "critic_enabled",
        "shield_mode",
    }:
        raise ValueError("Resume metadata method_profile fields are malformed")
    specialist_ids_value = method_profile_value["specialist_ids"]
    if not isinstance(specialist_ids_value, list) or not all(
        isinstance(agent_id, str) for agent_id in specialist_ids_value
    ):
        raise ValueError("Resume metadata method_profile.specialist_ids is malformed")
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
        observation_dtype=_require_string(values["observation_dtype"], "observation_dtype"),
        action_schema_version=_require_int(
            values["action_schema_version"], "action_schema_version"
        ),
        action_count=_require_int(values["action_count"], "action_count"),
        action_order=tuple(action_order_value),
        method_profile=MethodProfileSnapshot(
            method_id=_require_string(method_profile_value["method_id"], "method_profile.method_id"),
            policy_kind=_require_string(
                method_profile_value["policy_kind"], "method_profile.policy_kind"
            ),
            specialist_ids=tuple(specialist_ids_value),
            critic_enabled=method_profile_value["critic_enabled"],
            shield_mode=_require_string(
                method_profile_value["shield_mode"], "method_profile.shield_mode"
            ),
        ),
        curriculum_state=values["curriculum_state"],
        checkpoint_curriculum_artifacts=values["checkpoint_curriculum_artifacts"],
        episode_seed_artifacts=values.get("episode_seed_artifacts", ()),
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
        or metadata.observation_dtype != OBSERVATION_DTYPE
    ):
        raise ValueError("Resume observation schema mismatch")
    if (
        metadata.action_schema_version != ACTION_SCHEMA_VERSION
        or metadata.action_count != len(ACTION_ORDER)
        or metadata.action_order != ACTION_ORDER
    ):
        raise ValueError("Resume action schema mismatch")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Resume metadata JSON contains duplicate key {key!r}")
        values[key] = value
    return values


def _load_run_metadata(path: Path) -> RunMetadata:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Resume metadata JSON number must be finite: {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
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
        if path not in parent_values
        or path not in current_values
        or not _json_equal(parent_values[path], current_values[path])
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
        config_bytes, _config_digest = read_stable_artifact_bytes(config_path)
        parent_payload = load_unique_yaml(config_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Resume source resolved config is malformed: {config_path}") from exc
    parent_config = _require_dict(parent_payload, "resolved config")
    if not _json_equal(parent_config, metadata.resolved_config):
        raise ValueError("Resume metadata resolved config does not match config_resolved.yaml")
    try:
        parent_app_config = AppConfig.model_validate(parent_config)
    except Exception as exc:
        raise ValueError(f"Resume source resolved config is malformed: {config_path}") from exc
    current_payload = current_config.model_dump(mode="json")
    config_diff = _allowed_config_diff(parent_config, current_payload)
    final_curriculum_summary = metadata.curriculum_state
    final_curriculum_path = run_dir / cast(str, final_curriculum_summary["path"])
    expected_final_digest = cast(str, final_curriculum_summary["sha256"])
    final_curriculum_state, _final_digest = read_curriculum_state_artifact(
        final_curriculum_path,
        expected_sha256=expected_final_digest,
    )
    expected_final_state = CurriculumState(
        level=final_curriculum_summary["level"],
        consecutive_passes=final_curriculum_summary["consecutive_passes"],
        evaluations=final_curriculum_summary["evaluations"],
    )
    if final_curriculum_state != expected_final_state:
        raise ValueError("Resume final curriculum state does not match run metadata")
    CurriculumController(
        parent_app_config.scenarios.curriculum,
        final_curriculum_state,
    )
    checkpoint_relative_path = checkpoint_path.relative_to(run_dir).as_posix()
    matching_artifacts = [
        artifact
        for artifact in metadata.checkpoint_curriculum_artifacts
        if artifact["checkpoint_path"] == checkpoint_relative_path
    ]
    if len(matching_artifacts) != 1:
        raise ValueError("Resume checkpoint does not have exactly one curriculum state binding")
    curriculum_summary = matching_artifacts[0]
    expected_checkpoint_digest = cast(str, curriculum_summary["checkpoint_sha256"])
    checkpoint_bytes, checkpoint_digest = read_stable_artifact_bytes(checkpoint_path)
    if checkpoint_digest != expected_checkpoint_digest:
        raise ValueError("Resume checkpoint hash does not match its curriculum binding")
    curriculum_path = run_dir / cast(str, curriculum_summary["state_path"])
    if not curriculum_path.is_file():
        raise ValueError(f"Resume checkpoint curriculum state not found: {curriculum_path}")
    expected_state_digest = cast(str, curriculum_summary["state_sha256"])
    curriculum_state, _state_digest = read_checkpoint_curriculum_artifact(
        curriculum_path,
        expected_checkpoint_sha256=expected_checkpoint_digest,
        expected_sha256=expected_state_digest,
    )
    expected_state = CurriculumState(
        level=curriculum_summary["level"],
        consecutive_passes=curriculum_summary["consecutive_passes"],
        evaluations=curriculum_summary["evaluations"],
    )
    if curriculum_state != expected_state:
        raise ValueError("Resume source curriculum state does not match run metadata")
    CurriculumController(current_config.scenarios.curriculum, curriculum_state)
    return ResumeSource(
        checkpoint=checkpoint_path,
        run_dir=run_dir.resolve(strict=True),
        metadata=metadata,
        resolved_config=parent_config,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=expected_checkpoint_digest,
        config_diff=config_diff,
        curriculum_state=curriculum_state,
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
    canonical = FloatSchedule(expected)
    if type(schedule) is not type(canonical) or vars(schedule).keys() != vars(canonical).keys():
        raise ValueError(
            f"Resume {name} mismatch: expected the pinned SB3 constant schedule wrapper"
        )
    value_schedule = cast(FloatSchedule, schedule).value_schedule
    canonical_value_schedule = canonical.value_schedule
    if (
        type(value_schedule) is not ConstantSchedule
        or type(value_schedule) is not type(canonical_value_schedule)
        or vars(value_schedule).keys() != vars(canonical_value_schedule).keys()
    ):
        raise ValueError(f"Resume {name} mismatch: expected the pinned SB3 constant schedule value")
    _require_equal(name, value_schedule.val, expected)


def write_run_metadata(metadata: RunMetadata, destination: Path) -> None:
    """Atomically replace metadata through a cleaned sibling temporary file."""

    resume_payload: object = None
    if metadata.resume is not None:
        resume = ResumeMetadata(
            parent_checkpoint_path=metadata.resume.parent_checkpoint_path,
            parent_checkpoint_sha256=metadata.resume.parent_checkpoint_sha256,
            parent_run_dir=metadata.resume.parent_run_dir,
            parent_config=metadata.resume.parent_config,
            config_diff=metadata.resume.config_diff,
            start_num_timesteps=metadata.resume.start_num_timesteps,
        )
        resume_payload = {
            "parent_checkpoint_path": resume.parent_checkpoint_path,
            "parent_checkpoint_sha256": resume.parent_checkpoint_sha256,
            "parent_run_dir": resume.parent_run_dir,
            "parent_config": _thaw_json(resume.parent_config, "parent_config"),
            "config_diff": _thaw_json(resume.config_diff, "config_diff"),
            "start_num_timesteps": resume.start_num_timesteps,
        }
    validated = RunMetadata(
        resolved_config=metadata.resolved_config,
        method_profile=metadata.method_profile,
        resume=metadata.resume,
        research_contract_version=metadata.research_contract_version,
        observation_schema_version=metadata.observation_schema_version,
        observation_shape=metadata.observation_shape,
        observation_dtype=metadata.observation_dtype,
        action_schema_version=metadata.action_schema_version,
        action_count=metadata.action_count,
        action_order=metadata.action_order,
        curriculum_state=metadata.curriculum_state,
        checkpoint_curriculum_artifacts=metadata.checkpoint_curriculum_artifacts,
        episode_seed_artifacts=metadata.episode_seed_artifacts,
    )
    payload = {
        "resolved_config": _thaw_json(validated.resolved_config, "resolved_config"),
        "resume": resume_payload,
        "research_contract_version": validated.research_contract_version,
        "observation_schema_version": validated.observation_schema_version,
        "observation_shape": list(validated.observation_shape),
        "observation_dtype": validated.observation_dtype,
        "action_schema_version": validated.action_schema_version,
        "action_count": validated.action_count,
        "action_order": list(validated.action_order),
        "method_profile": {
            "method_id": validated.method_profile.method_id,
            "policy_kind": validated.method_profile.policy_kind,
            "specialist_ids": list(validated.method_profile.specialist_ids),
            "critic_enabled": validated.method_profile.critic_enabled,
            "shield_mode": validated.method_profile.shield_mode,
        },
        "curriculum_state": _thaw_json(validated.curriculum_state, "curriculum_state"),
        "checkpoint_curriculum_artifacts": _thaw_json(
            validated.checkpoint_curriculum_artifacts,
            "checkpoint_curriculum_artifacts",
        ),
        "episode_seed_artifacts": _thaw_json(
            validated.episode_seed_artifacts,
            "episode_seed_artifacts",
        ),
    }
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
                payload,
                output,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
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
    expected_method_profile = MethodProfileSnapshot.from_method_id(config.method.id)
    if metadata.method_profile != expected_method_profile:
        raise ValueError("Resume method profile mismatch")
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
    "MethodProfileSnapshot",
    "ResumeMetadata",
    "RunMetadata",
    "curriculum_state_artifact",
    "sha256_file",
    "validate_resume_contract",
]
