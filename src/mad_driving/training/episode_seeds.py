"""Durable actual-reset seed artifacts for PPO training environments."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import gymnasium as gym

from mad_driving.scenarios import EnvironmentRole

EPISODE_SEED_ARTIFACT_SCHEMA_VERSION: Final = 1
_ARTIFACT_DIRECTORY: Final = "episode_seeds"
_RECORD_FIELDS: Final = frozenset(
    {
        "role",
        "worker_index",
        "episode_rng_seed",
        "metadrive_scenario_index",
        "scenario_parameter_seed",
    }
)
_SEED_FIELDS: Final = (
    "episode_rng_seed",
    "metadrive_scenario_index",
    "scenario_parameter_seed",
)


def _validate_identity(role: object, worker_index: object) -> tuple[EnvironmentRole, int]:
    if role not in ("train", "validation", "test"):
        raise ValueError(f"Environment role is invalid: {role!r}")
    if isinstance(worker_index, bool) or not isinstance(worker_index, Integral):
        raise ValueError("worker_index must be a non-bool integer")
    normalized_worker = int(worker_index)
    if normalized_worker < 0:
        raise ValueError("worker_index must be non-negative")
    return role, normalized_worker  # type: ignore[return-value]


def _artifact_name(role: EnvironmentRole, worker_index: int) -> str:
    return f"{role}-worker-{worker_index:03d}.jsonl"


def _artifact_paths(
    workspace: Path,
    role: EnvironmentRole,
    worker_index: int,
) -> tuple[Path, Path]:
    workspace_root = workspace.resolve(strict=True)
    if not workspace_root.is_dir():
        raise NotADirectoryError(f"Run workspace is not a directory: {workspace_root}")
    artifact_root = workspace_root / _ARTIFACT_DIRECTORY
    artifact_root.mkdir(exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    if artifact_root.parent != workspace_root:
        raise RuntimeError(f"Refusing seed artifact directory outside workspace: {workspace_root}")
    artifact = artifact_root / _artifact_name(role, worker_index)
    if artifact.parent != artifact_root:
        raise RuntimeError(f"Refusing seed artifact outside workspace: {workspace_root}")
    return artifact_root, artifact


def _validate_reset_info(
    info: object,
    *,
    role: EnvironmentRole,
    worker_index: int,
) -> dict[str, object]:
    if not isinstance(info, Mapping):
        raise ValueError("Environment reset seed info must be a mapping")
    seeds: dict[str, int] = {}
    for field in _SEED_FIELDS:
        value = info.get(field)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(
                f"Environment reset seed info {field} must be a non-negative non-bool integer"
            )
        seeds[field] = int(value)
    return {
        "role": role,
        "worker_index": worker_index,
        **seeds,
    }


class EpisodeSeedRecordingWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Record the seed identities returned by every wrapped environment reset."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        workspace: Path,
        role: EnvironmentRole,
        worker_index: int,
    ) -> None:
        super().__init__(env)
        normalized_role, normalized_worker = _validate_identity(role, worker_index)
        artifact_root, artifact = _artifact_paths(
            workspace,
            normalized_role,
            normalized_worker,
        )
        descriptor = os.open(artifact, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._artifact_root = artifact_root
        self._artifact = artifact
        self._role = normalized_role
        self._worker_index = normalized_worker

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if options is None:
            observation, info = self.env.reset(seed=seed)
        else:
            observation, info = self.env.reset(seed=seed, options=options)
        record = _validate_reset_info(
            info,
            role=self._role,
            worker_index=self._worker_index,
        )
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        resolved_artifact = self._artifact.resolve(strict=True)
        if resolved_artifact.parent != self._artifact_root or not resolved_artifact.is_file():
            raise RuntimeError(
                f"Refusing seed artifact write outside workspace: {self._artifact_root.parent}"
            )
        with resolved_artifact.open("ab", buffering=0) as output:
            output.write(encoded)
            os.fsync(output.fileno())
        return observation, dict(info)


def _load_artifact_records(
    artifact: Path,
    *,
    role: EnvironmentRole,
    worker_index: int,
) -> list[dict[str, object]]:
    try:
        lines = artifact.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Episode seed artifact is unreadable: {artifact}") from exc
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"Episode seed artifact contains a blank record: {artifact}")
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {constant}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Episode seed artifact record is malformed: {artifact}:{line_number}"
            ) from exc
        if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
            raise ValueError(
                f"Episode seed artifact record fields are malformed: {artifact}:{line_number}"
            )
        expected = _validate_reset_info(value, role=role, worker_index=worker_index)
        if value != expected:
            raise ValueError(
                f"Episode seed artifact identity is malformed: {artifact}:{line_number}"
            )
        records.append(expected)
    return records


def summarize_episode_seed_artifacts(
    workspace: Path,
    *,
    expected_identities: Sequence[tuple[EnvironmentRole, int]],
) -> tuple[dict[str, object], ...]:
    """Validate owned artifacts and return deterministic metadata summaries."""

    workspace_root = workspace.resolve(strict=True)
    summaries: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    for raw_role, raw_worker_index in expected_identities:
        role, worker_index = _validate_identity(raw_role, raw_worker_index)
        artifact_root, artifact = _artifact_paths(workspace_root, role, worker_index)
        resolved_artifact = artifact.resolve(strict=True)
        if resolved_artifact.parent != artifact_root or not resolved_artifact.is_file():
            raise ValueError(f"Episode seed artifact is outside the run workspace: {artifact}")
        expected_paths.add(resolved_artifact)
        records = _load_artifact_records(
            resolved_artifact,
            role=role,
            worker_index=worker_index,
        )
        with resolved_artifact.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        summaries.append(
            {
                "path": resolved_artifact.relative_to(workspace_root).as_posix(),
                "record_count": len(records),
                "role": role,
                "schema_version": EPISODE_SEED_ARTIFACT_SCHEMA_VERSION,
                "sha256": digest,
                "worker_index": worker_index,
            }
        )
    artifact_root = (workspace_root / _ARTIFACT_DIRECTORY).resolve(strict=True)
    actual_paths = {path.resolve(strict=True) for path in artifact_root.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError("Episode seed artifact inventory does not match expected environments")
    return tuple(summaries)
