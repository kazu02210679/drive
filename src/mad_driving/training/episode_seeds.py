"""Durable actual-reset seed artifacts for PPO training environments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Final, NamedTuple

import gymnasium as gym

from mad_driving.scenarios import EnvironmentRole

EPISODE_SEED_ARTIFACT_SCHEMA_VERSION: Final = 3
_ARTIFACT_DIRECTORY: Final = "episode_seeds"
_HEADER_RECORD_TYPE: Final = "episode_seed_artifact"
_RECORD_FIELDS: Final = frozenset(
    {
        "role",
        "worker_index",
        "environment_seed",
        "scenario_selection_seed",
        "scenario_parameter_seed",
        "scenario_id",
        "difficulty_level",
        "scenario_parameters",
    }
)
_SEED_FIELDS: Final = (
    "environment_seed",
    "scenario_selection_seed",
    "scenario_parameter_seed",
)


class EpisodeSeedArtifactDescriptor(NamedTuple):
    """Parent-transferable identity captured from a live writer descriptor."""

    role: EnvironmentRole
    worker_index: int
    relative_path: str
    device: int
    inode: int


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
        if type(value) is not int or value < 0:
            raise ValueError(
                f"Environment reset seed info {field} must be a non-negative non-bool integer"
            )
        seeds[field] = int(value)
    scenario_id = info.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Environment reset seed info scenario_id must be a non-empty string")
    difficulty_level = info.get("difficulty_level")
    if type(difficulty_level) is not int or not 0 <= difficulty_level <= 3:
        raise ValueError(
            "Environment reset seed info difficulty_level must be an integer from 0 through 3"
        )
    scenario_parameters = _validated_json_value(
        info.get("scenario_parameters"),
        "scenario_parameters",
    )
    if not isinstance(scenario_parameters, dict):
        raise ValueError("Environment reset seed info scenario_parameters must be a JSON object")
    return {
        "role": role,
        "worker_index": worker_index,
        **seeds,
        "scenario_id": scenario_id,
        "difficulty_level": int(difficulty_level),
        "scenario_parameters": scenario_parameters,
    }


def _validated_json_value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Environment reset seed info {name} keys must be strings")
            result[key] = _validated_json_value(nested, f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _validated_json_value(nested, f"{name}[]")
            for nested in value
        ]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(
        f"Environment reset seed info {name} must contain only finite JSON-safe values"
    )


def _file_identity(stat_result: os.stat_result) -> tuple[int, int]:
    device = int(stat_result.st_dev)
    inode = int(stat_result.st_ino)
    if device < 0 or inode <= 0:
        raise RuntimeError("Episode seed artifact has no verifiable file identity")
    return device, inode


def _identity_payload(identity: tuple[int, int]) -> dict[str, int]:
    return {"device": identity[0], "inode": identity[1]}


def _encode_json_line(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parsed_json_integer(
    value: object,
    name: str,
    *,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Episode seed artifact {name} must be an integer >= {minimum}")
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Episode seed artifact append made no progress")
        remaining = remaining[written:]


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
        descriptor = os.open(
            artifact,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            identity = _file_identity(os.fstat(descriptor))
            _write_all(
                descriptor,
                _encode_json_line(
                    {
                        "file_identity": _identity_payload(identity),
                        "record_type": _HEADER_RECORD_TYPE,
                        "role": normalized_role,
                        "schema_version": EPISODE_SEED_ARTIFACT_SCHEMA_VERSION,
                        "worker_index": normalized_worker,
                    }
                ),
            )
            os.fsync(descriptor)
        except BaseException as primary_error:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"Episode seed descriptor cleanup also failed: {cleanup_error}"
                )
            raise
        self._artifact_root = artifact_root
        self._artifact = artifact
        self._role = normalized_role
        self._worker_index = normalized_worker
        self._descriptor: int | None = descriptor
        self._identity = identity
        self._artifact_descriptor = EpisodeSeedArtifactDescriptor(
            role=normalized_role,
            worker_index=normalized_worker,
            relative_path=artifact.relative_to(artifact_root.parent).as_posix(),
            device=identity[0],
            inode=identity[1],
        )
        self._closed = False

    @property
    def episode_seed_artifact_descriptor(self) -> EpisodeSeedArtifactDescriptor:
        """Return the immutable identity proven by the still-open writer."""

        descriptor = self._open_descriptor()
        self._assert_artifact_identity(descriptor)
        return self._artifact_descriptor

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        descriptor = self._open_descriptor()
        try:
            self._assert_artifact_identity(descriptor)
            if options is None:
                observation, info = self.env.reset(seed=seed)
            else:
                observation, info = self.env.reset(seed=seed, options=options)
            record = _validate_reset_info(
                info,
                role=self._role,
                worker_index=self._worker_index,
            )
            self._assert_artifact_identity(descriptor)
            _write_all(descriptor, _encode_json_line(record))
            os.fsync(descriptor)
        except BaseException as primary_error:
            if isinstance(primary_error, RuntimeError) and "artifact identity" in str(
                primary_error
            ):
                try:
                    self.close()
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        f"Episode seed recorder cleanup also failed: {cleanup_error}"
                    )
            raise
        return observation, dict(info)

    def close(self) -> None:
        """Close the wrapped environment and writer once without retrying resources."""

        if self._closed:
            return
        self._closed = True
        descriptor = self._descriptor
        self._descriptor = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            self.env.close()
        except BaseException as error:
            primary_error = error
        if descriptor is not None:
            try:
                self._assert_artifact_identity(descriptor)
                os.fsync(descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
        if primary_error is not None:
            for cleanup_error in cleanup_errors:
                primary_error.add_note(
                    f"Episode seed artifact cleanup also failed: {cleanup_error}"
                )
            raise primary_error
        if cleanup_errors:
            first, *remaining = cleanup_errors
            for cleanup_error in remaining:
                first.add_note(f"Episode seed artifact cleanup also failed: {cleanup_error}")
            raise first

    def _open_descriptor(self) -> int:
        descriptor = self._descriptor
        if self._closed or descriptor is None:
            raise RuntimeError("Episode seed recorder is closed")
        return descriptor

    def _assert_artifact_identity(self, descriptor: int) -> None:
        descriptor_identity = _file_identity(os.fstat(descriptor))
        try:
            path_stat = os.stat(self._artifact, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(
                "Episode seed artifact identity no longer matches its path"
            ) from error
        if (
            descriptor_identity != self._identity
            or _file_identity(path_stat) != self._identity
            or not self._artifact.is_file()
        ):
            raise RuntimeError("Episode seed artifact identity no longer matches its path")


def _parse_artifact_records(
    payload: bytes,
    artifact: Path,
    *,
    role: EnvironmentRole,
    worker_index: int,
    identity: tuple[int, int],
) -> list[dict[str, object]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"Episode seed artifact is unreadable: {artifact}") from exc
    if not payload.endswith(b"\n"):
        raise ValueError(f"Episode seed artifact has an incomplete final record: {artifact}")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Episode seed artifact header is missing: {artifact}")
    try:
        header = json.loads(
            lines[0],
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {constant}")
            ),
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Episode seed artifact header is malformed: {artifact}") from exc
    header_fields = {
        "file_identity",
        "record_type",
        "role",
        "schema_version",
        "worker_index",
    }
    if not isinstance(header, dict) or set(header) != header_fields:
        raise ValueError(f"Episode seed artifact file identity header is malformed: {artifact}")
    header_identity = header["file_identity"]
    if not isinstance(header_identity, dict) or set(header_identity) != {"device", "inode"}:
        raise ValueError(f"Episode seed artifact file identity header is malformed: {artifact}")
    parsed_identity = (
        _parsed_json_integer(header_identity["device"], "header device", minimum=0),
        _parsed_json_integer(header_identity["inode"], "header inode", minimum=1),
    )
    parsed_schema_version = _parsed_json_integer(
        header["schema_version"],
        "header schema_version",
        minimum=0,
    )
    parsed_worker_index = _parsed_json_integer(
        header["worker_index"],
        "header worker_index",
        minimum=0,
    )
    if (
        parsed_identity != identity
        or header["record_type"] != _HEADER_RECORD_TYPE
        or type(header["record_type"]) is not str
        or header["role"] != role
        or type(header["role"]) is not str
        or parsed_schema_version != EPISODE_SEED_ARTIFACT_SCHEMA_VERSION
        or parsed_worker_index != worker_index
    ):
        raise ValueError(f"Episode seed artifact file identity header is malformed: {artifact}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            raise ValueError(f"Episode seed artifact contains a blank record: {artifact}")
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {constant}")
                ),
                object_pairs_hook=_strict_json_object,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Episode seed artifact record is malformed: {artifact}:{line_number}"
            ) from exc
        if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
            raise ValueError(
                f"Episode seed artifact record fields are malformed: {artifact}:{line_number}"
            )
        record_role = value["role"]
        record_worker_index = _parsed_json_integer(
            value["worker_index"],
            "record worker_index",
            minimum=0,
        )
        if (
            type(record_role) is not str
            or record_role != role
            or record_worker_index != worker_index
        ):
            raise ValueError(
                f"Episode seed artifact identity is malformed: {artifact}:{line_number}"
            )
        expected = _validate_reset_info(value, role=role, worker_index=worker_index)
        if value != expected:
            raise ValueError(
                f"Episode seed artifact identity is malformed: {artifact}:{line_number}"
            )
        records.append(expected)
    return records


def _stat_signature(stat_result: os.stat_result) -> tuple[tuple[int, int], int, int, int]:
    return (
        _file_identity(stat_result),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _path_identity(artifact: Path) -> tuple[int, int]:
    try:
        return _file_identity(os.stat(artifact, follow_symlinks=False))
    except OSError as exc:
        raise ValueError(f"Episode seed artifact identity is unreadable: {artifact}") from exc


def _read_parse_and_hash_artifact(
    artifact: Path,
    *,
    descriptor: EpisodeSeedArtifactDescriptor,
) -> tuple[list[dict[str, object]], str, tuple[int, int]]:
    expected_identity = (descriptor.device, descriptor.inode)
    try:
        with artifact.open("rb", buffering=0) as source:
            before = os.fstat(source.fileno())
            identity = _file_identity(before)
            if identity != expected_identity:
                raise ValueError(f"Episode seed artifact identity changed: {artifact}")
            if _path_identity(artifact) != identity:
                raise ValueError(f"Episode seed artifact identity changed: {artifact}")
            payload = source.read()
            records = _parse_artifact_records(
                payload,
                artifact,
                role=descriptor.role,
                worker_index=descriptor.worker_index,
                identity=expected_identity,
            )
            digest = hashlib.sha256(payload).hexdigest()
            after = os.fstat(source.fileno())
            if _stat_signature(after) != _stat_signature(before):
                raise ValueError(
                    f"Episode seed artifact changed while being inventoried: {artifact}"
                )
            if _path_identity(artifact) != identity:
                raise ValueError(f"Episode seed artifact identity changed: {artifact}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Episode seed artifact is unreadable: {artifact}") from exc
    return records, digest, identity


def summarize_episode_seed_artifacts(
    workspace: Path,
    *,
    expected_descriptors: Sequence[EpisodeSeedArtifactDescriptor],
) -> tuple[dict[str, object], ...]:
    """Validate owned artifacts and return deterministic metadata summaries."""

    workspace_root = workspace.resolve(strict=True)
    summaries: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    expected_environments: set[tuple[EnvironmentRole, int]] = set()
    for descriptor in expected_descriptors:
        if not isinstance(descriptor, EpisodeSeedArtifactDescriptor):
            raise ValueError("Episode seed artifact descriptor is malformed")
        role, worker_index = _validate_identity(descriptor.role, descriptor.worker_index)
        expected_relative_path = f"{_ARTIFACT_DIRECTORY}/{_artifact_name(role, worker_index)}"
        if descriptor.relative_path != expected_relative_path:
            raise ValueError("Episode seed artifact descriptor path is malformed")
        _file_identity_payload(descriptor.device, descriptor.inode)
        environment_identity = (role, worker_index)
        if environment_identity in expected_environments:
            raise ValueError("Episode seed artifact descriptors contain a duplicate environment")
        expected_environments.add(environment_identity)
        try:
            artifact_root = (workspace_root / _ARTIFACT_DIRECTORY).resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "Episode seed artifact inventory does not match expected environments"
            ) from exc
        artifact = workspace_root / Path(*descriptor.relative_path.split("/"))
        try:
            resolved_artifact = artifact.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "Episode seed artifact inventory does not match expected environments"
            ) from exc
        if resolved_artifact.parent != artifact_root or not resolved_artifact.is_file():
            raise ValueError(f"Episode seed artifact is outside the run workspace: {artifact}")
        if resolved_artifact in expected_paths:
            raise ValueError("Episode seed artifact descriptors contain a duplicate path")
        expected_paths.add(resolved_artifact)
        records, digest, opened_identity = _read_parse_and_hash_artifact(
            resolved_artifact,
            descriptor=descriptor,
        )
        summaries.append(
            {
                "file_identity": _identity_payload(opened_identity),
                "path": resolved_artifact.relative_to(workspace_root).as_posix(),
                "record_count": len(records),
                "role": role,
                "schema_version": EPISODE_SEED_ARTIFACT_SCHEMA_VERSION,
                "sha256": digest,
                "worker_index": worker_index,
            }
        )
    try:
        artifact_root = (workspace_root / _ARTIFACT_DIRECTORY).resolve(strict=True)
        actual_paths = {path.resolve(strict=True) for path in artifact_root.iterdir()}
    except OSError as exc:
        raise ValueError(
            "Episode seed artifact inventory does not match expected environments"
        ) from exc
    if actual_paths != expected_paths:
        raise ValueError("Episode seed artifact inventory does not match expected environments")
    return tuple(summaries)


def _file_identity_payload(device: object, inode: object) -> tuple[int, int]:
    if (
        isinstance(device, bool)
        or not isinstance(device, Integral)
        or isinstance(inode, bool)
        or not isinstance(inode, Integral)
    ):
        raise ValueError("Episode seed artifact descriptor identity is malformed")
    normalized = (int(device), int(inode))
    if normalized[0] < 0 or normalized[1] <= 0:
        raise ValueError("Episode seed artifact descriptor identity is malformed")
    return normalized
