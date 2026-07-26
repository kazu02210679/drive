"""Authenticated checkpoint discovery and fixed-validation model selection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from mad_driving.evaluation.metrics import EpisodeMetricRecord
from mad_driving.evaluation.models import EVALUATION_CASES
from mad_driving.methods import MethodProfileSnapshot

CheckpointKind = Literal["periodic", "level_best", "final"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PERIODIC_PATTERN = re.compile(r"ppo_checkpoint_([0-9]+)_steps\.zip\Z")
_LEVEL_BEST_PATTERN = re.compile(r"best_model_level_[0-3]\.zip\Z")
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
# SB3's `data` member contains JSON metadata, not tensor parameters. These limits leave
# ample room for normal serialized spaces/schedules while bounding hostile ZIP metadata.
_MAX_SB3_ARCHIVE_MEMBERS = 128
_MAX_SB3_DATA_BYTES = 16 * 1024 * 1024
_MAX_SB3_DATA_COMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_SB3_DATA_COMPRESSION_RATIO = 500.0
_FILE_READ_CHUNK_BYTES = 1024 * 1024
_DATA_READ_CHUNK_BYTES = 64 * 1024
_MODEL_SELECTION_COLUMNS = (
    "method_id",
    "policy_seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "training_timestep",
    "validation_plan_sha256",
    "mean_episode_reward",
    "collision_rate",
    "success_rate",
    "mean_route_completion",
    "selected",
)


@dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    sha256: str
    method_id: str
    policy_seed: int
    checkpoint_kind: CheckpointKind
    curriculum_level: int
    training_timestep: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not str(path) or path.suffix != ".zip":
            raise ValueError("candidate path must identify a ZIP checkpoint")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("candidate sha256 must be a lowercase SHA-256 digest")
        try:
            profile = MethodProfileSnapshot.from_method_id(self.method_id)
        except ValueError as error:
            raise ValueError("candidate method_id is unknown") from error
        if profile.policy_kind != "ppo":
            raise ValueError("candidate method_id must identify a PPO method")
        if type(self.policy_seed) is not int or self.policy_seed < 0:
            raise ValueError("candidate policy_seed must be a non-negative integer")
        if self.checkpoint_kind not in ("periodic", "level_best", "final"):
            raise ValueError("candidate checkpoint_kind is unknown")
        if type(self.curriculum_level) is not int or not 0 <= self.curriculum_level <= 3:
            raise ValueError("candidate curriculum_level must be from 0 through 3")
        if type(self.training_timestep) is not int or self.training_timestep < 0:
            raise ValueError("candidate training_timestep must be a non-negative integer")
        object.__setattr__(self, "path", path)


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ValidationPhysicalIdentity:
    """Canonical ordered identity of one physically instantiated validation episode."""

    case_id: str
    episode_rng_seed: int
    scenario_id: str
    difficulty_level: int
    metadrive_scenario_index: int
    scenario_selection_seed: int
    scenario_parameter_seed: int
    canonical_sampled_scenario_parameters: str

    @classmethod
    def from_record(cls, record: EpisodeMetricRecord) -> ValidationPhysicalIdentity:
        episode = record.episode
        return cls(
            case_id=episode.case_id,
            episode_rng_seed=episode.episode_rng_seed,
            scenario_id=episode.scenario_id,
            difficulty_level=episode.difficulty_level,
            metadrive_scenario_index=episode.metadrive_scenario_index,
            scenario_selection_seed=episode.scenario_selection_seed,
            scenario_parameter_seed=episode.scenario_parameter_seed,
            canonical_sampled_scenario_parameters=json.dumps(
                _plain_json(episode.sampled_scenario_parameters),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )


@dataclass(frozen=True)
class CheckpointScore:
    candidate: CheckpointCandidate
    training_timestep: int
    validation_plan_sha256: str
    validation_plan_rows: tuple[tuple[str, int], ...]
    episodes: tuple[EpisodeMetricRecord, ...]

    def __post_init__(self) -> None:
        if type(self.candidate) is not CheckpointCandidate:
            raise TypeError("candidate must be a CheckpointCandidate")
        if type(self.training_timestep) is not int or self.training_timestep < 0:
            raise ValueError("training_timestep must be a non-negative integer")
        if self.training_timestep != self.candidate.training_timestep:
            raise ValueError("training_timestep does not match the authenticated candidate")
        if (
            not isinstance(self.validation_plan_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.validation_plan_sha256) is None
        ):
            raise ValueError("validation_plan_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.episodes, tuple) or not self.episodes:
            raise ValueError("episodes must be a non-empty tuple")
        if any(type(record) is not EpisodeMetricRecord for record in self.episodes):
            raise TypeError("episodes must contain EpisodeMetricRecord values")
        if not isinstance(self.validation_plan_rows, tuple) or any(
            type(row) is not tuple
            or len(row) != 2
            or not isinstance(row[0], str)
            or type(row[1]) is not int
            or row[1] < 0
            for row in self.validation_plan_rows
        ):
            raise ValueError("validation_plan_rows must be ordered case/RNG pairs")
        expected_cases = {case.case_id for case in EVALUATION_CASES}
        actual_cases = {record.episode.case_id for record in self.episodes}
        if actual_cases != expected_cases:
            raise ValueError("checkpoint score must cover the fixed all-level validation cases")
        matrix = self.validation_matrix
        if matrix != self.validation_plan_rows:
            raise ValueError("checkpoint score episodes do not bind the validation plan rows")
        physical_identities = self.physical_identities
        if len(physical_identities) != len(set(physical_identities)):
            raise ValueError("checkpoint score contains a duplicate physical identity")
        if len(matrix) != len(set(matrix)):
            raise ValueError("checkpoint score contains a duplicate validation plan row")
        tracks = {record.episode.episode_key.track for record in self.episodes}
        if len(tracks) != 1:
            raise ValueError("checkpoint score must use one validation track")
        for record in self.episodes:
            episode = record.episode
            key = episode.episode_key
            metrics = record.metrics
            metrics.__post_init__()
            record.__post_init__()
            if key.role != "validation":
                raise ValueError("checkpoint scores accept validation episodes only")
            if (
                key.method_id != self.candidate.method_id
                or key.policy_seed != self.candidate.policy_seed
            ):
                raise ValueError("checkpoint score method/policy seed mismatch")
            if (
                episode.checkpoint_path is None
                or Path(episode.checkpoint_path) != self.candidate.path
                or episode.checkpoint_sha256 != self.candidate.sha256
            ):
                raise ValueError("checkpoint score candidate identity mismatch")
        object.__setattr__(self, "episodes", tuple(self.episodes))

    @property
    def validation_matrix(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (record.episode.case_id, record.episode.episode_rng_seed) for record in self.episodes
        )

    @property
    def validation_track(self) -> str:
        return self.episodes[0].episode.episode_key.track

    @property
    def physical_identities(self) -> tuple[ValidationPhysicalIdentity, ...]:
        return tuple(ValidationPhysicalIdentity.from_record(record) for record in self.episodes)

    @property
    def mean_episode_reward(self) -> float:
        return sum(record.metrics.episode_reward for record in self.episodes) / len(self.episodes)

    @property
    def collision_rate(self) -> float:
        return sum(record.metrics.collision for record in self.episodes) / len(self.episodes)

    @property
    def success_rate(self) -> float:
        return sum(record.metrics.scenario_success for record in self.episodes) / len(self.episodes)

    @property
    def mean_route_completion(self) -> float:
        return sum(record.metrics.final_route_completion for record in self.episodes) / len(
            self.episodes
        )


def _validated_directory(path: Path, name: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} cannot be a symbolic link")
    if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"{name} cannot be a reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be a directory")
    return path.resolve(strict=True)


def _validated_regular_file(path: Path, name: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} cannot be a symbolic link")
    if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"{name} cannot be a reparse point")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file")
    return path


def _json_equivalent(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _json_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
            return False
        return len(left) == len(right) and all(
            _json_equivalent(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_json_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _file_stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _path_regular_file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"checkpoint path identity is unavailable: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ValueError(f"checkpoint path identity is not a stable regular file: {path}")
    return _file_stat_signature(metadata)


def _assert_checkpoint_unchanged(
    checkpoint: Path,
    descriptor: int,
    expected_signature: tuple[int, int, int, int, int],
) -> None:
    try:
        descriptor_signature = _file_stat_signature(os.fstat(descriptor))
        path_signature = _path_regular_file_signature(checkpoint)
    except OSError as error:
        raise ValueError(
            f"checkpoint changed or was replaced while reading: {checkpoint}"
        ) from error
    if descriptor_signature != expected_signature or path_signature != expected_signature:
        raise ValueError(f"checkpoint changed or was replaced while reading: {checkpoint}")


def _bounded_sb3_data(archive: zipfile.ZipFile) -> bytes:
    members = archive.infolist()
    if len(members) > _MAX_SB3_ARCHIVE_MEMBERS:
        raise ValueError("SB3 checkpoint archive has too many members")
    if any(member.is_dir() and member.filename.rstrip("/") == "data" for member in members):
        raise ValueError("SB3 checkpoint data member cannot be a directory")
    data_members = [member for member in members if member.filename == "data"]
    if len(data_members) != 1:
        raise ValueError("SB3 checkpoint must contain exactly one canonical data member")
    member = data_members[0]
    if member.is_dir():
        raise ValueError("SB3 checkpoint data member cannot be a directory")
    if member.flag_bits & (1 | 0x40):
        raise ValueError("SB3 checkpoint data member cannot be encrypted")
    if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError("SB3 checkpoint data member uses unsupported compression")
    if member.file_size < 0 or member.file_size > _MAX_SB3_DATA_BYTES:
        raise ValueError("SB3 checkpoint data uncompressed size exceeds the limit")
    if member.compress_size < 0 or member.compress_size > _MAX_SB3_DATA_COMPRESSED_BYTES:
        raise ValueError("SB3 checkpoint data compressed size exceeds the limit")
    if member.file_size and member.compress_size == 0:
        raise ValueError("SB3 checkpoint data compression ratio exceeds the limit")
    if (
        member.compress_size
        and member.file_size / member.compress_size > _MAX_SB3_DATA_COMPRESSION_RATIO
    ):
        raise ValueError("SB3 checkpoint data compression ratio exceeds the limit")

    chunks: list[bytes] = []
    actual_size = 0
    with archive.open(member, "r") as data_source:
        while True:
            chunk = data_source.read(
                min(_DATA_READ_CHUNK_BYTES, _MAX_SB3_DATA_BYTES + 1 - actual_size)
            )
            if not chunk:
                break
            actual_size += len(chunk)
            if actual_size > _MAX_SB3_DATA_BYTES:
                raise ValueError("SB3 checkpoint actual streamed data exceeds the limit")
            chunks.append(chunk)
    if actual_size != member.file_size:
        raise ValueError("SB3 checkpoint actual data size does not match its declaration")
    return b"".join(chunks)


def _checkpoint_training_timestep(checkpoint: Path, expected_sha256: str) -> int:
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("authenticated checkpoint SHA-256 is malformed")
    try:
        with checkpoint.open("rb", buffering=0) as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("authenticated checkpoint descriptor is not a regular file")
            signature = _file_stat_signature(before)
            _assert_checkpoint_unchanged(checkpoint, source.fileno(), signature)
            digest = hashlib.sha256()
            while chunk := source.read(_FILE_READ_CHUNK_BYTES):
                digest.update(chunk)
            _assert_checkpoint_unchanged(checkpoint, source.fileno(), signature)
            if digest.hexdigest() != expected_sha256:
                raise ValueError(f"authenticated checkpoint SHA-256 mismatch: {checkpoint}")
            source.seek(0)
            with zipfile.ZipFile(source, "r") as archive:
                payload = _bounded_sb3_data(archive)
            _assert_checkpoint_unchanged(checkpoint, source.fileno(), signature)
    except ValueError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise ValueError(f"SB3 checkpoint ZIP data is malformed: {checkpoint}") from error
    try:
        data = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {constant}")
            ),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("SB3 checkpoint data member is malformed") from error
    if not isinstance(data, dict):
        raise ValueError("SB3 checkpoint data member must be a JSON object")
    timestep = data.get("num_timesteps")
    if type(timestep) is not int or timestep < 0:
        raise ValueError("SB3 checkpoint num_timesteps must be a non-negative integer")
    return timestep


def validate_ppo_checkpoint_archive(checkpoint_path: Path) -> str:
    """Validate one stable SB3 ZIP boundary and return its authenticated digest."""

    checkpoint = _validated_regular_file(Path(checkpoint_path), "PPO checkpoint")
    try:
        with checkpoint.open("rb", buffering=0) as source:
            signature = _file_stat_signature(os.fstat(source.fileno()))
            _assert_checkpoint_unchanged(checkpoint, source.fileno(), signature)
            digest = hashlib.sha256()
            while chunk := source.read(_FILE_READ_CHUNK_BYTES):
                digest.update(chunk)
            _assert_checkpoint_unchanged(checkpoint, source.fileno(), signature)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"PPO checkpoint is unreadable: {checkpoint}") from error
    result = digest.hexdigest()
    _checkpoint_training_timestep(checkpoint, result)
    return result


def _resolved_method_and_seed(config: Mapping[str, object]) -> tuple[str, int]:
    method = config.get("method")
    if not isinstance(method, Mapping) or set(method) != {"id"}:
        raise ValueError("resolved training method is malformed")
    method_id = method["id"]
    if not isinstance(method_id, str):
        raise ValueError("resolved training method is malformed")
    training = config.get("training")
    if not isinstance(training, Mapping) or "seed" not in training:
        raise ValueError("resolved training policy seed is missing")
    policy_seed = training["seed"]
    if type(policy_seed) is not int or policy_seed < 0:
        raise ValueError("resolved training policy seed is malformed")
    return method_id, policy_seed


def _validate_checkpoint_directory_entries(checkpoints_dir: Path) -> None:
    for entry in checkpoints_dir.iterdir():
        _validated_regular_file(entry, "checkpoint inventory entry")
        if not (entry.name.endswith(".zip") or entry.name.endswith(".zip.curriculum.yaml")):
            raise ValueError("checkpoint inventory contains an unsupported entry")


def discover_checkpoint_candidates(
    completed_run_dir: Path,
) -> tuple[CheckpointCandidate, ...]:
    """Discover only authenticated, supported checkpoints from a completed PPO run."""

    import yaml

    from mad_driving.config.parsing import load_unique_yaml
    from mad_driving.scenarios import EnvironmentRole
    from mad_driving.training import metadata as training_metadata
    from mad_driving.training.curriculum import (
        read_curriculum_state_artifact,
        read_stable_artifact_bytes,
    )
    from mad_driving.training.episode_seeds import (
        EpisodeSeedArtifactDescriptor,
        summarize_episode_seed_artifacts,
    )

    run_dir = _validated_directory(Path(completed_run_dir), "completed training run")
    metadata_path = _validated_regular_file(run_dir / "run_metadata.json", "training metadata")
    metadata = training_metadata._load_run_metadata(metadata_path)
    if not metadata.checkpoint_curriculum_artifacts or not metadata.episode_seed_artifacts:
        raise ValueError("training run is not complete")
    if metadata.method_profile.policy_kind != "ppo":
        raise ValueError("checkpoint candidates require a PPO training method")

    config_path = _validated_regular_file(
        run_dir / "config_resolved.yaml", "resolved training config"
    )
    try:
        config_bytes, _config_digest = read_stable_artifact_bytes(config_path)
        config_payload = load_unique_yaml(config_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("resolved training config is malformed") from error
    if not isinstance(config_payload, Mapping):
        raise ValueError("resolved training config is malformed")
    metadata_method, metadata_seed = _resolved_method_and_seed(metadata.resolved_config)
    config_method, config_seed = _resolved_method_and_seed(config_payload)
    if config_method != metadata_method:
        raise ValueError("resolved training method mismatch")
    if config_seed != metadata_seed:
        raise ValueError("resolved training policy seed mismatch")
    if not _json_equivalent(config_payload, metadata.resolved_config):
        raise ValueError("resolved training config does not match run metadata")

    try:
        seed_descriptors = tuple(
            EpisodeSeedArtifactDescriptor(
                role=cast(EnvironmentRole, summary["role"]),
                worker_index=_strict_json_integer(
                    summary["worker_index"], "episode seed worker_index"
                ),
                relative_path=str(summary["path"]),
                device=_strict_json_integer(
                    cast(Mapping[str, object], summary["file_identity"])["device"],
                    "episode seed device",
                ),
                inode=_strict_json_integer(
                    cast(Mapping[str, object], summary["file_identity"])["inode"],
                    "episode seed inode",
                    minimum=1,
                ),
            )
            for summary in metadata.episode_seed_artifacts
        )
        actual_seed_artifacts = summarize_episode_seed_artifacts(
            run_dir,
            expected_descriptors=seed_descriptors,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("episode seed artifact authentication failed") from error
    if not _json_equivalent(actual_seed_artifacts, metadata.episode_seed_artifacts):
        raise ValueError("episode seed artifact inventory does not match run metadata")

    state_summary = metadata.curriculum_state
    state_path = _validated_regular_file(
        run_dir / str(state_summary["path"]), "final curriculum state"
    )
    state, _state_digest = read_curriculum_state_artifact(
        state_path,
        expected_sha256=str(state_summary["sha256"]),
    )
    if (
        state.level != state_summary["level"]
        or state.consecutive_passes != state_summary["consecutive_passes"]
        or state.evaluations != state_summary["evaluations"]
    ):
        raise ValueError("final curriculum state does not match run metadata")

    checkpoints_dir = _validated_directory(run_dir / "checkpoints", "checkpoints directory")
    _validate_checkpoint_directory_entries(checkpoints_dir)
    actual_inventory = training_metadata.checkpoint_curriculum_artifact_inventory(checkpoints_dir)
    if not _json_equivalent(actual_inventory, metadata.checkpoint_curriculum_artifacts):
        raise ValueError("checkpoint inventory does not match run metadata")

    candidates: list[CheckpointCandidate] = []
    for artifact in metadata.checkpoint_curriculum_artifacts:
        relative_path = str(artifact["checkpoint_path"])
        name = Path(relative_path).name
        checkpoint_sha256 = str(artifact["checkpoint_sha256"])
        training_timestep = _checkpoint_training_timestep(
            run_dir / Path(relative_path), checkpoint_sha256
        )
        kind: CheckpointKind
        if _LEVEL_BEST_PATTERN.fullmatch(name) is not None:
            kind = "level_best"
        elif name == "final_model.zip":
            kind = "final"
        elif (periodic_match := _PERIODIC_PATTERN.fullmatch(name)) is not None:
            kind = "periodic"
            if training_timestep != int(periodic_match.group(1)):
                raise ValueError("periodic checkpoint filename timestep mismatch")
        elif name == "best_model.zip":
            continue
        else:
            raise ValueError(f"checkpoint inventory contains unsupported candidate: {name}")
        candidates.append(
            CheckpointCandidate(
                path=run_dir / Path(relative_path),
                sha256=checkpoint_sha256,
                method_id=metadata_method,
                policy_seed=metadata_seed,
                checkpoint_kind=kind,
                curriculum_level=int(artifact["level"]),
                training_timestep=training_timestep,
            )
        )
    if not candidates:
        raise ValueError("completed training run has no supported checkpoint candidates")
    return tuple(candidates)


def select_checkpoint(scores: Sequence[CheckpointScore]) -> CheckpointScore:
    """Select one candidate by the fixed deterministic lexicographic ordering."""

    values = tuple(scores)
    if not values:
        raise ValueError("scores must be non-empty")
    if any(type(score) is not CheckpointScore for score in values):
        raise TypeError("scores must contain CheckpointScore values")
    first = values[0]
    for score in values[1:]:
        if (
            score.candidate.method_id != first.candidate.method_id
            or score.candidate.policy_seed != first.candidate.policy_seed
        ):
            raise ValueError("scores must share one method/policy seed")
        if score.validation_plan_sha256 != first.validation_plan_sha256:
            raise ValueError("scores must share one validation plan hash")
        if (
            score.validation_plan_rows != first.validation_plan_rows
            or score.physical_identities != first.physical_identities
            or score.validation_track != first.validation_track
        ):
            raise ValueError("scores use a different scenario/seed matrix or physical identities")
    return min(
        values,
        key=lambda score: (
            -score.mean_episode_reward,
            score.collision_rate,
            -score.success_rate,
            -score.mean_route_completion,
            score.training_timestep,
            score.candidate.sha256,
        ),
    )


def write_selection_artifacts(
    output_dir: Path, scores: Sequence[CheckpointScore]
) -> tuple[Path, Path]:
    """Write fixed-column scores and strict selected-checkpoint identities."""

    directory = _validated_directory(Path(output_dir), "selection output directory")
    values = tuple(scores)
    if not values:
        raise ValueError("scores must be non-empty")
    if any(type(score) is not CheckpointScore for score in values):
        raise TypeError("scores must contain CheckpointScore values")
    reference = values[0]
    for score in values[1:]:
        if score.validation_plan_sha256 != reference.validation_plan_sha256:
            raise ValueError("scores do not share one validation plan hash")
        if (
            score.validation_plan_rows != reference.validation_plan_rows
            or score.physical_identities != reference.physical_identities
        ):
            raise ValueError("scores use a different scenario/seed matrix or physical identities")
    identities = tuple((score.candidate.path, score.candidate.sha256) for score in values)
    if len(identities) != len(set(identities)):
        raise ValueError("scores contain a duplicate checkpoint candidate")

    groups: dict[tuple[str, int], list[CheckpointScore]] = {}
    for score in values:
        key = (score.candidate.method_id, score.candidate.policy_seed)
        groups.setdefault(key, []).append(score)
    selected_by_group = {key: select_checkpoint(group) for key, group in sorted(groups.items())}
    ordered = sorted(
        values,
        key=lambda score: (
            score.candidate.method_id,
            score.candidate.policy_seed,
            score.training_timestep,
            score.candidate.sha256,
        ),
    )
    csv_path = directory / "model_selection.csv"
    json_path = directory / "selected_checkpoints.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("selection artifacts already exist")

    with csv_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=_MODEL_SELECTION_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for score in ordered:
            selected = selected_by_group[(score.candidate.method_id, score.candidate.policy_seed)]
            writer.writerow(
                {
                    "method_id": score.candidate.method_id,
                    "policy_seed": score.candidate.policy_seed,
                    "checkpoint_path": str(score.candidate.path),
                    "checkpoint_sha256": score.candidate.sha256,
                    "training_timestep": score.training_timestep,
                    "validation_plan_sha256": score.validation_plan_sha256,
                    "mean_episode_reward": score.mean_episode_reward,
                    "collision_rate": score.collision_rate,
                    "success_rate": score.success_rate,
                    "mean_route_completion": score.mean_route_completion,
                    "selected": "true" if score == selected else "false",
                }
            )

    selected_payload = {
        "schema_version": 1,
        "selected_checkpoints": [
            {
                "checkpoint_path": str(score.candidate.path),
                "checkpoint_sha256": score.candidate.sha256,
                "method_id": score.candidate.method_id,
                "policy_seed": score.candidate.policy_seed,
                "validation_plan_sha256": score.validation_plan_sha256,
            }
            for score in selected_by_group.values()
        ],
    }
    with json_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(
            selected_payload,
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output.write("\n")
    return csv_path, json_path


def write_unselected_smoke_checkpoint_artifacts(
    output_dir: Path,
    candidates: Sequence[CheckpointCandidate],
) -> tuple[Path, Path]:
    """Record authenticated smoke checkpoints without inventing validation metrics."""

    directory = _validated_directory(Path(output_dir), "selection output directory")
    values = tuple(candidates)
    if not values:
        raise ValueError("smoke checkpoint candidates must be non-empty")
    if any(type(candidate) is not CheckpointCandidate for candidate in values):
        raise TypeError("candidates must contain CheckpointCandidate values")
    identities = tuple(
        (candidate.method_id, candidate.policy_seed, candidate.path, candidate.sha256)
        for candidate in values
    )
    if len(identities) != len(set(identities)):
        raise ValueError("smoke checkpoint candidates contain a duplicate identity")
    method_seeds = tuple((candidate.method_id, candidate.policy_seed) for candidate in values)
    if len(method_seeds) != len(set(method_seeds)):
        raise ValueError("smoke checkpoint candidates contain a duplicate method/policy seed")

    csv_path = directory / "model_selection.csv"
    json_path = directory / "selected_checkpoints.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("selection artifacts already exist")
    ordered = sorted(
        values,
        key=lambda candidate: (
            candidate.method_id,
            candidate.policy_seed,
            candidate.training_timestep,
            candidate.sha256,
        ),
    )
    with csv_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=_MODEL_SELECTION_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in ordered:
            writer.writerow(
                {
                    "method_id": candidate.method_id,
                    "policy_seed": candidate.policy_seed,
                    "checkpoint_path": str(candidate.path),
                    "checkpoint_sha256": candidate.sha256,
                    "training_timestep": candidate.training_timestep,
                    "validation_plan_sha256": "",
                    "mean_episode_reward": "",
                    "collision_rate": "",
                    "success_rate": "",
                    "mean_route_completion": "",
                    "selected": "false",
                }
            )
    with json_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(
            {"schema_version": 1, "selected_checkpoints": []},
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output.write("\n")
    return csv_path, json_path


__all__ = [
    "CheckpointCandidate",
    "CheckpointScore",
    "ValidationPhysicalIdentity",
    "discover_checkpoint_candidates",
    "select_checkpoint",
    "validate_ppo_checkpoint_archive",
    "write_selection_artifacts",
    "write_unselected_smoke_checkpoint_artifacts",
]
