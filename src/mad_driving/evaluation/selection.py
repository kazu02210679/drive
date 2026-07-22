"""Authenticated checkpoint discovery and fixed-validation model selection."""

from __future__ import annotations

import csv
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mad_driving.evaluation.metrics import EpisodeMetricRecord
from mad_driving.evaluation.models import EVALUATION_CASES
from mad_driving.methods import MethodProfileSnapshot

CheckpointKind = Literal["periodic", "level_best", "final"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PERIODIC_PATTERN = re.compile(r"ppo_checkpoint_[0-9]+_steps\.zip\Z")
_LEVEL_BEST_PATTERN = re.compile(r"best_model_level_[0-3]\.zip\Z")
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
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
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class CheckpointScore:
    candidate: CheckpointCandidate
    training_timestep: int
    validation_plan_sha256: str
    episodes: tuple[EpisodeMetricRecord, ...]

    def __post_init__(self) -> None:
        if type(self.candidate) is not CheckpointCandidate:
            raise TypeError("candidate must be a CheckpointCandidate")
        if type(self.training_timestep) is not int or self.training_timestep < 0:
            raise ValueError("training_timestep must be a non-negative integer")
        if (
            not isinstance(self.validation_plan_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.validation_plan_sha256) is None
        ):
            raise ValueError("validation_plan_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.episodes, tuple) or not self.episodes:
            raise ValueError("episodes must be a non-empty tuple")
        if any(type(record) is not EpisodeMetricRecord for record in self.episodes):
            raise TypeError("episodes must contain EpisodeMetricRecord values")
        expected_cases = {case.case_id for case in EVALUATION_CASES}
        actual_cases = {record.episode.case_id for record in self.episodes}
        if actual_cases != expected_cases:
            raise ValueError("checkpoint score must cover the fixed all-level validation cases")
        matrix = self.validation_matrix
        if len(matrix) != len(set(matrix)):
            raise ValueError("checkpoint score contains a duplicate validation matrix row")
        tracks = {record.episode.episode_key.track for record in self.episodes}
        if len(tracks) != 1:
            raise ValueError("checkpoint score must use one validation track")
        for record in self.episodes:
            episode = record.episode
            key = episode.episode_key
            metrics = record.metrics
            if type(metrics.collision) is not bool or type(metrics.scenario_success) is not bool:
                raise ValueError("checkpoint ranking indicators must be boolean")
            for name, value in (
                ("episode_reward", metrics.episode_reward),
                ("final_route_completion", metrics.final_route_completion),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"checkpoint ranking {name} must be finite")
            if not 0.0 <= metrics.final_route_completion <= 1.0:
                raise ValueError("checkpoint ranking final_route_completion is out of range")
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
    from mad_driving.training import metadata as training_metadata
    from mad_driving.training.curriculum import (
        read_curriculum_state_artifact,
        read_stable_artifact_bytes,
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
        kind: CheckpointKind
        if _LEVEL_BEST_PATTERN.fullmatch(name) is not None:
            kind = "level_best"
        elif name == "final_model.zip":
            kind = "final"
        elif _PERIODIC_PATTERN.fullmatch(name) is not None:
            kind = "periodic"
        elif name == "best_model.zip":
            continue
        else:
            raise ValueError(f"checkpoint inventory contains unsupported candidate: {name}")
        candidates.append(
            CheckpointCandidate(
                path=run_dir / Path(relative_path),
                sha256=str(artifact["checkpoint_sha256"]),
                method_id=metadata_method,
                policy_seed=metadata_seed,
                checkpoint_kind=kind,
                curriculum_level=int(artifact["level"]),
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
            score.validation_matrix != first.validation_matrix
            or score.validation_track != first.validation_track
        ):
            raise ValueError("scores use a different scenario/seed matrix")
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
            score.validation_matrix != reference.validation_matrix
            or score.validation_track != reference.validation_track
        ):
            raise ValueError("scores use a different scenario/seed matrix")
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


__all__ = [
    "CheckpointCandidate",
    "CheckpointScore",
    "discover_checkpoint_candidates",
    "select_checkpoint",
    "write_selection_artifacts",
]
