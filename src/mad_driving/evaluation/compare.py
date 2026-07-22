"""Strict offline matching and aggregation of completed evaluation metrics."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import fmean, stdev
from types import MappingProxyType
from typing import Final

from mad_driving.config.models import MethodId
from mad_driving.evaluation.metrics import EpisodeMetricRecord, EpisodeMetrics
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    EvaluationTrack,
    ScenarioCellId,
    expected_runtime_shield_mode,
)
from mad_driving.methods import MethodProfileSnapshot

SMOKE_RESULT_LABEL: Final = "SMOKE - NOT A RESEARCH RESULT"

_METHODS_BY_TRACK: Final[Mapping[EvaluationTrack, tuple[MethodId, ...]]] = MappingProxyType(
    {
        "decision": ("b1_nominal", "b2_multi_no_review", "proposed"),
        "system": ("b0_rule", "b1_nominal", "b2_multi_no_review", "proposed"),
        "ablation": (
            "proposed",
            "proposed_no_critic",
            "proposed_no_shield",
            "proposed_no_hazard",
        ),
    }
)
_TRACK_ORDER: Final = {track: index for index, track in enumerate(_METHODS_BY_TRACK)}
_CASE_ORDER: Final = {case.case_id: index for index, case in enumerate(EVALUATION_CASES)}
_METRIC_NAMES: Final = tuple(field.name for field in fields(EpisodeMetrics))

COMPARISON_CSV_COLUMNS: Final = (
    "result_label",
    "is_formal",
    "track",
    "case_id",
    "method_id",
    "metric",
    "physical_episode_count",
    "policy_replicate_count",
    "mean",
    "policy_seed_stdev",
)

EVAL_METRICS_CSV_COLUMNS: Final = (
    "result_label",
    "is_formal",
    "record_schema_version",
    "research_contract_version",
    "track",
    "method_id",
    "policy_seed",
    "case_id",
    "episode_index",
    "test_seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "policy_kind",
    "specialist_ids",
    "critic_enabled",
    "shield_mode",
    "metadrive_scenario_index",
    "scenario_selection_seed",
    "scenario_parameter_seed",
    "scenario_id",
    "difficulty_level",
    *_METRIC_NAMES,
)


@dataclass(frozen=True)
class ComparisonRow:
    """One method/track/scenario metric aggregated over independent policies."""

    is_formal: bool
    track: EvaluationTrack
    case_id: ScenarioCellId
    method_id: MethodId
    metric: str
    physical_episode_count: int
    policy_replicate_count: int
    mean: float | None
    policy_seed_stdev: float | None

    def __post_init__(self) -> None:
        if type(self.is_formal) is not bool:
            raise TypeError("is_formal must be a bool")
        if self.track not in _METHODS_BY_TRACK:
            raise ValueError("track is unknown")
        if self.method_id not in _METHODS_BY_TRACK[self.track]:
            raise ValueError("method_id is outside the track matrix")
        if self.case_id not in _CASE_ORDER:
            raise ValueError("case_id is unknown")
        if self.metric not in _METRIC_NAMES:
            raise ValueError("metric is unknown")
        for name in ("physical_episode_count", "policy_replicate_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("mean", "policy_seed_stdev"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when available")
        if self.policy_replicate_count == 1 and self.policy_seed_stdev is not None:
            raise ValueError("one policy replicate has undefined sample standard deviation")

    @property
    def result_label(self) -> str:
        return "" if self.is_formal else SMOKE_RESULT_LABEL


def _physical_key(record: EpisodeMetricRecord) -> tuple[object, ...]:
    episode = record.episode
    key = episode.episode_key
    return (key.track, key.case_id, episode.episode_index, key.episode_rng_seed)


def _record_key(record: EpisodeMetricRecord) -> tuple[object, ...]:
    episode = record.episode
    key = episode.episode_key
    return (*_physical_key(record), key.method_id, key.policy_seed)


def _physical_identity(record: EpisodeMetricRecord) -> tuple[object, ...]:
    episode = record.episode
    return (
        episode.metadrive_scenario_index,
        episode.scenario_selection_seed,
        episode.scenario_parameter_seed,
        episode.case_id,
        episode.scenario_id,
        episode.difficulty_level,
        episode.sampled_scenario_parameters,
    )


def validate_matched_episodes(records: Sequence[EpisodeMetricRecord]) -> None:
    """Reject anything other than complete, exact track-matrix matched sets."""

    values = tuple(records)
    if not values:
        raise ValueError("records must be non-empty")
    if any(type(record) is not EpisodeMetricRecord for record in values):
        raise TypeError("records must contain EpisodeMetricRecord values")

    identities = tuple(_record_key(record) for record in values)
    if len(identities) != len(set(identities)):
        raise ValueError("comparison records contain a duplicate episode identity")

    is_formal = values[0].episode.is_formal
    if any(record.episode.is_formal is not is_formal for record in values[1:]):
        raise ValueError("comparison records disagree on explicit is_formal")

    for record in values:
        episode = record.episode
        key = episode.episode_key
        if key.role != "test":
            raise ValueError("comparison accepts only test episode records")
        expected_profile = MethodProfileSnapshot.from_method_id(key.method_id)
        if episode.method_profile != expected_profile:
            raise ValueError("episode method profile is invalid")
        if key.method_id not in _METHODS_BY_TRACK[key.track]:
            raise ValueError("episode method is outside the required track matrix")
        if episode.shield_mode != expected_runtime_shield_mode(key.track, key.method_id):
            raise ValueError("episode shield_mode is outside the required track matrix")

    tracks = {record.episode.episode_key.track for record in values}
    required_tracks = set(_METHODS_BY_TRACK)
    if tracks != required_tracks:
        raise ValueError(
            "comparison required tracks mismatch: "
            f"expected {sorted(required_tracks)}, got {sorted(tracks)}"
        )
    for track in tracks:
        track_records = tuple(
            record for record in values if record.episode.episode_key.track == track
        )
        actual_methods = {record.episode.episode_key.method_id for record in track_records}
        expected_methods = set(_METHODS_BY_TRACK[track])
        if actual_methods != expected_methods:
            raise ValueError(
                f"track {track} required methods mismatch: "
                f"expected {sorted(expected_methods)}, got {sorted(actual_methods)}"
            )

        ppo_seed_sets = {
            method_id: {
                record.episode.episode_key.policy_seed
                for record in track_records
                if record.episode.episode_key.method_id == method_id
            }
            for method_id in _METHODS_BY_TRACK[track]
            if method_id != "b0_rule"
        }
        if len({frozenset(seeds) for seeds in ppo_seed_sets.values()}) != 1:
            raise ValueError(f"track {track} PPO methods have different policy seeds")
        ppo_seeds = next(iter(ppo_seed_sets.values()))

        physical_sets: dict[tuple[str, int | None], set[tuple[object, ...]]] = {}
        for method_id in _METHODS_BY_TRACK[track]:
            seeds = {None} if method_id == "b0_rule" else ppo_seeds
            for policy_seed in seeds:
                physical_sets[(method_id, policy_seed)] = {
                    _physical_key(record)
                    for record in track_records
                    if record.episode.episode_key.method_id == method_id
                    and record.episode.episode_key.policy_seed == policy_seed
                }
        if len({frozenset(keys) for keys in physical_sets.values()}) != 1:
            raise ValueError(f"track {track} does not contain a full matched physical set")

        by_physical: dict[tuple[object, ...], list[EpisodeMetricRecord]] = defaultdict(list)
        for record in track_records:
            by_physical[_physical_key(record)].append(record)
        for physical_key, matched in by_physical.items():
            expected_identity = _physical_identity(matched[0])
            if any(_physical_identity(record) != expected_identity for record in matched[1:]):
                raise ValueError(
                    f"matched episode {physical_key!r} has differing physical identity"
                )

        bindings: dict[tuple[str, int | None], tuple[str | None, str | None]] = {}
        for record in track_records:
            episode = record.episode
            binding_key = (episode.episode_key.method_id, episode.episode_key.policy_seed)
            binding = (episode.checkpoint_path, episode.checkpoint_sha256)
            previous = bindings.setdefault(binding_key, binding)
            if previous != binding:
                raise ValueError("method/policy replicate checkpoint contract is inconsistent")


def _numeric_metric(metrics: EpisodeMetrics, name: str) -> float | None:
    value = getattr(metrics, name)
    if value is None:
        return None
    return float(value)


def build_comparison_rows(
    records: Sequence[EpisodeMetricRecord],
) -> tuple[ComparisonRow, ...]:
    """Aggregate physical episodes first, then independent policy-seed means."""

    validate_matched_episodes(records)
    grouped: dict[tuple[EvaluationTrack, ScenarioCellId, MethodId], list[EpisodeMetricRecord]] = (
        defaultdict(list)
    )
    for record in records:
        episode = record.episode
        grouped[(episode.episode_key.track, episode.case_id, episode.episode_key.method_id)].append(
            record
        )

    output: list[ComparisonRow] = []
    for (track, case_id, method_id), method_records in sorted(
        grouped.items(),
        key=lambda item: (
            _TRACK_ORDER[item[0][0]],
            _CASE_ORDER[item[0][1]],
            _METHODS_BY_TRACK[item[0][0]].index(item[0][2]),
        ),
    ):
        by_policy: dict[int | None, list[EpisodeMetricRecord]] = defaultdict(list)
        for record in method_records:
            by_policy[record.episode.episode_key.policy_seed].append(record)
        policy_seeds = sorted(by_policy, key=lambda seed: -1 if seed is None else seed)
        for metric_name in _METRIC_NAMES:
            replicate_means: list[float] = []
            for policy_seed in policy_seeds:
                available = [
                    value
                    for record in by_policy[policy_seed]
                    if (value := _numeric_metric(record.metrics, metric_name)) is not None
                ]
                if available:
                    replicate_means.append(fmean(available))
            output.append(
                ComparisonRow(
                    is_formal=method_records[0].episode.is_formal,
                    track=track,
                    case_id=case_id,
                    method_id=method_id,
                    metric=metric_name,
                    physical_episode_count=len(by_policy[policy_seeds[0]]),
                    policy_replicate_count=len(policy_seeds),
                    mean=fmean(replicate_means) if replicate_means else None,
                    policy_seed_stdev=(
                        stdev(replicate_means) if len(replicate_means) >= 2 else None
                    ),
                )
            )
    return tuple(output)


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CSV numeric cells must be finite")
    return value


def _write_csv(path: Path, columns: tuple[str, ...], rows: Sequence[Mapping[str, object]]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row[name]) for name in columns})


def write_comparison_csv(path: Path, rows: Sequence[ComparisonRow]) -> None:
    """Write deterministic fixed-column comparison rows."""

    values = tuple(rows)
    if any(type(row) is not ComparisonRow for row in values):
        raise TypeError("rows must contain ComparisonRow values")
    payload = tuple(
        {
            "result_label": row.result_label,
            "is_formal": row.is_formal,
            "track": row.track,
            "case_id": row.case_id,
            "method_id": row.method_id,
            "metric": row.metric,
            "physical_episode_count": row.physical_episode_count,
            "policy_replicate_count": row.policy_replicate_count,
            "mean": row.mean,
            "policy_seed_stdev": row.policy_seed_stdev,
        }
        for row in values
    )
    _write_csv(path, COMPARISON_CSV_COLUMNS, payload)


def write_eval_metrics_csv(path: Path, records: Sequence[EpisodeMetricRecord]) -> None:
    """Write canonical one-row-per-physical-episode evaluation metrics."""

    validate_matched_episodes(records)
    ordered = sorted(records, key=_record_key)
    payload: list[dict[str, object]] = []
    for record in ordered:
        episode = record.episode
        key = episode.episode_key
        profile = episode.method_profile
        row: dict[str, object] = {
            "result_label": "" if episode.is_formal else SMOKE_RESULT_LABEL,
            "is_formal": episode.is_formal,
            "record_schema_version": episode.record_schema_version,
            "research_contract_version": episode.research_contract_version,
            "track": key.track,
            "method_id": key.method_id,
            "policy_seed": key.policy_seed,
            "case_id": key.case_id,
            "episode_index": episode.episode_index,
            "test_seed": key.episode_rng_seed,
            "checkpoint_path": episode.checkpoint_path,
            "checkpoint_sha256": episode.checkpoint_sha256,
            "policy_kind": profile.policy_kind,
            "specialist_ids": ";".join(profile.specialist_ids),
            "critic_enabled": profile.critic_enabled,
            "shield_mode": episode.shield_mode,
            "metadrive_scenario_index": episode.metadrive_scenario_index,
            "scenario_selection_seed": episode.scenario_selection_seed,
            "scenario_parameter_seed": episode.scenario_parameter_seed,
            "scenario_id": episode.scenario_id,
            "difficulty_level": episode.difficulty_level,
        }
        row.update({name: getattr(record.metrics, name) for name in _METRIC_NAMES})
        payload.append(row)
    _write_csv(path, EVAL_METRICS_CSV_COLUMNS, payload)


__all__ = [
    "COMPARISON_CSV_COLUMNS",
    "EVAL_METRICS_CSV_COLUMNS",
    "ComparisonRow",
    "build_comparison_rows",
    "validate_matched_episodes",
    "write_comparison_csv",
    "write_eval_metrics_csv",
]
