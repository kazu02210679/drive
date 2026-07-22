"""Deterministic Markdown reports from verified canonical evaluation artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import fmean, stdev
from typing import Final, cast

from mad_driving.config.models import MethodId
from mad_driving.evaluation.compare import (
    COMPARISON_CSV_COLUMNS,
    EVAL_METRICS_CSV_COLUMNS,
)
from mad_driving.evaluation.metrics import EpisodeMetrics
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    TEST_SEED_START,
    TEST_SEED_STOP,
    EvaluationCase,
    EvaluationTrack,
    expected_runtime_shield_mode,
)
from mad_driving.evaluation.training_metrics import (
    REQUIRED_TENSORBOARD_TAGS,
    TRAIN_METRICS_CSV_COLUMNS,
)
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.visualization import (
    METHOD_ORDER,
    PLOT_INVENTORY,
    SMOKE_RESULT_LABEL,
    _reject_output_in_source_bundle,
    _unique_json_object,
    _VerifiedBundle,
    _verify_bundle,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TRACK_ORDER: Final = ("decision", "system", "ablation")
_TRACK_TITLES: Final = {
    "decision": "Decision track",
    "system": "System track",
    "ablation": "Ablation track",
}
_METHODS_BY_TRACK: Final[Mapping[str, tuple[str, ...]]] = {
    "decision": ("b1_nominal", "b2_multi_no_review", "proposed"),
    "system": ("b0_rule", "b1_nominal", "b2_multi_no_review", "proposed"),
    "ablation": (
        "proposed",
        "proposed_no_critic",
        "proposed_no_shield",
        "proposed_no_hazard",
    ),
}
_CASES_BY_ID: Final[Mapping[str, EvaluationCase]] = {
    case.case_id: case for case in EVALUATION_CASES
}
_CASE_ORDER: Final[Mapping[str, int]] = {
    case.case_id: index for index, case in enumerate(EVALUATION_CASES)
}
_METRIC_GROUPS: Final[Mapping[str, tuple[str, ...]]] = {
    "Safety": (
        "collision",
        "crossing_actor_collision",
        "near_miss",
        "minimum_actual_ttc_s",
        "negative_stopping_margin",
        "minimum_stopping_margin_m",
        "hard_rule_violation",
        "raw_unsafe_request_rate",
        "shield_intervention_rate",
        "off_road",
    ),
    "Efficiency": (
        "scenario_success",
        "final_route_completion",
        "average_speed_mps",
        "simulated_travel_time_s",
        "unnecessary_braking_event_count",
        "unnecessary_stop_duration_s",
        "episode_reward",
    ),
    "Comfort": (
        "longitudinal_acceleration_rms_mps2",
        "maximum_deceleration_mps2",
        "longitudinal_jerk_rms_mps3",
    ),
    "Multi-Agent": (
        "agent_disagreement_eligible_steps",
        "agent_disagreement_count",
        "agent_disagreement_rate",
        "critic_challenge_eligible_steps",
        "critic_challenge_count",
        "critic_challenge_rate",
        "critic_found_missed_danger_count",
        "critic_found_missed_danger_rate",
        "critic_false_challenge_count",
        "critic_false_challenge_rate",
        "agent_failure_fallback_count",
    ),
    "Latency": (
        "decision_latency_p50_ms",
        "decision_latency_p95_ms",
        "decision_latency_p99_ms",
    ),
}
_METRIC_NAMES: Final = tuple(field.name for field in fields(EpisodeMetrics))
_BOOL_METRICS: Final = frozenset(
    {
        "collision",
        "crossing_actor_collision",
        "near_miss",
        "negative_stopping_margin",
        "hard_rule_violation",
        "off_road",
        "scenario_success",
    }
)
_COUNT_METRICS: Final = frozenset(
    {
        "unnecessary_braking_event_count",
        "agent_disagreement_eligible_steps",
        "agent_disagreement_count",
        "critic_challenge_eligible_steps",
        "critic_challenge_count",
        "critic_found_missed_danger_count",
        "critic_false_challenge_count",
        "agent_failure_fallback_count",
    }
)
_OPTIONAL_METRICS: Final = frozenset(
    {
        "minimum_actual_ttc_s",
        "minimum_stopping_margin_m",
        "longitudinal_jerk_rms_mps3",
        "agent_disagreement_rate",
        "critic_challenge_rate",
        "critic_found_missed_danger_rate",
        "critic_false_challenge_rate",
    }
)
_TRAIN_METRICS: Final = (*REQUIRED_TENSORBOARD_TAGS, "policy_entropy")
_EXPECTED_COMPARISON_IDENTITIES: Final = tuple(
    (track, case.case_id, method_id, metric)
    for track in _TRACK_ORDER
    for case in EVALUATION_CASES
    for method_id in _METHODS_BY_TRACK[track]
    for metric in _METRIC_NAMES
)
_EXPECTED_COMPARISON_GROUPS: Final = tuple(
    (track, case.case_id, method_id)
    for track in _TRACK_ORDER
    for case in EVALUATION_CASES
    for method_id in _METHODS_BY_TRACK[track]
)


@dataclass(frozen=True)
class _TrainCsvRow:
    result_label: str
    is_formal: bool
    run_id: str
    method_id: str
    policy_seed: int
    timestep: int
    metric: str
    value: float | None


@dataclass(frozen=True)
class _EvalCsvRow:
    cells: Mapping[str, str]
    result_label: str
    is_formal: bool
    track: str
    method_id: str
    policy_seed: int | None
    case_id: str
    episode_index: int
    test_seed: int
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    metrics: EpisodeMetrics


@dataclass(frozen=True)
class _ComparisonCsvRow:
    result_label: str
    is_formal: bool
    track: str
    case_id: str
    method_id: str
    metric: str
    physical_episode_count: int
    policy_replicate_count: int
    mean: float | None
    policy_seed_stdev: float | None


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value: {value}")


def _raw_csv_rows(
    payload: bytes,
    label: str,
    columns: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or tuple(reader.fieldnames) != columns:
        raise ValueError(f"{label} must use the canonical fixed CSV columns")
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"{label} is malformed CSV") from error
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} must contain complete canonical rows")
    return rows


def _integer(value: str, label: str, *, minimum: int = 0) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"{label} must be a canonical integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _boolean(value: str, label: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"{label} must be a canonical boolean")
    return value == "True"


def _number(value: str, label: str, *, allow_empty: bool = True) -> float | None:
    if value == "" and allow_empty:
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric when available") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite when available")
    return number


def _row_formality(result_label: str, raw_formal: str | None, label: str) -> bool:
    if result_label not in {"", SMOKE_RESULT_LABEL}:
        raise ValueError(f"{label} result_label is invalid")
    is_formal = result_label == ""
    if raw_formal is not None and _boolean(raw_formal, f"{label} is_formal") is not is_formal:
        raise ValueError(f"{label} smoke/formal result_label is inconsistent")
    return is_formal


def _require_one_formality(values: Sequence[bool], label: str) -> None:
    if len(set(values)) != 1:
        raise ValueError(f"{label} mixes smoke and formal rows")


def _parse_train_metrics_csv(payload: bytes) -> tuple[_TrainCsvRow, ...]:
    raw_rows = _raw_csv_rows(payload, "train_metrics.csv", TRAIN_METRICS_CSV_COLUMNS)
    rows: list[_TrainCsvRow] = []
    identities: set[tuple[str, str, int, int, str]] = set()
    for raw in raw_rows:
        is_formal = _row_formality(raw["result_label"], None, "train_metrics.csv")
        run_id = raw["run_id"]
        if not run_id or run_id.strip() != run_id:
            raise ValueError("train_metrics.csv run_id must be a canonical non-empty string")
        method_id = raw["method_id"]
        if method_id not in METHOD_ORDER:
            raise ValueError("train_metrics.csv method_id is unknown")
        profile = MethodProfileSnapshot.from_method_id(method_id)
        if profile.policy_kind != "ppo":
            raise ValueError("train_metrics.csv accepts only PPO method rows")
        policy_seed = _integer(raw["policy_seed"], "training policy_seed")
        timestep = _integer(raw["timestep"], "training timestep")
        metric = raw["metric"]
        if metric not in _TRAIN_METRICS:
            raise ValueError("train_metrics.csv metric is unknown")
        value = _number(raw["value"], "training metric value")
        if is_formal and value is None:
            raise ValueError("formal training metric value is required")
        identity = (run_id, method_id, policy_seed, timestep, metric)
        if identity in identities:
            raise ValueError("train_metrics.csv contains a duplicate row identity")
        identities.add(identity)
        rows.append(
            _TrainCsvRow(
                raw["result_label"],
                is_formal,
                run_id,
                method_id,
                policy_seed,
                timestep,
                metric,
                value,
            )
        )
    _require_one_formality([row.is_formal for row in rows], "train_metrics.csv")
    grouped: dict[tuple[str, str, int], list[_TrainCsvRow]] = {}
    for row in rows:
        grouped.setdefault((row.run_id, row.method_id, row.policy_seed), []).append(row)
    required = frozenset(_TRAIN_METRICS)
    for run_identity, run_rows in grouped.items():
        by_metric = {
            metric: [row for row in run_rows if row.metric == metric] for metric in required
        }
        missing = sorted(metric for metric, points in by_metric.items() if not points)
        if missing:
            mode = "formal" if run_rows[0].is_formal else "smoke"
            raise ValueError(
                f"{mode} training run {run_identity!r} is missing required series: "
                + ", ".join(missing)
            )
        for metric, points in by_metric.items():
            unavailable = [point for point in points if point.value is None]
            if unavailable and (
                run_rows[0].is_formal or len(points) != 1 or points[0].timestep != 0
            ):
                raise ValueError(
                    f"training series {metric} must be wholly available or one explicit "
                    "smoke unavailable row at timestep 0"
                )
        entropy = {row.timestep: row.value for row in by_metric["train/entropy_loss"]}
        derived = {row.timestep: row.value for row in by_metric["policy_entropy"]}
        expected_derived = {
            timestep: None if value is None else -value for timestep, value in entropy.items()
        }
        if derived != expected_derived:
            raise ValueError(
                "policy_entropy must exactly equal the negated train/entropy_loss series"
            )
    return tuple(rows)


def _episode_metrics(raw: Mapping[str, str]) -> EpisodeMetrics:
    values: dict[str, object] = {}
    for name in _METRIC_NAMES:
        if name in _BOOL_METRICS:
            values[name] = _boolean(raw[name], f"eval_metrics.csv {name}")
        elif name in _COUNT_METRICS:
            values[name] = _integer(raw[name], f"eval_metrics.csv {name}")
        else:
            value = _number(
                raw[name],
                f"eval_metrics.csv {name}",
                allow_empty=name in _OPTIONAL_METRICS,
            )
            values[name] = value
    try:
        return EpisodeMetrics(**values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"eval_metrics.csv metric domains are invalid: {error}") from error


def _parse_eval_metrics_csv(payload: bytes) -> tuple[_EvalCsvRow, ...]:
    raw_rows = _raw_csv_rows(payload, "eval_metrics.csv", EVAL_METRICS_CSV_COLUMNS)
    rows: list[_EvalCsvRow] = []
    identities: set[tuple[str, str, int | None, str, int, int]] = set()
    for raw in raw_rows:
        is_formal = _row_formality(raw["result_label"], raw["is_formal"], "eval_metrics.csv")
        track = raw["track"]
        method_id = raw["method_id"]
        if track not in _METHODS_BY_TRACK or method_id not in _METHODS_BY_TRACK[track]:
            raise ValueError("eval_metrics.csv track/method is outside the fixed matrix")
        profile = MethodProfileSnapshot.from_method_id(method_id)
        policy_seed = (
            None
            if raw["policy_seed"] == ""
            else _integer(raw["policy_seed"], "evaluation policy_seed")
        )
        if (method_id == "b0_rule") is not (policy_seed is None):
            raise ValueError("eval_metrics.csv policy_seed does not match the method")
        case_id = raw["case_id"]
        if case_id not in _CASES_BY_ID:
            raise ValueError("eval_metrics.csv case_id is unknown")
        episode_index = _integer(raw["episode_index"], "evaluation episode_index")
        test_seed = _integer(raw["test_seed"], "evaluation test_seed")
        if not TEST_SEED_START <= test_seed < TEST_SEED_STOP:
            raise ValueError("eval_metrics.csv test_seed is outside the test split")
        if _integer(raw["record_schema_version"], "record_schema_version") != 1:
            raise ValueError("eval_metrics.csv record_schema_version must be 1")
        if _integer(raw["research_contract_version"], "research_contract_version") != 7:
            raise ValueError("eval_metrics.csv research_contract_version must be 7")
        checkpoint_path = raw["checkpoint_path"] or None
        checkpoint_sha256 = raw["checkpoint_sha256"] or None
        if method_id == "b0_rule":
            if checkpoint_path is not None or checkpoint_sha256 is not None:
                raise ValueError("B0 eval_metrics.csv rows cannot bind a checkpoint")
        elif (
            checkpoint_path is None
            or checkpoint_sha256 is None
            or _SHA256_PATTERN.fullmatch(checkpoint_sha256) is None
        ):
            raise ValueError("PPO eval_metrics.csv rows require checkpoint path and SHA-256")
        if raw["policy_kind"] != profile.policy_kind:
            raise ValueError("eval_metrics.csv policy_kind does not match the method")
        if raw["specialist_ids"] != ";".join(profile.specialist_ids):
            raise ValueError("eval_metrics.csv specialist_ids do not match the method")
        critic_enabled = _boolean(raw["critic_enabled"], "eval_metrics.csv critic_enabled")
        if critic_enabled is not profile.critic_enabled:
            raise ValueError("eval_metrics.csv critic_enabled does not match the method")
        expected_shield = expected_runtime_shield_mode(
            cast(EvaluationTrack, track), cast(MethodId, method_id)
        )
        if raw["shield_mode"] != expected_shield:
            raise ValueError("eval_metrics.csv shield_mode does not match the fixed matrix")
        case = _CASES_BY_ID[case_id]
        if raw["scenario_id"] != case.scenario_id:
            raise ValueError("eval_metrics.csv scenario_id does not match case_id")
        if _integer(raw["difficulty_level"], "difficulty_level") != case.difficulty_level:
            raise ValueError("eval_metrics.csv difficulty_level does not match case_id")
        for name in (
            "metadrive_scenario_index",
            "scenario_selection_seed",
            "scenario_parameter_seed",
        ):
            _integer(raw[name], f"eval_metrics.csv {name}")
        identity = (track, method_id, policy_seed, case_id, episode_index, test_seed)
        if identity in identities:
            raise ValueError("eval_metrics.csv contains a duplicate episode identity")
        identities.add(identity)
        rows.append(
            _EvalCsvRow(
                raw,
                raw["result_label"],
                is_formal,
                track,
                method_id,
                policy_seed,
                case_id,
                episode_index,
                test_seed,
                checkpoint_path,
                checkpoint_sha256,
                _episode_metrics(raw),
            )
        )
    _require_one_formality([row.is_formal for row in rows], "eval_metrics.csv")
    return tuple(rows)


def _parse_comparison_csv(payload: bytes) -> tuple[_ComparisonCsvRow, ...]:
    raw_rows = _raw_csv_rows(payload, "comparison.csv", COMPARISON_CSV_COLUMNS)
    rows: list[_ComparisonCsvRow] = []
    identities: set[tuple[str, str, str, str]] = set()
    for raw in raw_rows:
        is_formal = _row_formality(raw["result_label"], raw["is_formal"], "comparison.csv")
        track = raw["track"]
        method_id = raw["method_id"]
        case_id = raw["case_id"]
        metric = raw["metric"]
        if track not in _METHODS_BY_TRACK or method_id not in _METHODS_BY_TRACK[track]:
            raise ValueError("comparison.csv track/method is outside the fixed matrix")
        if case_id not in _CASES_BY_ID:
            raise ValueError("comparison.csv case_id is unknown")
        if metric not in _METRIC_NAMES:
            raise ValueError("comparison.csv metric is unknown")
        physical_count = _integer(
            raw["physical_episode_count"], "comparison physical episode count", minimum=1
        )
        replicate_count = _integer(
            raw["policy_replicate_count"], "comparison policy replicate count", minimum=1
        )
        mean = _number(raw["mean"], "comparison mean")
        stdev = _number(raw["policy_seed_stdev"], "comparison policy seed stdev")
        if stdev is not None and stdev < 0.0:
            raise ValueError("comparison policy seed stdev must be non-negative")
        if replicate_count == 1 and stdev is not None:
            raise ValueError("one policy replicate has unavailable sample standard deviation")
        if mean is None and stdev is not None:
            raise ValueError("unavailable comparison mean cannot have a standard deviation")
        bounded_metric = (
            metric in _BOOL_METRICS
            or metric.endswith("_rate")
            or metric == "final_route_completion"
        )
        if bounded_metric:
            if mean is not None and not 0.0 <= mean <= 1.0:
                raise ValueError(f"comparison mean for {metric} must be in [0, 1]")
        if metric in _COUNT_METRICS and mean is not None and mean < 0.0:
            raise ValueError(f"comparison mean for {metric} must be non-negative")
        identity = (track, case_id, method_id, metric)
        if identity in identities:
            raise ValueError("comparison.csv contains a duplicate row identity")
        identities.add(identity)
        rows.append(
            _ComparisonCsvRow(
                raw["result_label"],
                is_formal,
                track,
                case_id,
                method_id,
                metric,
                physical_count,
                replicate_count,
                mean,
                stdev,
            )
        )
    _require_one_formality([row.is_formal for row in rows], "comparison.csv")
    actual_identities = tuple((row.track, row.case_id, row.method_id, row.metric) for row in rows)
    if actual_identities != _EXPECTED_COMPARISON_IDENTITIES:
        raise ValueError(
            "comparison.csv required matrix is missing, extra, or outside canonical order"
        )
    return tuple(rows)


def _read_checkpoints(bundle: _VerifiedBundle) -> tuple[dict[str, object], ...]:
    path = bundle.root / "selected_checkpoints.json"
    try:
        text = bundle.read_bytes(path).decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("selected checkpoint JSON is malformed") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "selected_checkpoints"}:
        raise ValueError("selected checkpoint JSON fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("selected checkpoint schema_version must be 1")
    raw = payload["selected_checkpoints"]
    if not isinstance(raw, list):
        raise ValueError("selected_checkpoints must be a list")
    checkpoints: list[dict[str, object]] = []
    expected_fields = {
        "checkpoint_path",
        "checkpoint_sha256",
        "method_id",
        "policy_seed",
        "validation_plan_sha256",
    }
    identities: set[tuple[str, int]] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("selected checkpoint fields are invalid")
        method_id = item["method_id"]
        policy_seed = item["policy_seed"]
        path_value = item["checkpoint_path"]
        checkpoint_hash = item["checkpoint_sha256"]
        plan_hash = item["validation_plan_sha256"]
        if method_id not in METHOD_ORDER or method_id == "b0_rule":
            raise ValueError("selected checkpoint method is invalid")
        if type(policy_seed) is not int or policy_seed < 0:
            raise ValueError("selected checkpoint policy_seed is invalid")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("selected checkpoint path is invalid")
        if (
            not isinstance(checkpoint_hash, str)
            or _SHA256_PATTERN.fullmatch(checkpoint_hash) is None
        ):
            raise ValueError("selected checkpoint SHA-256 is invalid")
        if not isinstance(plan_hash, str) or _SHA256_PATTERN.fullmatch(plan_hash) is None:
            raise ValueError("selected checkpoint validation-plan SHA-256 is invalid")
        identity = (method_id, policy_seed)
        if identity in identities:
            raise ValueError("selected checkpoint identity is duplicated")
        identities.add(identity)
        checkpoints.append(item)
    ordered = tuple(
        sorted(
            checkpoints,
            key=lambda item: (
                METHOD_ORDER.index(str(item["method_id"])),
                cast(int, item["policy_seed"]),
            ),
        )
    )
    plan_hashes = {str(item["validation_plan_sha256"]) for item in ordered}
    if len(plan_hashes) > 1:
        raise ValueError("selected checkpoints have inconsistent validation-plan provenance")
    return ordered


def _validate_checkpoint_provenance(
    eval_rows: Sequence[_EvalCsvRow],
    checkpoints: Sequence[Mapping[str, object]],
    *,
    smoke: bool,
) -> None:
    selected = {
        (str(item["method_id"]), cast(int, item["policy_seed"])): (
            str(item["checkpoint_path"]),
            str(item["checkpoint_sha256"]),
        )
        for item in checkpoints
    }
    observed: dict[tuple[str, int], tuple[str, str]] = {}
    used_selected: set[tuple[str, int]] = set()
    for row in eval_rows:
        if row.method_id == "b0_rule":
            continue
        assert row.policy_seed is not None
        assert row.checkpoint_path is not None
        assert row.checkpoint_sha256 is not None
        key = (row.method_id, row.policy_seed)
        binding = (row.checkpoint_path, row.checkpoint_sha256)
        previous = observed.setdefault(key, binding)
        if previous != binding:
            raise ValueError("evaluation rows disagree on PPO checkpoint identity")
        selected_binding = selected.get(key)
        if selected_binding is None:
            if not smoke:
                raise ValueError("formal evaluation PPO checkpoint is absent from selection")
            continue
        if selected_binding != binding:
            raise ValueError("evaluation checkpoint path/hash does not match selected checkpoint")
        used_selected.add(key)
    if not smoke and used_selected != set(selected):
        raise ValueError("formal selected checkpoint inventory does not match evaluation")
    if smoke and not set(selected) <= set(observed):
        raise ValueError("smoke selected checkpoints must belong to evaluated PPO identities")


def _evaluation_physical_key(row: _EvalCsvRow) -> tuple[str, int, int]:
    return (row.case_id, row.episode_index, row.test_seed)


def _evaluation_physical_identity(row: _EvalCsvRow) -> tuple[str, ...]:
    return (
        row.cells["metadrive_scenario_index"],
        row.cells["scenario_selection_seed"],
        row.cells["scenario_parameter_seed"],
        row.cells["scenario_id"],
        row.cells["difficulty_level"],
    )


def _validate_matched_eval_rows(eval_rows: Sequence[_EvalCsvRow]) -> None:
    actual_groups = {(row.track, row.case_id, row.method_id) for row in eval_rows}
    if actual_groups != set(_EXPECTED_COMPARISON_GROUPS):
        raise ValueError("comparison groups are not exactly backed by eval_metrics.csv episodes")
    for track in _TRACK_ORDER:
        track_rows = [row for row in eval_rows if row.track == track]
        seed_sets = {
            method_id: {row.policy_seed for row in track_rows if row.method_id == method_id}
            for method_id in _METHODS_BY_TRACK[track]
            if method_id != "b0_rule"
        }
        if len({frozenset(seeds) for seeds in seed_sets.values()}) != 1:
            raise ValueError(f"eval_metrics.csv track {track} has unmatched PPO policy seeds")
        ppo_seeds = next(iter(seed_sets.values()))
        physical_sets: list[frozenset[tuple[str, int, int]]] = []
        for method_id in _METHODS_BY_TRACK[track]:
            seeds = {None} if method_id == "b0_rule" else ppo_seeds
            for policy_seed in seeds:
                physical_sets.append(
                    frozenset(
                        _evaluation_physical_key(row)
                        for row in track_rows
                        if row.method_id == method_id and row.policy_seed == policy_seed
                    )
                )
        if len(set(physical_sets)) != 1:
            raise ValueError(f"eval_metrics.csv track {track} has unmatched physical episodes")
        by_physical: dict[tuple[str, int, int], list[_EvalCsvRow]] = {}
        for row in track_rows:
            by_physical.setdefault(_evaluation_physical_key(row), []).append(row)
        for physical_key, matched in by_physical.items():
            identities = {_evaluation_physical_identity(row) for row in matched}
            if len(identities) != 1:
                raise ValueError(
                    f"eval_metrics.csv matched episode {physical_key!r} has differing provenance"
                )


def _validate_comparison_derivation(
    eval_rows: Sequence[_EvalCsvRow],
    comparison_rows: Sequence[_ComparisonCsvRow],
) -> None:
    _validate_matched_eval_rows(eval_rows)
    by_group: dict[tuple[str, str, str], dict[int | None, list[_EvalCsvRow]]] = {}
    for row in eval_rows:
        policies = by_group.setdefault((row.track, row.case_id, row.method_id), {})
        policies.setdefault(row.policy_seed, []).append(row)
    aggregates = {
        (row.track, row.case_id, row.method_id, row.metric): row for row in comparison_rows
    }
    for group in _EXPECTED_COMPARISON_GROUPS:
        by_policy = by_group[group]
        physical_counts = {len(rows) for rows in by_policy.values()}
        if len(physical_counts) != 1:
            raise ValueError(f"eval_metrics.csv group {group!r} has unequal physical counts")
        physical_count = next(iter(physical_counts))
        replicate_count = len(by_policy)
        for metric in _METRIC_NAMES:
            replicate_means: list[float] = []
            for policy_rows in by_policy.values():
                available = [
                    float(value)
                    for row in policy_rows
                    if (value := getattr(row.metrics, metric)) is not None
                ]
                if available:
                    replicate_means.append(fmean(available))
            expected_mean = fmean(replicate_means) if replicate_means else None
            expected_stdev = stdev(replicate_means) if len(replicate_means) >= 2 else None
            aggregate = aggregates[(*group, metric)]
            if aggregate.physical_episode_count != physical_count:
                raise ValueError("comparison physical episode count does not match eval rows")
            if aggregate.policy_replicate_count != replicate_count:
                raise ValueError("comparison policy replicate count does not match eval rows")
            if aggregate.mean != expected_mean:
                raise ValueError("comparison mean is not derived from eval rows")
            if aggregate.policy_seed_stdev != expected_stdev:
                raise ValueError("comparison policy-seed stdev is not derived from eval rows")


def _validate_training_provenance(
    training_rows: Sequence[_TrainCsvRow],
    eval_rows: Sequence[_EvalCsvRow],
    *,
    smoke: bool,
) -> None:
    evaluated = {
        (row.method_id, row.policy_seed) for row in eval_rows if row.policy_seed is not None
    }
    trained = {(row.method_id, row.policy_seed) for row in training_rows}
    if not trained <= evaluated:
        raise ValueError("training provenance contains a method/seed absent from evaluation")
    if not smoke and trained != evaluated:
        raise ValueError("formal evaluation requires training provenance for every PPO method/seed")


def _markdown_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in rows),
    ]


def _relative_link(output: Path, target: Path) -> str:
    relative = os.path.relpath(
        Path(os.path.abspath(target)),
        Path(os.path.abspath(output.parent)),
    )
    return relative.replace("\\", "/")


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else format(value, ".6g")


def _metric_tables(comparison_rows: Sequence[_ComparisonCsvRow], track: str) -> list[str]:
    track_rows = [row for row in comparison_rows if row.track == track]
    identities = sorted(
        {(row.case_id, row.method_id) for row in track_rows},
        key=lambda item: (_CASE_ORDER[item[0]], METHOD_ORDER.index(item[1])),
    )
    by_metric = {(row.case_id, row.method_id, row.metric): row.mean for row in track_rows}
    lines: list[str] = []
    for group, metrics in _METRIC_GROUPS.items():
        lines.extend((f"### {group}", ""))
        rows = [
            (
                case_id,
                method_id,
                *(_format_metric(by_metric[(case_id, method_id, metric)]) for metric in metrics),
            )
            for case_id, method_id in identities
        ]
        lines.extend(_table(("Case", "Method", *metrics), rows))
        lines.append("")
    return lines


def _required_artifacts(bundle: _VerifiedBundle) -> tuple[Path, ...]:
    required = (
        "evaluation_plan.yaml",
        "model_selection.csv",
        "selected_checkpoints.json",
        "metrics/train_metrics.csv",
        "metrics/eval_metrics.csv",
        "metrics/comparison.csv",
        *(f"plots/{name}" for name in PLOT_INVENTORY),
    )
    paths = tuple(bundle.root / relative for relative in required)
    for path in paths:
        bundle.relative_artifact(path)
    return paths


def write_markdown_report(bundle_dir: Path, output_md: Path) -> None:
    """Write a deterministic report using only a fully verified artifact bundle."""

    output = Path(output_md)
    bundle = _verify_bundle(Path(bundle_dir))
    _reject_output_in_source_bundle(bundle.root, output)
    if output.exists():
        raise FileExistsError(output)
    _required_artifacts(bundle)
    training_rows = _parse_train_metrics_csv(
        bundle.read_bytes(bundle.root / "metrics/train_metrics.csv")
    )
    eval_rows = _parse_eval_metrics_csv(bundle.read_bytes(bundle.root / "metrics/eval_metrics.csv"))
    comparison_rows = _parse_comparison_csv(
        bundle.read_bytes(bundle.root / "metrics/comparison.csv")
    )
    checkpoints = _read_checkpoints(bundle)
    formalities = {row.is_formal for row in training_rows}
    formalities.update(row.is_formal for row in eval_rows)
    formalities.update(row.is_formal for row in comparison_rows)
    if len(formalities) != 1:
        raise ValueError("canonical metrics disagree on smoke/formal provenance")
    smoke = formalities == {False}
    _validate_comparison_derivation(eval_rows, comparison_rows)
    _validate_training_provenance(training_rows, eval_rows, smoke=smoke)
    _validate_checkpoint_provenance(eval_rows, checkpoints, smoke=smoke)

    def link(relative: str) -> str:
        return _relative_link(output, bundle.root / relative)

    schema_versions = ", ".join(sorted({row.cells["record_schema_version"] for row in eval_rows}))
    contract_versions = ", ".join(
        sorted({row.cells["research_contract_version"] for row in eval_rows})
    )
    lines = ["# Phase 6 Evaluation Report", ""]
    if smoke:
        lines.extend(
            (
                f"> **{SMOKE_RESULT_LABEL}**",
                "> This smoke run is pipeline validation, not a formal scientific conclusion.",
                "",
            )
        )
    lines.extend(
        (
            "## Provenance",
            "",
            f"- [Artifact manifest]({link('evaluation_manifest.json')})",
            f"- [Evaluation plan]({link('evaluation_plan.yaml')})",
            f"- [Selected checkpoints]({link('selected_checkpoints.json')})",
            f"- [Model selection matrix]({link('model_selection.csv')})",
            f"- [Training metrics]({link('metrics/train_metrics.csv')})",
            f"- [Episode metrics]({link('metrics/eval_metrics.csv')})",
            f"- [Matched comparison]({link('metrics/comparison.csv')})",
            "",
            f"Record schema versions: {schema_versions}.",
            f"Research contract versions: {contract_versions}.",
            "Observation schema remains `(24,)`; action order remains "
            "`KEEP`, `SLOW`, `PREPARE_STOP`, `STOP`.",
            "",
            "## Selected checkpoint hashes",
            "",
        )
    )
    checkpoint_rows = [
        (
            item["method_id"],
            item["policy_seed"],
            item["checkpoint_path"],
            item["checkpoint_sha256"],
            item["validation_plan_sha256"],
        )
        for item in checkpoints
    ]
    lines.extend(
        _table(
            (
                "Method",
                "Policy seed",
                "Checkpoint",
                "Checkpoint SHA-256",
                "Validation plan SHA-256",
            ),
            checkpoint_rows,
        )
    )
    lines.extend(("", "## Exact evaluation matrix", ""))
    matrix = sorted(
        [
            (
                row.track,
                row.method_id,
                row.policy_seed if row.policy_seed is not None else "rule",
                row.case_id,
                row.cells["scenario_id"],
                row.test_seed,
                row.cells["shield_mode"],
            )
            for row in eval_rows
        ],
        key=lambda row: (
            _TRACK_ORDER.index(row[0]),
            METHOD_ORDER.index(row[1]),
            _CASE_ORDER[row[3]],
            row[5],
        ),
    )
    lines.extend(
        _table(
            ("Track", "Method", "Policy seed", "Case", "Scenario", "Episode seed", "Shield"),
            matrix,
        )
    )
    lines.append("")
    for track in _TRACK_ORDER:
        lines.extend((f"## {_TRACK_TITLES[track]}", ""))
        lines.extend(_metric_tables(comparison_rows, track))

    lines.extend(("## Plots", ""))
    for plot_name in PLOT_INVENTORY:
        lines.append(f"- [{plot_name}]({link(f'plots/{plot_name}')})")
    lines.extend(("", "## Representative episodes", ""))
    renders = sorted(
        relative
        for relative in bundle.artifacts
        if relative.startswith("renders/") and relative.endswith(".gif")
    )
    if renders:
        lines.extend(f"- [{Path(relative).name}]({link(relative)})" for relative in renders)
    else:
        lines.append("N/A — no representative episode GIF was declared in the manifest.")
    lines.extend(
        (
            "",
            "## N/A interpretation",
            "",
            "N/A means the canonical source cell was empty because the metric was "
            "unavailable; it is never interpreted as zero or NaN.",
            "",
            "## Failures and exclusions",
            "",
            "None recorded in the verified evaluation manifest.",
            "",
        )
    )
    encoded = "\n".join(lines).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(encoded)


__all__ = ["write_markdown_report"]
