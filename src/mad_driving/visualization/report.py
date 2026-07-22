"""Deterministic Markdown reports from verified canonical evaluation artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from mad_driving.evaluation.compare import (
    COMPARISON_CSV_COLUMNS,
    EVAL_METRICS_CSV_COLUMNS,
)
from mad_driving.evaluation.training_metrics import TRAIN_METRICS_CSV_COLUMNS
from mad_driving.visualization import (
    METHOD_ORDER,
    PLOT_INVENTORY,
    SMOKE_RESULT_LABEL,
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
_EVAL_NUMERIC_COLUMNS: Final = frozenset(
    {
        "record_schema_version",
        "research_contract_version",
        "policy_seed",
        "episode_index",
        "test_seed",
        "metadrive_scenario_index",
        "scenario_selection_seed",
        "scenario_parameter_seed",
        "difficulty_level",
        *(metric for metrics in _METRIC_GROUPS.values() for metric in metrics),
    }
)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_csv(
    bundle: _VerifiedBundle,
    relative: str,
    columns: tuple[str, ...],
) -> list[dict[str, str]]:
    text = bundle.read_text(bundle.root / relative)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or tuple(reader.fieldnames) != columns:
        raise ValueError(f"{relative} must use the canonical fixed CSV columns")
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"{relative} is malformed CSV") from error
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{relative} must contain complete canonical rows")
    return rows


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


def _validate_training_rows(rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        _number(row["policy_seed"], "training policy_seed", allow_empty=False)
        _number(row["timestep"], "training timestep", allow_empty=False)
        _number(row["value"], "training metric value")


def _validate_eval_rows(rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        if row["track"] not in _TRACK_ORDER or row["method_id"] not in METHOD_ORDER:
            raise ValueError("eval_metrics.csv contains an unknown track or method")
        if row["is_formal"] not in {"True", "False"}:
            raise ValueError("eval_metrics.csv is_formal must be a canonical boolean")
        for column in _EVAL_NUMERIC_COLUMNS:
            _number(row[column], f"eval_metrics.csv {column}")


def _validate_comparison_rows(rows: Sequence[Mapping[str, str]]) -> None:
    valid_metrics = {metric for metrics in _METRIC_GROUPS.values() for metric in metrics}
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row["track"] not in _TRACK_ORDER or row["method_id"] not in METHOD_ORDER:
            raise ValueError("comparison.csv contains an unknown track or method")
        if row["metric"] not in valid_metrics:
            raise ValueError("comparison.csv contains an unknown metric")
        key = (row["track"], row["case_id"], row["method_id"], row["metric"])
        if key in seen:
            raise ValueError("comparison.csv contains duplicate grouped metrics")
        seen.add(key)
        _number(
            row["physical_episode_count"],
            "comparison physical episode count",
            allow_empty=False,
        )
        _number(
            row["policy_replicate_count"],
            "comparison policy replicate count",
            allow_empty=False,
        )
        _number(row["mean"], "comparison mean")
        _number(row["policy_seed_stdev"], "comparison policy seed stdev")


def _read_checkpoints(bundle: _VerifiedBundle) -> tuple[dict[str, object], ...]:
    path = bundle.root / "selected_checkpoints.json"
    try:
        payload = json.loads(
            bundle.read_text(path),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
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
    return tuple(
        sorted(
            checkpoints,
            key=lambda item: (
                METHOD_ORDER.index(str(item["method_id"])),
                cast(int, item["policy_seed"]),
            ),
        )
    )


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


def _format_metric(value: str) -> str:
    number = _number(value, "comparison mean")
    return "N/A" if number is None else format(number, ".6g")


def _metric_tables(comparison_rows: Sequence[Mapping[str, str]], track: str) -> list[str]:
    track_rows = [row for row in comparison_rows if row["track"] == track]
    identities = sorted(
        {(row["case_id"], row["method_id"]) for row in track_rows},
        key=lambda item: (item[0], METHOD_ORDER.index(item[1])),
    )
    by_metric = {
        (row["case_id"], row["method_id"], row["metric"]): row["mean"] for row in track_rows
    }
    lines: list[str] = []
    for group, metrics in _METRIC_GROUPS.items():
        lines.extend((f"### {group}", ""))
        rows = [
            (
                case_id,
                method_id,
                *(
                    _format_metric(by_metric.get((case_id, method_id, metric), ""))
                    for metric in metrics
                ),
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
    if output.exists():
        raise FileExistsError(output)
    bundle = _verify_bundle(Path(bundle_dir))
    _required_artifacts(bundle)
    training_rows = _read_csv(bundle, "metrics/train_metrics.csv", TRAIN_METRICS_CSV_COLUMNS)
    eval_rows = _read_csv(bundle, "metrics/eval_metrics.csv", EVAL_METRICS_CSV_COLUMNS)
    comparison_rows = _read_csv(bundle, "metrics/comparison.csv", COMPARISON_CSV_COLUMNS)
    _validate_training_rows(training_rows)
    _validate_eval_rows(eval_rows)
    _validate_comparison_rows(comparison_rows)
    checkpoints = _read_checkpoints(bundle)
    labels = {row["result_label"] for row in (*training_rows, *eval_rows, *comparison_rows)}
    if not labels <= {"", SMOKE_RESULT_LABEL} or len(labels) != 1:
        raise ValueError("canonical metrics disagree on smoke/formal provenance")
    smoke = labels == {SMOKE_RESULT_LABEL}

    def link(relative: str) -> str:
        return _relative_link(output, bundle.root / relative)

    schema_versions = ", ".join(sorted({row["record_schema_version"] for row in eval_rows}))
    contract_versions = ", ".join(sorted({row["research_contract_version"] for row in eval_rows}))
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
        {
            (
                row["track"],
                row["method_id"],
                row["policy_seed"] or "rule",
                row["case_id"],
                row["scenario_id"],
                row["test_seed"],
                row["shield_mode"],
            )
            for row in eval_rows
        },
        key=lambda row: (
            _TRACK_ORDER.index(row[0]),
            METHOD_ORDER.index(row[1]),
            row[3],
            int(row[5]),
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
