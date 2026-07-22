from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mad_driving.evaluation.models import (
    REWARD_COMPONENT_KEYS,
    EvaluationEpisodeKey,
    EvaluationStepRecord,
)
from mad_driving.evaluation.serialization import read_jsonl_strict, write_jsonl_strict
from mad_driving.interfaces import CriticReview
from mad_driving.training import MethodProfileSnapshot


def make_step(step_index: int = 0, **overrides: Any) -> EvaluationStepRecord:
    key = EvaluationEpisodeKey(
        method_id="b0_rule",
        track="system",
        role="test",
        policy_seed=None,
        case_id="level0_nominal",
        episode_rng_seed=20_000,
    )
    values: dict[str, Any] = {
        "record_schema_version": 1,
        "research_contract_version": 7,
        "episode_key": key,
        "method_profile": MethodProfileSnapshot.from_method_id("b0_rule"),
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "episode_index": 2,
        "is_formal": False,
        "step_index": step_index,
        "simulation_time_s": step_index * 0.1,
        "decision_interval_s": 0.1,
        "episode_rng_seed": 20_000,
        "metadrive_scenario_index": 0,
        "scenario_selection_seed": 11,
        "scenario_parameter_seed": 13,
        "case_id": "level0_nominal",
        "scenario_id": "nominal",
        "difficulty_level": 0,
        "requested_action": 0,
        "required_action": 0,
        "executed_action": 0,
        "unsafe_request": False,
        "shield_intervened": False,
        "shield_reasons": (),
        "target_speed_mps": 10.0,
        "ego_speed_mps": 10.0,
        "ego_speed_limit_mps": 13.0,
        "ego_longitudinal_acceleration_mps2": 0.0,
        "route_completion": 0.1,
        "route_progress_m": 5.0,
        "lane_offset_m": 0.0,
        "collision_occurred": False,
        "collision_kind": None,
        "minimum_actual_ttc_s": None,
        "minimum_actual_stopping_margin_m": None,
        "pre_step_hard_rule_constraint": False,
        "post_step_rule_violation_event": False,
        "scenario_success": False,
        "scenario_failure": False,
        "arrived": False,
        "off_road": False,
        "terminated": False,
        "truncated": False,
        "cumulative_unnecessary_stop_duration_s": 0.0,
        "reward_total": 0.0,
        "reward_components": {name: 0.0 for name in REWARD_COMPONENT_KEYS},
        "claims": (),
        "review": CriticReview(0.0, False, 0.0, (), (), ()),
        "expected_agent_ids": (),
        "failed_agent_ids": (),
        "errors": (),
        "policy_inference_latency_ms": 0.2,
        "agent_analysis_latency_ms": 0.1,
        "shield_latency_ms": 0.1,
        "total_decision_latency_ms": 0.5,
        "frame_path": None,
    }
    values.update(overrides)
    return EvaluationStepRecord(**values)


def test_jsonl_writer_is_compact_sorted_utf8_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "steps.jsonl"
    row = {"z": "日本語", "a": 1}

    write_jsonl_strict(destination, (row,))

    assert destination.read_bytes() == b'{"a":1,"z":"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"}\n'
    with pytest.raises(FileExistsError):
        write_jsonl_strict(destination, (row,))


def test_jsonl_writer_rejects_nonfinite_values_without_leaving_destination(tmp_path: Path) -> None:
    destination = tmp_path / "bad.jsonl"

    with pytest.raises(ValueError):
        write_jsonl_strict(destination, ({"value": float("nan")},))
    assert not destination.exists()


def test_jsonl_reader_round_trips_steps_and_enforces_contiguous_indices(tmp_path: Path) -> None:
    destination = tmp_path / "steps.jsonl"
    records = (make_step(0), make_step(1))
    write_jsonl_strict(destination, (record.to_dict() for record in records))

    assert read_jsonl_strict(destination, EvaluationStepRecord) == records

    bad = tmp_path / "noncontiguous.jsonl"
    write_jsonl_strict(bad, (make_step(0).to_dict(), make_step(2).to_dict()))
    with pytest.raises(ValueError, match="contiguous"):
        read_jsonl_strict(bad, EvaluationStepRecord)


def test_step_reader_rejects_empty_and_mixed_episode_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_jsonl_strict(empty, EvaluationStepRecord)

    mixed = tmp_path / "mixed.jsonl"
    other_key = EvaluationEpisodeKey(
        method_id="b0_rule",
        track="system",
        role="test",
        policy_seed=None,
        case_id="level1_lead_brake",
        episode_rng_seed=20_001,
    )
    write_jsonl_strict(
        mixed,
        (
            make_step(0).to_dict(),
            make_step(
                1,
                episode_key=other_key,
                episode_rng_seed=20_001,
                case_id="level1_lead_brake",
                scenario_id="lead_brake",
                difficulty_level=1,
            ).to_dict(),
        ),
    )
    with pytest.raises(ValueError, match="episode key"):
        read_jsonl_strict(mixed, EvaluationStepRecord)


@pytest.mark.parametrize(
    ("field", "value"),
    [("episode_index", 3), ("is_formal", True)],
)
def test_step_reader_rejects_mixed_explicit_evaluation_provenance(
    tmp_path: Path, field: str, value: object
) -> None:
    destination = tmp_path / f"mixed-{field}.jsonl"
    write_jsonl_strict(
        destination,
        (make_step(0).to_dict(), make_step(1, **{field: value}).to_dict()),
    )

    with pytest.raises(ValueError, match=field):
        read_jsonl_strict(destination, EvaluationStepRecord)


def test_jsonl_reader_rejects_duplicate_keys_at_every_nesting_level(tmp_path: Path) -> None:
    payload = make_step().to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    duplicated_nested = encoded.replace(
        '"episode_rng_seed":20000',
        '"episode_rng_seed":20000,"episode_rng_seed":20000',
        1,
    )
    destination = tmp_path / "duplicate.jsonl"
    destination.write_text(duplicated_nested + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        read_jsonl_strict(destination, EvaluationStepRecord)


def test_jsonl_reader_rejects_nan_blank_lines_and_incomplete_final_records(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.jsonl"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_jsonl_strict(nonfinite, EvaluationStepRecord)

    blank = tmp_path / "blank.jsonl"
    blank.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank"):
        read_jsonl_strict(blank, EvaluationStepRecord)

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="trailing newline"):
        read_jsonl_strict(incomplete, EvaluationStepRecord)
