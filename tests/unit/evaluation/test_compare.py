from __future__ import annotations

import csv
from dataclasses import replace
from math import sqrt
from pathlib import Path

import pytest

from mad_driving.evaluation.compare import (
    COMPARISON_CSV_COLUMNS,
    EVAL_METRICS_CSV_COLUMNS,
    build_comparison_rows,
    validate_matched_episodes,
    write_comparison_csv,
    write_eval_metrics_csv,
)
from mad_driving.evaluation.metrics import EpisodeMetricRecord, EpisodeMetrics
from mad_driving.evaluation.models import EvaluationEpisodeKey, EvaluationEpisodeRecord
from mad_driving.methods import MethodProfileSnapshot

SYSTEM_METHODS = ("b0_rule", "b1_nominal", "b2_multi_no_review", "proposed")


def _metrics(reward: float, *, disagreement_rate: float | None = None) -> EpisodeMetrics:
    eligible = 0 if disagreement_rate is None else 2
    disagreement_count = 0 if disagreement_rate is None else int(disagreement_rate * eligible)
    return EpisodeMetrics(
        collision=False,
        crossing_actor_collision=False,
        near_miss=False,
        minimum_actual_ttc_s=None,
        negative_stopping_margin=False,
        minimum_stopping_margin_m=None,
        hard_rule_violation=False,
        raw_unsafe_request_rate=0.0,
        shield_intervention_rate=0.0,
        off_road=False,
        scenario_success=True,
        final_route_completion=1.0,
        average_speed_mps=5.0,
        simulated_travel_time_s=0.2,
        unnecessary_braking_event_count=0,
        unnecessary_stop_duration_s=0.0,
        longitudinal_acceleration_rms_mps2=0.0,
        maximum_deceleration_mps2=0.0,
        longitudinal_jerk_rms_mps3=0.0,
        agent_disagreement_eligible_steps=eligible,
        agent_disagreement_count=disagreement_count,
        agent_disagreement_rate=disagreement_rate,
        critic_challenge_eligible_steps=0,
        critic_challenge_count=0,
        critic_challenge_rate=None,
        critic_found_missed_danger_count=0,
        critic_found_missed_danger_rate=None,
        critic_false_challenge_count=0,
        critic_false_challenge_rate=None,
        agent_failure_fallback_count=0,
        decision_latency_p50_ms=1.0,
        decision_latency_p95_ms=1.0,
        decision_latency_p99_ms=1.0,
        episode_reward=reward,
    )


def _record(
    method_id: str,
    policy_seed: int | None,
    episode_index: int,
    reward: float,
    *,
    is_formal: bool = True,
    disagreement_rate: float | None = None,
) -> EpisodeMetricRecord:
    test_seed = 20_010 + episode_index
    checkpoint_path = None if method_id == "b0_rule" else f"runs/{method_id}/{policy_seed}.zip"
    checkpoint_sha256 = None if method_id == "b0_rule" else f"{policy_seed:064x}"
    episode = EvaluationEpisodeRecord(
        record_schema_version=1,
        research_contract_version=7,
        episode_key=EvaluationEpisodeKey(
            method_id=method_id,  # type: ignore[arg-type]
            track="system",
            role="test",
            policy_seed=policy_seed,
            case_id="level1_lead_brake",
            episode_rng_seed=test_seed,
        ),
        method_profile=MethodProfileSnapshot.from_method_id(method_id),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        episode_index=episode_index,
        is_formal=is_formal,
        episode_rng_seed=test_seed,
        metadrive_scenario_index=episode_index,
        scenario_selection_seed=100 + episode_index,
        scenario_parameter_seed=200 + episode_index,
        case_id="level1_lead_brake",
        scenario_id="lead_brake",
        difficulty_level=1,
        sampled_scenario_parameters={"initial_gap_m": 35.0 + episode_index},
        step_count=2,
        final_step_index=1,
        simulated_duration_s=0.2,
        cumulative_reward=reward,
        collision_occurred=False,
        collision_kind=None,
        scenario_success=True,
        scenario_failure=False,
        arrived=False,
        off_road=False,
        terminated=True,
        truncated=False,
        complete=True,
    )
    return EpisodeMetricRecord(
        episode,
        _metrics(reward, disagreement_rate=disagreement_rate),
    )


def _matched_records(*, is_formal: bool = True) -> tuple[EpisodeMetricRecord, ...]:
    rows: list[EpisodeMetricRecord] = []
    rewards = {42: (1.0, 3.0), 43: (5.0, 7.0)}
    for episode_index in range(2):
        rows.append(
            _record("b0_rule", None, episode_index, (2.0, 4.0)[episode_index], is_formal=is_formal)
        )
        for method_id in SYSTEM_METHODS[1:]:
            for policy_seed in (42, 43):
                rows.append(
                    _record(
                        method_id,
                        policy_seed,
                        episode_index,
                        rewards[policy_seed][episode_index],
                        is_formal=is_formal,
                    )
                )
    return tuple(rows)


def test_comparison_uses_matched_physical_episodes_then_policy_replicates() -> None:
    rows = build_comparison_rows(_matched_records())
    proposed_reward = next(
        row for row in rows if row.method_id == "proposed" and row.metric == "episode_reward"
    )
    b0_reward = next(
        row for row in rows if row.method_id == "b0_rule" and row.metric == "episode_reward"
    )

    assert proposed_reward.physical_episode_count == 2
    assert proposed_reward.policy_replicate_count == 2
    assert proposed_reward.mean == 4.0
    assert proposed_reward.policy_seed_stdev == pytest.approx(sqrt(8.0))
    assert b0_reward.physical_episode_count == 2
    assert b0_reward.policy_replicate_count == 1
    assert b0_reward.mean == 3.0
    assert b0_reward.policy_seed_stdev is None


def test_one_seed_smoke_has_no_policy_dispersion_and_preserves_undefined_rates() -> None:
    records = tuple(
        record
        for record in _matched_records(is_formal=False)
        if record.episode.episode_key.policy_seed in (None, 42)
    )

    rows = build_comparison_rows(records)
    reward = next(
        row for row in rows if row.method_id == "proposed" and row.metric == "episode_reward"
    )
    unavailable = next(
        row for row in rows if row.method_id == "proposed" and row.metric == "critic_challenge_rate"
    )

    assert reward.policy_replicate_count == 1
    assert reward.policy_seed_stdev is None
    assert unavailable.mean is None
    assert unavailable.policy_seed_stdev is None
    assert reward.result_label == "SMOKE - NOT A RESEARCH RESULT"


def test_validation_rejects_missing_duplicate_and_unmatched_rows() -> None:
    records = _matched_records()
    with pytest.raises(ValueError, match="required methods"):
        validate_matched_episodes(
            tuple(
                record for record in records if record.episode.episode_key.method_id != "proposed"
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_matched_episodes((*records, records[0]))

    unmatched = list(records)
    target = next(
        index
        for index, record in enumerate(unmatched)
        if record.episode.episode_key.method_id == "proposed"
        and record.episode.episode_key.policy_seed == 43
        and record.episode.episode_index == 1
    )
    changed = replace(unmatched[target].episode, episode_index=9)
    unmatched[target] = EpisodeMetricRecord(changed, unmatched[target].metrics)
    with pytest.raises(ValueError, match="matched"):
        validate_matched_episodes(tuple(unmatched))


def test_validation_rejects_policy_and_physical_contract_mismatches() -> None:
    records = list(_matched_records())

    pseudo_b0 = replace(
        records[0].episode,
        episode_key=replace(records[0].episode.episode_key, method_id="b1_nominal", policy_seed=99),
        method_profile=MethodProfileSnapshot.from_method_id("b1_nominal"),
        checkpoint_path="runs/b1_nominal/99.zip",
        checkpoint_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="policy seeds|matched"):
        validate_matched_episodes((*records, EpisodeMetricRecord(pseudo_b0, records[0].metrics)))

    drifted = replace(records[-1].episode, scenario_parameter_seed=999)
    with pytest.raises(ValueError, match="physical identity"):
        validate_matched_episodes(
            (*records[:-1], EpisodeMetricRecord(drifted, records[-1].metrics))
        )

    mixed_formality = replace(records[-1].episode, is_formal=False)
    with pytest.raises(ValueError, match="is_formal"):
        validate_matched_episodes(
            (*records[:-1], EpisodeMetricRecord(mixed_formality, records[-1].metrics))
        )

    object.__setattr__(records[-1].episode.method_profile, "shield_mode", "off")
    with pytest.raises(ValueError, match="profile|Shield"):
        validate_matched_episodes(tuple(records))


def test_csv_writers_use_fixed_columns_empty_none_cells_and_smoke_label(tmp_path: Path) -> None:
    records = tuple(
        record
        for record in _matched_records(is_formal=False)
        if record.episode.episode_key.policy_seed in (None, 42)
    )
    comparison_path = tmp_path / "metrics" / "comparison.csv"
    evaluation_path = tmp_path / "metrics" / "eval_metrics.csv"

    write_comparison_csv(comparison_path, build_comparison_rows(records))
    write_eval_metrics_csv(evaluation_path, records)

    with comparison_path.open(encoding="utf-8", newline="") as source:
        comparison_rows = list(csv.DictReader(source))
    with evaluation_path.open(encoding="utf-8", newline="") as source:
        evaluation_rows = list(csv.DictReader(source))
    assert tuple(comparison_rows[0]) == COMPARISON_CSV_COLUMNS
    assert tuple(evaluation_rows[0]) == EVAL_METRICS_CSV_COLUMNS
    b0 = next(row for row in evaluation_rows if row["method_id"] == "b0_rule")
    assert b0["policy_seed"] == ""
    assert b0["critic_challenge_rate"] == ""
    assert b0["result_label"] == "SMOKE - NOT A RESEARCH RESULT"
    assert all(row["result_label"] == "SMOKE - NOT A RESEARCH RESULT" for row in comparison_rows)
    assert comparison_path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in comparison_path.read_bytes()
