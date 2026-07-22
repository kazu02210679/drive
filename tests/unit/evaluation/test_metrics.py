from __future__ import annotations

from dataclasses import fields, replace
from math import nan, sqrt
from typing import Any

import pytest

from mad_driving import evaluation
from mad_driving.evaluation.metrics import (
    EpisodeMetricRecord,
    EpisodeMetrics,
    reduce_episode,
)
from mad_driving.evaluation.models import (
    REWARD_COMPONENT_KEYS,
    EvaluationEpisodeKey,
    EvaluationEpisodeRecord,
    EvaluationStepRecord,
)
from mad_driving.interfaces import CriticReview, RiskClaim
from mad_driving.methods import MethodProfileSnapshot


def make_claim(
    agent_id: str,
    *,
    claim_id: str | None = None,
    speed_cap_mps: float = 13.0,
    hard_stop_required: bool = False,
) -> RiskClaim:
    return RiskClaim(
        claim_id=claim_id or f"{agent_id}:claim",
        agent_id=agent_id,
        event_type="rear_end",
        target_actor_id="lead",
        probability=0.5,
        confidence=0.9,
        severity=0.5,
        time_horizon_s=3.0,
        min_ttc_s=2.0,
        stopping_margin_m=1.0,
        recommended_max_speed_mps=speed_cap_mps,
        hard_stop_required=hard_stop_required,
        evidence=("visible actor",),
        assumptions=("constant velocity",),
        valid_until_step=99,
    )


def make_step(step_index: int = 0, **overrides: Any) -> EvaluationStepRecord:
    profile = MethodProfileSnapshot.from_method_id("proposed")
    episode_key = EvaluationEpisodeKey(
        method_id="proposed",
        track="system",
        role="validation",
        policy_seed=42,
        case_id="level1_lead_brake",
        episode_rng_seed=10_001,
    )
    components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
    components["progress_reward"] = 1.0
    values: dict[str, Any] = {
        "record_schema_version": 1,
        "research_contract_version": 7,
        "episode_key": episode_key,
        "method_profile": profile,
        "checkpoint_path": "runs/proposed/seed-42/model.zip",
        "checkpoint_sha256": "a" * 64,
        "step_index": step_index,
        "simulation_time_s": step_index * 0.1,
        "decision_interval_s": 0.1,
        "episode_rng_seed": 10_001,
        "metadrive_scenario_index": 17,
        "scenario_selection_seed": 31,
        "scenario_parameter_seed": 37,
        "case_id": "level1_lead_brake",
        "scenario_id": "lead_brake",
        "difficulty_level": 1,
        "requested_action": 0,
        "required_action": 0,
        "executed_action": 0,
        "unsafe_request": False,
        "shield_intervened": False,
        "shield_reasons": (),
        "target_speed_mps": 13.0,
        "ego_speed_mps": 5.0,
        "ego_speed_limit_mps": 13.0,
        "ego_longitudinal_acceleration_mps2": 0.0,
        "route_completion": 0.25,
        "route_progress_m": 10.0,
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
        "terminated": True,
        "truncated": False,
        "cumulative_unnecessary_stop_duration_s": 0.0,
        "reward_total": 1.0,
        "reward_components": components,
        "claims": (),
        "review": CriticReview(0.0, False, 0.0, (), (), ()),
        "expected_agent_ids": profile.specialist_ids,
        "failed_agent_ids": (),
        "errors": (),
        "policy_inference_latency_ms": 1.0,
        "agent_analysis_latency_ms": 2.0,
        "shield_latency_ms": 0.5,
        "total_decision_latency_ms": 4.0,
        "frame_path": None,
    }
    values.update(overrides)
    return EvaluationStepRecord(**values)


def make_episode() -> EvaluationEpisodeRecord:
    step = make_step()
    return EvaluationEpisodeRecord(
        record_schema_version=1,
        research_contract_version=7,
        episode_key=step.episode_key,
        method_profile=step.method_profile,
        checkpoint_path=step.checkpoint_path,
        checkpoint_sha256=step.checkpoint_sha256,
        episode_rng_seed=step.episode_rng_seed,
        metadrive_scenario_index=step.metadrive_scenario_index,
        scenario_selection_seed=step.scenario_selection_seed,
        scenario_parameter_seed=step.scenario_parameter_seed,
        case_id=step.case_id,
        scenario_id=step.scenario_id,
        difficulty_level=step.difficulty_level,
        sampled_scenario_parameters={"initial_gap_m": 30.0},
        step_count=1,
        final_step_index=0,
        simulated_duration_s=0.1,
        cumulative_reward=1.0,
        collision_occurred=False,
        collision_kind=None,
        scenario_success=False,
        scenario_failure=False,
        arrived=False,
        off_road=False,
        terminated=True,
        truncated=False,
        complete=True,
    )


def reward_components(total: float) -> dict[str, float]:
    components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
    components["progress_reward"] = total
    return components


def test_one_step_episode_preserves_undefined_metrics_as_none() -> None:
    metrics = reduce_episode((make_step(),), 0.1)

    assert tuple(field.name for field in fields(metrics)) == (
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
        "scenario_success",
        "final_route_completion",
        "average_speed_mps",
        "simulated_travel_time_s",
        "unnecessary_braking_event_count",
        "unnecessary_stop_duration_s",
        "longitudinal_acceleration_rms_mps2",
        "maximum_deceleration_mps2",
        "longitudinal_jerk_rms_mps3",
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
        "decision_latency_p50_ms",
        "decision_latency_p95_ms",
        "decision_latency_p99_ms",
        "episode_reward",
    )
    assert metrics.minimum_actual_ttc_s is None
    assert metrics.minimum_stopping_margin_m is None
    assert metrics.longitudinal_jerk_rms_mps3 is None
    assert metrics.agent_disagreement_rate is None
    assert metrics.critic_challenge_rate is None
    assert metrics.critic_found_missed_danger_rate is None
    assert metrics.critic_false_challenge_rate is None
    assert metrics.simulated_travel_time_s == 0.1
    assert metrics.average_speed_mps == 5.0
    assert metrics.decision_latency_p50_ms == 4.0
    assert metrics.episode_reward == 1.0


def test_episode_metric_record_requires_exact_concrete_members() -> None:
    metrics = reduce_episode((make_step(),), 0.1)
    paired = EpisodeMetricRecord(make_episode(), metrics)

    assert paired.episode.episode_key == make_episode().episode_key
    assert paired.metrics == metrics
    with pytest.raises(TypeError, match="episode"):
        EpisodeMetricRecord(object(), metrics)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metrics"):
        EpisodeMetricRecord(make_episode(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("records", "decision_dt_s", "near_miss_ttc_s", "message"),
    [
        ((), 0.1, 3.0, "non-empty"),
        ((make_step(0), make_step(2)), 0.1, 3.0, "contiguous"),
        ((make_step(decision_interval_s=0.2),), 0.1, 3.0, "decision_interval_s"),
        ((make_step(),), 0.0, 3.0, "decision_dt_s"),
        ((make_step(),), nan, 3.0, "decision_dt_s"),
        ((make_step(),), 0.1, 0.0, "near_miss_ttc_s"),
        ((make_step(),), 0.1, nan, "near_miss_ttc_s"),
    ],
)
def test_reducer_rejects_invalid_episode_sequences_and_thresholds(
    records: tuple[EvaluationStepRecord, ...],
    decision_dt_s: float,
    near_miss_ttc_s: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reduce_episode(records, decision_dt_s, near_miss_ttc_s=near_miss_ttc_s)


def test_reducer_rejects_records_from_more_than_one_episode() -> None:
    other_key = replace(make_step().episode_key, episode_rng_seed=10_002)
    other = make_step(1, episode_key=other_key, episode_rng_seed=10_002)

    with pytest.raises(ValueError, match="single episode"):
        reduce_episode((make_step(), other), 0.1)


def test_reducer_computes_hand_calculated_trajectory_metrics() -> None:
    speeds = (2.0, 4.0, 6.0, 8.0, 10.0)
    accelerations = (1.0, -2.0, -3.0, 1.0, 1.0)
    stop_durations = (0.0, 0.1, 0.2, 0.2, 0.3)
    latencies = (50.0, 10.0, 40.0, 20.0, 30.0)
    records = tuple(
        make_step(
            index,
            simulation_time_s=index * 0.5,
            decision_interval_s=0.5,
            ego_speed_mps=speeds[index],
            ego_longitudinal_acceleration_mps2=accelerations[index],
            cumulative_unnecessary_stop_duration_s=stop_durations[index],
            total_decision_latency_ms=latencies[index],
            unsafe_request=index in (1, 3),
            required_action=3 if index in (1, 3) else 0,
            executed_action=3 if index == 1 else 0,
            shield_intervened=index == 1,
            pre_step_hard_rule_constraint=index == 3,
            minimum_actual_ttc_s=(4.0, 2.0, 1.5, None, 2.5)[index],
            minimum_actual_stopping_margin_m=(3.0, 1.0, -1.0, None, 2.0)[index],
            collision_occurred=index == 4,
            collision_kind="crossing_actor" if index == 4 else None,
            off_road=index == 2,
            scenario_success=index == 4,
            route_completion=(index + 1) / 5,
            reward_total=float(index + 1),
            reward_components=reward_components(float(index + 1)),
        )
        for index in range(5)
    )

    metrics = reduce_episode(records, 0.5)

    assert metrics.collision is True
    assert metrics.crossing_actor_collision is True
    assert metrics.near_miss is False
    assert metrics.minimum_actual_ttc_s == 1.5
    assert metrics.negative_stopping_margin is True
    assert metrics.minimum_stopping_margin_m == -1.0
    assert metrics.hard_rule_violation is True
    assert metrics.raw_unsafe_request_rate == 0.4
    assert metrics.shield_intervention_rate == 0.2
    assert metrics.off_road is True
    assert metrics.scenario_success is True
    assert metrics.final_route_completion == 1.0
    assert metrics.average_speed_mps == 6.0
    assert metrics.simulated_travel_time_s == 2.5
    assert metrics.unnecessary_braking_event_count == 2
    assert metrics.unnecessary_stop_duration_s == 0.3
    assert metrics.longitudinal_acceleration_rms_mps2 == pytest.approx(sqrt(3.2))
    assert metrics.maximum_deceleration_mps2 == 3.0
    assert metrics.longitudinal_jerk_rms_mps3 == pytest.approx(sqrt(26.0))
    assert metrics.decision_latency_p50_ms == 30.0
    assert metrics.decision_latency_p95_ms == 50.0
    assert metrics.decision_latency_p99_ms == 50.0
    assert metrics.episode_reward == 15.0


@pytest.mark.parametrize(
    ("ttc_s", "threshold_s", "collision", "expected"),
    [
        (2.999, 3.0, False, True),
        (3.0, 3.0, False, False),
        (1.0, 3.0, True, False),
    ],
)
def test_near_miss_is_strictly_below_threshold_and_excludes_collisions(
    ttc_s: float,
    threshold_s: float,
    collision: bool,
    expected: bool,
) -> None:
    record = make_step(
        minimum_actual_ttc_s=ttc_s,
        collision_occurred=collision,
        collision_kind="vehicle" if collision else None,
    )

    assert reduce_episode((record,), 0.1, near_miss_ttc_s=threshold_s).near_miss is expected


def test_reducer_counts_disagreement_review_and_fallback_steps() -> None:
    nominal_keep = make_claim("nominal", speed_cap_mps=13.0)
    hazard_slow = make_claim("hazard", speed_cap_mps=10.0)
    rule_stop = make_claim("rule", speed_cap_mps=20.0, hard_stop_required=True)
    nominal_prepare = make_claim("nominal", claim_id="nominal:prepare", speed_cap_mps=4.0)
    hazard_prepare = make_claim("hazard", claim_id="hazard:prepare", speed_cap_mps=4.0)
    rule_only = make_claim("rule", claim_id="rule:only", speed_cap_mps=20.0)
    records = (
        make_step(
            0,
            ego_speed_limit_mps=20.0,
            claims=(nominal_keep, hazard_slow, rule_stop),
            review=CriticReview(
                0.7,
                True,
                0.5,
                ("hazard",),
                (nominal_keep.claim_id,),
                ("hazard is stricter",),
            ),
            minimum_actual_ttc_s=2.0,
            minimum_actual_stopping_margin_m=-0.5,
            terminated=False,
        ),
        make_step(
            1,
            ego_speed_limit_mps=20.0,
            claims=(nominal_prepare, hazard_prepare),
            review=CriticReview(
                0.4,
                False,
                0.5,
                (),
                (hazard_prepare.claim_id,),
                ("challenge on safe step",),
            ),
            minimum_actual_ttc_s=3.0,
            minimum_actual_stopping_margin_m=0.0,
            terminated=False,
        ),
        make_step(
            2,
            claims=(rule_only,),
            review=CriticReview(0.0, False, 0.5, ("rule",), (), ()),
            terminated=False,
        ),
        make_step(
            3,
            failed_agent_ids=("nominal", "hazard"),
            errors=(
                "nominal:RuntimeError:timeout",
                "hazard:ValueError:malformed",
            ),
        ),
    )

    metrics = reduce_episode(records, 0.1)

    assert metrics.agent_disagreement_eligible_steps == 2
    assert metrics.agent_disagreement_count == 1
    assert metrics.agent_disagreement_rate == 0.5
    assert metrics.critic_challenge_eligible_steps == 3
    assert metrics.critic_challenge_count == 2
    assert metrics.critic_challenge_rate == pytest.approx(2 / 3)
    assert metrics.critic_found_missed_danger_count == 1
    assert metrics.critic_found_missed_danger_rate == 0.5
    assert metrics.critic_false_challenge_count == 1
    assert metrics.critic_false_challenge_rate == 0.5
    assert metrics.agent_failure_fallback_count == 1


def test_challenge_subrates_are_none_when_no_challenge_occurred() -> None:
    record = make_step(claims=(make_claim("nominal"),))

    metrics = reduce_episode((record,), 0.1)

    assert metrics.critic_challenge_eligible_steps == 1
    assert metrics.critic_challenge_count == 0
    assert metrics.critic_challenge_rate == 0.0
    assert metrics.critic_found_missed_danger_rate is None
    assert metrics.critic_false_challenge_rate is None


def test_evaluation_package_exports_task_6_contracts() -> None:
    assert evaluation.EpisodeMetrics is EpisodeMetrics
    assert evaluation.EpisodeMetricRecord is EpisodeMetricRecord
    assert evaluation.reduce_episode is reduce_episode
    assert evaluation.CheckpointCandidate.__name__ == "CheckpointCandidate"
    assert evaluation.CheckpointScore.__name__ == "CheckpointScore"
    assert evaluation.discover_checkpoint_candidates.__name__ == ("discover_checkpoint_candidates")
    assert evaluation.select_checkpoint.__name__ == "select_checkpoint"
    assert evaluation.write_selection_artifacts.__name__ == "write_selection_artifacts"
