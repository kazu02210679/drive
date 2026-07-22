from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from math import inf, nan
from pathlib import Path
from typing import Any

import pytest

from mad_driving.evaluation.models import (
    REWARD_COMPONENT_KEYS,
    EvaluationEpisodeKey,
    EvaluationEpisodeRecord,
    EvaluationStepRecord,
)
from mad_driving.interfaces import CriticReview, RiskClaim
from mad_driving.training import MethodProfileSnapshot


def make_claim() -> RiskClaim:
    return RiskClaim(
        claim_id="nominal:lead",
        agent_id="nominal",
        event_type="rear_end",
        target_actor_id="lead",
        probability=0.25,
        confidence=0.9,
        severity=0.4,
        time_horizon_s=3.0,
        min_ttc_s=2.5,
        stopping_margin_m=4.0,
        recommended_max_speed_mps=8.0,
        hard_stop_required=False,
        evidence=("lead vehicle",),
        assumptions=("constant speed",),
        valid_until_step=3,
    )


def make_step(**overrides: Any) -> EvaluationStepRecord:
    components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
    components["progress_reward"] = 0.5
    values: dict[str, Any] = {
        "record_schema_version": 1,
        "research_contract_version": 7,
        "episode_key": EvaluationEpisodeKey(
            method_id="b1_nominal",
            track="decision",
            role="test",
            policy_seed=42,
            case_id="level1_lead_brake",
            episode_rng_seed=20_001,
        ),
        "method_profile": MethodProfileSnapshot.from_method_id("b1_nominal"),
        "checkpoint_path": "runs/b1/seed-42/best_model.zip",
        "checkpoint_sha256": "a" * 64,
        "step_index": 0,
        "simulation_time_s": 0.0,
        "decision_interval_s": 0.1,
        "episode_rng_seed": 20_001,
        "metadrive_scenario_index": 17,
        "scenario_selection_seed": 31,
        "scenario_parameter_seed": 37,
        "case_id": "level1_lead_brake",
        "scenario_id": "lead_brake",
        "difficulty_level": 1,
        "requested_action": 1,
        "required_action": 2,
        "executed_action": 2,
        "unsafe_request": True,
        "shield_intervened": True,
        "shield_reasons": ("minimum_ttc",),
        "target_speed_mps": 8.0,
        "ego_speed_mps": 10.0,
        "ego_speed_limit_mps": 13.0,
        "ego_longitudinal_acceleration_mps2": -1.0,
        "route_completion": 0.2,
        "route_progress_m": 12.5,
        "lane_offset_m": -0.1,
        "collision_occurred": False,
        "collision_kind": None,
        "minimum_actual_ttc_s": 2.4,
        "minimum_actual_stopping_margin_m": 3.5,
        "pre_step_hard_rule_constraint": False,
        "post_step_rule_violation_event": False,
        "scenario_success": False,
        "scenario_failure": False,
        "arrived": False,
        "off_road": False,
        "terminated": False,
        "truncated": False,
        "cumulative_unnecessary_stop_duration_s": 0.0,
        "reward_total": 0.5,
        "reward_components": components,
        "claims": (make_claim(),),
        "review": CriticReview(0.0, False, 0.4, ("nominal",), (), ()),
        "expected_agent_ids": ("nominal",),
        "failed_agent_ids": (),
        "errors": (),
        "policy_inference_latency_ms": 1.0,
        "agent_analysis_latency_ms": 2.0,
        "shield_latency_ms": 0.5,
        "total_decision_latency_ms": 4.0,
        "frame_path": "frames/episode-1/000000.png",
    }
    values.update(overrides)
    return EvaluationStepRecord(**values)


def make_episode(**overrides: Any) -> EvaluationEpisodeRecord:
    values: dict[str, Any] = {
        "record_schema_version": 1,
        "research_contract_version": 7,
        "episode_key": EvaluationEpisodeKey(
            method_id="b1_nominal",
            track="decision",
            role="test",
            policy_seed=42,
            case_id="level1_lead_brake",
            episode_rng_seed=20_001,
        ),
        "method_profile": MethodProfileSnapshot.from_method_id("b1_nominal"),
        "checkpoint_path": "runs/b1/seed-42/best_model.zip",
        "checkpoint_sha256": "b" * 64,
        "episode_rng_seed": 20_001,
        "metadrive_scenario_index": 17,
        "scenario_selection_seed": 31,
        "scenario_parameter_seed": 37,
        "case_id": "level1_lead_brake",
        "scenario_id": "lead_brake",
        "difficulty_level": 1,
        "sampled_scenario_parameters": {"initial_gap_m": 35.0, "nested": {"mild": True}},
        "step_count": 2,
        "final_step_index": 1,
        "simulated_duration_s": 0.2,
        "cumulative_reward": 1.5,
        "collision_occurred": False,
        "collision_kind": None,
        "scenario_success": True,
        "scenario_failure": False,
        "arrived": False,
        "off_road": False,
        "terminated": True,
        "truncated": False,
        "complete": True,
    }
    values.update(overrides)
    return EvaluationEpisodeRecord(**values)


def test_step_record_is_frozen_and_round_trips_all_required_fields() -> None:
    record = make_step()

    assert EvaluationStepRecord.from_dict(record.to_dict()) == record
    assert tuple(record.reward_components) == REWARD_COMPONENT_KEYS
    with pytest.raises(FrozenInstanceError):
        record.step_index = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.reward_components["progress_reward"] = 99.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_schema_version", 2),
        ("research_contract_version", 6),
        ("step_index", -1),
        ("simulation_time_s", -0.1),
        ("decision_interval_s", 0.0),
        ("requested_action", 4),
        ("collision_kind", "pedestrian"),
        ("route_completion", 1.1),
        ("policy_inference_latency_ms", -1.0),
        ("total_decision_latency_ms", 1.9),
    ],
)
def test_step_record_rejects_invalid_versions_enums_ranges_and_latencies(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        make_step(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simulation_time_s", nan),
        ("ego_speed_mps", inf),
        ("reward_total", nan),
        ("minimum_actual_stopping_margin_m", -inf),
        ("agent_analysis_latency_ms", nan),
    ],
)
def test_step_record_rejects_nonfinite_numbers(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_step(**{field: value})


@pytest.mark.parametrize("value", [-0.1, nan, inf])
def test_step_record_requires_finite_non_negative_ego_speed_limit(value: float) -> None:
    with pytest.raises(ValueError, match="ego_speed_limit_mps"):
        make_step(ego_speed_limit_mps=value)


def test_step_record_rejects_inconsistent_actions_rewards_and_outcomes() -> None:
    with pytest.raises(ValueError, match="unsafe_request"):
        make_step(unsafe_request=False)
    with pytest.raises(ValueError, match="shield_intervened"):
        make_step(shield_intervened=False)
    with pytest.raises(ValueError, match="reward"):
        make_step(reward_total=0.6)
    with pytest.raises(ValueError, match="success"):
        make_step(scenario_success=True, scenario_failure=True)
    with pytest.raises(ValueError, match="collision_kind"):
        make_step(collision_occurred=False, collision_kind="vehicle")


def test_step_record_requires_exact_reward_component_keys() -> None:
    missing = dict(make_step().reward_components)
    missing.pop("jerk_penalty")
    with pytest.raises(ValueError, match="reward_components"):
        make_step(reward_components=missing)

    extra = dict(make_step().reward_components)
    extra["bonus"] = 0.0
    with pytest.raises(ValueError, match="reward_components"):
        make_step(reward_components=extra)


def test_step_record_requires_reward_total_to_equal_exact_component_sum() -> None:
    with pytest.raises(ValueError, match="reward_total"):
        make_step(reward_total=0.500_000_000_5)


def test_step_record_accepts_exact_reward_calculator_sum_in_canonical_order() -> None:
    components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
    components["progress_reward"] = 0.1
    components["collision_penalty"] = -0.6
    components["near_miss_penalty"] = -0.3
    producer_total = sum(components[name] for name in REWARD_COMPONENT_KEYS)
    assert producer_total == -0.8

    record = make_step(reward_components=components, reward_total=producer_total)

    assert record.reward_total == producer_total


@pytest.mark.parametrize(
    "frame_path",
    ["", "/tmp/frame.png", "../frame.png", "frames/../frame.png", "C:/frames/frame.png"],
)
def test_step_record_rejects_unsafe_frame_paths(frame_path: str) -> None:
    with pytest.raises(ValueError, match="frame_path"):
        make_step(frame_path=frame_path)


def test_step_record_validates_checkpoint_and_profile_identity() -> None:
    b0_key = EvaluationEpisodeKey(
        method_id="b0_rule",
        track="system",
        role="test",
        policy_seed=None,
        case_id="level1_lead_brake",
        episode_rng_seed=20_001,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        make_step(
            episode_key=b0_key,
            method_profile=MethodProfileSnapshot.from_method_id("b0_rule"),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        make_step(checkpoint_sha256="A" * 64)
    with pytest.raises(ValueError, match="method_profile"):
        make_step(method_profile=MethodProfileSnapshot.from_method_id("proposed"))


def test_step_record_rejects_bad_failure_mapping_and_agent_identity() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        make_step(failed_agent_ids=("nominal",), errors=())
    with pytest.raises(ValueError, match="expected_agent_ids"):
        make_step(expected_agent_ids=("hazard",))
    with pytest.raises(ValueError, match="claim"):
        make_step(claims=(make_claim(), make_claim()))


def test_strict_from_dict_rejects_missing_and_unknown_fields() -> None:
    payload = make_step().to_dict()
    payload.pop("frame_path")
    with pytest.raises(ValueError, match="fields"):
        EvaluationStepRecord.from_dict(payload)

    payload = make_step().to_dict()
    payload["future_field"] = True
    with pytest.raises(ValueError, match="fields"):
        EvaluationStepRecord.from_dict(payload)


def test_step_from_dict_rejects_boolean_difficulty_level() -> None:
    payload = make_step().to_dict()
    payload["difficulty_level"] = True

    with pytest.raises(ValueError, match="difficulty_level"):
        EvaluationStepRecord.from_dict(payload)


def test_episode_from_dict_rejects_boolean_difficulty_level() -> None:
    payload = make_episode().to_dict()
    payload["difficulty_level"] = True

    with pytest.raises(ValueError, match="difficulty_level"):
        EvaluationEpisodeRecord.from_dict(payload)


def test_strict_from_dict_rejects_non_boolean_claim_and_review_fields() -> None:
    payload = make_step().to_dict()
    claims = payload["claims"]
    assert isinstance(claims, list)
    assert isinstance(claims[0], dict)
    claims[0]["hard_stop_required"] = 1
    with pytest.raises(ValueError, match="RiskClaim"):
        EvaluationStepRecord.from_dict(payload)

    payload = make_step().to_dict()
    review = payload["review"]
    assert isinstance(review, dict)
    review["unresolved_conflict"] = 0
    with pytest.raises(ValueError, match="CriticReview"):
        EvaluationStepRecord.from_dict(payload)


@pytest.mark.parametrize(
    "error",
    (
        "nominal:Evil\x00:message",
        "nominal:Evil-Type:message",
        "nominal:Runtime Error:message",
        f"nominal:{'E' * 129}:message",
    ),
)
def test_step_record_rejects_unsanitized_or_malformed_exception_type(error: str) -> None:
    with pytest.raises(ValueError, match="exception type"):
        make_step(failed_agent_ids=("nominal",), errors=(error,))


def test_importing_evaluation_does_not_load_simulator_or_training_framework() -> None:
    script = """
import sys
import mad_driving.evaluation

for forbidden in ("metadrive", "stable_baselines3"):
    assert not any(
        name == forbidden or name.startswith(forbidden + ".")
        for name in sys.modules
    ), forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_episode_record_round_trips_and_freezes_nested_json() -> None:
    source = {"initial_gap_m": 35.0, "nested": {"mild": True}}
    record = make_episode(sampled_scenario_parameters=source)
    source["initial_gap_m"] = 99.0

    assert record.sampled_scenario_parameters["initial_gap_m"] == 35.0
    assert EvaluationEpisodeRecord.from_dict(record.to_dict()) == record
    with pytest.raises(TypeError):
        record.sampled_scenario_parameters["new"] = 1  # type: ignore[index]


def test_episode_record_rejects_terminal_and_index_contradictions() -> None:
    with pytest.raises(ValueError, match="final_step_index"):
        make_episode(final_step_index=0)
    with pytest.raises(ValueError, match="terminal"):
        make_episode(terminated=False, truncated=False)
    with pytest.raises(ValueError, match="success"):
        make_episode(scenario_success=True, scenario_failure=True)
    with pytest.raises(ValueError, match="complete"):
        make_episode(complete=False)
    with pytest.raises(ValueError, match="collision_kind"):
        make_episode(collision_occurred=True, collision_kind=None)


def test_episode_record_accepts_gymnasium_double_terminal_flags() -> None:
    record = make_episode(terminated=True, truncated=True)

    assert record.terminated is True
    assert record.truncated is True


def test_episode_key_enforces_b0_and_ppo_policy_seed_rules() -> None:
    with pytest.raises(ValueError, match="B0"):
        replace(make_step().episode_key, method_id="b0_rule", policy_seed=42)
    with pytest.raises(ValueError, match="PPO"):
        replace(make_step().episode_key, policy_seed=None)


def test_episode_record_rejects_non_json_or_nonfinite_parameters() -> None:
    with pytest.raises(ValueError, match="JSON"):
        make_episode(sampled_scenario_parameters={"path": Path("not-json")})
    with pytest.raises(ValueError, match="finite"):
        make_episode(sampled_scenario_parameters={"gap": nan})
