import json
import math
from dataclasses import FrozenInstanceError, asdict, fields, replace
from typing import Any

import pytest

from mad_driving.interfaces.actor_state import ActorState
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.decision_trace import DecisionTrace
from mad_driving.interfaces.risk_claim import RiskClaim
from mad_driving.interfaces.scene_frame import (
    OcclusionRegion,
    PrivilegedWorldState,
    RoadContext,
    SceneFrame,
    stopping_margin_m,
)
from mad_driving.interfaces.scene_snapshot import EgoState, SceneObservation
from mad_driving.scenarios import EpisodeSeeds


def make_actor(**overrides: Any) -> ActorState:
    values: dict[str, Any] = {
        "actor_id": "vehicle-1",
        "actor_type": "vehicle",
        "position_xy_m": (10.0, 1.0),
        "velocity_xy_mps": (5.0, 0.0),
        "acceleration_xy_mps2": (0.0, 0.0),
        "heading_rad": 0.0,
        "length_m": 4.5,
        "width_m": 1.8,
        "relative_longitudinal_m": 10.0,
        "relative_lateral_m": 1.0,
        "same_lane": True,
        "visible": True,
        "occluded": False,
    }
    values.update(overrides)
    return ActorState(**values)


def make_ego(**overrides: Any) -> EgoState:
    values: dict[str, Any] = {
        "position_xy_m": (0.0, 0.0),
        "speed_mps": 8.0,
        "acceleration_mps2": 0.0,
        "heading_rad": 0.0,
        "lane_offset_m": 0.0,
        "route_progress": 0.25,
        "speed_limit_mps": 12.0,
    }
    values.update(overrides)
    return EgoState(**values)


def make_seeds(**overrides: Any) -> EpisodeSeeds:
    values: dict[str, Any] = {
        "episode_rng_seed": 42,
        "metadrive_scenario_index": 7,
        "scenario_selection_seed": 9,
        "scenario_parameter_seed": 11,
    }
    values.update(overrides)
    return EpisodeSeeds(**values)


def make_road_context(**overrides: Any) -> RoadContext:
    values: dict[str, Any] = {
        "stop_required": False,
        "distance_to_conflict_point_m": None,
        "intersection_entry_prohibited": False,
    }
    values.update(overrides)
    return RoadContext(**values)


def make_snapshot(**overrides: Any) -> SceneObservation:
    values: dict[str, Any] = {
        "step_index": 1,
        "sim_time_s": 0.1,
        "ego": make_ego(),
        "visible_actors": (make_actor(),),
        "occlusion_regions": (),
        "road_context": make_road_context(),
        "previous_executed_action": 0,
        "previous_shield_intervention": False,
    }
    values.update(overrides)
    return SceneObservation(**values)


def make_frame(**overrides: Any) -> SceneFrame:
    values: dict[str, Any] = {
        "scenario_id": "test",
        "seeds": make_seeds(),
        "observation": make_snapshot(),
        "privileged": PrivilegedWorldState(
            all_actors=(make_actor(),),
            collision_occurred=False,
            collision_kind=None,
            off_road=False,
            arrived=False,
            scenario_success=False,
            scenario_failure=False,
            minimum_actual_ttc_s=None,
            minimum_actual_stopping_margin_m=None,
            hard_rule_constraint=False,
        ),
    }
    values.update(overrides)
    return SceneFrame(**values)


def test_agent_visible_observation_excludes_scenario_identity_and_seeds() -> None:
    observation_fields = {field.name for field in fields(SceneObservation)}
    frame_fields = {field.name for field in fields(SceneFrame)}

    assert "scenario_id" not in observation_fields
    assert "seeds" not in observation_fields
    assert {"scenario_id", "seeds"} <= frame_fields


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_privileged_oracle_ttc_requires_a_non_negative_finite_value(value: float) -> None:
    with pytest.raises(ValueError, match="minimum_actual_ttc_s"):
        replace(make_frame().privileged, minimum_actual_ttc_s=value)


@pytest.mark.parametrize(
    (
        "ego_speed_mps",
        "minimum_actual_ttc_s",
        "reaction_delay_s",
        "safe_deceleration_mps2",
        "expected",
    ),
    [
        (8.0, None, 0.5, 6.0, None),
        (0.0, 2.0, 0.5, 6.0, 0.0),
        (8.0, 3.0, 0.5, 4.0, 12.0),
        (10.0, 1.0, 0.5, 5.0, -5.0),
    ],
)
def test_stopping_margin_oracle_uses_fixed_truth_formula(
    ego_speed_mps: float,
    minimum_actual_ttc_s: float | None,
    reaction_delay_s: float,
    safe_deceleration_mps2: float,
    expected: float | None,
) -> None:
    result = stopping_margin_m(
        ego_speed_mps=ego_speed_mps,
        minimum_actual_ttc_s=minimum_actual_ttc_s,
        reaction_delay_s=reaction_delay_s,
        safe_deceleration_mps2=safe_deceleration_mps2,
    )

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize("safe_deceleration_mps2", [0.0, -1.0])
def test_stopping_margin_oracle_requires_positive_safe_deceleration(
    safe_deceleration_mps2: float,
) -> None:
    with pytest.raises(ValueError, match="safe_deceleration_mps2"):
        stopping_margin_m(
            ego_speed_mps=8.0,
            minimum_actual_ttc_s=2.0,
            reaction_delay_s=0.5,
            safe_deceleration_mps2=safe_deceleration_mps2,
        )


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_privileged_stopping_margin_requires_a_finite_value(value: float) -> None:
    with pytest.raises(ValueError, match="minimum_actual_stopping_margin_m"):
        replace(make_frame().privileged, minimum_actual_stopping_margin_m=value)


def test_privileged_rule_constraint_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="hard_rule_constraint"):
        replace(make_frame().privileged, hard_rule_constraint=1)  # type: ignore[arg-type]


def make_claim(**overrides: Any) -> RiskClaim:
    values: dict[str, Any] = {
        "claim_id": "nominal-1",
        "agent_id": "nominal",
        "event_type": "lead_vehicle",
        "target_actor_id": "vehicle-1",
        "probability": 0.2,
        "confidence": 0.8,
        "severity": 0.3,
        "time_horizon_s": 5.0,
        "min_ttc_s": 4.0,
        "stopping_margin_m": 10.0,
        "recommended_max_speed_mps": 10.0,
        "hard_stop_required": False,
        "evidence": ("relative_position",),
        "assumptions": ("constant_velocity",),
        "valid_until_step": 2,
    }
    values.update(overrides)
    return RiskClaim(**values)


def make_review(**overrides: Any) -> CriticReview:
    values: dict[str, Any] = {
        "conflict_score": 0.1,
        "unresolved_conflict": False,
        "max_severity": 0.3,
        "supported_agent_ids": ("nominal",),
        "challenged_claim_ids": (),
        "reasons": (),
    }
    values.update(overrides)
    return CriticReview(**values)


def make_trace(**overrides: Any) -> DecisionTrace:
    values: dict[str, Any] = {
        "step_index": 1,
        "raw_action": 0,
        "required_action": 0,
        "executed_action": 0,
        "intervention_required": False,
        "target_speed_mps": 8.0,
        "shield_intervened": False,
        "shield_reasons": (),
        "claims": (make_claim(),),
        "review": make_review(),
        "reward_components": {"progress": 0.1},
        "expected_agent_ids": ("nominal", "hazard", "rule"),
        "analysis_latency_ms": 0.0,
        "shield_latency_ms": 0.0,
    }
    values.update(overrides)
    return DecisionTrace(**values)


def test_decision_trace_requires_complete_scenario_metadata() -> None:
    with pytest.raises(ValueError, match="scenario trace metadata must be complete"):
        replace(make_trace(), scenario_id="lead_brake", difficulty_level=None)
    with pytest.raises(ValueError, match="scenario trace metadata must be complete"):
        replace(make_trace(), scenario_id=None, difficulty_level=1)


@pytest.mark.parametrize(
    ("scenario_id", "difficulty_level", "message"),
    [
        ("", 1, "scenario_id"),
        (7, 1, "scenario_id"),
        ("lead_brake", True, "difficulty_level"),
        ("lead_brake", -1, "difficulty_level"),
        ("lead_brake", 4, "difficulty_level"),
    ],
)
def test_decision_trace_strictly_validates_complete_scenario_metadata(
    scenario_id: object,
    difficulty_level: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_trace(  # type: ignore[arg-type]
            scenario_id=scenario_id,
            difficulty_level=difficulty_level,
        )


def test_decision_trace_preserves_complete_phase5_episode_metadata() -> None:
    trace = make_trace(
        episode_rng_seed=42,
        metadrive_scenario_index=7,
        scenario_selection_seed=9,
        scenario_parameter_seed=11,
        role="train",
        worker_index=0,
        scenario_id="lead_brake",
        difficulty_level=1,
    )

    assert trace.scenario_id == "lead_brake"
    assert trace.difficulty_level == 1
    json.dumps(asdict(trace))


def test_decision_trace_freezes_expected_agents_and_validates_timing() -> None:
    trace = make_trace(
        expected_agent_ids=["nominal", "rule"],
        analysis_latency_ms=1.25,
        shield_latency_ms=0.5,
    )

    assert trace.expected_agent_ids == ("nominal", "rule")
    assert trace.analysis_latency_ms == 1.25
    assert trace.shield_latency_ms == 0.5

    for field, value in (
        ("analysis_latency_ms", -0.1),
        ("analysis_latency_ms", math.inf),
        ("shield_latency_ms", math.nan),
    ):
        with pytest.raises(ValueError, match=field):
            make_trace(**{field: value})


def test_models_are_frozen_and_json_serializable() -> None:
    frame = make_frame()
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        required_action=1,
        executed_action=1,
        intervention_required=True,
        target_speed_mps=7.0,
        shield_intervened=True,
        shield_reasons=("margin",),
        claims=(make_claim(),),
        review=make_review(),
        reward_components={"progress": 0.1},
        expected_agent_ids=("nominal", "hazard", "rule"),
        analysis_latency_ms=0.0,
        shield_latency_ms=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        frame.observation.step_index = 2  # type: ignore[misc]
    json.dumps(asdict(trace))


def test_scene_frame_keeps_privileged_labels_out_of_observation() -> None:
    frame = make_frame(
        privileged=PrivilegedWorldState(
            all_actors=(make_actor(),),
            collision_occurred=True,
            collision_kind="vehicle",
            off_road=True,
            arrived=True,
            scenario_success=True,
            scenario_failure=False,
            minimum_actual_ttc_s=1.5,
            minimum_actual_stopping_margin_m=-1.5,
            hard_rule_constraint=True,
        )
    )

    assert frame.privileged.collision_occurred is True
    assert frame.privileged.off_road is True
    assert frame.privileged.arrived is True
    assert frame.observation.road_context.intersection_entry_prohibited is False
    assert not hasattr(frame.observation, "collision_occurred")


def test_scene_observation_rejects_hidden_actor_kinematics() -> None:
    with pytest.raises(ValueError, match="visible_actors"):
        make_snapshot(visible_actors=(make_actor(visible=False),))


def test_scene_observation_defensively_freezes_and_validates_inputs() -> None:
    visible_actors = [make_actor()]
    occlusion_regions: list[OcclusionRegion] = []
    observation = make_snapshot(
        visible_actors=visible_actors,
        occlusion_regions=occlusion_regions,
    )
    visible_actors.clear()
    occlusion_regions.append(
        OcclusionRegion(
            region_id="building-corner",
            boundary_points_xy_m=((0.0, 0.0), (1.0, 0.0)),
        )
    )

    assert observation.visible_actors == (make_actor(),)
    assert observation.occlusion_regions == ()
    with pytest.raises(ValueError, match="seeds"):
        make_frame(seeds={"episode_rng_seed": 42})
    with pytest.raises(ValueError, match="road_context"):
        make_snapshot(road_context={"stop_required": False})


def test_decision_trace_copies_reward_components() -> None:
    components = {"progress": 0.1}
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        required_action=0,
        executed_action=0,
        intervention_required=False,
        target_speed_mps=8.0,
        shield_intervened=False,
        shield_reasons=(),
        claims=(make_claim(),),
        review=make_review(),
        reward_components=components,
        expected_agent_ids=("nominal", "hazard", "rule"),
        analysis_latency_ms=0.0,
        shield_latency_ms=0.0,
    )

    components["progress"] = 99.0
    assert trace.reward_components == {"progress": 0.1}
    with pytest.raises(TypeError):
        trace.reward_components["progress"] = 2.0  # type: ignore[index]


def test_decision_trace_preserves_monitor_mode_shield_requirement() -> None:
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        required_action=3,
        executed_action=0,
        intervention_required=True,
        target_speed_mps=8.0,
        shield_intervened=False,
        shield_reasons=("imminent_ttc",),
        claims=(make_claim(),),
        review=make_review(),
        reward_components={},
        expected_agent_ids=("nominal", "hazard", "rule"),
        analysis_latency_ms=0.0,
        shield_latency_ms=0.0,
    )

    assert trace.required_action == 3
    assert trace.intervention_required is True
    assert trace.shield_intervened is False


def test_decision_trace_rejects_inconsistent_shield_diagnostics() -> None:
    with pytest.raises(ValueError, match="intervention_required"):
        DecisionTrace(
            step_index=1,
            raw_action=0,
            required_action=3,
            executed_action=0,
            intervention_required=False,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(make_claim(),),
            review=make_review(),
            reward_components={},
            expected_agent_ids=("nominal", "hazard", "rule"),
            analysis_latency_ms=0.0,
            shield_latency_ms=0.0,
        )


def test_decision_trace_preserves_low_level_control_fail_safe() -> None:
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        required_action=0,
        executed_action=0,
        intervention_required=False,
        target_speed_mps=8.0,
        shield_intervened=False,
        shield_reasons=(),
        claims=(make_claim(),),
        review=make_review(),
        reward_components={},
        expected_agent_ids=("nominal", "hazard", "rule"),
        analysis_latency_ms=0.0,
        shield_latency_ms=0.0,
        control_fail_safe=True,
        control_fail_safe_reason="ValueError",
    )

    assert trace.control_fail_safe is True
    assert trace.control_fail_safe_reason == "ValueError"

    with pytest.raises(ValueError, match="control_fail_safe_reason"):
        DecisionTrace(
            step_index=1,
            raw_action=0,
            required_action=0,
            executed_action=0,
            intervention_required=False,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(make_claim(),),
            review=make_review(),
            reward_components={},
            expected_agent_ids=("nominal", "hazard", "rule"),
            analysis_latency_ms=0.0,
            shield_latency_ms=0.0,
            control_fail_safe=False,
            control_fail_safe_reason="ValueError",
        )


def test_decision_trace_freezes_and_validates_analysis_diagnostics() -> None:
    failed_agent_ids = ["hazard"]
    errors = ["hazard:RuntimeError:failed"]
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        required_action=0,
        executed_action=0,
        intervention_required=False,
        target_speed_mps=8.0,
        shield_intervened=False,
        shield_reasons=[],  # type: ignore[arg-type]
        claims=[make_claim()],  # type: ignore[arg-type]
        review=make_review(),
        reward_components={"progress": 0.1},
        expected_agent_ids=("nominal", "hazard", "rule"),
        analysis_latency_ms=0.0,
        shield_latency_ms=0.0,
        failed_agent_ids=failed_agent_ids,  # type: ignore[arg-type]
        errors=errors,  # type: ignore[arg-type]
    )

    failed_agent_ids.clear()
    errors.clear()
    assert trace.failed_agent_ids == ("hazard",)
    assert trace.errors == ("hazard:RuntimeError:failed",)
    assert trace.shield_reasons == ()
    assert isinstance(trace.claims, tuple)

    with pytest.raises(ValueError, match="one-to-one|match failed_agent_ids"):
        DecisionTrace(
            step_index=1,
            raw_action=0,
            required_action=0,
            executed_action=0,
            intervention_required=False,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(make_claim(),),
            review=make_review(),
            reward_components={},
            expected_agent_ids=("nominal", "hazard", "rule"),
            analysis_latency_ms=0.0,
            shield_latency_ms=0.0,
            failed_agent_ids=("hazard",),
            errors=("rule:RuntimeError:failed",),
        )


def test_decision_trace_deeply_freezes_claim_and_review_sequences() -> None:
    evidence = ["relative_position"]
    assumptions = ["constant_velocity"]
    supported_agent_ids = ["nominal"]
    challenged_claim_ids = ["hazard-1"]
    reasons = ["supported"]
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        required_action=0,
        executed_action=0,
        intervention_required=False,
        target_speed_mps=8.0,
        shield_intervened=False,
        shield_reasons=(),
        claims=(
            make_claim(
                evidence=evidence,
                assumptions=assumptions,
            ),
        ),
        review=make_review(
            supported_agent_ids=supported_agent_ids,
            challenged_claim_ids=challenged_claim_ids,
            reasons=reasons,
        ),
        reward_components={"progress": 0.1},
        expected_agent_ids=("nominal", "hazard", "rule"),
        analysis_latency_ms=0.0,
        shield_latency_ms=0.0,
    )

    evidence.append("caller_mutation")
    assumptions.clear()
    supported_agent_ids.clear()
    challenged_claim_ids.append("caller-mutation")
    reasons[0] = "caller_mutation"

    assert trace.claims[0].evidence == ("relative_position",)
    assert trace.claims[0].assumptions == ("constant_velocity",)
    assert trace.review.supported_agent_ids == ("nominal",)
    assert trace.review.challenged_claim_ids == ("hazard-1",)
    assert trace.review.reasons == ("supported",)
    json.dumps(asdict(trace))


@pytest.mark.parametrize(
    ("claim_overrides", "review_overrides", "message"),
    [
        ({"evidence": ["valid", 1]}, {}, "evidence"),
        ({}, {"reasons": "bare string"}, "reasons"),
    ],
)
def test_decision_trace_rejects_invalid_nested_string_sequences(
    claim_overrides: dict[str, Any],
    review_overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DecisionTrace(
            step_index=1,
            raw_action=0,
            required_action=0,
            executed_action=0,
            intervention_required=False,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(make_claim(**claim_overrides),),
            review=make_review(**review_overrides),
            reward_components={},
            expected_agent_ids=("nominal", "hazard", "rule"),
            analysis_latency_ms=0.0,
            shield_latency_ms=0.0,
        )


@pytest.mark.parametrize("actor_type", ["pedestrian", "", "VEHICLE"])
def test_actor_type_is_restricted(actor_type: str) -> None:
    with pytest.raises(ValueError, match="actor_type"):
        make_actor(actor_type=actor_type)


@pytest.mark.parametrize("field", ["length_m", "width_m"])
def test_actor_dimensions_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_actor(**{field: 0.0})


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (make_actor, "position_xy_m", (math.nan, 0.0)),
        (make_actor, "heading_rad", math.inf),
        (make_ego, "speed_mps", math.inf),
        (make_ego, "lane_offset_m", math.nan),
        (make_snapshot, "sim_time_s", math.inf),
        (make_road_context, "distance_to_conflict_point_m", math.nan),
        (make_claim, "time_horizon_s", math.inf),
        (make_claim, "min_ttc_s", math.nan),
        (make_review, "conflict_score", math.inf),
    ],
)
def test_non_finite_values_are_rejected(factory: Any, field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        factory(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("probability", -0.1),
        ("probability", 1.1),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("severity", -0.1),
        ("severity", 1.1),
        ("recommended_max_speed_mps", -0.1),
    ],
)
def test_risk_claim_ranges_are_validated(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_claim(**{field: value})


@pytest.mark.parametrize("field", ["conflict_score", "max_severity"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_critic_review_ranges_are_validated(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_review(**{field: value})


@pytest.mark.parametrize("route_progress", [-0.1, 1.1])
def test_route_progress_is_normalized(route_progress: float) -> None:
    with pytest.raises(ValueError, match="route_progress"):
        make_ego(route_progress=route_progress)


@pytest.mark.parametrize("previous_executed_action", [-1, 4])
def test_scene_observation_action_is_discrete_four(previous_executed_action: int) -> None:
    with pytest.raises(ValueError, match="previous_executed_action"):
        make_snapshot(previous_executed_action=previous_executed_action)


@pytest.mark.parametrize("heading_rad", [math.pi, 3.0 * math.pi])
def test_ego_heading_must_be_normalized_at_project_boundary(heading_rad: float) -> None:
    with pytest.raises(ValueError, match="heading_rad"):
        make_ego(heading_rad=heading_rad)


@pytest.mark.parametrize("field", ["raw_action", "executed_action"])
@pytest.mark.parametrize("value", [-1, 4])
def test_decision_trace_actions_are_discrete_four(field: str, value: int) -> None:
    values: dict[str, Any] = {
        "step_index": 1,
        "raw_action": 0,
        "required_action": 0,
        "executed_action": 0,
        "intervention_required": False,
        "target_speed_mps": 8.0,
        "shield_intervened": False,
        "shield_reasons": (),
        "claims": (make_claim(),),
        "review": make_review(),
        "reward_components": {"progress": 0.1},
        "expected_agent_ids": ("nominal", "hazard", "rule"),
        "analysis_latency_ms": 0.0,
        "shield_latency_ms": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        DecisionTrace(**values)


def test_reward_components_must_be_finite() -> None:
    with pytest.raises(ValueError, match="reward_components"):
        DecisionTrace(
            step_index=1,
            raw_action=0,
            required_action=0,
            executed_action=0,
            intervention_required=False,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(make_claim(),),
            review=make_review(),
            reward_components={"progress": math.nan},
            expected_agent_ids=("nominal", "hazard", "rule"),
            analysis_latency_ms=0.0,
            shield_latency_ms=0.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region_id", ""),
        ("boundary_points_xy_m", ((0.0, 0.0),)),
        ("boundary_points_xy_m", ((0.0, 0.0), (math.nan, 1.0))),
    ],
)
def test_occlusion_region_requires_a_valid_boundary(field: str, value: Any) -> None:
    values: dict[str, Any] = {
        "region_id": "building-corner",
        "boundary_points_xy_m": ((0.0, 0.0), (1.0, 0.0)),
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        OcclusionRegion(**values)


def test_road_context_requires_a_finite_optional_conflict_distance() -> None:
    with pytest.raises(ValueError, match="distance_to_conflict_point_m"):
        RoadContext(
            stop_required=False,
            distance_to_conflict_point_m=math.inf,
            intersection_entry_prohibited=False,
        )
