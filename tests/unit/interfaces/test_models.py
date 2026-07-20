import json
import math
from dataclasses import FrozenInstanceError, asdict
from typing import Any

import pytest

from mad_driving.interfaces.actor_state import ActorState
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.decision_trace import DecisionTrace
from mad_driving.interfaces.risk_claim import RiskClaim
from mad_driving.interfaces.scene_frame import OcclusionRegion, RoadContext
from mad_driving.interfaces.scene_snapshot import EgoState, SceneSnapshot


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


def make_snapshot(**overrides: Any) -> SceneSnapshot:
    values: dict[str, Any] = {
        "step_index": 1,
        "sim_time_s": 0.1,
        "scenario_id": "test",
        "seed": 42,
        "ego": make_ego(),
        "actors": (make_actor(),),
        "stop_required": False,
        "occlusion_present": False,
        "distance_to_conflict_point_m": None,
        "previous_action": 0,
        "previous_shield_intervention": False,
        "collision_occurred": False,
        "off_road": False,
        "intersection_entry_prohibited": False,
    }
    values.update(overrides)
    return SceneSnapshot(**values)


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


def test_models_are_frozen_and_json_serializable() -> None:
    snapshot = make_snapshot()
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        executed_action=1,
        target_speed_mps=7.0,
        shield_intervened=True,
        shield_reasons=("margin",),
        claims=(make_claim(),),
        review=make_review(),
        reward_components={"progress": 0.1},
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.step_index = 2  # type: ignore[misc]
    json.dumps(asdict(trace))


def test_scene_snapshot_contains_explicit_rule_state() -> None:
    snapshot = make_snapshot(
        collision_occurred=True,
        off_road=True,
        intersection_entry_prohibited=True,
    )

    assert snapshot.collision_occurred is True
    assert snapshot.off_road is True
    assert snapshot.intersection_entry_prohibited is True


def test_decision_trace_copies_reward_components() -> None:
    components = {"progress": 0.1}
    trace = DecisionTrace(
        step_index=1,
        raw_action=0,
        executed_action=0,
        target_speed_mps=8.0,
        shield_intervened=False,
        shield_reasons=(),
        claims=(make_claim(),),
        review=make_review(),
        reward_components=components,
    )

    components["progress"] = 99.0
    assert trace.reward_components == {"progress": 0.1}


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
        (make_snapshot, "distance_to_conflict_point_m", math.nan),
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


@pytest.mark.parametrize("previous_action", [-1, 4])
def test_snapshot_action_is_discrete_four(previous_action: int) -> None:
    with pytest.raises(ValueError, match="previous_action"):
        make_snapshot(previous_action=previous_action)


@pytest.mark.parametrize("field", ["raw_action", "executed_action"])
@pytest.mark.parametrize("value", [-1, 4])
def test_decision_trace_actions_are_discrete_four(field: str, value: int) -> None:
    values: dict[str, Any] = {
        "step_index": 1,
        "raw_action": 0,
        "executed_action": 0,
        "target_speed_mps": 8.0,
        "shield_intervened": False,
        "shield_reasons": (),
        "claims": (make_claim(),),
        "review": make_review(),
        "reward_components": {"progress": 0.1},
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        DecisionTrace(**values)


def test_reward_components_must_be_finite() -> None:
    with pytest.raises(ValueError, match="reward_components"):
        DecisionTrace(
            step_index=1,
            raw_action=0,
            executed_action=0,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(make_claim(),),
            review=make_review(),
            reward_components={"progress": math.nan},
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
