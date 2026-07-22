from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from mad_driving.control import DrivingAction
from mad_driving.evaluation.policies import PpoPolicyAdapter, VisibleTtcRulePolicy
from mad_driving.interfaces import (
    ActorState,
    EgoState,
    OcclusionRegion,
    RoadContext,
    SceneObservation,
)
from mad_driving.methods import MethodProfileSnapshot


def actor(
    *,
    longitudinal_m: float,
    lateral_m: float = 0.0,
    velocity_xy_mps: tuple[float, float] = (0.0, 0.0),
    same_lane: bool = True,
    actor_type: str = "vehicle",
) -> ActorState:
    return ActorState(
        actor_id="visible",
        actor_type=actor_type,  # type: ignore[arg-type]
        position_xy_m=(longitudinal_m, lateral_m),
        velocity_xy_mps=velocity_xy_mps,
        acceleration_xy_mps2=(0.0, 0.0),
        heading_rad=0.0,
        length_m=4.0,
        width_m=2.0,
        relative_longitudinal_m=longitudinal_m,
        relative_lateral_m=lateral_m,
        same_lane=same_lane,
        visible=True,
        occluded=False,
    )


def scene(
    *,
    actors: tuple[ActorState, ...] = (),
    stop_required: bool = False,
    entry_prohibited: bool = False,
    occluded: bool = False,
) -> SceneObservation:
    regions = (OcclusionRegion("hidden-zone", ((4.0, -2.0), (4.0, 2.0))),) if occluded else ()
    return SceneObservation(
        step_index=0,
        sim_time_s=0.0,
        ego=EgoState((0.0, 0.0), 10.0, 0.0, 0.0, 0.0, 0.0, 15.0),
        visible_actors=actors,
        occlusion_regions=regions,
        road_context=RoadContext(stop_required, 20.0, entry_prohibited),
        previous_executed_action=int(DrivingAction.KEEP),
        previous_shield_intervention=False,
    )


@pytest.mark.parametrize(
    ("ttc_s", "expected"),
    [
        (1.0, DrivingAction.STOP),
        (3.0, DrivingAction.PREPARE_STOP),
        (5.0, DrivingAction.SLOW),
        (5.001, DrivingAction.KEEP),
    ],
)
def test_b0_uses_fixed_inclusive_visible_ttc_boundaries(
    ttc_s: float, expected: DrivingAction
) -> None:
    policy = VisibleTtcRulePolicy()
    observation = scene(actors=(actor(longitudinal_m=10.0 * ttc_s),))

    assert policy.predict(observation) == int(expected)
    assert policy.predict(observation) == int(expected)


@pytest.mark.parametrize("field", ["stop_required", "entry_prohibited"])
def test_b0_stops_for_visible_road_rule_requirements(field: str) -> None:
    assert VisibleTtcRulePolicy().predict(scene(**{field: True})) == int(DrivingAction.STOP)


def test_b0_considers_intersecting_trajectories_and_ignores_occlusion_without_visibility() -> None:
    crossing = actor(
        longitudinal_m=20.0,
        lateral_m=-20.0,
        velocity_xy_mps=(0.0, 10.0),
        same_lane=False,
        actor_type="crossing_actor",
    )

    assert VisibleTtcRulePolicy().predict(scene(actors=(crossing,))) == int(
        DrivingAction.PREPARE_STOP
    )
    assert VisibleTtcRulePolicy().predict(scene(occluded=True)) == int(DrivingAction.KEEP)


class FakeModel:
    def __init__(self, actions: tuple[object, ...] = (np.array([2]),)) -> None:
        self.actions = actions
        self.calls: list[tuple[np.ndarray, bool, object, object]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        state: object = None,
        episode_start: object = None,
    ) -> tuple[object, str]:
        self.calls.append((observation, deterministic, state, episode_start))
        action = self.actions[min(len(self.calls) - 1, len(self.actions) - 1)]
        return action, f"state-{len(self.calls)}"


def metadata() -> dict[str, Any]:
    resolved = {"method": {"id": "proposed"}, "seed": 42}
    return {
        "research_contract_version": 7,
        "observation_schema_version": 1,
        "observation_shape": (24,),
        "observation_dtype": "float32",
        "action_schema_version": 1,
        "action_count": 4,
        "action_order": ("KEEP", "SLOW", "PREPARE_STOP", "STOP"),
        "method_profile": MethodProfileSnapshot.from_method_id("proposed"),
        "resolved_config": resolved,
        "checkpoint_path": "runs/proposed/checkpoints/final.zip",
        "checkpoint_sha256": "a" * 64,
    }


def make_ppo_adapter(
    model: FakeModel | None = None,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
    checkpoint_path: str = "runs/proposed/checkpoints/final.zip",
    checkpoint_sha256: str = "a" * 64,
) -> PpoPolicyAdapter:
    values = metadata() if checkpoint_metadata is None else checkpoint_metadata
    return PpoPolicyAdapter(
        model or FakeModel(),
        method_id="proposed",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        resolved_config={"method": {"id": "proposed"}, "seed": 42},
        checkpoint_metadata=values,
    )


def test_ppo_prediction_is_deterministic_scalar_and_reset_clears_recurrent_state() -> None:
    model = FakeModel()
    adapter = make_ppo_adapter(model)
    observation = np.zeros(24, dtype=np.float32)

    assert adapter.predict(observation) == int(DrivingAction.PREPARE_STOP)
    adapter.reset()
    assert adapter.predict(observation) == int(DrivingAction.PREPARE_STOP)

    assert all(call[1] is True for call in model.calls)
    assert model.calls[0][2] is None
    assert model.calls[1][2] is None


@pytest.mark.parametrize("action", [np.array([4]), np.array([1, 2]), np.array([1.5])])
def test_ppo_rejects_non_scalar_or_out_of_range_actions(action: object) -> None:
    adapter = make_ppo_adapter(FakeModel((action,)))

    with pytest.raises(ValueError, match="action"):
        adapter.predict(np.zeros(24, dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method_profile", MethodProfileSnapshot.from_method_id("b1_nominal")),
        ("observation_schema_version", 2),
        ("observation_shape", (25,)),
        ("observation_dtype", "float64"),
        ("action_schema_version", 2),
        ("checkpoint_sha256", "b" * 64),
        ("checkpoint_path", "other.zip"),
        ("resolved_config", {"method": {"id": "b1_nominal"}, "seed": 42}),
    ],
)
def test_ppo_rejects_checkpoint_contract_mismatches(field: str, value: object) -> None:
    changed = deepcopy(metadata())
    changed[field] = value

    with pytest.raises(ValueError, match="metadata|checkpoint|config|profile|schema"):
        make_ppo_adapter(checkpoint_metadata=changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_path", ""),
        ("checkpoint_path", "   "),
        ("checkpoint_sha256", ""),
        ("checkpoint_sha256", "A" * 64),
        ("checkpoint_sha256", "g" * 64),
        ("checkpoint_sha256", "a" * 63),
    ],
)
def test_ppo_rejects_malformed_checkpoint_identity_before_prediction(
    field: str, value: str
) -> None:
    model = FakeModel()
    changed = metadata()
    changed[field] = value
    arguments = {field: value}

    with pytest.raises(ValueError, match="checkpoint"):
        make_ppo_adapter(model, checkpoint_metadata=changed, **arguments)

    assert model.calls == []
