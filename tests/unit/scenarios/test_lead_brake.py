from __future__ import annotations

import pytest

from mad_driving.config.models import LeadBrakeScenarioConfig
from mad_driving.scenarios import (
    ActorCommand,
    EpisodeSeeds,
    LeadBrakeRuntime,
    NominalScenarioRuntime,
    RoadGeometry,
    ScenarioActorState,
    ScenarioParameterSampler,
    ScenarioStepResult,
)


class FakeEnvironment:
    def __init__(self) -> None:
        self.spawns: list[object] = []
        self.commands: list[tuple[str, ActorCommand]] = []
        self.actor_ids = {"lead-brake"}
        self.collided_actor_ids: set[str] = set()
        self.lead_speed_mps = 8.0

    def scenario_road_geometry(self) -> RoadGeometry:
        return RoadGeometry((">", ">>", 0), 10.0, 0.0, 10.0, 0.1)

    def scenario_spawn_lane_vehicle(self, spawn: object) -> str:
        self.spawns.append(spawn)
        return "lead-brake"

    def scenario_command_actor(self, actor_id: str, command: ActorCommand) -> None:
        if actor_id not in self.actor_ids:
            raise RuntimeError(f"missing scenario Actor: {actor_id}")
        self.commands.append((actor_id, command))

    def scenario_actor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.actor_ids))

    def scenario_ego_collided_with(self, actor_id: str) -> bool:
        return actor_id in self.collided_actor_ids

    def scenario_actor_state(self, actor_id: str) -> ScenarioActorState:
        if actor_id not in self.actor_ids:
            raise RuntimeError(f"missing scenario Actor: {actor_id}")
        return ScenarioActorState(
            actor_id=actor_id,
            position_xy_m=(40.0, 0.0),
            velocity_xy_mps=(self.lead_speed_mps, 0.0),
            acceleration_xy_mps2=(0.0, 0.0),
            heading_rad=0.0,
        )

    def remove(self, actor_id: str) -> None:
        self.actor_ids.remove(actor_id)


def no_collision_info() -> dict[str, object]:
    return {"crash_vehicle": False}


def reset_lead_brake(
    *, trigger_s: float = 1.0, deceleration_mps2: float = 4.0, survival_s: float = 4.0
) -> tuple[LeadBrakeRuntime, FakeEnvironment, object]:
    config = LeadBrakeScenarioConfig(
        trigger_s={"minimum": trigger_s, "maximum": trigger_s},
        mild_deceleration_mps2={"minimum": deceleration_mps2, "maximum": deceleration_mps2},
        survival_s=survival_s,
    )
    runtime = LeadBrakeRuntime(config, ScenarioParameterSampler(7), difficulty_level=1)
    environment = FakeEnvironment()
    state = runtime.reset(environment, seeds=EpisodeSeeds(1, 2, 3))
    state = runtime.after_simulator_reset(environment, state)
    return runtime, environment, state


def test_lead_brake_spawns_ahead_in_the_ego_lane() -> None:
    _, environment, state = reset_lead_brake()

    spawn = environment.spawns[-1]
    assert spawn.lane_index == (">", ">>", 0)
    assert spawn.longitudinal_m == 10.0 + state.parameters["initial_gap_m"]


def test_lead_brake_triggers_sampled_deceleration() -> None:
    runtime, environment, state = reset_lead_brake(trigger_s=1.0, deceleration_mps2=4.0)
    runtime.before_step(environment, state, step_index=10)

    assert environment.commands[-1] == ("lead-brake", ActorCommand.longitudinal(-4.0))


def test_lead_brake_sustains_deceleration_until_the_lead_stops() -> None:
    runtime, environment, state = reset_lead_brake(trigger_s=1.0, deceleration_mps2=4.0)

    runtime.before_step(environment, state, step_index=10)
    runtime.before_step(environment, state, step_index=11)
    environment.lead_speed_mps = 0.0
    runtime.before_step(environment, state, step_index=12)

    assert environment.commands == [
        ("lead-brake", ActorCommand.longitudinal(-4.0)),
        ("lead-brake", ActorCommand.longitudinal(-4.0)),
    ]


def test_lead_brake_succeeds_after_survival_window() -> None:
    runtime, environment, state = reset_lead_brake(trigger_s=1.0, survival_s=4.0)
    transition = runtime.after_step(
        environment, state, step_index=50, raw_info=no_collision_info()
    )

    assert transition.outcome == ScenarioStepResult(success=True, failure=False)


def test_lead_brake_does_not_succeed_when_ego_is_off_road() -> None:
    runtime, environment, state = reset_lead_brake(trigger_s=1.0, survival_s=4.0)

    transition = runtime.after_step(
        environment,
        state,
        step_index=50,
        raw_info={"crash_vehicle": False, "out_of_road": True},
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=False)


def test_nominal_does_not_succeed_when_ego_is_off_road() -> None:
    environment = FakeEnvironment()
    runtime = NominalScenarioRuntime(survival_s=4.0)
    state = runtime.reset(environment, seeds=EpisodeSeeds(1, 2, 3))
    state = runtime.after_simulator_reset(environment, state)

    transition = runtime.after_step(
        environment,
        state,
        step_index=40,
        raw_info={"crash_vehicle": False, "off_road": True},
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=False)


def test_lead_brake_does_not_fail_for_a_collision_with_another_vehicle() -> None:
    runtime, environment, state = reset_lead_brake()

    transition = runtime.after_step(
        environment,
        state,
        step_index=1,
        raw_info={"crash_vehicle": True},
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=False)


def test_lead_brake_fails_only_when_the_ego_contacts_its_actor() -> None:
    runtime, environment, state = reset_lead_brake()
    environment.collided_actor_ids.add("lead-brake")

    transition = runtime.after_step(
        environment,
        state,
        step_index=1,
        raw_info={"crash_vehicle": True},
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=True)


def test_lead_brake_requires_the_typed_vehicle_collision_outcome() -> None:
    runtime, environment, state = reset_lead_brake()
    environment.collided_actor_ids.add("lead-brake")

    transition = runtime.after_step(
        environment,
        state,
        step_index=1,
        raw_info={"crash_vehicle": False},
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=False)


def test_missing_spawned_actor_is_internal_error() -> None:
    runtime, environment, state = reset_lead_brake()
    environment.remove("lead-brake")

    with pytest.raises(RuntimeError, match="missing scenario Actor"):
        runtime.before_step(environment, state, step_index=1)
