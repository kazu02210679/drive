"""Behavioral tests for the seeded occluded crossing scenario."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mad_driving.config.models import OccludedCrossingScenarioConfig
from mad_driving.scenarios import (
    EpisodeSeeds,
    KinematicActorSpawn,
    LaneVehicleSpawn,
    RoadGeometry,
    ScenarioActorState,
    ScenarioParameterSampler,
    StaticOccluderSpawn,
    VelocityCommand,
)
from mad_driving.scenarios.occluded_crossing import OccludedCrossingRuntime
from mad_driving.scenarios.runtime import ScenarioStepResult


class FakeEnvironment:
    """Simulator-neutral environment with controllable crossing actor truth."""

    def __init__(self) -> None:
        self.geometry = RoadGeometry((">", ">>", 0), 10.0, 0.0, 10.0, 0.1, lane_width_m=4.0)
        self.spawns: list[object] = []
        self.commands: list[tuple[str, object]] = []
        self.actor_ids = {"crossing-cyclist", "static-occluder", "crossing-lead"}
        self.actor_states = {
            "crossing-cyclist": ScenarioActorState(
                "crossing-cyclist", (40.0, 8.0), (0.0, 0.0), (0.0, 0.0), 0.0
            )
        }
        self.collided_actor_ids: set[str] = set()
        self.visible_ids = {"ego", "traffic-1", *self.actor_ids}
        self.spawn_return_ids: dict[str, str] = {}

    def scenario_road_geometry(self) -> RoadGeometry:
        return self.geometry

    def scenario_lane_position(
        self, lane_index: tuple[str, str, int], longitudinal_m: float, lateral_m: float
    ) -> tuple[float, float]:
        assert lane_index == self.geometry.ego_lane_index
        return longitudinal_m, lateral_m

    def scenario_spawn_lane_vehicle(self, spawn: LaneVehicleSpawn) -> str:
        self.spawns.append(spawn)
        return self.spawn_return_ids.get(spawn.actor_id, spawn.actor_id)

    def scenario_spawn_crossing_actor(self, spawn: KinematicActorSpawn) -> str:
        self.spawns.append(spawn)
        self.actor_states[spawn.actor_id] = ScenarioActorState(
            spawn.actor_id,
            spawn.position_xy_m,
            spawn.velocity_xy_mps,
            (0.0, 0.0),
            spawn.heading_rad,
        )
        return self.spawn_return_ids.get(spawn.actor_id, spawn.actor_id)

    def scenario_spawn_occluder(self, spawn: StaticOccluderSpawn) -> str:
        self.spawns.append(spawn)
        return self.spawn_return_ids.get(spawn.actor_id, spawn.actor_id)

    def scenario_command_actor(self, actor_id: str, command: object) -> None:
        self.commands.append((actor_id, command))

    def scenario_actor_state(self, actor_id: str) -> ScenarioActorState:
        return self.actor_states[actor_id]

    def scenario_actor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.actor_ids))

    def scenario_visible_actor_ids(self) -> frozenset[str]:
        return frozenset(self.visible_ids)

    def scenario_ego_collided_with(self, actor_id: str) -> bool:
        return actor_id in self.collided_actor_ids


def crossing_config(
    *, trigger_s: float = 1.0, survival_s: float = 2.0
) -> OccludedCrossingScenarioConfig:
    return OccludedCrossingScenarioConfig(
        conflict_distance_m={"minimum": 30.0, "maximum": 30.0},
        crossing_start_offset_m={"minimum": 8.0, "maximum": 8.0},
        crossing_speed_mps={"minimum": 4.0, "maximum": 4.0},
        trigger_s={"minimum": trigger_s, "maximum": trigger_s},
        survival_s=survival_s,
        secondary_lead_gap_m={"minimum": 40.0, "maximum": 40.0},
        secondary_lead_speed_fraction={"minimum": 0.9, "maximum": 0.9},
        reveal_lateral_m=3.0,
    )


def reset_crossing(
    *, actor_lateral_m: float = 8.0, reveal_lateral_m: float = 3.0
) -> tuple[OccludedCrossingRuntime, FakeEnvironment, object]:
    runtime = OccludedCrossingRuntime(
        crossing_config(), ScenarioParameterSampler(7), difficulty_level=3
    )
    environment = FakeEnvironment()
    state = runtime.reset(environment, seeds=EpisodeSeeds(1, 2, 3))
    state = runtime.after_simulator_reset(environment, state)
    environment.actor_states["crossing-cyclist"] = replace(
        environment.actor_states["crossing-cyclist"],
        position_xy_m=(40.0, actor_lateral_m),
    )
    state = replace(state, parameters={**state.parameters, "reveal_lateral_m": reveal_lateral_m})
    return runtime, environment, state


def no_collision_info() -> dict[str, object]:
    return {
        "crash_vehicle": False,
        "crash_human": False,
        "crash_object": False,
        "crash_sidewalk": False,
        "crash_building": False,
    }


def test_crossing_actor_is_hidden_but_privileged_before_reveal() -> None:
    runtime, _environment, state = reset_crossing(actor_lateral_m=8.0, reveal_lateral_m=3.0)

    context = runtime.observation_context(state)

    assert "crossing-cyclist" not in context.visible_actor_ids
    assert "static-occluder" in context.visible_actor_ids
    assert "crossing-lead" in context.visible_actor_ids
    assert "traffic-1" in context.visible_actor_ids
    assert context.distance_to_conflict_point_m == pytest.approx(30.0)


def test_context_recomputes_conflict_distance_from_current_ego_lane_coordinates() -> None:
    runtime, environment, state = reset_crossing()
    environment.geometry = replace(environment.geometry, ego_longitudinal_m=17.5)

    context = runtime.observation_context(state)

    assert context.distance_to_conflict_point_m == pytest.approx(22.5)


def test_crossing_actor_becomes_visible_at_boundary() -> None:
    runtime, environment, state = reset_crossing(actor_lateral_m=2.9, reveal_lateral_m=3.0)

    transition = runtime.after_step(environment, state, step_index=20, raw_info=no_collision_info())
    context = runtime.observation_context(transition.state)

    assert "crossing-cyclist" in context.visible_actor_ids


def test_crossing_spawns_cyclist_occluder_and_secondary_lead_from_seeded_parameters() -> None:
    _runtime, environment, state = reset_crossing()

    cyclist, occluder, lead = environment.spawns

    assert isinstance(cyclist, KinematicActorSpawn)
    assert cyclist.actor_id == "crossing-cyclist"
    assert cyclist.actor_kind == "crossing_actor"
    assert isinstance(occluder, StaticOccluderSpawn)
    assert abs(occluder.position_xy_m[1]) >= environment.geometry.lane_width_m / 2.0 + 0.5
    assert isinstance(lead, LaneVehicleSpawn)
    assert lead.actor_id == "crossing-lead"
    assert lead.lane_index == environment.geometry.ego_lane_index
    assert lead.longitudinal_m == pytest.approx(50.0)
    assert lead.speed_mps == pytest.approx(9.0)
    assert state.parameters["conflict_distance_m"] == 30.0


def test_crossing_releases_cyclist_only_at_trigger() -> None:
    runtime, environment, state = reset_crossing()

    runtime.before_step(environment, state, step_index=9)
    runtime.before_step(environment, state, step_index=10)

    assert len(environment.commands) == 1
    assert environment.commands[0][0] == "crossing-cyclist"
    assert isinstance(environment.commands[-1][1], VelocityCommand)


def test_crossing_collision_is_failure() -> None:
    runtime, environment, state = reset_crossing()
    environment.collided_actor_ids.add("crossing-cyclist")

    transition = runtime.after_step(
        environment, state, step_index=20, raw_info={**no_collision_info(), "crash_human": True}
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=True)


def test_crossing_fails_fast_when_its_cyclist_disappears() -> None:
    runtime, environment, state = reset_crossing()
    environment.actor_ids.remove("crossing-cyclist")

    with pytest.raises(RuntimeError, match="missing scenario Actor: crossing-cyclist"):
        runtime.after_step(environment, state, step_index=1, raw_info=no_collision_info())


def test_crossing_rejects_an_unexpected_spawn_identity() -> None:
    runtime = OccludedCrossingRuntime(
        crossing_config(), ScenarioParameterSampler(7), difficulty_level=3
    )
    environment = FakeEnvironment()
    environment.spawn_return_ids["crossing-cyclist"] = "wrong-id"
    state = runtime.reset(environment, seeds=EpisodeSeeds(1, 2, 3))

    with pytest.raises(RuntimeError, match="unexpected ID"):
        runtime.after_simulator_reset(environment, state)


def test_unrelated_collision_suppresses_success_without_mislabelling_crossing_failure() -> None:
    runtime, environment, state = reset_crossing()
    cleared_state = replace(state, parameters={**state.parameters, "cleared_step": 1})

    transition = runtime.after_step(
        environment,
        cleared_state,
        step_index=21,
        raw_info={**no_collision_info(), "crash_vehicle": True},
    )

    assert transition.outcome == ScenarioStepResult(success=False, failure=False)


def test_crossing_succeeds_only_after_clearance_and_survival_window() -> None:
    runtime, environment, state = reset_crossing()
    environment.actor_states["crossing-cyclist"] = replace(
        environment.actor_states["crossing-cyclist"], position_xy_m=(40.0, -2.3)
    )

    cleared = runtime.after_step(environment, state, step_index=20, raw_info=no_collision_info())
    complete = runtime.after_step(
        environment, cleared.state, step_index=40, raw_info=no_collision_info()
    )

    assert cleared.outcome == ScenarioStepResult(success=False, failure=False)
    assert complete.outcome == ScenarioStepResult(success=True, failure=False)
