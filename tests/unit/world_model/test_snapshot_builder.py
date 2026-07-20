import math
from dataclasses import asdict
from typing import Any

import pytest

from mad_driving.scenarios import (
    EpisodeSeeds,
    ScenarioObservationContext,
    ScenarioStepResult,
)
from mad_driving.world_model.snapshot_builder import SceneSnapshotBuilder
from mad_driving.world_model.validation import decision_interval_s


class FakeLane:
    def __init__(
        self,
        index: tuple[str, str, int],
        speed_limit: float = 36.0,
        width: float | None = 3.5,
    ) -> None:
        self.index = index
        self.speed_limit = speed_limit
        self.width = width

    def local_coordinates(self, position: tuple[float, float]) -> tuple[float, float]:
        return position[0], position[1]


class FakeNavigation:
    def __init__(self, lane: FakeLane, route_completion: float = 0.25) -> None:
        self.current_lane = lane
        self.route_completion = route_completion


class FakeVehicle:
    LENGTH = 4.5
    WIDTH = 1.8

    def __init__(
        self,
        name: str,
        *,
        position: tuple[float, float],
        velocity: tuple[float, float],
        last_velocity: tuple[float, float],
        heading_theta: float = 0.0,
        lane: FakeLane | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.name = name
        self.position = position
        self.velocity = velocity
        self.last_velocity = last_velocity
        self.heading_theta = heading_theta
        self.navigation = FakeNavigation(lane) if lane is not None else None
        self.lane = lane
        self.lane_index = lane.index if lane is not None else None
        self.max_speed_m_s = 20.0
        self.crash_vehicle = False
        self.crash_human = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False
        self.on_lane = True
        if random_seed is not None:
            self.random_seed = random_seed

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)


class FakeEngine:
    def __init__(self, objects: list[FakeVehicle]) -> None:
        self._objects = {obj.name: obj for obj in objects}

    def get_objects(self) -> dict[str, FakeVehicle]:
        return self._objects


class FakeEnv:
    def __init__(self, ego: FakeVehicle, actors: list[FakeVehicle]) -> None:
        self.vehicle = ego
        self.engine = FakeEngine([ego, *actors])
        self.config: dict[str, Any] = {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "map_config": {"lane_width": 3.5},
        }


class MetaDriveStyleConfig:
    """MetaDrive Config has get(), but does not implement Mapping."""

    def __init__(self) -> None:
        self.values = {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "map_config": {"lane_width": 3.5},
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def make_env() -> FakeEnv:
    ego_lane = FakeLane(("A", "B", 0))
    ego = FakeVehicle(
        "ego",
        position=(0.0, 0.5),
        velocity=(10.0, 0.0),
        last_velocity=(9.0, 0.0),
        lane=ego_lane,
    )
    actor_z = FakeVehicle(
        "z-vehicle",
        position=(12.0, 1.5),
        velocity=(5.0, 0.0),
        last_velocity=(5.0, 0.0),
        lane=ego_lane,
    )
    actor_a = FakeVehicle(
        "a-vehicle",
        position=(8.0, -2.0),
        velocity=(4.0, 0.0),
        last_velocity=(3.5, 0.0),
        lane=FakeLane(("A", "B", 1)),
    )
    return FakeEnv(ego, [actor_z, actor_a])


def build_frame(
    env: FakeEnv,
    *,
    visible_actor_ids: frozenset[str] | None = None,
    raw_info: dict[str, object] | None = None,
):
    return SceneSnapshotBuilder().build(
        env,
        step_index=2,
        seeds=EpisodeSeeds(
            episode_rng_seed=42,
            metadrive_scenario_index=7,
            scenario_parameter_seed=11,
        ),
        context=ScenarioObservationContext(
            scenario_id="unit",
            visible_actor_ids=visible_actor_ids,
        ),
        scenario_result=ScenarioStepResult(success=False, failure=False),
        raw_info={} if raw_info is None else raw_info,
        previous_executed_action=1,
        previous_shield_intervention=False,
    )


def test_builds_si_ego_state_and_stable_actor_order() -> None:
    observation = build_frame(make_env()).observation

    assert observation.sim_time_s == pytest.approx(0.2)
    assert observation.ego.speed_mps == pytest.approx(10.0)
    assert observation.ego.acceleration_mps2 == pytest.approx(10.0)
    assert observation.ego.lane_offset_m == pytest.approx(0.5)
    assert observation.ego.route_progress == pytest.approx(0.25)
    assert observation.ego.speed_limit_mps == pytest.approx(10.0)
    assert [actor.actor_id for actor in observation.visible_actors] == ["a-vehicle", "z-vehicle"]


def test_projects_relative_coordinates_and_lane_membership() -> None:
    observation = build_frame(make_env()).observation
    actors = {actor.actor_id: actor for actor in observation.visible_actors}

    assert actors["z-vehicle"].relative_longitudinal_m == pytest.approx(12.0)
    assert actors["z-vehicle"].relative_lateral_m == pytest.approx(1.0)
    assert actors["z-vehicle"].same_lane is True
    assert actors["a-vehicle"].same_lane is False
    assert actors["a-vehicle"].acceleration_xy_mps2 == pytest.approx((5.0, 0.0))


def test_identical_runtime_state_produces_identical_snapshot() -> None:
    first = build_frame(make_env())
    second = build_frame(make_env())

    assert asdict(first) == asdict(second)


def test_metadrive_random_seed_stabilizes_regenerated_actor_ids() -> None:
    lane = FakeLane(("A", "B", 0))
    first = make_env()
    second = make_env()
    first.engine = FakeEngine(
        [
            first.vehicle,
            FakeVehicle(
                "49d65b79-497d-43be-973e-3e72b4c4a2b9",
                position=(12.0, 0.0),
                velocity=(5.0, 0.0),
                last_velocity=(5.0, 0.0),
                lane=lane,
                random_seed=1234,
            ),
        ]
    )
    second.engine = FakeEngine(
        [
            second.vehicle,
            FakeVehicle(
                "a30aca0f-ca69-4e16-a5d8-1f6b109d8be4",
                position=(12.0, 0.0),
                velocity=(5.0, 0.0),
                last_velocity=(5.0, 0.0),
                lane=lane,
                random_seed=1234,
            ),
        ]
    )

    first_actor = build_frame(first).observation.visible_actors[0]
    second_actor = build_frame(second).observation.visible_actors[0]

    assert first_actor.actor_id == second_actor.actor_id == "metadrive-1234"


def test_missing_navigation_uses_safe_finite_defaults() -> None:
    ego = FakeVehicle(
        "ego",
        position=(0.0, 0.0),
        velocity=(2.0, 0.0),
        last_velocity=(2.0, 0.0),
        lane=None,
    )
    observation = build_frame(FakeEnv(ego, [])).observation

    assert observation.ego.lane_offset_m == 0.0
    assert observation.ego.route_progress == 0.0
    assert observation.ego.speed_limit_mps == 20.0


def test_non_finite_simulator_state_is_rejected() -> None:
    env = make_env()
    env.vehicle.velocity = (math.nan, 0.0)

    with pytest.raises(ValueError, match="speed_mps|velocity"):
        build_frame(env)


def test_accepts_metadrive_config_object_with_get_method() -> None:
    env = make_env()
    env.config = MetaDriveStyleConfig()  # type: ignore[assignment]

    assert build_frame(env).observation.sim_time_s == pytest.approx(0.2)


def test_decision_interval_requires_explicit_runtime_timing() -> None:
    with pytest.raises(ValueError, match="physics_world_step_size"):
        decision_interval_s({"decision_repeat": 5})

    with pytest.raises(ValueError, match="decision_repeat"):
        decision_interval_s({"physics_world_step_size": 0.02})


def test_prefers_current_agent_api_without_accessing_deprecated_vehicle() -> None:
    base_env = make_env()

    class AgentApiEnv:
        agent = base_env.vehicle
        engine = base_env.engine
        config = base_env.config

        @property
        def vehicle(self) -> FakeVehicle:
            raise AssertionError("deprecated env.vehicle was accessed")

    assert build_frame(AgentApiEnv()).observation.ego.speed_mps == pytest.approx(10.0)  # type: ignore[arg-type]


def test_builder_uses_scenario_context_only_for_observable_road_facts() -> None:
    frame = SceneSnapshotBuilder().build(
        make_env(),
        step_index=1,
        seeds=EpisodeSeeds(42, 7, 11),
        context=ScenarioObservationContext(
            scenario_id="scenario_flags",
            stop_required=True,
            distance_to_conflict_point_m=12.0,
            intersection_entry_prohibited=True,
        ),
        scenario_result=ScenarioStepResult(success=False, failure=False),
        raw_info={},
        previous_executed_action=0,
        previous_shield_intervention=False,
    )

    assert frame.observation.road_context.stop_required is True
    assert frame.observation.road_context.distance_to_conflict_point_m == 12.0
    assert frame.observation.road_context.intersection_entry_prohibited is True
    assert frame.privileged.collision_occurred is False
    assert frame.privileged.off_road is False


@pytest.mark.parametrize(
    "collision_flag",
    (
        "crash_vehicle",
        "crash_human",
        "crash_object",
        "crash_sidewalk",
        "crash_building",
    ),
)
def test_builder_maps_any_metadrive_collision_flag(collision_flag: str) -> None:
    env = make_env()
    setattr(env.vehicle, collision_flag, True)

    assert build_frame(env).privileged.collision_occurred is True


def test_builder_maps_explicit_outside_lane_state_only() -> None:
    env = make_env()
    env.vehicle.on_lane = False
    assert build_frame(env).privileged.off_road is True

    env.vehicle.on_lane = True
    assert build_frame(env).privileged.off_road is False

    del env.vehicle.on_lane
    assert build_frame(env).privileged.off_road is False


def test_hidden_actor_exists_only_in_privileged_state() -> None:
    frame = build_frame(
        make_env(),
        visible_actor_ids=frozenset({"z-vehicle"}),
    )

    assert tuple(actor.actor_id for actor in frame.observation.visible_actors) == (
        "z-vehicle",
    )
    assert tuple(actor.actor_id for actor in frame.privileged.all_actors) == (
        "a-vehicle",
        "z-vehicle",
    )
    assert not hasattr(frame.observation, "collision_occurred")


def test_privileged_outcomes_combine_raw_info_and_vehicle_state() -> None:
    frame = build_frame(
        make_env(),
        raw_info={"out_of_road": True, "arrive_dest": True},
    )

    assert frame.privileged.off_road is True
    assert frame.privileged.arrived is True


def test_raw_collision_info_sets_privileged_occurrence_and_kind() -> None:
    frame = build_frame(make_env(), raw_info={"crash_vehicle": True})

    assert frame.privileged.collision_occurred is True
    assert frame.privileged.collision_kind == "vehicle"


def test_builder_normalizes_heading_and_uses_ego_left_lateral_coordinates() -> None:
    env = make_env()
    env.vehicle.heading_theta = math.pi / 2.0
    env.engine.get_objects()["z-vehicle"].heading_theta = -3.0 * math.pi
    env.engine.get_objects()["z-vehicle"].position = (-1.0, 0.0)

    observation = build_frame(env).observation
    actor = next(actor for actor in observation.visible_actors if actor.actor_id == "z-vehicle")

    assert observation.ego.heading_rad == pytest.approx(math.pi / 2.0)
    assert actor.heading_rad == pytest.approx(-math.pi)
    assert actor.relative_lateral_m == pytest.approx(1.0)


def test_builder_normalizes_ego_heading_to_half_open_pi_interval() -> None:
    env = make_env()
    env.vehicle.heading_theta = 3.0 * math.pi

    assert build_frame(env).observation.ego.heading_rad == pytest.approx(-math.pi)


def test_same_lane_requires_lane_identity_and_lateral_position_inside_lane_width() -> None:
    env = make_env()
    env.engine.get_objects()["z-vehicle"].position = (12.0, 2.0)

    actors = {
        actor.actor_id: actor for actor in build_frame(env).observation.visible_actors
    }

    assert actors["z-vehicle"].same_lane is False
    assert actors["a-vehicle"].same_lane is False


def test_same_lane_uses_configured_nested_lane_width_when_lane_width_is_not_exposed() -> None:
    env = make_env()
    env.config["map_config"]["lane_width"] = 4.0
    env.vehicle.navigation.current_lane.width = None
    env.engine.get_objects()["z-vehicle"].position = (12.0, 1.9)

    actors = {
        actor.actor_id: actor for actor in build_frame(env).observation.visible_actors
    }

    assert actors["z-vehicle"].same_lane is True
