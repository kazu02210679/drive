import math
from dataclasses import asdict
from typing import Any

import pytest

from mad_driving.world_model.snapshot_builder import SceneSnapshotBuilder


class FakeLane:
    def __init__(self, index: tuple[str, str, int], speed_limit: float = 36.0) -> None:
        self.index = index
        self.speed_limit = speed_limit

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
        }


class MetaDriveStyleConfig:
    """MetaDrive Config has get(), but does not implement Mapping."""

    def __init__(self) -> None:
        self.values = {"physics_world_step_size": 0.02, "decision_repeat": 5}

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


def build(env: FakeEnv):
    return SceneSnapshotBuilder().build(
        env,
        step_index=2,
        scenario_id="unit",
        seed=42,
        previous_action=1,
        previous_shield_intervention=False,
    )


def test_builds_si_ego_state_and_stable_actor_order() -> None:
    snapshot = build(make_env())

    assert snapshot.sim_time_s == pytest.approx(0.2)
    assert snapshot.ego.speed_mps == pytest.approx(10.0)
    assert snapshot.ego.acceleration_mps2 == pytest.approx(10.0)
    assert snapshot.ego.lane_offset_m == pytest.approx(0.5)
    assert snapshot.ego.route_progress == pytest.approx(0.25)
    assert snapshot.ego.speed_limit_mps == pytest.approx(10.0)
    assert [actor.actor_id for actor in snapshot.actors] == ["a-vehicle", "z-vehicle"]


def test_projects_relative_coordinates_and_lane_membership() -> None:
    snapshot = build(make_env())
    actors = {actor.actor_id: actor for actor in snapshot.actors}

    assert actors["z-vehicle"].relative_longitudinal_m == pytest.approx(12.0)
    assert actors["z-vehicle"].relative_lateral_m == pytest.approx(1.0)
    assert actors["z-vehicle"].same_lane is True
    assert actors["a-vehicle"].same_lane is False
    assert actors["a-vehicle"].acceleration_xy_mps2 == pytest.approx((5.0, 0.0))


def test_identical_runtime_state_produces_identical_snapshot() -> None:
    first = build(make_env())
    second = build(make_env())

    assert asdict(first) == asdict(second)


def test_missing_navigation_uses_safe_finite_defaults() -> None:
    ego = FakeVehicle(
        "ego",
        position=(0.0, 0.0),
        velocity=(2.0, 0.0),
        last_velocity=(2.0, 0.0),
        lane=None,
    )
    snapshot = build(FakeEnv(ego, []))

    assert snapshot.ego.lane_offset_m == 0.0
    assert snapshot.ego.route_progress == 0.0
    assert snapshot.ego.speed_limit_mps == 20.0


def test_non_finite_simulator_state_is_rejected() -> None:
    env = make_env()
    env.vehicle.velocity = (math.nan, 0.0)

    with pytest.raises(ValueError, match="speed_mps|velocity"):
        build(env)


def test_accepts_metadrive_config_object_with_get_method() -> None:
    env = make_env()
    env.config = MetaDriveStyleConfig()  # type: ignore[assignment]

    assert build(env).sim_time_s == pytest.approx(0.2)


def test_prefers_current_agent_api_without_accessing_deprecated_vehicle() -> None:
    base_env = make_env()

    class AgentApiEnv:
        agent = base_env.vehicle
        engine = base_env.engine
        config = base_env.config

        @property
        def vehicle(self) -> FakeVehicle:
            raise AssertionError("deprecated env.vehicle was accessed")

    assert build(AgentApiEnv()).ego.speed_mps == pytest.approx(10.0)  # type: ignore[arg-type]
