from __future__ import annotations

import pytest

from mad_driving.scenarios import (
    ActorCommand,
    LanePoseCommand,
    LaneVehicleSpawn,
    ScenarioActorManager,
)


class FakeActor:
    def __init__(self, actor_id: str) -> None:
        self.id = actor_id
        self.position = (40.0, 0.0)
        self.velocity = (8.0, 0.0)
        self.last_velocity = (8.0, 0.0)
        self.heading_theta = 0.0
        self.LENGTH = 4.5
        self.WIDTH = 1.8
        self.commands: list[ActorCommand] = []
        self.position_calls: list[tuple[float, float]] = []
        self.chassis = FakeChassis()

    def set_longitudinal_acceleration(self, command: ActorCommand) -> None:
        self.commands.append(command)

    def set_position(self, position: tuple[float, float]) -> None:
        self.position_calls.append(position)
        self.position = position


class FakeChassis:
    def __init__(self) -> None:
        self._node = object()

    def node(self) -> object:
        return self._node


class FakeContact:
    def __init__(self, node0: object, node1: object) -> None:
        self._node0 = node0
        self._node1 = node1

    def getNode0(self) -> object:
        return self._node0

    def getNode1(self) -> object:
        return self._node1


class FakeContactResult:
    def __init__(self, contacts: list[FakeContact]) -> None:
        self._contacts = contacts

    def getContacts(self) -> list[FakeContact]:
        return self._contacts


class FakeDynamicWorld:
    def __init__(self) -> None:
        self.contacts: list[FakeContact] = []
        self.queries: list[tuple[object, bool]] = []

    def contactTest(self, node: object, include_descendants: bool) -> FakeContactResult:
        self.queries.append((node, include_descendants))
        return FakeContactResult(self.contacts)


class FakePhysicsWorld:
    def __init__(self) -> None:
        self.dynamic_world = FakeDynamicWorld()


class FakeLane:
    def position(self, longitudinal_m: float, lateral_m: float) -> tuple[float, float]:
        return (longitudinal_m, lateral_m)


class FakeRoadNetwork:
    def get_lane(self, lane_index: tuple[str, str, int]) -> FakeLane:
        del lane_index
        return FakeLane()


class FakeMap:
    road_network = FakeRoadNetwork()


class FakeEngine:
    def __init__(self) -> None:
        self.objects: dict[str, FakeActor] = {}
        self.spawn_calls: list[dict[str, object]] = []
        self.global_config = {"physics_world_step_size": 0.02, "decision_repeat": 5}
        self.current_map = FakeMap()
        self.physics_world = FakePhysicsWorld()

    def spawn_object(self, object_class: object, **kwargs: object) -> FakeActor:
        del object_class
        self.spawn_calls.append(kwargs)
        actor_id = str(kwargs["name"])
        actor = FakeActor(actor_id)
        self.objects[actor_id] = actor
        return actor

    def clear_objects(self, actor_ids: list[str], **kwargs: object) -> list[str]:
        del kwargs
        for actor_id in actor_ids:
            self.objects.pop(actor_id, None)
        return []

    def get_objects(self, actor_ids: list[str] | None = None) -> dict[str, FakeActor]:
        if actor_ids is None:
            return dict(self.objects)
        return {
            actor_id: self.objects[actor_id] for actor_id in actor_ids if actor_id in self.objects
        }


def manager_with_fake_engine() -> ScenarioActorManager:
    return ScenarioActorManager(engine=FakeEngine())


def test_actor_manager_owns_and_clears_spawned_actor() -> None:
    manager = manager_with_fake_engine()
    actor_id = manager.spawn_lane_vehicle(LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0))

    assert actor_id == "lead"
    assert manager.actor_ids() == ("lead",)
    manager.before_reset()
    assert manager.actor_ids() == ()


def test_command_rejects_unknown_actor() -> None:
    manager = manager_with_fake_engine()

    with pytest.raises(KeyError, match="unknown scenario Actor"):
        manager.command_actor("missing", ActorCommand.longitudinal(-4.0))


def test_actor_manager_forwards_pending_command_before_step() -> None:
    manager = manager_with_fake_engine()
    manager.spawn_lane_vehicle(LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0))
    command = ActorCommand.longitudinal(-4.0)

    manager.command_actor("lead", command)
    manager.before_step()

    actor = manager.engine.get_objects(["lead"])["lead"]
    assert actor.commands == [command]


def test_actor_manager_applies_lane_pose_command_before_step() -> None:
    manager = manager_with_fake_engine()
    manager.spawn_lane_vehicle(LaneVehicleSpawn("cut-in", (">", ">>", 1), 40.0, 0.0, 8.0))

    manager.command_actor("cut-in", LanePoseCommand((">", ">>", 0), 42.0, 1.5))
    manager.before_step()

    actor = manager.engine.get_objects(["cut-in"])["cut-in"]
    assert actor.position_calls == [(42.0, 1.5)]


def test_lane_pose_state_derives_observed_kinematics_from_position_delta() -> None:
    manager = manager_with_fake_engine()
    manager.spawn_lane_vehicle(LaneVehicleSpawn("cut-in", (">", ">>", 1), 40.0, 0.0, 8.0))
    actor = manager.engine.get_objects(["cut-in"])["cut-in"]

    manager.command_actor("cut-in", LanePoseCommand((">", ">>", 0), 40.8, -0.5))
    manager.before_step()
    manager.after_step()

    first_state = manager.actor_state("cut-in")
    assert first_state.velocity_xy_mps == pytest.approx((8.0, -5.0))
    assert first_state.acceleration_xy_mps2 == pytest.approx((0.0, -50.0))
    assert actor.velocity == (8.0, 0.0)

    manager.command_actor("cut-in", LanePoseCommand((">", ">>", 0), 41.6, -1.2))
    manager.before_step()
    manager.after_step()

    second_state = manager.actor_state("cut-in")
    assert second_state.velocity_xy_mps == pytest.approx((8.0, -7.0))
    assert second_state.acceleration_xy_mps2 == pytest.approx((0.0, -20.0))
    assert actor.velocity == (8.0, 0.0)


def test_actor_manager_attributes_an_ego_contact_to_the_requested_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = manager_with_fake_engine()
    manager.spawn_lane_vehicle(LaneVehicleSpawn("cut-in", (">", ">>", 1), 40.0, 0.0, 8.0))
    actor = manager.engine.get_objects(["cut-in"])["cut-in"]
    ego = FakeActor("ego")
    manager.engine.physics_world.dynamic_world.contacts = [
        FakeContact(ego.chassis.node(), actor.chassis.node())
    ]
    monkeypatch.setattr(
        "mad_driving.scenarios.actor_manager.get_object_from_node",
        lambda node: actor if node is actor.chassis.node() else None,
        raising=False,
    )

    assert manager.ego_collided_with(ego, "cut-in") is True
    assert manager.engine.physics_world.dynamic_world.queries == [(ego.chassis.node(), True)]


def test_actor_manager_rejects_a_contact_with_a_different_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = manager_with_fake_engine()
    manager.spawn_lane_vehicle(LaneVehicleSpawn("cut-in", (">", ">>", 1), 40.0, 0.0, 8.0))
    ego = FakeActor("ego")
    other = FakeActor("traffic")
    manager.engine.physics_world.dynamic_world.contacts = [
        FakeContact(ego.chassis.node(), other.chassis.node())
    ]
    monkeypatch.setattr(
        "mad_driving.scenarios.actor_manager.get_object_from_node",
        lambda node: other if node is other.chassis.node() else None,
        raising=False,
    )

    assert manager.ego_collided_with(ego, "cut-in") is False


def test_actor_manager_rejects_duplicate_actor_ids() -> None:
    manager = manager_with_fake_engine()
    spawn = LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0)
    manager.spawn_lane_vehicle(spawn)

    with pytest.raises(ValueError, match="duplicate scenario Actor"):
        manager.spawn_lane_vehicle(spawn)


class VelocityOnlyActor:
    def __init__(self, actor_id: str) -> None:
        self.id = actor_id
        self.position = (40.0, 0.0)
        self.velocity = (8.0, 0.0)
        self.last_velocity = (8.0, 0.0)
        self.heading_theta = 0.0
        self.velocity_calls: list[tuple[tuple[float, float], float]] = []

    @property
    def speed(self) -> float:
        return self.velocity[0]

    def set_velocity(self, direction: tuple[float, float], speed: float) -> None:
        self.velocity_calls.append((direction, speed))
        self.velocity = (speed, 0.0)


class VelocityOnlyEngine(FakeEngine):
    def spawn_object(self, object_class: object, **kwargs: object) -> VelocityOnlyActor:
        del object_class
        actor_id = str(kwargs["name"])
        actor = VelocityOnlyActor(actor_id)
        self.objects[actor_id] = actor  # type: ignore[assignment]
        return actor


def test_fallback_command_converts_acceleration_using_decision_interval() -> None:
    engine = VelocityOnlyEngine()
    manager = ScenarioActorManager(engine=engine)
    manager.spawn_lane_vehicle(LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0))

    manager.command_actor("lead", ActorCommand.longitudinal(-4.0))
    manager.before_step()

    actor = engine.get_objects(["lead"])["lead"]
    assert actor.velocity_calls[-1] == ((1.0, 0.0), 7.6)


def test_actor_state_reports_acceleration_per_second() -> None:
    engine = VelocityOnlyEngine()
    manager = ScenarioActorManager(engine=engine)
    manager.spawn_lane_vehicle(LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0))
    actor = engine.get_objects(["lead"])["lead"]
    manager.before_step()
    actor.velocity = (7.6, 0.0)
    manager.after_step()

    state = manager.actor_state("lead")

    assert state.acceleration_xy_mps2 == pytest.approx((-4.0, 0.0))
