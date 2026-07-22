from __future__ import annotations

import pytest

from mad_driving.scenarios import ActorCommand, LaneVehicleSpawn, ScenarioActorManager


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

    def set_longitudinal_acceleration(self, command: ActorCommand) -> None:
        self.commands.append(command)


class FakeEngine:
    def __init__(self) -> None:
        self.objects: dict[str, FakeActor] = {}
        self.spawn_calls: list[dict[str, object]] = []
        self.global_config = {"physics_world_step_size": 0.02, "decision_repeat": 5}

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
            actor_id: self.objects[actor_id]
            for actor_id in actor_ids
            if actor_id in self.objects
        }


def manager_with_fake_engine() -> ScenarioActorManager:
    return ScenarioActorManager(engine=FakeEngine())


def test_actor_manager_owns_and_clears_spawned_actor() -> None:
    manager = manager_with_fake_engine()
    actor_id = manager.spawn_lane_vehicle(
        LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0)
    )

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
    actor.last_velocity = (8.0, 0.0)
    actor.velocity = (7.6, 0.0)

    state = manager.actor_state("lead")

    assert state.acceleration_xy_mps2 == pytest.approx((-4.0, 0.0))
