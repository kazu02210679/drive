import pytest

from mad_driving.config.models import CutInScenarioConfig
from mad_driving.scenarios import (
    CutInRuntime as ExportedCutInRuntime,
)
from mad_driving.scenarios import (
    EpisodeSeeds,
    LanePoseCommand,
    RoadGeometry,
    ScenarioActorState,
    ScenarioParameterSampler,
)
from mad_driving.scenarios.cut_in import CutInRuntime, smoothstep


@pytest.mark.parametrize(("progress", "expected"), [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_smoothstep_endpoints_and_midpoint(progress: float, expected: float) -> None:
    assert smoothstep(progress) == pytest.approx(expected)


def test_cut_in_runtime_is_part_of_the_scenario_public_api() -> None:
    assert ExportedCutInRuntime is CutInRuntime


class FakeEnvironment:
    def __init__(self, geometry: RoadGeometry | None = None) -> None:
        self.spawns: list[object] = []
        self.commands: list[tuple[str, LanePoseCommand]] = []
        self.actor_ids = {"cut-in"}
        self.collided_actor_ids: set[str] = set()
        self.geometry = geometry or RoadGeometry(
            (">", ">>", 0),
            10.0,
            0.0,
            10.0,
            0.1,
            adjacent_lane_indices=((">", ">>", 1),),
            lane_width_m=3.5,
        )

    def scenario_road_geometry(self) -> RoadGeometry:
        return self.geometry

    def scenario_spawn_lane_vehicle(self, spawn: object) -> str:
        self.spawns.append(spawn)
        return "cut-in"

    def scenario_command_actor(self, actor_id: str, command: LanePoseCommand) -> None:
        self.commands.append((actor_id, command))

    def scenario_actor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.actor_ids))

    def scenario_ego_collided_with(self, actor_id: str) -> bool:
        return actor_id in self.collided_actor_ids

    def scenario_actor_state(self, actor_id: str) -> ScenarioActorState:
        return ScenarioActorState(
            actor_id=actor_id,
            position_xy_m=(40.0, 0.0),
            velocity_xy_mps=(10.0, 0.0),
            acceleration_xy_mps2=(0.0, 0.0),
            heading_rad=0.0,
        )


def reset_cut_in(
    *, trigger_s: float, merge_s: float
) -> tuple[CutInRuntime, FakeEnvironment, object]:
    config = CutInScenarioConfig(
        initial_gap_m={"minimum": 20.0, "maximum": 20.0},
        trigger_s={"minimum": trigger_s, "maximum": trigger_s},
        merge_duration_s={"minimum": merge_s, "maximum": merge_s},
        speed_fraction={"minimum": 1.0, "maximum": 1.0},
    )
    runtime = CutInRuntime(config, ScenarioParameterSampler(7), difficulty_level=2)
    environment = FakeEnvironment()
    state = runtime.reset(environment, seeds=EpisodeSeeds(1, 2, 3))
    state = runtime.after_simulator_reset(environment, state)
    return runtime, environment, state


def test_cut_in_moves_from_adjacent_lane_to_ego_lane() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    runtime.before_step(environment, state, step_index=10)
    start = environment.commands[-1][1]
    runtime.before_step(environment, state, step_index=20)
    middle = environment.commands[-1][1]
    runtime.before_step(environment, state, step_index=30)
    end = environment.commands[-1][1]

    assert abs(start.lateral_m) > abs(middle.lateral_m) > abs(end.lateral_m)
    assert end.lateral_m == pytest.approx(0.0)
    assert start.longitudinal_m < middle.longitudinal_m < end.longitudinal_m


def test_cut_in_lane_pose_starts_at_completed_time_without_a_longitudinal_jump() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    runtime.before_step(environment, state, step_index=10)
    at_trigger = environment.commands[-1][1]
    runtime.before_step(environment, state, step_index=11)
    after_trigger = environment.commands[-1][1]
    runtime.before_step(environment, state, step_index=12)
    later = environment.commands[-1][1]

    initial_longitudinal_m = state.parameters["initial_longitudinal_m"]
    speed_mps = state.parameters["speed_mps"]
    decision_interval_s = state.parameters["decision_interval_s"]
    assert isinstance(initial_longitudinal_m, float)
    assert isinstance(speed_mps, float)
    assert isinstance(decision_interval_s, float)
    assert at_trigger.longitudinal_m == pytest.approx(
        initial_longitudinal_m + speed_mps * decision_interval_s * 9
    )
    assert after_trigger.longitudinal_m - at_trigger.longitudinal_m == pytest.approx(
        speed_mps * decision_interval_s
    )
    assert later.longitudinal_m - after_trigger.longitudinal_m == pytest.approx(
        speed_mps * decision_interval_s
    )


def test_cut_in_spawns_in_the_sampled_adjacent_lane() -> None:
    _, environment, _ = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    assert environment.spawns[-1].lane_index == (">", ">>", 1)


def test_cut_in_sampling_is_seeded_and_uses_sorted_adjacent_lanes() -> None:
    geometry = RoadGeometry(
        (">", ">>", 1),
        10.0,
        0.0,
        10.0,
        0.1,
        adjacent_lane_indices=((">", ">>", 0), (">", ">>", 2)),
        lane_width_m=3.5,
    )
    config = CutInScenarioConfig()

    first_runtime = CutInRuntime(config, ScenarioParameterSampler(17), difficulty_level=2)
    second_runtime = CutInRuntime(config, ScenarioParameterSampler(17), difficulty_level=2)
    first_environment = FakeEnvironment(geometry)
    second_environment = FakeEnvironment(geometry)
    first_state = first_runtime.after_simulator_reset(
        first_environment,
        first_runtime.reset(first_environment, seeds=EpisodeSeeds(1, 2, 3)),
    )
    second_state = second_runtime.after_simulator_reset(
        second_environment,
        second_runtime.reset(second_environment, seeds=EpisodeSeeds(1, 2, 3)),
    )

    assert first_state.parameters == second_state.parameters
    assert first_state.parameters["source_lane_index"] in geometry.adjacent_lane_indices


def test_cut_in_fails_fast_without_an_available_adjacent_lane() -> None:
    geometry = RoadGeometry((">", ">>", 0), 10.0, 0.0, 10.0, 0.1)
    environment = FakeEnvironment(geometry)
    runtime = CutInRuntime(CutInScenarioConfig(), ScenarioParameterSampler(7), difficulty_level=2)
    state = runtime.reset(environment, seeds=EpisodeSeeds(1, 2, 3))

    with pytest.raises(RuntimeError, match="available adjacent lane"):
        runtime.after_simulator_reset(environment, state)


def test_cut_in_succeeds_after_merge_and_survival_window() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    transition = runtime.after_step(
        environment, state, step_index=60, raw_info={"crash_vehicle": False}
    )

    assert transition.outcome.success is True
    assert transition.outcome.failure is False


def test_cut_in_does_not_fail_for_a_collision_with_another_vehicle() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    transition = runtime.after_step(
        environment, state, step_index=1, raw_info={"crash_vehicle": True}
    )

    assert transition.outcome.success is False
    assert transition.outcome.failure is False


def test_cut_in_does_not_succeed_at_the_boundary_after_another_vehicle_collision() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    transition = runtime.after_step(
        environment, state, step_index=60, raw_info={"crash_vehicle": True}
    )

    assert transition.outcome.success is False
    assert transition.outcome.failure is False


def test_cut_in_does_not_succeed_at_the_boundary_after_a_human_collision() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    transition = runtime.after_step(
        environment, state, step_index=60, raw_info={"crash_human": True}
    )

    assert transition.outcome.success is False
    assert transition.outcome.failure is False


def test_cut_in_does_not_succeed_at_the_boundary_when_off_road() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)

    transition = runtime.after_step(environment, state, step_index=60, raw_info={"off_road": True})

    assert transition.outcome.success is False
    assert transition.outcome.failure is False


def test_cut_in_fails_only_when_the_ego_contacts_its_actor() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)
    environment.collided_actor_ids.add("cut-in")

    transition = runtime.after_step(
        environment, state, step_index=1, raw_info={"crash_vehicle": True}
    )

    assert transition.outcome.success is False
    assert transition.outcome.failure is True


def test_cut_in_requires_the_typed_vehicle_collision_outcome() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)
    environment.collided_actor_ids.add("cut-in")

    transition = runtime.after_step(
        environment, state, step_index=1, raw_info={"crash_vehicle": False}
    )

    assert transition.outcome.failure is False
