from collections.abc import Mapping

import pytest

from mad_driving.config.models import ScenarioSplitsConfig
from mad_driving.scenarios import (
    EpisodeSeeds,
    ScenarioManagerRuntime,
    ScenarioObservationContext,
    ScenarioState,
    ScenarioStepResult,
    ScenarioTransition,
)


class FakeEnvironment:
    pass


class FakeRuntime:
    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id

    def reset(self, environment: object, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        return ScenarioState(self.scenario_id, seeds, {"runtime": self.scenario_id})

    def after_simulator_reset(self, environment: object, state: ScenarioState) -> ScenarioState:
        del environment
        return state

    def before_step(
        self, environment: object, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        del environment, step_index
        return state

    def after_step(
        self,
        environment: object,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        del environment, step_index, raw_info
        return ScenarioTransition(state, ScenarioStepResult(False, False))

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        return ScenarioObservationContext(state.scenario_id)


def fake_runtimes() -> dict[str, object]:
    return {
        scenario_id: (lambda level, sampler, scenario_id=scenario_id: FakeRuntime(scenario_id))
        for scenario_id in ("nominal", "lead_brake", "cut_in", "occluded_crossing")
    }


def make_manager(
    level: int,
    *,
    selection: str | None = None,
    automatic: bool = False,
) -> ScenarioManagerRuntime:
    resolved_selection = selection or (
        "auto"
        if automatic
        else {
            0: "nominal",
            1: "lead_brake",
            2: "auto",
            3: "occluded_crossing",
        }[level]
    )
    curriculum = (
        {"mode": "automatic", "initial_level": level}
        if automatic
        else {"mode": "fixed", "fixed_level": level}
    )
    return ScenarioManagerRuntime(
        ScenarioSplitsConfig(
            selection=resolved_selection,
            curriculum=curriculum,
        ),
        runtimes=fake_runtimes(),
    )


def runtime_state(seeds: EpisodeSeeds, *, level: int):
    return make_manager(level).reset(FakeEnvironment(), seeds=seeds)


def test_manager_scenario_choice_uses_only_selection_seed() -> None:
    first = runtime_state(EpisodeSeeds(1, 7, 99, 101), level=2)
    second = runtime_state(EpisodeSeeds(500, 800, 99, 202), level=2)

    assert first.scenario_id == second.scenario_id


def test_pending_level_applies_only_on_next_reset() -> None:
    runtime = make_manager(level=0, automatic=True)
    state = runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(1, 2, 3, 4))
    runtime.set_difficulty_level(2)

    assert state.parameters["difficulty_level"] == 0
    next_state = runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(4, 5, 6, 7))
    assert next_state.parameters["difficulty_level"] == 2


def test_concrete_selection_uses_only_the_configured_allowed_scenario() -> None:
    runtime = make_manager(level=2, selection="lead_brake")

    state = runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(1, 2, 3, 4))

    assert state.scenario_id == "lead_brake"


def test_explicit_validation_schedule_balances_level_two_independent_of_seed() -> None:
    runtime = make_manager(level=2)
    runtime.set_scenario_schedule(("lead_brake", "cut_in", "lead_brake", "cut_in"))

    states = tuple(
        runtime.reset(
            FakeEnvironment(),
            seeds=EpisodeSeeds(index, index + 10, index + 20, index + 30),
        )
        for index in range(4)
    )

    assert tuple(state.scenario_id for state in states) == (
        "lead_brake",
        "cut_in",
        "lead_brake",
        "cut_in",
    )


def test_explicit_validation_schedule_rejects_scenario_outside_pending_level() -> None:
    runtime = make_manager(level=1)

    with pytest.raises(ValueError, match="schedule|difficulty level"):
        runtime.set_scenario_schedule(("cut_in",))


def test_default_registry_builds_cut_in_at_level_two() -> None:
    runtime = ScenarioManagerRuntime(
        ScenarioSplitsConfig(
            selection="cut_in",
            curriculum={"mode": "fixed", "fixed_level": 2},
        )
    )

    state = runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(1, 2, 3, 4))

    assert state.scenario_id == "cut_in"


def test_concrete_selection_rejects_a_scenario_outside_the_active_level() -> None:
    with pytest.raises(ValueError, match="match its level"):
        make_manager(level=0, selection="lead_brake")


def test_unregistered_selected_scenario_fails_fast() -> None:
    runtime = ScenarioManagerRuntime(
        ScenarioSplitsConfig(
            selection="occluded_crossing",
            curriculum={"mode": "fixed", "fixed_level": 3},
        ),
        runtimes={},
    )

    with pytest.raises(RuntimeError, match="no registered runtime"):
        runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(1, 2, 3, 4))
