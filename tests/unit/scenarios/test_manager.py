from mad_driving.config.models import ScenarioSplitsConfig
from mad_driving.scenarios import EpisodeSeeds, ScenarioManagerRuntime


class FakeEnvironment:
    pass


def make_manager(level: int) -> ScenarioManagerRuntime:
    return ScenarioManagerRuntime(
        ScenarioSplitsConfig(curriculum={"mode": "fixed", "fixed_level": level})
    )


def runtime_state(seeds: EpisodeSeeds, *, level: int):
    return make_manager(level).reset(FakeEnvironment(), seeds=seeds)


def test_manager_uses_only_parameter_seed() -> None:
    first = runtime_state(EpisodeSeeds(1, 7, 99), level=2)
    second = runtime_state(EpisodeSeeds(500, 800, 99), level=2)

    assert first.scenario_id == second.scenario_id
    assert first.parameters == second.parameters


def test_pending_level_applies_only_on_next_reset() -> None:
    runtime = make_manager(level=0)
    state = runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(1, 2, 3))
    runtime.set_difficulty_level(2)

    assert state.parameters["difficulty_level"] == 0
    next_state = runtime.reset(FakeEnvironment(), seeds=EpisodeSeeds(4, 5, 6))
    assert next_state.parameters["difficulty_level"] == 2
