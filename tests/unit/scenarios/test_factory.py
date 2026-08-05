from mad_driving.config.models import AppConfig
from mad_driving.scenarios import (
    EpisodeSeeds,
    NoOpScenarioRuntime,
    ScenarioManagerRuntime,
    build_scenario_runtime_factory,
)


def make_config(scenario_id: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": scenario_id,
            "decision_steps": 8,
            "fixed_action": [0.0, 0.25],
            "metadrive": {},
        }
    )


def test_factory_uses_phase5_manager_only_for_phase5() -> None:
    phase5_factory = build_scenario_runtime_factory(make_config("phase5"))
    phase4_factory = build_scenario_runtime_factory(make_config("phase4"))

    assert isinstance(phase5_factory("phase5"), ScenarioManagerRuntime)
    assert isinstance(phase4_factory("phase4"), NoOpScenarioRuntime)


def test_phase5_factory_builds_the_occluded_crossing_runtime() -> None:
    payload = make_config("phase5").model_dump(mode="python")
    payload["scenarios"] = {
        "selection": "occluded_crossing",
        "curriculum": {"mode": "fixed", "fixed_level": 3},
    }
    config = AppConfig.model_validate(payload)

    runtime = build_scenario_runtime_factory(config)("phase5")

    state = runtime.reset(object(), seeds=EpisodeSeeds(1, 2, 3, 4))

    assert state.scenario_id == "occluded_crossing"


def test_evaluation_schedule_uses_phase5_manager_for_next_legacy_reset() -> None:
    factory = build_scenario_runtime_factory(make_config("phase1_smoke"))

    factory.set_difficulty_level(1)
    factory.set_scenario_schedule(("lead_brake",))

    assert isinstance(factory("phase1_smoke"), ScenarioManagerRuntime)
    assert isinstance(factory("phase1_smoke"), NoOpScenarioRuntime)
