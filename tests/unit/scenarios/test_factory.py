from mad_driving.config.models import AppConfig
from mad_driving.scenarios import (
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
