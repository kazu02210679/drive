"""Application-configured construction of scenario runtimes."""

from mad_driving.config.models import AppConfig
from mad_driving.scenarios.manager import ScenarioManagerRuntime
from mad_driving.scenarios.runtime import NoOpScenarioRuntime, ScenarioRuntime


class ScenarioRuntimeFactory:
    """Create Phase 5 manager runtimes while preserving earlier no-op scenarios."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._phase5_runtime = ScenarioManagerRuntime(config.scenarios)

    def __call__(self, scenario_id: str) -> ScenarioRuntime:
        if scenario_id == "phase5":
            return self._phase5_runtime
        return NoOpScenarioRuntime(scenario_id)

    def set_difficulty_level(self, level: int) -> None:
        """Queue a level for the shared Phase 5 runtime's next reset."""

        self._phase5_runtime.set_difficulty_level(level)


def build_scenario_runtime_factory(config: AppConfig) -> ScenarioRuntimeFactory:
    """Build the one stateful scenario factory owned by a Gym environment."""

    return ScenarioRuntimeFactory(config)
