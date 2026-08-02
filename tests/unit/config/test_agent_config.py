import math
from typing import Any

import pytest
from pydantic import ValidationError

from mad_driving.config.models import (
    AgentsConfig,
    AppConfig,
    CriticAgentConfig,
    HazardAgentConfig,
    NominalAgentConfig,
    RuleAgentConfig,
)


def base_payload() -> dict[str, Any]:
    return {
        "seed": 42,
        "scenario_id": "phase2_unit",
        "decision_steps": 2,
        "fixed_action": [0.0, 0.25],
        "metadrive": {},
    }


def test_agent_defaults_are_available_to_old_minimal_configs() -> None:
    config = AppConfig.model_validate(base_payload())

    assert config.agents.nominal.horizon_s == 5.0
    assert config.agents.nominal.time_step_s == 0.25
    assert config.agents.hazard.lead_max_deceleration_mps2 == -8.0
    assert config.agents.hazard.crossing_actor_max_speed_mps == 8.0
    assert config.agents.hazard.reaction_delay_s == 0.5
    assert config.agents.hazard.ego_max_safe_deceleration_mps2 == -6.0
    assert config.agents.critic.recommendation_spread_mps == 5.0
    assert config.agents.critic.low_confidence_threshold == 0.5


def test_agent_models_are_frozen_and_strict() -> None:
    config = AgentsConfig()

    with pytest.raises(ValidationError, match="frozen"):
        config.nominal.horizon_s = 4.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        NominalAgentConfig.model_validate({"unknown": 1})


@pytest.mark.parametrize(
    ("model", "values", "field"),
    [
        (NominalAgentConfig, {"horizon_s": 0.0}, "horizon_s"),
        (NominalAgentConfig, {"time_step_s": math.nan}, "time_step_s"),
        (NominalAgentConfig, {"horizon_s": 1.0, "time_step_s": 2.0}, "time_step_s"),
        (HazardAgentConfig, {"lead_max_deceleration_mps2": 0.0}, "lead_max"),
        (HazardAgentConfig, {"ego_max_safe_deceleration_mps2": 1.0}, "ego_max"),
        (HazardAgentConfig, {"crossing_actor_max_speed_mps": math.inf}, "crossing"),
        (CriticAgentConfig, {"low_confidence_threshold": 1.1}, "low_confidence"),
        (CriticAgentConfig, {"recommendation_spread_mps": -1.0}, "recommendation"),
    ],
)
def test_invalid_agent_config_is_rejected(
    model: type[NominalAgentConfig] | type[HazardAgentConfig] | type[CriticAgentConfig],
    values: dict[str, float],
    field: str,
) -> None:
    with pytest.raises(ValidationError, match=field):
        model.model_validate(values)


def test_explicit_agent_config_is_nested_under_app_config() -> None:
    payload = base_payload()
    payload["agents"] = {
        "nominal": {"horizon_s": 4.0, "time_step_s": 0.5},
        "hazard": {"occlusion_crawl_speed_mps": 1.5},
        "rule": {},
        "critic": {"recommendation_spread_mps": 4.0},
    }

    config = AppConfig.model_validate(payload)

    assert config.agents.nominal.horizon_s == 4.0
    assert config.agents.hazard.occlusion_crawl_speed_mps == 1.5
    assert isinstance(config.agents.rule, RuleAgentConfig)
    assert config.agents.critic.recommendation_spread_mps == 4.0
