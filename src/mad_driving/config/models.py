"""Validated configuration models."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, model_validator


class StrictFrozenModel(BaseModel):
    """Base class that rejects unknown settings and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MetaDriveConfig(StrictFrozenModel):
    """MetaDrive options used by the Phase 1 headless environment."""

    use_render: bool = False
    image_observation: bool = False
    num_scenarios: PositiveInt = 1
    start_seed: int = 0
    traffic_density: FiniteFloat = Field(default=0.1, ge=0.0, le=1.0)
    horizon: PositiveInt = 200


class NominalAgentConfig(StrictFrozenModel):
    """Constant-acceleration prediction settings."""

    horizon_s: FiniteFloat = Field(default=5.0, gt=0.0)
    time_step_s: FiniteFloat = Field(default=0.25, gt=0.0)
    ego_length_m: FiniteFloat = Field(default=4.5, gt=0.0)
    ego_width_m: FiniteFloat = Field(default=1.8, gt=0.0)
    lane_half_width_m: FiniteFloat = Field(default=1.75, gt=0.0)
    longitudinal_buffer_m: FiniteFloat = Field(default=0.5, ge=0.0)
    lateral_buffer_m: FiniteFloat = Field(default=0.25, ge=0.0)
    probability_ttc_scale_s: FiniteFloat = Field(default=3.0, gt=0.0)
    probability_distance_scale_m: FiniteFloat = Field(default=3.0, gt=0.0)

    @model_validator(mode="after")
    def validate_time_step(self) -> Self:
        if self.time_step_s > self.horizon_s:
            raise ValueError("time_step_s must not exceed horizon_s")
        return self


class HazardAgentConfig(StrictFrozenModel):
    """Worst-case braking, crossing, and occlusion settings."""

    lead_max_deceleration_mps2: FiniteFloat = Field(default=-8.0, lt=0.0)
    crossing_actor_max_speed_mps: FiniteFloat = Field(default=8.0, gt=0.0)
    reaction_delay_s: FiniteFloat = Field(default=0.5, gt=0.0)
    ego_max_safe_deceleration_mps2: FiniteFloat = Field(default=-6.0, lt=0.0)
    ego_length_m: FiniteFloat = Field(default=4.5, gt=0.0)
    safety_buffer_m: FiniteFloat = Field(default=2.0, ge=0.0)
    occlusion_crawl_speed_mps: FiniteFloat = Field(default=2.0, gt=0.0)
    severe_margin_scale_m: FiniteFloat = Field(default=10.0, gt=0.0)
    crossing_occupancy_allowance_s: FiniteFloat = Field(default=1.0, gt=0.0)


class RuleAgentConfig(StrictFrozenModel):
    """Reserved strict configuration boundary for deterministic rules."""


class CriticAgentConfig(StrictFrozenModel):
    """Thresholds for the fixed one-pass cross review."""

    nominal_low_risk_threshold: FiniteFloat = Field(default=0.3, ge=0.0, le=1.0)
    recommendation_spread_mps: FiniteFloat = Field(default=5.0, ge=0.0)
    low_confidence_threshold: FiniteFloat = Field(default=0.5, ge=0.0, le=1.0)
    definitive_speed_fraction: FiniteFloat = Field(default=0.5, ge=0.0, le=1.0)


class AgentsConfig(StrictFrozenModel):
    """Configuration for every fixed Phase 2 agent."""

    nominal: NominalAgentConfig = Field(default_factory=NominalAgentConfig)
    hazard: HazardAgentConfig = Field(default_factory=HazardAgentConfig)
    rule: RuleAgentConfig = Field(default_factory=RuleAgentConfig)
    critic: CriticAgentConfig = Field(default_factory=CriticAgentConfig)


class AppConfig(StrictFrozenModel):
    """Root configuration for a deterministic Phase 1 smoke run."""

    seed: int = Field(ge=0)
    scenario_id: str = Field(min_length=1)
    decision_steps: PositiveInt
    fixed_action: tuple[FiniteFloat, FiniteFloat]
    metadrive: MetaDriveConfig
    agents: AgentsConfig = Field(default_factory=AgentsConfig)

    def metadrive_dict(self) -> dict[str, Any]:
        """Return a plain dictionary accepted by MetaDrive."""

        return self.metadrive.model_dump()
