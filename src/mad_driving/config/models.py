"""Validated configuration models."""

from math import isclose
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, model_validator


class StrictFrozenModel(BaseModel):
    """Base class that rejects unknown settings and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StrictTypedFrozenModel(StrictFrozenModel):
    """Frozen model that also rejects Pydantic scalar type coercion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SeedRangeConfig(StrictTypedFrozenModel):
    """One non-empty half-open range of scenario seeds."""

    seed_start: int = Field(ge=0)
    seed_count: PositiveInt

    @property
    def range(self) -> range:
        """Return the configured half-open seed range."""

        return range(self.seed_start, self.seed_start + self.seed_count)


class ScenarioSplitsConfig(StrictTypedFrozenModel):
    """Disjoint scenario seed ranges for training and evaluation roles."""

    train: SeedRangeConfig = SeedRangeConfig(seed_start=0, seed_count=10_000)
    validation: SeedRangeConfig = SeedRangeConfig(seed_start=10_000, seed_count=1_000)
    test: SeedRangeConfig = SeedRangeConfig(seed_start=20_000, seed_count=1_000)

    @model_validator(mode="after")
    def validate_disjoint_ranges(self) -> Self:
        ranges = (self.train.range, self.validation.range, self.test.range)
        if any(
            first.start < second.stop and second.start < first.stop
            for index, first in enumerate(ranges)
            for second in ranges[index + 1 :]
        ):
            raise ValueError("scenario seed ranges must not overlap")
        return self


class MetaDriveConfig(StrictFrozenModel):
    """MetaDrive options used by the Phase 1 headless environment."""

    use_render: bool = False
    image_observation: bool = False
    num_scenarios: PositiveInt = 1
    start_seed: int = 0
    traffic_density: FiniteFloat = Field(default=0.1, ge=0.0, le=1.0)
    horizon: PositiveInt = 200
    physics_dt_s: FiniteFloat = Field(default=0.02, gt=0.0)
    decision_repeat: PositiveInt = 5
    decision_dt_s: FiniteFloat = Field(default=0.10, gt=0.0)
    lane_width_m: FiniteFloat = Field(default=3.5, gt=0.0)

    @model_validator(mode="after")
    def validate_decision_timing(self) -> Self:
        if not isclose(self.decision_dt_s, self.physics_dt_s * self.decision_repeat):
            raise ValueError("decision_dt_s must equal physics_dt_s * decision_repeat")
        return self


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


class CoordinatorConfig(StrictTypedFrozenModel):
    """Thresholds for the deterministic Phase 3 baseline Coordinator."""

    conflict_min_action: int = Field(default=1, ge=0, le=3)
    severe_min_action: int = Field(default=2, ge=0, le=3)
    severe_threshold: FiniteFloat = Field(default=0.75, ge=0.0, le=1.0)


class ShieldConfig(StrictTypedFrozenModel):
    """Modes and physical thresholds for the deterministic Safety Shield."""

    mode: Literal["off", "monitor", "enforce"] = "enforce"
    imminent_ttc_s: FiniteFloat = Field(default=1.0, gt=0.0)
    caution_ttc_s: FiniteFloat = Field(default=3.0, gt=0.0)
    emergency_margin_m: FiniteFloat = 0.0
    caution_margin_m: FiniteFloat = 5.0
    missing_agent_action: int = Field(default=2, ge=0, le=3)
    multiple_missing_action: int = Field(default=3, ge=0, le=3)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        if self.imminent_ttc_s > self.caution_ttc_s:
            raise ValueError("imminent_ttc_s must not exceed caution_ttc_s")
        if self.emergency_margin_m > self.caution_margin_m:
            raise ValueError("emergency_margin_m must not exceed caution_margin_m")
        if self.multiple_missing_action < self.missing_agent_action:
            raise ValueError("multiple_missing_action must not be less than missing_agent_action")
        return self


class SpeedPIDConfig(StrictTypedFrozenModel):
    """Longitudinal PID gains and acceleration command limits."""

    kp: FiniteFloat = Field(default=0.50, ge=0.0)
    ki: FiniteFloat = Field(default=0.05, ge=0.0)
    kd: FiniteFloat = Field(default=0.10, ge=0.0)
    integral_limit: FiniteFloat = Field(default=10.0, gt=0.0)
    max_acceleration_mps2: FiniteFloat = Field(default=2.5, gt=0.0)
    normal_deceleration_mps2: FiniteFloat = Field(default=-3.0, lt=0.0)
    emergency_deceleration_mps2: FiniteFloat = Field(default=-6.0, lt=0.0)

    @model_validator(mode="after")
    def validate_deceleration_order(self) -> Self:
        if self.emergency_deceleration_mps2 > self.normal_deceleration_mps2:
            raise ValueError("emergency_deceleration_mps2 must not exceed normal_deceleration_mps2")
        return self


class SteeringPIDConfig(StrictTypedFrozenModel):
    """Heading and lateral lane-centering PID settings."""

    heading_kp: FiniteFloat = Field(default=1.7, ge=0.0)
    heading_ki: FiniteFloat = Field(default=0.01, ge=0.0)
    heading_kd: FiniteFloat = Field(default=3.5, ge=0.0)
    lateral_kp: FiniteFloat = Field(default=0.3, ge=0.0)
    lateral_ki: FiniteFloat = Field(default=0.002, ge=0.0)
    lateral_kd: FiniteFloat = Field(default=0.05, ge=0.0)
    integral_limit: FiniteFloat = Field(default=5.0, gt=0.0)
    lookahead_m: FiniteFloat = Field(default=1.0, gt=0.0)


class ControlConfig(StrictTypedFrozenModel):
    """Complete low-level lane and speed control configuration."""

    speed: SpeedPIDConfig = Field(default_factory=SpeedPIDConfig)
    steering: SteeringPIDConfig = Field(default_factory=SteeringPIDConfig)


class ObservationConfig(StrictTypedFrozenModel):
    """Normalization limits for the fixed coordinator observation."""

    max_speed_mps: FiniteFloat = Field(default=40.0, gt=0.0)
    max_abs_acceleration_mps2: FiniteFloat = Field(default=10.0, gt=0.0)
    max_abs_lane_offset_m: FiniteFloat = Field(default=3.5, gt=0.0)
    max_ttc_s: FiniteFloat = Field(default=10.0, gt=0.0)
    max_abs_stopping_margin_m: FiniteFloat = Field(default=50.0, gt=0.0)


class RewardConfig(StrictTypedFrozenModel):
    """Weights and thresholds for the Phase 4 transition reward."""

    progress_per_meter: FiniteFloat = Field(default=0.10, ge=0.0)
    arrival: FiniteFloat = Field(default=100.0, ge=0.0)
    collision_vehicle: FiniteFloat = Field(default=200.0, ge=0.0)
    collision_crossing_actor: FiniteFloat = Field(default=500.0, ge=0.0)
    near_miss_max: FiniteFloat = Field(default=50.0, ge=0.0)
    near_miss_ttc_s: FiniteFloat = Field(default=3.0, gt=0.0)
    offroad: FiniteFloat = Field(default=100.0, ge=0.0)
    hard_rule_violation: FiniteFloat = Field(default=100.0, ge=0.0)
    jerk_scale: FiniteFloat = Field(default=0.05, ge=0.0)
    unnecessary_brake_scale: FiniteFloat = Field(default=0.20, ge=0.0)
    unnecessary_brake_severity_threshold: FiniteFloat = Field(default=0.25, ge=0.0, le=1.0)
    unnecessary_brake_safe_ttc_s: FiniteFloat = Field(default=5.0, gt=0.0)
    standstill_per_second: FiniteFloat = Field(default=0.50, ge=0.0)
    standstill_speed_mps: FiniteFloat = Field(default=0.10, gt=0.0)
    shield_intervention: FiniteFloat = Field(default=2.0, ge=0.0)


class PPOConfig(StrictTypedFrozenModel):
    """Strict PPO hyperparameters and training artifact settings."""

    algorithm: Literal["PPO"] = "PPO"
    policy: Literal["MlpPolicy"] = "MlpPolicy"
    learning_rate: FiniteFloat = Field(default=0.0003, gt=0.0)
    n_steps: PositiveInt = 2048
    batch_size: PositiveInt = 64
    n_epochs: PositiveInt = 10
    gamma: FiniteFloat = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: FiniteFloat = Field(default=0.95, ge=0.0, le=1.0)
    clip_range: FiniteFloat = Field(default=0.2, gt=0.0)
    ent_coef: FiniteFloat = Field(default=0.01, ge=0.0)
    vf_coef: FiniteFloat = Field(default=0.5, ge=0.0)
    max_grad_norm: FiniteFloat = Field(default=0.5, gt=0.0)
    seed: int = Field(default=42, ge=0)
    smoke_timesteps: PositiveInt = 5_000
    total_timesteps: PositiveInt = 500_000
    num_envs: PositiveInt = 1
    checkpoint_interval_steps: PositiveInt = 10_000
    eval_interval_steps: PositiveInt = 10_000
    eval_episodes: PositiveInt = 5
    run_root: str = Field(default="runs", min_length=1)

    @model_validator(mode="after")
    def validate_rollout_size(self) -> Self:
        if self.n_steps * self.num_envs % self.batch_size != 0:
            raise ValueError("n_steps * num_envs must be divisible by batch_size")
        return self


class AppConfig(StrictFrozenModel):
    """Root configuration for a deterministic Phase 1 smoke run."""

    seed: int = Field(ge=0)
    scenario_id: str = Field(min_length=1)
    decision_steps: PositiveInt
    fixed_action: tuple[FiniteFloat, FiniteFloat]
    metadrive: MetaDriveConfig
    scenarios: ScenarioSplitsConfig = Field(default_factory=ScenarioSplitsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    shield: ShieldConfig = Field(default_factory=ShieldConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    training: PPOConfig = Field(default_factory=PPOConfig)

    def metadrive_dict(self) -> dict[str, Any]:
        """Return a plain dictionary accepted by MetaDrive."""

        metadrive = self.metadrive.model_dump(
            exclude={"physics_dt_s", "decision_dt_s", "lane_width_m"}
        )
        metadrive["physics_world_step_size"] = self.metadrive.physics_dt_s
        metadrive["map_config"] = {"lane_width": self.metadrive.lane_width_m}
        return metadrive
