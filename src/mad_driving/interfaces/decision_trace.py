"""Per-step decision trace for later JSONL logging."""

from dataclasses import dataclass

from mad_driving.interfaces._validation import (
    require_action,
    require_finite,
    require_non_negative,
)
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.risk_claim import RiskClaim


@dataclass(frozen=True)
class DecisionTrace:
    step_index: int
    raw_action: int
    executed_action: int
    target_speed_mps: float
    shield_intervened: bool
    shield_reasons: tuple[str, ...]
    claims: tuple[RiskClaim, ...]
    review: CriticReview
    reward_components: dict[str, float]
    episode_rng_seed: int | None = None
    metadrive_scenario_index: int | None = None
    scenario_parameter_seed: int | None = None
    role: str | None = None
    worker_index: int | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        require_action("raw_action", self.raw_action)
        require_action("executed_action", self.executed_action)
        require_non_negative("target_speed_mps", self.target_speed_mps)
        for value in self.reward_components.values():
            require_finite("reward_components", value)
        episode_metadata = (
            self.episode_rng_seed,
            self.metadrive_scenario_index,
            self.scenario_parameter_seed,
            self.role,
            self.worker_index,
        )
        if any(value is not None for value in episode_metadata):
            if any(value is None for value in episode_metadata):
                raise ValueError("episode trace metadata must be complete")
            for name, metadata_value in (
                ("episode_rng_seed", self.episode_rng_seed),
                ("metadrive_scenario_index", self.metadrive_scenario_index),
                ("scenario_parameter_seed", self.scenario_parameter_seed),
                ("worker_index", self.worker_index),
            ):
                if (
                    isinstance(metadata_value, bool)
                    or not isinstance(metadata_value, int)
                    or metadata_value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")
            if self.role not in {"train", "validation", "test"}:
                raise ValueError("role must be train, validation, or test")
        object.__setattr__(self, "reward_components", dict(self.reward_components))
