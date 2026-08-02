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

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        require_action("raw_action", self.raw_action)
        require_action("executed_action", self.executed_action)
        require_non_negative("target_speed_mps", self.target_speed_mps)
        for value in self.reward_components.values():
            require_finite("reward_components", value)
        object.__setattr__(self, "reward_components", dict(self.reward_components))
