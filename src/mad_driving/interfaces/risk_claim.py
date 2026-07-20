"""Structured agent risk claim."""

from dataclasses import dataclass

from mad_driving.interfaces._validation import (
    require_finite,
    require_non_empty,
    require_non_negative,
    require_probability,
)


@dataclass(frozen=True)
class RiskClaim:
    claim_id: str
    agent_id: str
    event_type: str
    target_actor_id: str | None
    probability: float | None
    confidence: float
    severity: float
    time_horizon_s: float
    min_ttc_s: float | None
    stopping_margin_m: float | None
    recommended_max_speed_mps: float
    hard_stop_required: bool
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    valid_until_step: int

    def __post_init__(self) -> None:
        require_non_empty("claim_id", self.claim_id)
        require_non_empty("agent_id", self.agent_id)
        require_non_empty("event_type", self.event_type)
        if self.probability is not None:
            require_probability("probability", self.probability)
        require_probability("confidence", self.confidence)
        require_probability("severity", self.severity)
        require_non_negative("time_horizon_s", self.time_horizon_s)
        if self.min_ttc_s is not None:
            require_non_negative("min_ttc_s", self.min_ttc_s)
        if self.stopping_margin_m is not None:
            require_finite("stopping_margin_m", self.stopping_margin_m)
        require_non_negative("recommended_max_speed_mps", self.recommended_max_speed_mps)
        if self.valid_until_step < 0:
            raise ValueError("valid_until_step must be non-negative")
