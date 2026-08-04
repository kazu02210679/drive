"""Structured cross-review result."""

from dataclasses import dataclass

from mad_driving.interfaces._validation import canonical_string_tuple, require_probability


@dataclass(frozen=True)
class CriticReview:
    conflict_score: float
    unresolved_conflict: bool
    max_severity: float
    supported_agent_ids: tuple[str, ...]
    challenged_claim_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        require_probability("conflict_score", self.conflict_score)
        require_probability("max_severity", self.max_severity)
        object.__setattr__(
            self,
            "supported_agent_ids",
            canonical_string_tuple("supported_agent_ids", self.supported_agent_ids),
        )
        object.__setattr__(
            self,
            "challenged_claim_ids",
            canonical_string_tuple("challenged_claim_ids", self.challenged_claim_ids),
        )
        object.__setattr__(self, "reasons", canonical_string_tuple("reasons", self.reasons))
