"""Fixed-order orchestration for deterministic driving analysis agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol
from unicodedata import category

from mad_driving.agents.critic import CriticAgent
from mad_driving.agents.hazard import HazardAgent
from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.agents.protocol import DrivingAgent
from mad_driving.agents.rule import RuleAgent
from mad_driving.config.models import AgentsConfig
from mad_driving.interfaces import CriticReview, RiskClaim, SceneObservation
from mad_driving.interfaces.defensive_validation import valid_claim

_MAX_SANITIZED_EXCEPTION_MESSAGE_LENGTH = 256
_REPLACED_EXCEPTION_CHARACTER_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


class ReviewingAgent(Protocol):
    """Contract for a single review over already-computed claims."""

    def review(
        self,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        *,
        failed_agent_ids: Sequence[str],
    ) -> CriticReview: ...


class AnalysisSuite(Protocol):
    """Contract consumed by the smoke runner."""

    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult: ...


class SuiteFactory(Protocol):
    def __call__(self, config: AgentsConfig) -> AnalysisSuite: ...


@dataclass(frozen=True)
class AgentAnalysisResult:
    """Frozen, defensively validated specialist claims and their one Critic review."""

    claims: tuple[RiskClaim, ...]
    failed_agent_ids: tuple[str, ...]
    errors: tuple[str, ...]
    review: CriticReview

    def __post_init__(self) -> None:
        claims = tuple(_canonical_claim(claim) for claim in self.claims)
        failed_agent_ids = _canonical_strings(self.failed_agent_ids, "failed_agent_ids")
        errors = _canonical_strings(self.errors, "errors")
        claim_ids = tuple(claim.claim_id for claim in claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claims must have unique claim_id values")
        if not all(agent_id for agent_id in failed_agent_ids):
            raise ValueError("failed_agent_ids must contain non-empty strings")
        if len(failed_agent_ids) != len(set(failed_agent_ids)):
            raise ValueError("failed_agent_ids must be unique")
        _validate_error_mapping(failed_agent_ids, errors)
        review = _canonical_review(self.review)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "failed_agent_ids", failed_agent_ids)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "review", review)


def analyze_safely(
    suite: AnalysisSuite,
    observation: SceneObservation,
) -> AgentAnalysisResult:
    """Analyze one observation with specialist-level failure isolation."""

    return suite.analyze(observation)


@dataclass(frozen=True)
class AgentSuite:
    """Call Nominal, Hazard, Rule, and Critic exactly once in fixed order."""

    nominal: DrivingAgent | None
    hazard: DrivingAgent | None
    rule: DrivingAgent | None
    critic: ReviewingAgent

    @classmethod
    def from_config(cls, config: AgentsConfig) -> AgentSuite:
        return cls(
            nominal=NominalMotionAgent(config.nominal),
            hazard=HazardAgent(config.hazard),
            rule=RuleAgent(config.rule),
            critic=CriticAgent(config.critic),
        )

    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        claims: list[RiskClaim] = []
        failed_agent_ids: list[str] = []
        errors: list[str] = []
        seen_claim_ids: set[str] = set()
        for agent in (self.nominal, self.hazard, self.rule):
            if agent is None:
                continue
            try:
                agent_claims = tuple(agent.analyze(observation))
                self._validate_agent_claims(agent, agent_claims, seen_claim_ids)
            except Exception as error:
                failed_agent_ids.append(agent.agent_id)
                errors.append(_format_agent_error(agent.agent_id, error))
                continue
            claims.extend(agent_claims)
            seen_claim_ids.update(claim.claim_id for claim in agent_claims)

        failed = tuple(failed_agent_ids)
        review = self.critic.review(observation, tuple(claims), failed_agent_ids=failed)
        if failed:
            review = replace(
                review,
                reasons=review.reasons
                + tuple(f"agent_analysis_failed:{agent_id}" for agent_id in failed),
            )
        return AgentAnalysisResult(tuple(claims), failed, tuple(errors), review)

    @staticmethod
    def _validate_agent_claims(
        agent: DrivingAgent,
        claims: tuple[RiskClaim, ...],
        seen_claim_ids: set[str],
    ) -> None:
        if not 1 <= len(claims) <= 3:
            raise ValueError("an agent must return one to three claims")
        if not all(valid_claim(claim) for claim in claims):
            raise ValueError("agent returned an invalid claim")
        if any(claim.agent_id != agent.agent_id for claim in claims):
            raise ValueError("agent returned a claim with a mismatched agent_id")
        claim_ids = tuple(claim.claim_id for claim in claims)
        if len(claim_ids) != len(set(claim_ids)) or seen_claim_ids.intersection(claim_ids):
            raise ValueError("agent returned a duplicate claim_id")


def _canonical_claim(claim: RiskClaim) -> RiskClaim:
    """Reconstruct a claim so mutable caller-owned sequences cannot leak through."""

    try:
        return RiskClaim(
            claim_id=claim.claim_id,
            agent_id=claim.agent_id,
            event_type=claim.event_type,
            target_actor_id=claim.target_actor_id,
            probability=claim.probability,
            confidence=claim.confidence,
            severity=claim.severity,
            time_horizon_s=claim.time_horizon_s,
            min_ttc_s=claim.min_ttc_s,
            stopping_margin_m=claim.stopping_margin_m,
            recommended_max_speed_mps=claim.recommended_max_speed_mps,
            hard_stop_required=claim.hard_stop_required,
            evidence=_canonical_strings(claim.evidence, "claim evidence"),
            assumptions=_canonical_strings(claim.assumptions, "claim assumptions"),
            valid_until_step=claim.valid_until_step,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("claims must contain only valid RiskClaim values") from error


def _canonical_review(review: CriticReview) -> CriticReview:
    """Reconstruct the review with immutable identifier and reason sequences."""

    try:
        return CriticReview(
            conflict_score=review.conflict_score,
            unresolved_conflict=review.unresolved_conflict,
            max_severity=review.max_severity,
            supported_agent_ids=_canonical_strings(
                review.supported_agent_ids, "review supported_agent_ids"
            ),
            challenged_claim_ids=_canonical_strings(
                review.challenged_claim_ids, "review challenged_claim_ids"
            ),
            reasons=_canonical_strings(review.reasons, "review reasons"),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("review must be a valid CriticReview") from error


def _canonical_strings(values: object, field_name: str) -> tuple[str, ...]:
    """Freeze a sequence of strings or reject malformed scalar and element values."""

    if isinstance(values, str | bytes):
        raise ValueError(f"{field_name} must be a sequence of strings")
    try:
        result: tuple[object, ...] = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of strings") from error
    if not all(isinstance(value, str) for value in result):
        raise ValueError(f"{field_name} must contain only strings")
    return tuple(value for value in result if isinstance(value, str))


def _validate_error_mapping(
    failed_agent_ids: tuple[str, ...], errors: tuple[str, ...]
) -> None:
    """Require one sanitized error in the same order for each failed specialist."""

    if len(failed_agent_ids) != len(errors) or len(errors) != len(set(errors)):
        raise ValueError("errors must map one-to-one to failed_agent_ids")
    for agent_id, error in zip(failed_agent_ids, errors, strict=True):
        prefix = f"{agent_id}:"
        if not error.startswith(prefix):
            raise ValueError("errors must match failed_agent_ids in order")
        exception_type, separator, message = error[len(prefix) :].partition(":")
        if not exception_type or not separator:
            raise ValueError("errors must use agent:type:message format")
        if len(message) > _MAX_SANITIZED_EXCEPTION_MESSAGE_LENGTH or any(
            category(character) in _REPLACED_EXCEPTION_CHARACTER_CATEGORIES
            for character in message
        ):
            raise ValueError("errors must contain bounded sanitized messages")


def _format_agent_error(agent_id: str, error: Exception) -> str:
    """Return ``agent:type:message`` with controls replaced and text bounded to 256 chars."""

    try:
        message = str(error)
    except BaseException:
        message = "<unprintable>"
    sanitized = "".join(
        " " if category(character) in _REPLACED_EXCEPTION_CHARACTER_CATEGORIES else character
        for character in message
    )
    return (
        f"{agent_id}:{type(error).__name__}:"
        f"{sanitized[:_MAX_SANITIZED_EXCEPTION_MESSAGE_LENGTH]}"
    )
