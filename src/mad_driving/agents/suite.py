"""Fixed-order orchestration for deterministic driving analysis agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from mad_driving.agents.critic import CriticAgent
from mad_driving.agents.hazard import HazardAgent
from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.agents.protocol import DrivingAgent
from mad_driving.agents.rule import RuleAgent
from mad_driving.config.models import AgentsConfig
from mad_driving.interfaces import CriticReview, RiskClaim, SceneObservation
from mad_driving.interfaces.defensive_validation import valid_claim, valid_review


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
        claims = tuple(self.claims)
        failed_agent_ids = tuple(self.failed_agent_ids)
        errors = tuple(self.errors)
        if not all(valid_claim(claim) for claim in claims):
            raise ValueError("claims must contain only valid RiskClaim values")
        claim_ids = tuple(claim.claim_id for claim in claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claims must have unique claim_id values")
        if not all(isinstance(agent_id, str) and agent_id for agent_id in failed_agent_ids):
            raise ValueError("failed_agent_ids must contain non-empty strings")
        if len(failed_agent_ids) != len(set(failed_agent_ids)):
            raise ValueError("failed_agent_ids must be unique")
        if not all(isinstance(error, str) for error in errors):
            raise ValueError("errors must contain strings")
        if not valid_review(self.review):
            raise ValueError("review must be a valid CriticReview")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "failed_agent_ids", failed_agent_ids)
        object.__setattr__(self, "errors", errors)


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
                errors.append(f"{agent.agent_id}:{type(error).__name__}:{error}")
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
