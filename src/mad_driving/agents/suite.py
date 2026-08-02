"""Fixed-order orchestration for deterministic driving analysis agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mad_driving.agents.critic import CriticAgent
from mad_driving.agents.hazard import HazardAgent
from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.agents.protocol import DrivingAgent
from mad_driving.agents.rule import RuleAgent
from mad_driving.config.models import AgentsConfig
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot


class ReviewingAgent(Protocol):
    """Contract for a single review over already-computed claims."""

    def review(self, snapshot: SceneSnapshot, claims: Sequence[RiskClaim]) -> CriticReview: ...


class AnalysisSuite(Protocol):
    """Contract consumed by the smoke runner."""

    def analyze(self, snapshot: SceneSnapshot) -> tuple[tuple[RiskClaim, ...], CriticReview]: ...


class SuiteFactory(Protocol):
    def __call__(self, config: AgentsConfig) -> AnalysisSuite: ...


@dataclass(frozen=True)
class AgentSuite:
    """Call Nominal, Hazard, Rule, and Critic exactly once in fixed order."""

    nominal: DrivingAgent
    hazard: DrivingAgent
    rule: DrivingAgent
    critic: ReviewingAgent

    @classmethod
    def from_config(cls, config: AgentsConfig) -> AgentSuite:
        return cls(
            nominal=NominalMotionAgent(config.nominal),
            hazard=HazardAgent(config.hazard),
            rule=RuleAgent(config.rule),
            critic=CriticAgent(config.critic),
        )

    def analyze(self, snapshot: SceneSnapshot) -> tuple[tuple[RiskClaim, ...], CriticReview]:
        claims = (
            self.nominal.analyze(snapshot),
            self.hazard.analyze(snapshot),
            self.rule.analyze(snapshot),
        )
        return claims, self.critic.review(snapshot, claims)
