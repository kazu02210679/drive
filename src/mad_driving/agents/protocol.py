"""Common contract for deterministic claim-producing agents."""

from typing import Protocol, runtime_checkable

from mad_driving.interfaces import RiskClaim, SceneSnapshot


@runtime_checkable
class DrivingAgent(Protocol):
    """Analyze one immutable scene without side effects."""

    agent_id: str

    def analyze(self, snapshot: SceneSnapshot) -> RiskClaim: ...
