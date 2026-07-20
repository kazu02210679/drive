"""Validation shared by fixed-schema Coordinator claim consumers."""

from collections.abc import Sequence

from mad_driving.interfaces import RiskClaim

SPECIALIST_AGENT_IDS = frozenset({"nominal", "hazard", "rule"})


def validate_specialist_claim_agent_ids(claims: Sequence[RiskClaim]) -> None:
    """Reject claims outside the fixed Coordinator specialist schema."""

    if any(
        type(claim.agent_id) is not str or claim.agent_id not in SPECIALIST_AGENT_IDS
        for claim in claims
    ):
        raise ValueError("invalid claim agent_id")
