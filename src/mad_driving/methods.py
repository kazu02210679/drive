"""Central immutable composition profiles for Phase 6 comparison methods."""

from dataclasses import dataclass
from typing import Literal, get_args

from mad_driving.agents.critic import CriticAgent
from mad_driving.agents.hazard import HazardAgent
from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.agents.noop_critic import NoOpCritic
from mad_driving.agents.rule import RuleAgent
from mad_driving.agents.suite import AgentSuite
from mad_driving.config.models import AgentsConfig, MethodId

SpecialistId = Literal["nominal", "hazard", "rule"]


@dataclass(frozen=True)
class MethodProfile:
    """Immutable method composition selected solely by its closed ID."""

    method_id: MethodId
    policy_kind: Literal["rule", "ppo"]
    specialist_ids: tuple[SpecialistId, ...]
    critic_enabled: bool
    default_shield_mode: Literal["off", "monitor", "enforce"]


_METHOD_PROFILES: dict[MethodId, MethodProfile] = {
    "b0_rule": MethodProfile("b0_rule", "rule", (), False, "enforce"),
    "b1_nominal": MethodProfile("b1_nominal", "ppo", ("nominal",), False, "enforce"),
    "b2_multi_no_review": MethodProfile(
        "b2_multi_no_review", "ppo", ("nominal", "hazard", "rule"), False, "enforce"
    ),
    "proposed": MethodProfile(
        "proposed", "ppo", ("nominal", "hazard", "rule"), True, "enforce"
    ),
    "proposed_no_critic": MethodProfile(
        "proposed_no_critic", "ppo", ("nominal", "hazard", "rule"), False, "enforce"
    ),
    "proposed_no_shield": MethodProfile(
        "proposed_no_shield", "ppo", ("nominal", "hazard", "rule"), True, "off"
    ),
    "proposed_no_hazard": MethodProfile(
        "proposed_no_hazard", "ppo", ("nominal", "rule"), True, "enforce"
    ),
}

if set(_METHOD_PROFILES) != set(get_args(MethodId)):
    raise RuntimeError("method profile registry must match MethodId values exactly")


def get_method_profile(method_id: MethodId) -> MethodProfile:
    """Return the immutable central profile for a validated method ID."""

    return _METHOD_PROFILES[method_id]


def build_method_suite(config: AgentsConfig, method_id: MethodId) -> AgentSuite:
    """Construct exactly the specialists and reviewer selected by one profile."""

    profile = get_method_profile(method_id)
    specialist_ids = set(profile.specialist_ids)
    return AgentSuite(
        nominal=NominalMotionAgent(config.nominal) if "nominal" in specialist_ids else None,
        hazard=HazardAgent(config.hazard) if "hazard" in specialist_ids else None,
        rule=RuleAgent(config.rule) if "rule" in specialist_ids else None,
        critic=CriticAgent(config.critic) if profile.critic_enabled else NoOpCritic(),
    )


__all__ = [
    "MethodId",
    "MethodProfile",
    "build_method_suite",
    "get_method_profile",
]
