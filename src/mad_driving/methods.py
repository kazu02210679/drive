"""Central immutable composition profiles for Phase 6 comparison methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, get_args

from mad_driving.config.models import AgentsConfig, MethodId

if TYPE_CHECKING:
    from mad_driving.agents.suite import AgentSuite

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


@dataclass(frozen=True)
class MethodProfileSnapshot:
    """Immutable runtime composition identity recorded with every run."""

    method_id: str
    policy_kind: str
    specialist_ids: tuple[str, ...]
    critic_enabled: bool
    shield_mode: str

    @classmethod
    def from_method_id(cls, method_id: str) -> MethodProfileSnapshot:
        if method_id not in _METHOD_PROFILES:
            raise ValueError("method_profile.method_id is unknown")
        profile = get_method_profile(cast(MethodId, method_id))
        return cls(
            method_id=profile.method_id,
            policy_kind=profile.policy_kind,
            specialist_ids=profile.specialist_ids,
            critic_enabled=profile.critic_enabled,
            shield_mode=profile.default_shield_mode,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_profile.method_id must be a non-empty string")
        if not isinstance(self.policy_kind, str) or not self.policy_kind:
            raise ValueError("method_profile.policy_kind must be a non-empty string")
        if (
            not isinstance(self.specialist_ids, list | tuple)
            or not all(isinstance(agent_id, str) and agent_id for agent_id in self.specialist_ids)
        ):
            raise ValueError("method_profile.specialist_ids must be non-empty strings")
        if not isinstance(self.critic_enabled, bool):
            raise ValueError("method_profile.critic_enabled must be boolean")
        if not isinstance(self.shield_mode, str) or not self.shield_mode:
            raise ValueError("method_profile.shield_mode must be a non-empty string")
        if self.method_id not in _METHOD_PROFILES:
            raise ValueError("method_profile.method_id is unknown")
        profile = get_method_profile(cast(MethodId, self.method_id))
        expected = (
            profile.method_id,
            profile.policy_kind,
            profile.specialist_ids,
            profile.critic_enabled,
            profile.default_shield_mode,
        )
        actual = (
            self.method_id,
            self.policy_kind,
            tuple(self.specialist_ids),
            self.critic_enabled,
            self.shield_mode,
        )
        if actual != expected:
            raise ValueError("method_profile must equal the central method profile")
        object.__setattr__(self, "specialist_ids", profile.specialist_ids)


def build_method_suite(config: AgentsConfig, method_id: MethodId) -> AgentSuite:
    """Construct exactly the specialists and reviewer selected by one profile."""

    from mad_driving.agents.critic import CriticAgent
    from mad_driving.agents.hazard import HazardAgent
    from mad_driving.agents.nominal import NominalMotionAgent
    from mad_driving.agents.noop_critic import NoOpCritic
    from mad_driving.agents.rule import RuleAgent
    from mad_driving.agents.suite import AgentSuite

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
    "MethodProfileSnapshot",
    "build_method_suite",
    "get_method_profile",
]
