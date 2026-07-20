"""Per-step decision trace for later JSONL logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Self
from unicodedata import category

from mad_driving.interfaces._validation import (
    require_action,
    require_finite,
    require_non_negative,
)
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.risk_claim import RiskClaim


class _FrozenRewardComponents(dict[str, float]):
    """JSON- and pickle-friendly immutable reward component mapping."""

    @staticmethod
    def _reject_mutation(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("reward_components is immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    clear = _reject_mutation
    update = _reject_mutation

    def pop(self, key: str, default: object = None) -> NoReturn:
        self._reject_mutation(key, default)

    def popitem(self) -> NoReturn:
        self._reject_mutation()

    def setdefault(self, key: str, default: float | None = None) -> NoReturn:
        self._reject_mutation(key, default)

    def __ior__(self, value: object, /) -> Self:  # type: ignore[override,misc]
        self._reject_mutation(value)

    def __reduce__(self) -> tuple[type[_FrozenRewardComponents], tuple[dict[str, float]]]:
        return type(self), (dict(self),)


_MAX_ANALYSIS_ERROR_MESSAGE_LENGTH = 256
_REPLACED_ERROR_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


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
    failed_agent_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    episode_rng_seed: int | None = None
    metadrive_scenario_index: int | None = None
    scenario_parameter_seed: int | None = None
    role: str | None = None
    worker_index: int | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        require_action("raw_action", self.raw_action)
        require_action("executed_action", self.executed_action)
        require_non_negative("target_speed_mps", self.target_speed_mps)
        shield_reasons = tuple(self.shield_reasons)
        claims = tuple(self.claims)
        failed_agent_ids = tuple(self.failed_agent_ids)
        errors = tuple(self.errors)
        if not all(isinstance(value, str) for value in shield_reasons):
            raise ValueError("shield_reasons must contain only strings")
        if not all(isinstance(value, RiskClaim) for value in claims):
            raise ValueError("claims must contain only RiskClaim values")
        if not all(isinstance(value, str) and value for value in failed_agent_ids):
            raise ValueError("failed_agent_ids must contain non-empty strings")
        if len(failed_agent_ids) != len(set(failed_agent_ids)):
            raise ValueError("failed_agent_ids must be unique")
        self._validate_error_mapping(failed_agent_ids, errors)
        components = _FrozenRewardComponents(self.reward_components)
        for value in components.values():
            require_finite("reward_components", value)
        episode_metadata = (
            self.episode_rng_seed,
            self.metadrive_scenario_index,
            self.scenario_parameter_seed,
            self.role,
            self.worker_index,
        )
        if any(value is not None for value in episode_metadata):
            if any(value is None for value in episode_metadata):
                raise ValueError("episode trace metadata must be complete")
            for name, metadata_value in (
                ("episode_rng_seed", self.episode_rng_seed),
                ("metadrive_scenario_index", self.metadrive_scenario_index),
                ("scenario_parameter_seed", self.scenario_parameter_seed),
                ("worker_index", self.worker_index),
            ):
                if (
                    isinstance(metadata_value, bool)
                    or not isinstance(metadata_value, int)
                    or metadata_value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")
            if self.role not in {"train", "validation", "test"}:
                raise ValueError("role must be train, validation, or test")
        object.__setattr__(self, "shield_reasons", shield_reasons)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "failed_agent_ids", failed_agent_ids)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "reward_components", components)

    @staticmethod
    def _validate_error_mapping(
        failed_agent_ids: tuple[str, ...], errors: tuple[str, ...]
    ) -> None:
        if len(failed_agent_ids) != len(errors) or len(errors) != len(set(errors)):
            raise ValueError("errors must map one-to-one to failed_agent_ids")
        for agent_id, error in zip(failed_agent_ids, errors, strict=True):
            if not isinstance(error, str) or not error.startswith(f"{agent_id}:"):
                raise ValueError("errors must match failed_agent_ids in order")
            exception_type, separator, message = error[len(agent_id) + 1 :].partition(":")
            if not exception_type or not separator:
                raise ValueError("errors must use agent:type:message format")
            if len(message) > _MAX_ANALYSIS_ERROR_MESSAGE_LENGTH or any(
                category(character) in _REPLACED_ERROR_CATEGORIES for character in message
            ):
                raise ValueError("errors must contain bounded sanitized messages")
