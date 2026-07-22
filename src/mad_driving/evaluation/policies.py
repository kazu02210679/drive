"""Simulator-independent adapters for Phase 6 evaluation policies."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, isclose, sin
from numbers import Integral
from typing import Any, Final, Protocol, cast

import numpy as np

from mad_driving.config.models import MethodId
from mad_driving.evaluation.models import RESEARCH_CONTRACT_VERSION
from mad_driving.interfaces import ActorState, SceneObservation
from mad_driving.methods import MethodProfileSnapshot

_OBSERVATION_SCHEMA_VERSION: Final = 1
_OBSERVATION_SHAPE: Final = (24,)
_OBSERVATION_DTYPE: Final = "float32"
_ACTION_SCHEMA_VERSION: Final = 1
_ACTION_ORDER: Final = ("KEEP", "SLOW", "PREPARE_STOP", "STOP")
_KEEP: Final = 0
_SLOW: Final = 1
_PREPARE_STOP: Final = 2
_STOP: Final = 3


class EvaluationPolicy(Protocol):
    """One resettable policy behind the common online evaluation runner."""

    def reset(self) -> None: ...

    def predict(self, observation: SceneObservation | np.ndarray[Any, Any]) -> int: ...


class VisibleTtcRulePolicy:
    """Deterministic B0 policy using only the current Agent-visible scene."""

    STOP_TTC_S: Final[float] = 1.0
    PREPARE_STOP_TTC_S: Final[float] = 3.0
    SLOW_TTC_S: Final[float] = 5.0

    def reset(self) -> None:
        """Reset B0; the policy is intentionally stateless."""

    def predict(self, observation: SceneObservation | np.ndarray[Any, Any]) -> int:
        if not isinstance(observation, SceneObservation):
            raise TypeError("B0 requires a SceneObservation")
        road = observation.road_context
        if road.stop_required or road.intersection_entry_prohibited:
            return _STOP
        ttc = self._minimum_visible_ttc_s(observation)
        if ttc is None:
            return _KEEP
        if ttc <= self.STOP_TTC_S:
            return _STOP
        if ttc <= self.PREPARE_STOP_TTC_S:
            return _PREPARE_STOP
        if ttc <= self.SLOW_TTC_S:
            return _SLOW
        return _KEEP

    @classmethod
    def _minimum_visible_ttc_s(cls, observation: SceneObservation) -> float | None:
        values = tuple(
            ttc
            for actor in observation.visible_actors
            if (ttc := cls._actor_ttc_s(observation, actor)) is not None
        )
        return min(values, default=None)

    @staticmethod
    def _actor_ttc_s(observation: SceneObservation, actor: ActorState) -> float | None:
        heading = observation.ego.heading_rad
        actor_longitudinal_speed = (
            cos(heading) * actor.velocity_xy_mps[0] + sin(heading) * actor.velocity_xy_mps[1]
        )
        actor_lateral_speed = (
            -sin(heading) * actor.velocity_xy_mps[0] + cos(heading) * actor.velocity_xy_mps[1]
        )
        relative_velocity = (
            actor_longitudinal_speed - observation.ego.speed_mps,
            actor_lateral_speed,
        )
        if actor.same_lane:
            closing_speed = -relative_velocity[0]
            if actor.relative_longitudinal_m <= 0.0 or closing_speed <= 0.0:
                return None
            return actor.relative_longitudinal_m / closing_speed
        return VisibleTtcRulePolicy._point_intersection_time_s(
            (actor.relative_longitudinal_m, actor.relative_lateral_m), relative_velocity
        )

    @staticmethod
    def _point_intersection_time_s(
        position: tuple[float, float], velocity: tuple[float, float]
    ) -> float | None:
        times: list[float] = []
        for coordinate, rate in zip(position, velocity, strict=True):
            if rate == 0.0:
                if coordinate != 0.0:
                    return None
                continue
            times.append(-coordinate / rate)
        if not times or any(value < 0.0 for value in times):
            return None
        first = times[0]
        if any(not isclose(value, first, rel_tol=1e-9, abs_tol=1e-9) for value in times[1:]):
            return None
        return first


class PpoPolicyAdapter:
    """Validate one PPO checkpoint binding and expose deterministic scalar actions."""

    def __init__(
        self,
        model: object,
        *,
        method_id: MethodId,
        checkpoint_path: str,
        checkpoint_sha256: str,
        resolved_config: Mapping[str, object],
        checkpoint_metadata: Mapping[str, object],
    ) -> None:
        self._model = model
        self.method_id = method_id
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.resolved_config = resolved_config
        self.method_profile = MethodProfileSnapshot.from_method_id(method_id)
        self._validate_binding(checkpoint_metadata)
        self._state: object = None
        self._episode_start = True

    def reset(self) -> None:
        self._state = None
        self._episode_start = True

    def predict(self, observation: SceneObservation | np.ndarray[Any, Any]) -> int:
        if not isinstance(observation, np.ndarray):
            raise TypeError("PPO requires the 24-dimensional NumPy observation")
        if observation.shape != _OBSERVATION_SHAPE or observation.dtype != np.dtype(
            _OBSERVATION_DTYPE
        ):
            raise ValueError("PPO observation does not match schema version 1")
        prediction = cast(Any, self._model).predict(
            observation,
            deterministic=True,
            state=self._state,
            episode_start=np.array([self._episode_start], dtype=np.bool_),
        )
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise ValueError("PPO model prediction must return action and state")
        raw_action, self._state = prediction
        self._episode_start = False
        values = np.asarray(raw_action)
        if values.size != 1:
            raise ValueError("PPO action must be scalar")
        scalar = values.reshape(-1)[0].item()
        if isinstance(scalar, bool) or not isinstance(scalar, Integral):
            raise ValueError("PPO action must be an integer")
        action = int(scalar)
        if not 0 <= action < len(_ACTION_ORDER):
            raise ValueError("PPO action must be in the configured action range")
        return action

    def _validate_binding(self, metadata: Mapping[str, object]) -> None:
        expected: dict[str, object] = {
            "research_contract_version": RESEARCH_CONTRACT_VERSION,
            "observation_schema_version": _OBSERVATION_SCHEMA_VERSION,
            "observation_shape": _OBSERVATION_SHAPE,
            "observation_dtype": _OBSERVATION_DTYPE,
            "action_schema_version": _ACTION_SCHEMA_VERSION,
            "action_count": len(_ACTION_ORDER),
            "action_order": _ACTION_ORDER,
            "method_profile": self.method_profile,
            "resolved_config": self.resolved_config,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        missing = set(expected) - set(metadata)
        if missing:
            raise ValueError(f"checkpoint metadata is missing required fields: {sorted(missing)!r}")
        for field, expected_value in expected.items():
            actual = metadata[field]
            if field in {"observation_shape", "action_order"} and isinstance(actual, list | tuple):
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(f"checkpoint metadata {field} mismatch")
        method_config = self.resolved_config.get("method")
        if not isinstance(method_config, Mapping) or method_config.get("id") != self.method_id:
            raise ValueError("resolved config method does not match PPO adapter")


__all__ = ["EvaluationPolicy", "PpoPolicyAdapter", "VisibleTtcRulePolicy"]
