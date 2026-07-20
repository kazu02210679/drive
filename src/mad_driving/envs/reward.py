"""Simulator-independent transition reward state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from mad_driving.config.models import RewardConfig
from mad_driving.control.actions import DrivingAction
from mad_driving.interfaces import RiskClaim, SceneSnapshot
from mad_driving.interfaces._validation import require_action, require_positive

CollisionKind = Literal["vehicle", "crossing_actor"]


@dataclass(frozen=True)
class RewardContext:
    """Immutable inputs for one post-step reward calculation."""

    previous_snapshot: SceneSnapshot
    next_snapshot: SceneSnapshot
    post_step_claims: tuple[RiskClaim, ...]
    executed_action: int
    shield_intervened: bool
    arrived: bool
    collision_kind: CollisionKind | None
    decision_interval_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "post_step_claims", tuple(self.post_step_claims))
        require_action("executed_action", self.executed_action)
        require_positive("decision_interval_s", self.decision_interval_s)
        if self.collision_kind not in (None, "vehicle", "crossing_actor"):
            raise ValueError("collision_kind must be vehicle, crossing_actor, or None")


@dataclass(frozen=True)
class RewardResult:
    """Signed reward total and a defensive copy of its component mapping."""

    total: float
    components: dict[str, float]

    def __post_init__(self) -> None:
        components = dict(self.components)
        if not math.isfinite(self.total):
            raise ValueError("reward total must be finite")
        if any(not math.isfinite(value) for value in components.values()):
            raise ValueError("reward components must be finite")
        if self.total != sum(components.values()):
            raise ValueError("reward total must equal the sum of components")
        object.__setattr__(self, "components", components)


class RewardCalculator:
    """Calculate signed reward components while tracking episode-local events."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        """Clear episode-local arrival and safe-braking state."""

        self._safe_brake_streak = 0
        self._arrival_rewarded = False

    def calculate(self, context: RewardContext) -> RewardResult:
        """Calculate all ten signed components for one transition."""

        dt = self._elapsed_time(context)
        rule_constraint = self._has_rule_constraint(context)
        min_ttc_s = self._minimum_ttc(context.post_step_claims)
        components = {
            "progress_reward": self._progress_reward(context),
            "arrival_reward": self._arrival_reward(context.arrived),
            "collision_penalty": self._collision_penalty(context.collision_kind),
            "near_miss_penalty": self._near_miss_penalty(min_ttc_s),
            "offroad_penalty": -self._config.offroad if context.next_snapshot.off_road else 0.0,
            "rule_violation_penalty": (
                -self._config.hard_rule_violation
                if rule_constraint and context.executed_action < DrivingAction.STOP
                else 0.0
            ),
            "jerk_penalty": self._jerk_penalty(context, dt),
            "unnecessary_brake_penalty": self._unnecessary_brake_penalty(
                context, min_ttc_s, rule_constraint
            ),
            "standstill_penalty": (
                -self._config.standstill_per_second * dt
                if context.next_snapshot.ego.speed_mps <= self._config.standstill_speed_mps
                else 0.0
            ),
            "shield_intervention_penalty": (
                -self._config.shield_intervention if context.shield_intervened else 0.0
            ),
        }
        total = sum(components.values())
        if not math.isfinite(total) or any(
            not math.isfinite(value) for value in components.values()
        ):
            raise ValueError("reward outputs must be finite")
        return RewardResult(total=total, components=components)

    @staticmethod
    def _elapsed_time(context: RewardContext) -> float:
        elapsed = context.next_snapshot.sim_time_s - context.previous_snapshot.sim_time_s
        return elapsed if elapsed > 0.0 else context.decision_interval_s

    def _progress_reward(self, context: RewardContext) -> float:
        previous_ego = context.previous_snapshot.ego
        next_ego = context.next_snapshot.ego
        delta_x = next_ego.position_xy_m[0] - previous_ego.position_xy_m[0]
        delta_y = next_ego.position_xy_m[1] - previous_ego.position_xy_m[1]
        forward_distance = delta_x * math.cos(previous_ego.heading_rad) + delta_y * math.sin(
            previous_ego.heading_rad
        )
        return self._config.progress_per_meter * max(0.0, forward_distance)

    def _arrival_reward(self, arrived: bool) -> float:
        if not arrived or self._arrival_rewarded:
            return 0.0
        self._arrival_rewarded = True
        return self._config.arrival

    def _collision_penalty(self, collision_kind: CollisionKind | None) -> float:
        if collision_kind == "crossing_actor":
            return -self._config.collision_crossing_actor
        if collision_kind == "vehicle":
            return -self._config.collision_vehicle
        return 0.0

    @staticmethod
    def _minimum_ttc(claims: tuple[RiskClaim, ...]) -> float | None:
        values = [claim.min_ttc_s for claim in claims if claim.min_ttc_s is not None]
        return min(values, default=None)

    def _near_miss_penalty(self, min_ttc_s: float | None) -> float:
        if min_ttc_s is None:
            return 0.0
        proximity = max(0.0, 1.0 - min_ttc_s / self._config.near_miss_ttc_s)
        return -self._config.near_miss_max * proximity**2

    @staticmethod
    def _has_rule_constraint(context: RewardContext) -> bool:
        snapshot = context.next_snapshot
        return (
            snapshot.stop_required
            or snapshot.intersection_entry_prohibited
            or any(
                claim.agent_id == "rule" and claim.hard_stop_required
                for claim in context.post_step_claims
            )
        )

    def _jerk_penalty(self, context: RewardContext, dt: float) -> float:
        acceleration_delta = abs(
            context.next_snapshot.ego.acceleration_mps2
            - context.previous_snapshot.ego.acceleration_mps2
        )
        return -self._config.jerk_scale * acceleration_delta / dt

    def _unnecessary_brake_penalty(
        self,
        context: RewardContext,
        min_ttc_s: float | None,
        rule_constraint: bool,
    ) -> float:
        hazard_claims = tuple(
            claim for claim in context.post_step_claims if claim.agent_id == "hazard"
        )
        has_rule_claim = any(claim.agent_id == "rule" for claim in context.post_step_claims)
        safe_post_step = (
            bool(hazard_claims)
            and has_rule_claim
            and all(
                claim.severity < self._config.unnecessary_brake_severity_threshold
                for claim in hazard_claims
            )
            and not rule_constraint
            and min_ttc_s is not None
            and min_ttc_s >= self._config.unnecessary_brake_safe_ttc_s
            and context.collision_kind is None
            and not context.next_snapshot.collision_occurred
            and not context.next_snapshot.off_road
            and not context.shield_intervened
        )
        braking = context.executed_action >= DrivingAction.SLOW
        if not (safe_post_step and braking):
            self._safe_brake_streak = 0
            return 0.0

        self._safe_brake_streak += 1
        if self._safe_brake_streak < self._config.unnecessary_brake_lookahead_steps:
            return 0.0
        return -self._config.unnecessary_brake_scale * context.executed_action
