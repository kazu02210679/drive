"""Simulator-independent transition reward state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mad_driving.config.models import RewardConfig
from mad_driving.control.actions import DrivingAction
from mad_driving.interfaces import CollisionKind, SceneFrame
from mad_driving.interfaces._validation import require_action, require_positive


@dataclass(frozen=True)
class RewardContext:
    """Immutable inputs for one post-step reward calculation."""

    previous_frame: SceneFrame
    next_frame: SceneFrame
    executed_action: int
    shield_intervened: bool
    decision_interval_s: float

    def __post_init__(self) -> None:
        require_action("executed_action", self.executed_action)
        require_positive("decision_interval_s", self.decision_interval_s)


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
        """Clear episode-local arrival state."""

        self._arrival_rewarded = False

    def calculate(self, context: RewardContext) -> RewardResult:
        """Calculate all ten signed components for one transition."""

        dt = self._elapsed_time(context)
        previous_oracle = context.previous_frame.privileged
        next_oracle = context.next_frame.privileged
        rule_constraint = previous_oracle.hard_rule_constraint
        previous_min_ttc_s = previous_oracle.minimum_actual_ttc_s
        next_min_ttc_s = next_oracle.minimum_actual_ttc_s
        privileged = context.next_frame.privileged
        components = {
            "progress_reward": self._progress_reward(context),
            "arrival_reward": self._arrival_reward(privileged.arrived),
            "collision_penalty": self._collision_penalty(privileged.collision_kind),
            "near_miss_penalty": self._near_miss_penalty(next_min_ttc_s),
            "offroad_penalty": -self._config.offroad if privileged.off_road else 0.0,
            "rule_violation_penalty": (
                -self._config.hard_rule_violation
                if rule_constraint and context.executed_action < DrivingAction.STOP
                else 0.0
            ),
            "jerk_penalty": self._jerk_penalty(context, dt),
            "unnecessary_brake_penalty": self._unnecessary_brake_penalty(
                context, previous_min_ttc_s, rule_constraint
            ),
            "standstill_penalty": (
                -self._config.standstill_per_second * dt
                if context.next_frame.observation.ego.speed_mps <= self._config.standstill_speed_mps
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
        elapsed = (
            context.next_frame.observation.sim_time_s
            - context.previous_frame.observation.sim_time_s
        )
        return elapsed if elapsed > 0.0 else context.decision_interval_s

    def _progress_reward(self, context: RewardContext) -> float:
        previous_ego = context.previous_frame.observation.ego
        next_ego = context.next_frame.observation.ego
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
        if collision_kind in {"vehicle", "object", "sidewalk", "building"}:
            return -self._config.collision_vehicle
        return 0.0

    def _near_miss_penalty(self, min_ttc_s: float | None) -> float:
        if min_ttc_s is None:
            return 0.0
        proximity = max(0.0, 1.0 - min_ttc_s / self._config.near_miss_ttc_s)
        return -self._config.near_miss_max * proximity**2

    def _jerk_penalty(self, context: RewardContext, dt: float) -> float:
        acceleration_delta = abs(
            context.next_frame.observation.ego.acceleration_mps2
            - context.previous_frame.observation.ego.acceleration_mps2
        )
        return -self._config.jerk_scale * acceleration_delta / dt

    def _unnecessary_brake_penalty(
        self,
        context: RewardContext,
        min_ttc_s: float | None,
        rule_constraint: bool,
    ) -> float:
        privileged = context.next_frame.privileged
        safe_ttc = min_ttc_s is None or min_ttc_s >= self._config.unnecessary_brake_safe_ttc_s
        safe_post_step = (
            not rule_constraint
            and safe_ttc
            and not privileged.collision_occurred
            and not privileged.off_road
            and not context.shield_intervened
        )
        braking = context.executed_action >= DrivingAction.SLOW
        if not (safe_post_step and braking):
            return 0.0
        return -self._config.unnecessary_brake_scale * context.executed_action
