"""Pure reduction of strict evaluation step records into episode metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from mad_driving.control import DrivingAction, action_for_speed_cap
from mad_driving.coordinator.observation import aggregate_agent_claims
from mad_driving.evaluation.models import EvaluationEpisodeRecord, EvaluationStepRecord


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def _require_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_finite(name: str, value: object, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_rate(name: str, value: object) -> float:
    result = _require_finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _require_optional_rate(
    name: str,
    value: object,
    *,
    count: int,
    denominator: int,
) -> None:
    if denominator == 0:
        if value is not None:
            raise ValueError(f"{name} must be None when its denominator is zero")
        return
    if value is None:
        raise ValueError(f"{name} must be defined when its denominator is non-zero")
    if _require_rate(name, value) != count / denominator:
        raise ValueError(f"{name} is inconsistent with its count and denominator")


@dataclass(frozen=True)
class EpisodeMetrics:
    collision: bool
    crossing_actor_collision: bool
    near_miss: bool
    minimum_actual_ttc_s: float | None
    negative_stopping_margin: bool
    minimum_stopping_margin_m: float | None
    hard_rule_violation: bool
    raw_unsafe_request_rate: float
    shield_intervention_rate: float
    off_road: bool
    scenario_success: bool
    final_route_completion: float
    average_speed_mps: float
    simulated_travel_time_s: float
    unnecessary_braking_event_count: int
    unnecessary_stop_duration_s: float
    longitudinal_acceleration_rms_mps2: float
    maximum_deceleration_mps2: float
    longitudinal_jerk_rms_mps3: float | None
    agent_disagreement_eligible_steps: int
    agent_disagreement_count: int
    agent_disagreement_rate: float | None
    critic_challenge_eligible_steps: int
    critic_challenge_count: int
    critic_challenge_rate: float | None
    critic_found_missed_danger_count: int
    critic_found_missed_danger_rate: float | None
    critic_false_challenge_count: int
    critic_false_challenge_rate: float | None
    agent_failure_fallback_count: int
    decision_latency_p50_ms: float
    decision_latency_p95_ms: float
    decision_latency_p99_ms: float
    episode_reward: float

    def __post_init__(self) -> None:
        for name in (
            "collision",
            "crossing_actor_collision",
            "near_miss",
            "negative_stopping_margin",
            "hard_rule_violation",
            "off_road",
            "scenario_success",
        ):
            _require_bool(name, getattr(self, name))

        counts = {
            name: _require_count(name, getattr(self, name))
            for name in (
                "unnecessary_braking_event_count",
                "agent_disagreement_eligible_steps",
                "agent_disagreement_count",
                "critic_challenge_eligible_steps",
                "critic_challenge_count",
                "critic_found_missed_danger_count",
                "critic_false_challenge_count",
                "agent_failure_fallback_count",
            )
        }
        if counts["agent_disagreement_count"] > counts["agent_disagreement_eligible_steps"]:
            raise ValueError("agent disagreement count exceeds eligible steps")
        if counts["critic_challenge_count"] > counts["critic_challenge_eligible_steps"]:
            raise ValueError("critic challenge count exceeds eligible steps")
        for name in (
            "critic_found_missed_danger_count",
            "critic_false_challenge_count",
        ):
            if counts[name] > counts["critic_challenge_count"]:
                raise ValueError(f"{name} exceeds critic challenge count")

        _require_rate("raw_unsafe_request_rate", self.raw_unsafe_request_rate)
        _require_rate("shield_intervention_rate", self.shield_intervention_rate)
        route_completion = _require_finite(
            "final_route_completion", self.final_route_completion, non_negative=True
        )
        if route_completion > 1.0:
            raise ValueError("final_route_completion must be in [0, 1]")
        for name in (
            "average_speed_mps",
            "simulated_travel_time_s",
            "unnecessary_stop_duration_s",
            "longitudinal_acceleration_rms_mps2",
            "maximum_deceleration_mps2",
            "decision_latency_p50_ms",
            "decision_latency_p95_ms",
            "decision_latency_p99_ms",
        ):
            _require_finite(name, getattr(self, name), non_negative=True)
        if self.longitudinal_jerk_rms_mps3 is not None:
            _require_finite(
                "longitudinal_jerk_rms_mps3",
                self.longitudinal_jerk_rms_mps3,
                non_negative=True,
            )
        if self.minimum_actual_ttc_s is not None:
            _require_finite("minimum_actual_ttc_s", self.minimum_actual_ttc_s, non_negative=True)
        if self.minimum_stopping_margin_m is not None:
            _require_finite("minimum_stopping_margin_m", self.minimum_stopping_margin_m)
        _require_finite("episode_reward", self.episode_reward)

        _require_optional_rate(
            "agent_disagreement_rate",
            self.agent_disagreement_rate,
            count=counts["agent_disagreement_count"],
            denominator=counts["agent_disagreement_eligible_steps"],
        )
        _require_optional_rate(
            "critic_challenge_rate",
            self.critic_challenge_rate,
            count=counts["critic_challenge_count"],
            denominator=counts["critic_challenge_eligible_steps"],
        )
        _require_optional_rate(
            "critic_found_missed_danger_rate",
            self.critic_found_missed_danger_rate,
            count=counts["critic_found_missed_danger_count"],
            denominator=counts["critic_challenge_count"],
        )
        _require_optional_rate(
            "critic_false_challenge_rate",
            self.critic_false_challenge_rate,
            count=counts["critic_false_challenge_count"],
            denominator=counts["critic_challenge_count"],
        )

        if self.crossing_actor_collision and not self.collision:
            raise ValueError("crossing_actor_collision requires collision")
        if self.near_miss and self.collision:
            raise ValueError("near_miss excludes collision")
        expected_negative_margin = (
            self.minimum_stopping_margin_m is not None and self.minimum_stopping_margin_m < 0.0
        )
        if self.negative_stopping_margin is not expected_negative_margin:
            raise ValueError("negative_stopping_margin is inconsistent with the minimum")


@dataclass(frozen=True)
class EpisodeMetricRecord:
    """Typed pairing between one terminal episode and its reduced metric row."""

    episode: EvaluationEpisodeRecord
    metrics: EpisodeMetrics

    def __post_init__(self) -> None:
        if type(self.episode) is not EvaluationEpisodeRecord:
            raise TypeError("episode must be an EvaluationEpisodeRecord")
        if type(self.metrics) is not EpisodeMetrics:
            raise TypeError("metrics must be EpisodeMetrics")
        if self.episode.collision_occurred is not self.metrics.collision:
            raise ValueError("episode collision is inconsistent with metrics")
        expected_crossing_collision = self.episode.collision_kind == "crossing_actor"
        if expected_crossing_collision is not self.metrics.crossing_actor_collision:
            raise ValueError("episode collision kind is inconsistent with metrics")
        if self.episode.scenario_success is not self.metrics.scenario_success:
            raise ValueError("episode scenario success is inconsistent with metrics")
        if self.episode.off_road is not self.metrics.off_road:
            raise ValueError("episode off-road status is inconsistent with metrics")
        if self.episode.cumulative_reward != self.metrics.episode_reward:
            raise ValueError("episode cumulative reward is inconsistent with metrics")
        if self.episode.simulated_duration_s != self.metrics.simulated_travel_time_s:
            raise ValueError("episode simulated duration is inconsistent with metrics")


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validated_records(
    records: Sequence[EvaluationStepRecord], decision_dt_s: float
) -> tuple[EvaluationStepRecord, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("records must be non-empty")
    if any(type(record) is not EvaluationStepRecord for record in values):
        raise TypeError("records must contain EvaluationStepRecord values")
    if tuple(record.step_index for record in values) != tuple(range(len(values))):
        raise ValueError("record step indices must be contiguous from zero")
    first = values[0]
    episode_contract = (
        first.checkpoint_path,
        first.checkpoint_sha256,
        first.method_profile,
        first.episode_key,
        first.episode_rng_seed,
        first.case_id,
        first.scenario_id,
        first.difficulty_level,
        first.metadrive_scenario_index,
        first.scenario_selection_seed,
        first.scenario_parameter_seed,
    )
    if any(
        (
            record.checkpoint_path,
            record.checkpoint_sha256,
            record.method_profile,
            record.episode_key,
            record.episode_rng_seed,
            record.case_id,
            record.scenario_id,
            record.difficulty_level,
            record.metadrive_scenario_index,
            record.scenario_selection_seed,
            record.scenario_parameter_seed,
        )
        != episode_contract
        for record in values[1:]
    ):
        raise ValueError("records must share a single episode contract")
    if any(record.decision_interval_s != decision_dt_s for record in values):
        raise ValueError("every decision_interval_s must equal decision_dt_s")
    return values


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[rank - 1]


def _unnecessary_braking_events(
    records: Sequence[EvaluationStepRecord], decision_dt_s: float
) -> int:
    previous_duration = 0.0
    previous_active = False
    events = 0
    for record in records:
        current_duration = record.cumulative_unnecessary_stop_duration_s
        if current_duration < previous_duration:
            raise ValueError("cumulative_unnecessary_stop_duration_s must not decrease")
        active = current_duration > previous_duration
        if active and current_duration > previous_duration + decision_dt_s:
            raise ValueError(
                "cumulative unnecessary-stop increment cannot exceed the decision interval"
            )
        if active and not previous_active:
            events += 1
        previous_duration = current_duration
        previous_active = active
    return events


def _specialist_actions(record: EvaluationStepRecord) -> dict[str, DrivingAction]:
    actions: dict[str, DrivingAction] = {}
    for agent_id in record.method_profile.specialist_ids:
        aggregate = aggregate_agent_claims(agent_id, record.claims)
        if aggregate is None:
            continue
        action = action_for_speed_cap(
            aggregate.recommended_max_speed_mps,
            record.ego_speed_limit_mps,
        )
        if aggregate.hard_stop_required:
            action = DrivingAction.STOP
        actions[agent_id] = action
    return actions


def _critic_found_missed_danger(
    record: EvaluationStepRecord, actions: dict[str, DrivingAction]
) -> bool:
    challenged = set(record.review.challenged_claim_ids)
    challenged_nominal = any(
        claim.agent_id == "nominal" and claim.claim_id in challenged for claim in record.claims
    )
    nominal_action = actions.get("nominal")
    if not challenged_nominal or nominal_action is None:
        return False
    return any(
        agent_id in record.review.supported_agent_ids
        and actions.get(agent_id, DrivingAction.KEEP) > nominal_action
        for agent_id in ("hazard", "rule")
    )


def _critic_false_challenge(record: EvaluationStepRecord, near_miss_ttc_s: float) -> bool:
    if not record.review.challenged_claim_ids:
        return False
    margin = record.minimum_actual_stopping_margin_m
    ttc = record.minimum_actual_ttc_s
    return (
        not record.pre_step_hard_rule_constraint
        and not record.collision_occurred
        and (margin is None or margin >= 0.0)
        and (ttc is None or ttc >= near_miss_ttc_s)
    )


def reduce_episode(
    records: Sequence[EvaluationStepRecord],
    decision_dt_s: float,
    *,
    near_miss_ttc_s: float = 3.0,
) -> EpisodeMetrics:
    """Reduce one fixed-interval episode without importing a simulator."""

    dt = _positive_finite("decision_dt_s", decision_dt_s)
    threshold = _positive_finite("near_miss_ttc_s", near_miss_ttc_s)
    values = _validated_records(records, dt)
    count = len(values)
    finite_ttc = tuple(
        record.minimum_actual_ttc_s for record in values if record.minimum_actual_ttc_s is not None
    )
    finite_margins = tuple(
        record.minimum_actual_stopping_margin_m
        for record in values
        if record.minimum_actual_stopping_margin_m is not None
    )
    collision = any(record.collision_occurred for record in values)
    minimum_ttc = min(finite_ttc, default=None)
    minimum_margin = min(finite_margins, default=None)
    duration = sum(record.decision_interval_s for record in values)
    weighted_speed = sum(record.ego_speed_mps * record.decision_interval_s for record in values)
    weighted_acceleration_square = sum(
        record.ego_longitudinal_acceleration_mps2**2 * record.decision_interval_s
        for record in values
    )
    latencies = tuple(record.total_decision_latency_ms for record in values)
    braking_events = _unnecessary_braking_events(values, dt)
    jerk_rms: float | None = None
    if len(values) > 1:
        jerk_square_time = sum(
            (
                (
                    current.ego_longitudinal_acceleration_mps2
                    - previous.ego_longitudinal_acceleration_mps2
                )
                / dt
            )
            ** 2
            * current.decision_interval_s
            for previous, current in pairwise(values)
        )
        jerk_duration = sum(record.decision_interval_s for record in values[1:])
        jerk_rms = math.sqrt(jerk_square_time / jerk_duration)
    specialist_actions = tuple(_specialist_actions(record) for record in values)
    disagreement_eligible = sum(len(actions) >= 2 for actions in specialist_actions)
    disagreement_count = sum(
        len(actions) >= 2 and len(set(actions.values())) > 1 for actions in specialist_actions
    )
    challenge_eligible = sum(
        record.method_profile.critic_enabled and bool(record.claims) for record in values
    )
    challenge_count = sum(
        record.method_profile.critic_enabled
        and bool(record.claims)
        and bool(record.review.challenged_claim_ids)
        for record in values
    )
    found_missed_danger_count = sum(
        record.method_profile.critic_enabled
        and bool(record.claims)
        and bool(record.review.challenged_claim_ids)
        and _critic_found_missed_danger(record, actions)
        for record, actions in zip(values, specialist_actions, strict=True)
    )
    false_challenge_count = sum(
        record.method_profile.critic_enabled
        and bool(record.claims)
        and _critic_false_challenge(record, threshold)
        for record in values
    )

    return EpisodeMetrics(
        collision=collision,
        crossing_actor_collision=any(
            record.collision_kind == "crossing_actor" for record in values
        ),
        near_miss=not collision and minimum_ttc is not None and minimum_ttc < threshold,
        minimum_actual_ttc_s=minimum_ttc,
        negative_stopping_margin=minimum_margin is not None and minimum_margin < 0.0,
        minimum_stopping_margin_m=minimum_margin,
        hard_rule_violation=any(
            record.pre_step_hard_rule_constraint and record.executed_action < DrivingAction.STOP
            for record in values
        ),
        raw_unsafe_request_rate=sum(record.unsafe_request for record in values) / count,
        shield_intervention_rate=sum(record.shield_intervened for record in values) / count,
        off_road=any(record.off_road for record in values),
        scenario_success=values[-1].scenario_success,
        final_route_completion=values[-1].route_completion,
        average_speed_mps=weighted_speed / duration,
        simulated_travel_time_s=duration,
        unnecessary_braking_event_count=braking_events,
        unnecessary_stop_duration_s=values[-1].cumulative_unnecessary_stop_duration_s,
        longitudinal_acceleration_rms_mps2=math.sqrt(weighted_acceleration_square / duration),
        maximum_deceleration_mps2=max(
            0.0,
            max(-record.ego_longitudinal_acceleration_mps2 for record in values),
        ),
        longitudinal_jerk_rms_mps3=jerk_rms,
        agent_disagreement_eligible_steps=disagreement_eligible,
        agent_disagreement_count=disagreement_count,
        agent_disagreement_rate=_rate(disagreement_count, disagreement_eligible),
        critic_challenge_eligible_steps=challenge_eligible,
        critic_challenge_count=challenge_count,
        critic_challenge_rate=_rate(challenge_count, challenge_eligible),
        critic_found_missed_danger_count=found_missed_danger_count,
        critic_found_missed_danger_rate=_rate(found_missed_danger_count, challenge_count),
        critic_false_challenge_count=false_challenge_count,
        critic_false_challenge_rate=_rate(false_challenge_count, challenge_count),
        agent_failure_fallback_count=sum(bool(record.failed_agent_ids) for record in values),
        decision_latency_p50_ms=_nearest_rank(latencies, 50),
        decision_latency_p95_ms=_nearest_rank(latencies, 95),
        decision_latency_p99_ms=_nearest_rank(latencies, 99),
        episode_reward=sum(record.reward_total for record in values),
    )


__all__ = ["EpisodeMetricRecord", "EpisodeMetrics", "reduce_episode"]
