"""Tests for the stateful ten-component transition reward."""

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from mad_driving.config.models import RewardConfig
from mad_driving.control.actions import DrivingAction
from mad_driving.envs.reward import RewardCalculator, RewardContext, RewardResult
from tests.unit.agents.factories import make_claim, make_snapshot

EXPECTED_COMPONENT_KEYS = (
    "progress_reward",
    "arrival_reward",
    "collision_penalty",
    "near_miss_penalty",
    "offroad_penalty",
    "rule_violation_penalty",
    "jerk_penalty",
    "unnecessary_brake_penalty",
    "standstill_penalty",
    "shield_intervention_penalty",
)


def make_context(**overrides: object) -> RewardContext:
    previous = make_snapshot(step_index=10, sim_time_s=1.0)
    next_snapshot = make_snapshot(step_index=11, sim_time_s=2.0)
    next_snapshot = replace(
        next_snapshot,
        ego=replace(
            next_snapshot.ego,
            position_xy_m=(10.0, 0.0),
            speed_mps=1.0,
        ),
    )
    values: dict[str, object] = {
        "previous_snapshot": previous,
        "next_snapshot": next_snapshot,
        "post_step_claims": (
            make_claim("hazard", min_ttc_s=10.0),
            make_claim("rule"),
        ),
        "executed_action": DrivingAction.KEEP,
        "shield_intervened": False,
        "arrived": False,
        "collision_kind": None,
        "decision_interval_s": 0.1,
    }
    values.update(overrides)
    return RewardContext(**values)  # type: ignore[arg-type]


def calculate(context: RewardContext, config: RewardConfig | None = None) -> RewardResult:
    return RewardCalculator(config or RewardConfig()).calculate(context)


def make_safe_braking_context(**overrides: object) -> RewardContext:
    values: dict[str, object] = {
        "post_step_claims": (
            make_claim("hazard", severity=0.1, min_ttc_s=6.0),
            make_claim("rule", hard_stop_required=False),
        ),
        "executed_action": DrivingAction.SLOW,
    }
    values.update(overrides)
    return make_context(**values)


def test_reward_components_are_signed_finite_and_sum_to_total() -> None:
    result = calculate(make_context())

    assert tuple(result.components) == EXPECTED_COMPONENT_KEYS
    assert result.total == pytest.approx(sum(result.components.values()))
    assert result.components["progress_reward"] > 0.0
    assert result.components["arrival_reward"] >= 0.0
    assert all(value <= 0.0 for key, value in result.components.items() if "penalty" in key)
    assert math.isfinite(result.total)
    assert all(math.isfinite(value) for value in result.components.values())


@pytest.mark.parametrize(
    ("displacement", "expected"),
    [((3.0, 4.0), 0.4), ((3.0, -4.0), 0.0)],
)
def test_progress_projects_forward_on_previous_heading_only(
    displacement: tuple[float, float], expected: float
) -> None:
    previous = make_snapshot(step_index=1, sim_time_s=1.0)
    previous = replace(previous, ego=replace(previous.ego, heading_rad=math.pi / 2.0))
    next_snapshot = make_snapshot(step_index=2, sim_time_s=2.0)
    next_snapshot = replace(
        next_snapshot,
        ego=replace(next_snapshot.ego, position_xy_m=displacement, speed_mps=1.0),
    )

    result = calculate(make_context(previous_snapshot=previous, next_snapshot=next_snapshot))

    assert result.components["progress_reward"] == pytest.approx(expected)


def test_arrival_reward_is_one_shot_and_resettable() -> None:
    calculator = RewardCalculator(RewardConfig())
    context = make_context(arrived=True)

    rewards = [calculator.calculate(context).components["arrival_reward"] for _ in range(2)]
    calculator.reset()
    after_reset = calculator.calculate(context).components["arrival_reward"]

    assert rewards == [100.0, 0.0]
    assert after_reset == 100.0


@pytest.mark.parametrize(
    ("collision_kind", "expected"),
    [("vehicle", -200.0), ("crossing_actor", -500.0), (None, 0.0)],
)
def test_collision_penalty_uses_collision_kind(collision_kind: str | None, expected: float) -> None:
    result = calculate(make_context(collision_kind=collision_kind))

    assert result.components["collision_penalty"] == expected


@pytest.mark.parametrize(
    ("ttc_s", "expected"),
    [(0.0, -50.0), (1.5, -12.5), (3.0, 0.0), (4.0, 0.0)],
)
def test_near_miss_penalty_is_continuous_quadratic(ttc_s: float, expected: float) -> None:
    claims = (
        make_claim("nominal", min_ttc_s=ttc_s),
        make_claim("hazard", min_ttc_s=ttc_s + 1.0),
    )

    result = calculate(make_context(post_step_claims=claims))

    assert result.components["near_miss_penalty"] == pytest.approx(expected)


@pytest.mark.parametrize(("off_road", "expected"), [(True, -100.0), (False, 0.0)])
def test_offroad_penalty_uses_post_step_snapshot(off_road: bool, expected: float) -> None:
    context = make_context()
    next_snapshot = replace(context.next_snapshot, off_road=off_road)

    result = calculate(replace(context, next_snapshot=next_snapshot))

    assert result.components["offroad_penalty"] == expected


@pytest.mark.parametrize("source", ["claim", "stop_required", "intersection_prohibited"])
def test_rule_violation_penalizes_non_stop_action_for_any_hard_constraint(source: str) -> None:
    context = make_context(executed_action=DrivingAction.PREPARE_STOP)
    claims = context.post_step_claims
    next_snapshot = context.next_snapshot
    if source == "claim":
        claims = (make_claim("rule", hard_stop_required=True),)
    elif source == "stop_required":
        next_snapshot = replace(next_snapshot, stop_required=True)
    else:
        next_snapshot = replace(next_snapshot, intersection_entry_prohibited=True)

    result = calculate(replace(context, post_step_claims=claims, next_snapshot=next_snapshot))

    assert result.components["rule_violation_penalty"] == -100.0


def test_rule_violation_allows_stop_action() -> None:
    context = make_context(
        post_step_claims=(make_claim("rule", hard_stop_required=True),),
        executed_action=DrivingAction.STOP,
    )

    result = calculate(context)

    assert result.components["rule_violation_penalty"] == 0.0


@pytest.mark.parametrize(
    ("previous_time", "next_time", "fallback_dt", "expected"),
    [(1.0, 1.5, 0.1, -0.2), (1.0, 1.0, 0.25, -0.4)],
)
def test_jerk_penalty_uses_elapsed_or_fallback_decision_interval(
    previous_time: float, next_time: float, fallback_dt: float, expected: float
) -> None:
    context = make_context(decision_interval_s=fallback_dt)
    previous = replace(
        context.previous_snapshot,
        sim_time_s=previous_time,
        ego=replace(context.previous_snapshot.ego, acceleration_mps2=1.0),
    )
    next_snapshot = replace(
        context.next_snapshot,
        sim_time_s=next_time,
        ego=replace(context.next_snapshot.ego, acceleration_mps2=3.0),
    )

    result = calculate(replace(context, previous_snapshot=previous, next_snapshot=next_snapshot))

    assert result.components["jerk_penalty"] == pytest.approx(expected)


def test_unnecessary_brake_penalty_starts_after_safe_lookahead_and_reset() -> None:
    calculator = RewardCalculator(RewardConfig(unnecessary_brake_lookahead_steps=3))

    penalties = [
        calculator.calculate(make_safe_braking_context()).components["unnecessary_brake_penalty"]
        for _ in range(4)
    ]
    calculator.reset()
    after_reset = calculator.calculate(make_safe_braking_context()).components[
        "unnecessary_brake_penalty"
    ]

    assert penalties == [0.0, 0.0, -0.2, -0.2]
    assert after_reset == 0.0


@pytest.mark.parametrize(
    "dangerous_context",
    [
        make_safe_braking_context(
            post_step_claims=(
                make_claim("hazard", severity=0.25, min_ttc_s=6.0),
                make_claim("rule"),
            )
        ),
        make_safe_braking_context(
            post_step_claims=(
                make_claim("hazard", severity=0.1, min_ttc_s=4.99),
                make_claim("rule"),
            )
        ),
        make_safe_braking_context(
            post_step_claims=(
                make_claim("hazard", severity=0.1, min_ttc_s=6.0),
                make_claim("rule", hard_stop_required=True),
            )
        ),
        replace(
            make_safe_braking_context(),
            next_snapshot=replace(make_safe_braking_context().next_snapshot, off_road=True),
        ),
        make_safe_braking_context(collision_kind="vehicle"),
        replace(
            make_safe_braking_context(),
            next_snapshot=replace(
                make_safe_braking_context().next_snapshot, collision_occurred=True
            ),
        ),
        make_safe_braking_context(shield_intervened=True),
    ],
    ids=[
        "hazard",
        "ttc",
        "rule",
        "offroad",
        "collision_kind",
        "snapshot_collision",
        "shield",
    ],
)
def test_dangerous_post_step_event_clears_unnecessary_brake_streak(
    dangerous_context: RewardContext,
) -> None:
    calculator = RewardCalculator(RewardConfig(unnecessary_brake_lookahead_steps=2))
    safe = make_safe_braking_context(executed_action=DrivingAction.STOP)
    assert calculator.calculate(safe).components["unnecessary_brake_penalty"] == 0.0

    assert calculator.calculate(dangerous_context).components["unnecessary_brake_penalty"] == 0.0
    after_danger = calculator.calculate(safe).components["unnecessary_brake_penalty"]

    assert after_danger == 0.0


def test_unnecessary_brake_penalty_scales_with_action_index() -> None:
    calculator = RewardCalculator(RewardConfig(unnecessary_brake_lookahead_steps=1))

    result = calculator.calculate(
        make_safe_braking_context(executed_action=DrivingAction.PREPARE_STOP)
    )

    assert result.components["unnecessary_brake_penalty"] == -0.4


@pytest.mark.parametrize(("speed_mps", "expected"), [(0.1, -0.25), (0.11, 0.0)])
def test_standstill_penalty_uses_post_step_speed_and_elapsed_time(
    speed_mps: float, expected: float
) -> None:
    context = make_context()
    previous = replace(context.previous_snapshot, sim_time_s=1.0)
    next_snapshot = replace(
        context.next_snapshot,
        sim_time_s=1.5,
        ego=replace(context.next_snapshot.ego, speed_mps=speed_mps),
    )

    result = calculate(replace(context, previous_snapshot=previous, next_snapshot=next_snapshot))

    assert result.components["standstill_penalty"] == pytest.approx(expected)


@pytest.mark.parametrize(("intervened", "expected"), [(True, -2.0), (False, 0.0)])
def test_shield_intervention_penalty_requires_actual_intervention(
    intervened: bool, expected: float
) -> None:
    result = calculate(make_context(shield_intervened=intervened))

    assert result.components["shield_intervention_penalty"] == expected


def test_non_finite_computed_output_is_rejected() -> None:
    config = RewardConfig(progress_per_meter=1e308)

    with pytest.raises(ValueError, match="finite"):
        calculate(make_context(), config)


def test_reward_result_copies_mutable_component_mapping() -> None:
    source = {"progress_reward": 1.0}

    result = RewardResult(total=1.0, components=source)
    source["progress_reward"] = 2.0

    assert result.components == {"progress_reward": 1.0}
    with pytest.raises(FrozenInstanceError):
        result.total = 2.0  # type: ignore[misc]
