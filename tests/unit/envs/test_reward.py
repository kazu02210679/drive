"""Tests for the frame-based, ten-component transition reward."""

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from mad_driving.config.models import RewardConfig
from mad_driving.control.actions import DrivingAction
from mad_driving.envs.reward import RewardCalculator, RewardContext, RewardResult
from tests.unit.agents.factories import make_analysis, make_claim, make_frame, make_snapshot

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
    previous_observation = make_snapshot(step_index=10, sim_time_s=1.0)
    next_observation = make_snapshot(step_index=11, sim_time_s=2.0, ego_speed_mps=1.0)
    next_observation = replace(
        next_observation,
        ego=replace(next_observation.ego, position_xy_m=(10.0, 0.0)),
    )
    analysis = overrides.pop(
        "analysis",
        make_analysis(claims=(make_claim("nominal"), make_claim("hazard"), make_claim("rule"))),
    )
    values: dict[str, object] = {
        "previous_frame": make_frame(observation=previous_observation),
        "next_frame": make_frame(observation=next_observation),
        "previous_analysis": analysis,
        "next_analysis": analysis,
        "executed_action": DrivingAction.KEEP,
        "shield_intervened": False,
        "decision_interval_s": 0.1,
    }
    values.update(overrides)
    return RewardContext(**values)  # type: ignore[arg-type]


def calculate(context: RewardContext, config: RewardConfig | None = None) -> RewardResult:
    return RewardCalculator(config or RewardConfig()).calculate(context)


def make_safe_braking_context(**overrides: object) -> RewardContext:
    values: dict[str, object] = {
        "analysis": make_analysis(
            claims=(
                make_claim("nominal"),
                make_claim("hazard", severity=0.1, min_ttc_s=6.0),
                make_claim("rule", hard_stop_required=False),
            )
        ),
        "executed_action": DrivingAction.SLOW,
    }
    values.update(overrides)
    return make_context(**values)


def test_reward_components_are_signed_finite_and_sum_to_total() -> None:
    result = calculate(make_context())

    assert tuple(result.components) == EXPECTED_COMPONENT_KEYS
    assert result.total == sum(result.components.values())
    assert result.components["progress_reward"] > 0.0
    assert result.components["arrival_reward"] >= 0.0
    assert all(value <= 0.0 for key, value in result.components.items() if "penalty" in key)
    assert math.isfinite(result.total)
    assert all(math.isfinite(value) for value in result.components.values())


@pytest.mark.parametrize(
    ("displacement", "expected"),
    [((3.0, 4.0), 0.4), ((3.0, -4.0), 0.0)],
)
def test_progress_uses_previous_heading_and_frame_transition(
    displacement: tuple[float, float], expected: float
) -> None:
    context = make_context()
    previous = replace(
        context.previous_frame.observation,
        ego=replace(context.previous_frame.observation.ego, heading_rad=math.pi / 2.0),
    )
    next_observation = replace(
        context.next_frame.observation,
        ego=replace(
            context.next_frame.observation.ego,
            position_xy_m=displacement,
            speed_mps=1.0,
        ),
    )

    result = calculate(
        replace(
            context,
            previous_frame=make_frame(observation=previous),
            next_frame=make_frame(observation=next_observation),
        )
    )

    assert result.components["progress_reward"] == pytest.approx(expected)


def test_arrival_reward_uses_next_privileged_state_once_per_episode() -> None:
    calculator = RewardCalculator(RewardConfig())
    context = make_context(next_frame=make_frame(arrived=True))

    rewards = [calculator.calculate(context).components["arrival_reward"] for _ in range(2)]
    calculator.reset()
    after_reset = calculator.calculate(context).components["arrival_reward"]

    assert rewards == [100.0, 0.0]
    assert after_reset == 100.0


@pytest.mark.parametrize(
    ("collision_kind", "expected"),
    [("vehicle", -200.0), ("crossing_actor", -500.0), (None, 0.0)],
)
def test_collision_penalty_uses_next_privileged_state(
    collision_kind: str | None, expected: float
) -> None:
    result = calculate(
        make_context(next_frame=make_frame(collision_occurred=True, collision_kind=collision_kind))
    )

    assert result.components["collision_penalty"] == expected


@pytest.mark.parametrize(
    ("ttc_s", "expected"),
    [(0.0, -50.0), (1.5, -12.5), (3.0, 0.0), (4.0, 0.0)],
)
def test_near_miss_penalty_uses_current_analysis(ttc_s: float, expected: float) -> None:
    result = calculate(
        make_context(
            analysis=make_analysis(
                claims=(
                    make_claim("nominal", min_ttc_s=ttc_s),
                    make_claim("hazard", min_ttc_s=ttc_s + 1.0),
                    make_claim("rule"),
                )
            )
        )
    )

    assert result.components["near_miss_penalty"] == pytest.approx(expected)


def test_raw_out_of_road_privileged_state_drives_penalty() -> None:
    result = calculate(make_context(next_frame=make_frame(off_road=True)))

    assert result.components["offroad_penalty"] == -100.0


@pytest.mark.parametrize("source", ["claim", "stop_required", "intersection_prohibited"])
def test_rule_violation_uses_current_rule_constraint(source: str) -> None:
    context = make_context(executed_action=DrivingAction.PREPARE_STOP)
    analysis = context.previous_analysis
    previous_observation = context.previous_frame.observation
    if source == "claim":
        analysis = make_analysis(claims=(make_claim("rule", hard_stop_required=True),))
    elif source == "stop_required":
        previous_observation = replace(
            previous_observation,
            road_context=replace(previous_observation.road_context, stop_required=True),
        )
    else:
        previous_observation = replace(
            previous_observation,
            road_context=replace(
                previous_observation.road_context,
                intersection_entry_prohibited=True,
            ),
        )

    result = calculate(
        replace(
            context,
            previous_analysis=analysis,
            previous_frame=make_frame(observation=previous_observation),
        )
    )

    assert result.components["rule_violation_penalty"] == -100.0


def test_action_validity_uses_pre_action_analysis_and_observation() -> None:
    previous_analysis = make_analysis(
        claims=(
            make_claim("hazard", severity=0.9, min_ttc_s=1.0),
            make_claim("rule", hard_stop_required=True),
        )
    )
    next_analysis = make_analysis(
        claims=(
            make_claim("hazard", severity=0.0, min_ttc_s=8.0),
            make_claim("rule", hard_stop_required=False),
        )
    )

    result = calculate(
        make_context(
            previous_analysis=previous_analysis,
            next_analysis=next_analysis,
            executed_action=DrivingAction.PREPARE_STOP,
        )
    )

    assert result.components["rule_violation_penalty"] == -100.0
    assert result.components["unnecessary_brake_penalty"] == 0.0


def test_unnecessary_brake_uses_pre_action_ttc_not_braking_outcome_ttc() -> None:
    result = calculate(
        make_context(
            previous_analysis=make_analysis(
                claims=(
                    make_claim("hazard", severity=0.1, min_ttc_s=1.0),
                    make_claim("rule"),
                )
            ),
            next_analysis=make_analysis(
                claims=(
                    make_claim("hazard", severity=0.1, min_ttc_s=8.0),
                    make_claim("rule"),
                )
            ),
            executed_action=DrivingAction.SLOW,
        )
    )

    assert result.components["unnecessary_brake_penalty"] == 0.0


def test_near_miss_uses_post_action_analysis() -> None:
    result = calculate(
        make_context(
            previous_analysis=make_analysis(
                claims=(make_claim("hazard", min_ttc_s=8.0), make_claim("rule"))
            ),
            next_analysis=make_analysis(
                claims=(make_claim("hazard", min_ttc_s=0.0), make_claim("rule"))
            ),
        )
    )

    assert result.components["near_miss_penalty"] == -50.0


def test_near_miss_ignores_non_physical_rule_claim_ttc() -> None:
    result = calculate(
        make_context(
            next_analysis=make_analysis(
                claims=(
                    make_claim("nominal", min_ttc_s=8.0),
                    make_claim("hazard", min_ttc_s=8.0),
                    make_claim("rule", min_ttc_s=0.0, hard_stop_required=True),
                )
            )
        )
    )

    assert result.components["near_miss_penalty"] == 0.0


@pytest.mark.parametrize(
    ("previous_time", "next_time", "fallback_dt", "expected"),
    [(1.0, 1.5, 0.1, -0.2), (1.0, 1.0, 0.25, -0.4)],
)
def test_jerk_penalty_uses_frame_elapsed_time_or_decision_interval(
    previous_time: float, next_time: float, fallback_dt: float, expected: float
) -> None:
    context = make_context(decision_interval_s=fallback_dt)
    previous_observation = replace(
        context.previous_frame.observation,
        sim_time_s=previous_time,
        ego=replace(context.previous_frame.observation.ego, acceleration_mps2=1.0),
    )
    next_observation = replace(
        context.next_frame.observation,
        sim_time_s=next_time,
        ego=replace(context.next_frame.observation.ego, acceleration_mps2=3.0),
    )

    result = calculate(
        replace(
            context,
            previous_frame=make_frame(observation=previous_observation),
            next_frame=make_frame(observation=next_observation),
        )
    )

    assert result.components["jerk_penalty"] == pytest.approx(expected)


def test_no_actor_safe_scene_penalizes_unnecessary_stop_immediately() -> None:
    context = make_context(
        analysis=make_analysis(
            claims=(
                make_claim("nominal", min_ttc_s=None, severity=0.0),
                make_claim("hazard", min_ttc_s=None, severity=0.0),
                make_claim("rule", min_ttc_s=None, severity=0.0),
            )
        ),
        executed_action=DrivingAction.STOP,
    )

    result = RewardCalculator(RewardConfig()).calculate(context)

    assert result.components["unnecessary_brake_penalty"] == pytest.approx(-0.6)


def test_current_safe_stop_does_not_wait_for_future_steps() -> None:
    calculator = RewardCalculator(RewardConfig())
    context = make_safe_braking_context(executed_action=DrivingAction.STOP)

    first = calculator.calculate(context).components["unnecessary_brake_penalty"]
    second = calculator.calculate(context).components["unnecessary_brake_penalty"]

    assert first == pytest.approx(-0.6)
    assert second == pytest.approx(-0.6)


@pytest.mark.parametrize(
    "analysis",
    [
        make_analysis(claims=(make_claim("nominal"), make_claim("rule"))),
        make_analysis(claims=(make_claim("nominal"), make_claim("hazard"))),
        make_analysis(
            claims=(make_claim("hazard"), make_claim("rule")),
            failed_agent_ids=("hazard",),
            errors=("hazard:RuntimeError:failed",),
        ),
        make_analysis(
            claims=(make_claim("hazard"), make_claim("rule")),
            failed_agent_ids=("rule",),
            errors=("rule:RuntimeError:failed",),
        ),
    ],
    ids=["missing_hazard", "missing_rule", "failed_hazard", "failed_rule"],
)
def test_missing_or_failed_specialist_suppresses_unnecessary_brake_judgment(
    analysis: object,
) -> None:
    result = calculate(make_safe_braking_context(analysis=analysis))

    assert result.components["unnecessary_brake_penalty"] == 0.0


@pytest.mark.parametrize(
    "case",
    [
        "severity",
        "ttc",
        "rule",
        "offroad",
        "collision",
        "shield",
    ],
)
def test_current_safety_event_suppresses_unnecessary_brake_penalty(case: str) -> None:
    if case == "severity":
        context = make_safe_braking_context(
            analysis=make_analysis(
                claims=(
                    make_claim("hazard", severity=0.25, min_ttc_s=6.0),
                    make_claim("rule"),
                )
            )
        )
    elif case == "ttc":
        context = make_safe_braking_context(
            analysis=make_analysis(
                claims=(
                    make_claim("hazard", severity=0.1, min_ttc_s=4.99),
                    make_claim("rule"),
                )
            )
        )
    elif case == "rule":
        context = make_safe_braking_context(
            analysis=make_analysis(
                claims=(
                    make_claim("hazard", severity=0.1, min_ttc_s=6.0),
                    make_claim("rule", hard_stop_required=True),
                )
            )
        )
    elif case == "offroad":
        context = make_safe_braking_context(next_frame=make_frame(off_road=True))
    elif case == "collision":
        context = make_safe_braking_context(
            next_frame=make_frame(collision_occurred=True, collision_kind="vehicle")
        )
    else:
        context = make_safe_braking_context(shield_intervened=True)

    result = calculate(context)

    assert result.components["unnecessary_brake_penalty"] == 0.0


def test_safe_ttc_threshold_is_inclusive() -> None:
    config = RewardConfig()
    result = calculate(
        make_safe_braking_context(
            analysis=make_analysis(
                claims=(
                    make_claim(
                        "hazard",
                        severity=0.1,
                        min_ttc_s=config.unnecessary_brake_safe_ttc_s,
                    ),
                    make_claim("rule"),
                )
            )
        ),
        config,
    )

    assert result.components["unnecessary_brake_penalty"] == -0.2


@pytest.mark.parametrize(("speed_mps", "expected"), [(0.1, -0.25), (0.11, 0.0)])
def test_standstill_penalty_uses_next_observation_speed(speed_mps: float, expected: float) -> None:
    context = make_context()
    next_observation = replace(
        context.next_frame.observation,
        sim_time_s=1.5,
        ego=replace(context.next_frame.observation.ego, speed_mps=speed_mps),
    )

    result = calculate(replace(context, next_frame=make_frame(observation=next_observation)))

    assert result.components["standstill_penalty"] == pytest.approx(expected)


@pytest.mark.parametrize(("intervened", "expected"), [(True, -2.0), (False, 0.0)])
def test_shield_intervention_penalty_requires_actual_intervention(
    intervened: bool, expected: float
) -> None:
    result = calculate(make_context(shield_intervened=intervened))

    assert result.components["shield_intervention_penalty"] == expected


@pytest.mark.parametrize("decision_interval_s", [0.0, -0.1, math.inf, math.nan])
def test_reward_context_requires_positive_finite_decision_interval(
    decision_interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="decision_interval_s"):
        make_context(decision_interval_s=decision_interval_s)


def test_non_finite_computed_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate(make_context(), RewardConfig(progress_per_meter=1e308))


def test_reward_result_copies_mutable_component_mapping() -> None:
    source = {"progress_reward": 1.0}
    result = RewardResult(total=1.0, components=source)
    source["progress_reward"] = 2.0

    assert result.components == {"progress_reward": 1.0}
    with pytest.raises(FrozenInstanceError):
        result.total = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("total", [math.inf, math.nan])
def test_reward_result_rejects_non_finite_total(total: float) -> None:
    with pytest.raises(ValueError, match="reward total must be finite"):
        RewardResult(total=total, components={"progress_reward": 0.0})


@pytest.mark.parametrize("component", [math.inf, math.nan])
def test_reward_result_rejects_non_finite_component(component: float) -> None:
    with pytest.raises(ValueError, match="reward components must be finite"):
        RewardResult(total=0.0, components={"progress_reward": component})


def test_reward_result_rejects_mismatched_total() -> None:
    with pytest.raises(ValueError, match="reward total must equal the sum of components"):
        RewardResult(total=2.0, components={"progress_reward": 1.0})


def test_reward_result_accepts_exact_component_sum() -> None:
    result = RewardResult(
        total=3.0,
        components={"progress_reward": 1.0, "arrival_reward": 2.0},
    )

    assert result.total == sum(result.components.values())
