import math

import pytest

from mad_driving.config.models import CoordinatorConfig
from mad_driving.control import DrivingAction
from mad_driving.coordinator import RuleBasedCoordinator
from mad_driving.interfaces import CriticReview
from tests.unit.agents.factories import make_claim, make_snapshot


def review(*, unresolved: bool = False, max_severity: float = 0.0) -> CriticReview:
    return CriticReview(
        conflict_score=1.0 if unresolved else 0.0,
        unresolved_conflict=unresolved,
        max_severity=max_severity,
        supported_agent_ids=(),
        challenged_claim_ids=(),
        reasons=("test_conflict",) if unresolved else (),
    )


def test_coordinator_uses_most_restrictive_claim_speed() -> None:
    action = RuleBasedCoordinator(CoordinatorConfig()).decide(
        make_snapshot(speed_limit_mps=20.0),
        (
            make_claim("nominal", recommended_max_speed_mps=20.0),
            make_claim("hazard", recommended_max_speed_mps=5.0),
            make_claim("rule", recommended_max_speed_mps=20.0),
        ),
        review(),
    )
    assert action is DrivingAction.PREPARE_STOP


def test_hard_stop_overrides_every_other_candidate() -> None:
    action = RuleBasedCoordinator(CoordinatorConfig()).decide(
        make_snapshot(),
        (make_claim("rule", hard_stop_required=True),),
        review(),
    )
    assert action is DrivingAction.STOP


def test_any_required_agent_missing_applies_prepare_stop_floor() -> None:
    action = RuleBasedCoordinator(CoordinatorConfig()).decide(
        make_snapshot(),
        (make_claim("nominal"), make_claim("rule")),
        review(),
    )
    assert action is DrivingAction.PREPARE_STOP


def test_adding_a_more_hazardous_claim_never_weakens_action() -> None:
    coordinator = RuleBasedCoordinator(CoordinatorConfig())
    baseline_claims = (
        make_claim("nominal", recommended_max_speed_mps=20.0),
        make_claim("hazard", recommended_max_speed_mps=20.0),
        make_claim("rule", recommended_max_speed_mps=20.0),
    )
    augmented_claims = (
        *baseline_claims,
        make_claim(
            "hazard",
            claim_id="hazard:2:none:test",
            recommended_max_speed_mps=0.0,
            hard_stop_required=True,
        ),
    )

    baseline = coordinator.decide(make_snapshot(speed_limit_mps=20.0), baseline_claims, review())
    augmented = coordinator.decide(make_snapshot(speed_limit_mps=20.0), augmented_claims, review())

    assert augmented >= baseline
    assert augmented is DrivingAction.STOP


def test_conflict_and_severity_apply_minimum_actions() -> None:
    coordinator = RuleBasedCoordinator(CoordinatorConfig())
    claims = (
        make_claim("nominal"),
        make_claim("hazard"),
        make_claim("rule"),
    )
    assert (
        coordinator.decide(make_snapshot(), claims, review(unresolved=True)) is DrivingAction.SLOW
    )
    assert (
        coordinator.decide(
            make_snapshot(),
            claims,
            review(max_severity=0.75),
        )
        is DrivingAction.PREPARE_STOP
    )


def test_identical_input_is_exactly_deterministic() -> None:
    coordinator = RuleBasedCoordinator(CoordinatorConfig())
    arguments = (make_snapshot(), (make_claim(),), review())
    assert coordinator.decide(*arguments) == coordinator.decide(*arguments)


def test_corrupted_claim_returns_prepare_stop() -> None:
    claims = [make_claim("nominal"), make_claim("hazard"), make_claim("rule")]
    object.__setattr__(claims[1], "recommended_max_speed_mps", math.nan)
    assert (
        RuleBasedCoordinator(CoordinatorConfig()).decide(make_snapshot(), tuple(claims), review())
        is DrivingAction.PREPARE_STOP
    )


def test_corrupted_snapshot_returns_prepare_stop() -> None:
    snapshot = make_snapshot()
    object.__setattr__(snapshot.ego, "speed_limit_mps", math.nan)
    assert (
        RuleBasedCoordinator(CoordinatorConfig()).decide(
            snapshot,
            (make_claim("nominal"), make_claim("hazard"), make_claim("rule")),
            review(),
        )
        is DrivingAction.PREPARE_STOP
    )


def test_corrupted_review_returns_prepare_stop() -> None:
    corrupted_review = review()
    object.__setattr__(corrupted_review, "max_severity", math.nan)
    assert (
        RuleBasedCoordinator(CoordinatorConfig()).decide(
            make_snapshot(),
            (make_claim("nominal"), make_claim("hazard"), make_claim("rule")),
            corrupted_review,
        )
        is DrivingAction.PREPARE_STOP
    )


@pytest.mark.parametrize(
    "invalid_agent_id",
    (math.nan, math.inf, -math.inf, 1, True, "", "unknown"),
)
def test_invalid_agent_id_is_rejected_before_action_selection(
    invalid_agent_id: object,
) -> None:
    malformed_claim = make_claim("hazard")
    object.__setattr__(malformed_claim, "agent_id", invalid_agent_id)

    with pytest.raises(ValueError, match="invalid claim agent_id"):
        RuleBasedCoordinator(CoordinatorConfig()).decide(
            make_snapshot(),
            (make_claim("nominal"), make_claim("rule"), malformed_claim),
            review(),
        )
