import math

import pytest

from mad_driving.config.models import ShieldConfig
from mad_driving.control import DrivingAction
from mad_driving.interfaces import RiskClaim, ShieldResult
from mad_driving.safety import SafetyShield
from tests.unit.agents.factories import make_claim, make_ego, make_frame, make_snapshot


def complete_claims(**hazard_overrides: object) -> tuple[RiskClaim, ...]:
    return (
        make_claim("nominal"),
        make_claim("hazard", **hazard_overrides),
        make_claim("rule"),
    )


def make_shield_observation(**overrides: object):
    values = dict(overrides)
    speed_limit_mps = float(values.pop("speed_limit_mps", 15.0))
    return make_snapshot(ego=make_ego(speed_limit_mps=speed_limit_mps), **values)


def test_enforce_never_relaxes_requested_action() -> None:
    shield = SafetyShield(ShieldConfig(mode="enforce"))
    for requested in DrivingAction:
        result = shield.filter(requested, make_snapshot(), complete_claims())
        assert result.executed_action >= requested


def test_modes_distinguish_candidate_from_real_intervention() -> None:
    claims = complete_claims(min_ttc_s=0.5)
    off = SafetyShield(ShieldConfig(mode="off")).filter(DrivingAction.KEEP, make_snapshot(), claims)
    monitor = SafetyShield(ShieldConfig(mode="monitor")).filter(
        DrivingAction.KEEP, make_snapshot(), claims
    )
    enforce = SafetyShield(ShieldConfig(mode="enforce")).filter(
        DrivingAction.KEEP, make_snapshot(), claims
    )
    assert off.required_action is DrivingAction.KEEP
    assert off.executed_action is DrivingAction.KEEP
    assert off.reasons == ()
    assert monitor.required_action is DrivingAction.STOP
    assert monitor.intervention_required is True
    assert monitor.intervened is False
    assert enforce.executed_action is DrivingAction.STOP
    assert enforce.intervened is True


def test_privileged_labels_cannot_change_an_observation_only_shield_decision() -> None:
    observation = make_snapshot()
    safe_frame = make_frame(observation=observation)
    labeled_frame = make_frame(
        observation=observation,
        collision_occurred=True,
        off_road=True,
        scenario_success=True,
        scenario_failure=True,
    )
    shield = SafetyShield(ShieldConfig())

    safe_result = shield.filter(DrivingAction.KEEP, safe_frame.observation, complete_claims())
    labeled_result = shield.filter(DrivingAction.KEEP, labeled_frame.observation, complete_claims())

    assert labeled_result == safe_result
    assert labeled_result.reasons == ()


def test_all_reasons_have_fixed_duplicate_free_order() -> None:
    claims = (
        make_claim(
            "nominal",
            min_ttc_s=0.5,
            recommended_max_speed_mps=0.0,
            hard_stop_required=True,
        ),
    )
    result = SafetyShield(ShieldConfig()).filter(
        DrivingAction.KEEP,
        make_snapshot(),
        claims,
    )
    assert result.reasons == (
        "hard_stop_required",
        "multiple_agents_missing",
        "imminent_ttc",
        "claim_speed_limit",
    )


@pytest.mark.parametrize(
    ("claims", "snapshot_overrides", "reason", "required"),
    [
        (complete_claims(hard_stop_required=True), {}, "hard_stop_required", 3),
        (complete_claims(min_ttc_s=1.0), {}, "imminent_ttc", 3),
        (complete_claims(min_ttc_s=3.0), {}, "caution_ttc", 2),
        (
            complete_claims(stopping_margin_m=-0.001),
            {},
            "negative_stopping_margin",
            3,
        ),
        (complete_claims(stopping_margin_m=0.0), {}, "low_stopping_margin", 2),
        (
            complete_claims(recommended_max_speed_mps=10.0),
            {"speed_limit_mps": 20.0},
            "claim_speed_limit",
            1,
        ),
        (
            (make_claim("nominal"), make_claim("rule")),
            {},
            "agent_missing",
            2,
        ),
        ((make_claim("nominal"),), {}, "multiple_agents_missing", 3),
    ],
)
def test_individual_safety_reasons(
    claims: tuple[RiskClaim, ...],
    snapshot_overrides: dict[str, object],
    reason: str,
    required: int,
) -> None:
    result = SafetyShield(ShieldConfig()).filter(
        DrivingAction.KEEP,
        make_shield_observation(**snapshot_overrides),
        claims,
    )
    assert reason in result.reasons
    assert result.required_action is DrivingAction(required)


def test_margin_boundaries_are_explicit() -> None:
    shield = SafetyShield(ShieldConfig())
    zero = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        complete_claims(stopping_margin_m=0.0),
    )
    negative = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        complete_claims(stopping_margin_m=-0.001),
    )
    clear = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        complete_claims(stopping_margin_m=5.0),
    )
    assert "negative_stopping_margin" not in zero.reasons
    assert "low_stopping_margin" in zero.reasons
    assert "negative_stopping_margin" in negative.reasons
    assert "low_stopping_margin" not in clear.reasons
    assert negative.executed_action >= zero.executed_action >= clear.executed_action


def test_ttc_boundaries_are_explicit() -> None:
    shield = SafetyShield(ShieldConfig())
    imminent = shield.filter(DrivingAction.KEEP, make_snapshot(), complete_claims(min_ttc_s=1.0))
    caution = shield.filter(DrivingAction.KEEP, make_snapshot(), complete_claims(min_ttc_s=3.0))
    clear = shield.filter(DrivingAction.KEEP, make_snapshot(), complete_claims(min_ttc_s=3.001))
    assert imminent.reasons == ("imminent_ttc",)
    assert caution.reasons == ("caution_ttc",)
    assert clear.reasons == ()


def test_invalid_claim_is_stopped_before_arithmetic() -> None:
    claim = make_claim()
    object.__setattr__(claim, "min_ttc_s", math.nan)
    result = SafetyShield(ShieldConfig()).filter(DrivingAction.KEEP, make_snapshot(), (claim,))
    assert result.executed_action is DrivingAction.STOP
    assert result.reasons == ("invalid_input", "multiple_agents_missing")


def test_invalid_snapshot_is_stopped_before_speed_mapping() -> None:
    snapshot = make_snapshot()
    object.__setattr__(snapshot.ego, "speed_limit_mps", math.nan)
    result = SafetyShield(ShieldConfig()).filter(DrivingAction.KEEP, snapshot, complete_claims())
    assert result.executed_action is DrivingAction.STOP
    assert result.reasons == ("invalid_input",)


def test_repeated_input_is_exactly_deterministic() -> None:
    shield = SafetyShield(ShieldConfig())
    arguments = (
        DrivingAction.KEEP,
        make_snapshot(),
        complete_claims(min_ttc_s=2.0, stopping_margin_m=1.0),
    )
    assert shield.filter(*arguments) == shield.filter(*arguments)


def test_generated_ttc_and_margin_matrix_is_monotone() -> None:
    shield = SafetyShield(ShieldConfig())
    ttc_values = (None, 5.0, 3.0, 1.0, 0.5)
    margin_values = (None, 10.0, 5.0, 0.0, -0.1)

    for requested in DrivingAction:
        for margin in margin_values:
            previous = requested
            for ttc in ttc_values:
                result = shield.filter(
                    requested,
                    make_snapshot(),
                    complete_claims(min_ttc_s=ttc, stopping_margin_m=margin),
                )
                assert result.executed_action >= previous
                previous = result.executed_action

        for ttc in ttc_values:
            previous = requested
            for margin in margin_values:
                result = shield.filter(
                    requested,
                    make_snapshot(),
                    complete_claims(min_ttc_s=ttc, stopping_margin_m=margin),
                )
                assert result.executed_action >= previous
                previous = result.executed_action


@pytest.mark.parametrize(
    ("one_missing", "multiple_missing"),
    [(one, multiple) for one in range(4) for multiple in range(one, 4)],
)
def test_missing_agent_count_is_monotone_for_every_valid_config(
    one_missing: int,
    multiple_missing: int,
) -> None:
    shield = SafetyShield(
        ShieldConfig(
            missing_agent_action=one_missing,
            multiple_missing_action=multiple_missing,
        )
    )
    complete = shield.filter(DrivingAction.KEEP, make_snapshot(), complete_claims())
    one = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        (make_claim("nominal"), make_claim("rule")),
    )
    multiple = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        (make_claim("nominal"),),
    )
    assert multiple.executed_action >= one.executed_action >= complete.executed_action


def test_shield_result_rejects_inconsistent_flags_and_duplicate_reasons() -> None:
    with pytest.raises(ValueError, match="intervention_required"):
        ShieldResult(
            DrivingAction.KEEP,
            DrivingAction.STOP,
            DrivingAction.STOP,
            False,
            True,
            (),
        )
    with pytest.raises(ValueError, match="duplicate-free"):
        ShieldResult(
            DrivingAction.KEEP,
            DrivingAction.KEEP,
            DrivingAction.KEEP,
            False,
            False,
            ("same", "same"),
        )
