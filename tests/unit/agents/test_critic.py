import math

from mad_driving.agents.critic import CriticAgent
from mad_driving.config.models import CriticAgentConfig
from mad_driving.interfaces import RiskClaim
from tests.unit.agents.factories import make_snapshot


def make_claim(
    agent_id: str = "nominal",
    *,
    claim_id: str | None = None,
    severity: float = 0.4,
    stopping_margin_m: float | None = 1.0,
    recommended_max_speed_mps: float = 10.0,
    hard_stop_required: bool = False,
    confidence: float = 1.0,
    evidence: tuple[str, ...] = ("observed_state",),
    valid_until_step: int = 1,
) -> RiskClaim:
    return RiskClaim(
        claim_id=claim_id or f"{agent_id}:1:none:test",
        agent_id=agent_id,
        event_type="test",
        target_actor_id=None,
        probability=0.5,
        confidence=confidence,
        severity=severity,
        time_horizon_s=1.0,
        min_ttc_s=None,
        stopping_margin_m=stopping_margin_m,
        recommended_max_speed_mps=recommended_max_speed_mps,
        hard_stop_required=hard_stop_required,
        evidence=evidence,
        assumptions=(),
        valid_until_step=valid_until_step,
    )


def critic() -> CriticAgent:
    return CriticAgent(CriticAgentConfig())


def test_critic_finds_nominal_hazard_disagreement() -> None:
    claims = (
        make_claim("nominal", severity=0.1),
        make_claim("hazard", stopping_margin_m=-1.0),
    )

    review = critic().review(make_snapshot(), claims)

    assert review.reasons == ("nominal_hazard_disagreement",)


def test_critic_finds_occlusion_underestimation() -> None:
    review = critic().review(
        make_snapshot(occlusion_present=True),
        (make_claim("nominal", severity=0.1),),
    )

    assert review.reasons == ("occlusion_underestimated",)


def test_critic_finds_hard_stop_conflict() -> None:
    claims = (
        make_claim("nominal", recommended_max_speed_mps=4.0),
        make_claim("rule", recommended_max_speed_mps=0.0, hard_stop_required=True),
    )

    review = critic().review(make_snapshot(), claims)

    assert review.reasons == ("hard_stop_conflict",)


def test_critic_finds_speed_recommendation_spread() -> None:
    claims = (
        make_claim("nominal", recommended_max_speed_mps=4.0),
        make_claim("hazard", recommended_max_speed_mps=10.0),
    )

    review = critic().review(make_snapshot(), claims)

    assert review.reasons == ("speed_recommendation_spread",)


def test_critic_finds_expired_claim_but_accepts_equal_boundary() -> None:
    expired = critic().review(
        make_snapshot(step_index=2),
        (make_claim(valid_until_step=1),),
    )
    current = critic().review(
        make_snapshot(step_index=2),
        (make_claim(valid_until_step=2),),
    )

    assert expired.reasons == ("claim_expired",)
    assert current.reasons == ()


def test_critic_finds_low_confidence_definitive_claim_at_half_speed() -> None:
    review = critic().review(
        make_snapshot(speed_limit_mps=10.0),
        (make_claim(confidence=0.4, recommended_max_speed_mps=5.0),),
    )

    assert review.reasons == ("low_confidence_definitive",)


def test_low_confidence_threshold_equality_does_not_trigger() -> None:
    review = critic().review(
        make_snapshot(speed_limit_mps=10.0),
        (make_claim(confidence=0.5, recommended_max_speed_mps=5.0),),
    )

    assert review.reasons == ()


def test_critic_finds_missing_evidence() -> None:
    review = critic().review(make_snapshot(), (make_claim(evidence=()),))

    assert review.reasons == ("missing_evidence",)


def test_invalid_claim_never_enters_review_arithmetic() -> None:
    claim = make_claim()
    object.__setattr__(claim, "severity", math.nan)

    review = critic().review(make_snapshot(), (claim,))

    assert review.max_severity == 1.0
    assert review.reasons == ("invalid_claim",)
    assert review.challenged_claim_ids == (claim.claim_id,)


def test_critic_finds_all_eight_rules_in_fixed_order() -> None:
    nominal = make_claim(
        "nominal",
        severity=0.1,
        recommended_max_speed_mps=10.0,
        evidence=(),
        valid_until_step=0,
    )
    hazard = make_claim(
        "hazard",
        stopping_margin_m=-1.0,
        recommended_max_speed_mps=5.0,
        confidence=0.4,
    )
    rule = make_claim("rule", recommended_max_speed_mps=0.0, hard_stop_required=True)
    invalid = make_claim("external", claim_id="external:1:none:invalid")
    object.__setattr__(invalid, "severity", math.inf)

    review = critic().review(
        make_snapshot(
            step_index=1,
            speed_limit_mps=10.0,
            occlusion_present=True,
        ),
        (nominal, hazard, rule, invalid),
    )

    assert review.reasons == (
        "nominal_hazard_disagreement",
        "occlusion_underestimated",
        "hard_stop_conflict",
        "speed_recommendation_spread",
        "claim_expired",
        "low_confidence_definitive",
        "missing_evidence",
        "invalid_claim",
    )
    assert review.conflict_score == 1.0
    assert review.max_severity == 1.0


def test_challenged_claim_ids_are_input_ordered_and_duplicate_free() -> None:
    first = make_claim("z", claim_id="shared", valid_until_step=0)
    second = make_claim("a", claim_id="shared", valid_until_step=0)

    review = critic().review(make_snapshot(), (first, second))

    assert review.challenged_claim_ids == ("shared",)


def test_supported_agents_are_sorted_and_empty_review_is_neutral() -> None:
    supported = critic().review(
        make_snapshot(),
        (make_claim("z"), make_claim("a")),
    )
    empty = critic().review(make_snapshot(), ())

    assert supported.supported_agent_ids == ("a", "z")
    assert supported.challenged_claim_ids == ()
    assert supported.max_severity == 0.4
    assert empty.conflict_score == 0.0
    assert empty.unresolved_conflict is False
    assert empty.max_severity == 0.0
    assert empty.supported_agent_ids == ()
