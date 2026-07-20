import math

import pytest

from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.config.models import NominalAgentConfig
from tests.unit.agents.factories import make_actor, make_snapshot


def test_nominal_selects_closing_same_lane_actor() -> None:
    snapshot = make_snapshot(
        ego_speed_mps=10.0,
        actors=(
            make_actor(
                "lead",
                longitudinal_m=15.0,
                longitudinal_speed_mps=2.0,
                same_lane=True,
            ),
        ),
    )

    (claim,) = NominalMotionAgent(NominalAgentConfig()).analyze(snapshot)

    assert claim.agent_id == "nominal"
    assert claim.event_type == "nominal_lead"
    assert claim.target_actor_id == "lead"
    assert claim.min_ttc_s is not None
    assert claim.probability is not None and claim.probability > 0.0
    assert claim.severity == claim.probability
    assert claim.recommended_max_speed_mps < snapshot.ego.speed_limit_mps
    assert claim.evidence
    assert claim.assumptions == ("constant_acceleration",)
    assert claim.time_horizon_s == 5.0


def test_nominal_detects_predicted_cut_in() -> None:
    actor = make_actor(
        "cut-in",
        longitudinal_m=12.0,
        lateral_m=3.0,
        longitudinal_speed_mps=8.0,
        lateral_speed_mps=-1.0,
        same_lane=False,
    )

    (claim,) = NominalMotionAgent(NominalAgentConfig()).analyze(make_snapshot(actors=(actor,)))

    assert claim.target_actor_id == "cut-in"
    assert claim.event_type == "nominal_cut_in"


def test_nominal_evaluates_crossing_actor() -> None:
    actor = make_actor(
        "crossing",
        actor_type="crossing_actor",
        longitudinal_m=8.0,
        lateral_m=4.0,
        longitudinal_speed_mps=0.0,
        lateral_speed_mps=-2.0,
        same_lane=False,
    )

    (claim,) = NominalMotionAgent(NominalAgentConfig()).analyze(make_snapshot(actors=(actor,)))

    assert claim.target_actor_id == "crossing"
    assert claim.event_type == "nominal_crossing"


def test_nominal_ignores_same_lane_actor_behind() -> None:
    (claim,) = NominalMotionAgent(NominalAgentConfig()).analyze(
        make_snapshot(actors=(make_actor("behind", longitudinal_m=-5.0),))
    )

    assert claim.event_type == "no_hazard"
    assert claim.target_actor_id is None
    assert claim.severity == 0.0


def test_nominal_returns_neutral_claim_without_actors() -> None:
    (claim,) = NominalMotionAgent(NominalAgentConfig()).analyze(make_snapshot())

    assert claim.claim_id == "nominal:1:none:no_hazard"
    assert claim.recommended_max_speed_mps == 15.0


def test_nominal_returns_up_to_three_safety_ordered_claims() -> None:
    actors = (
        make_actor("d", longitudinal_m=15.0, longitudinal_speed_mps=2.0),
        make_actor("c", longitudinal_m=15.0, longitudinal_speed_mps=2.0),
        make_actor("a", longitudinal_m=15.0, longitudinal_speed_mps=2.0),
        make_actor("b", longitudinal_m=15.0, longitudinal_speed_mps=2.0),
    )

    claims = NominalMotionAgent(NominalAgentConfig()).analyze(make_snapshot(actors=actors))

    assert tuple(claim.target_actor_id for claim in claims) == ("a", "b", "c")
    assert len(claims) == 3


def test_nominal_is_exactly_deterministic_and_finite() -> None:
    agent = NominalMotionAgent(NominalAgentConfig())
    snapshot = make_snapshot(
        actors=(make_actor("lead", longitudinal_m=14.0, longitudinal_speed_mps=4.0),)
    )

    first = agent.analyze(snapshot)
    second = agent.analyze(snapshot)

    assert first == second
    assert all(
        claim.probability is not None and math.isfinite(claim.probability) for claim in first
    )
    assert all(math.isfinite(claim.severity) for claim in first)
    assert all(math.isfinite(claim.recommended_max_speed_mps) for claim in first)


def test_nominal_uses_configured_horizon_and_step() -> None:
    agent = NominalMotionAgent(NominalAgentConfig(horizon_s=1.0, time_step_s=0.5))
    actor = make_actor("lead", longitudinal_m=9.0, longitudinal_speed_mps=0.0)

    (claim,) = agent.analyze(make_snapshot(actors=(actor,)))

    assert claim.time_horizon_s == 1.0
    assert claim.min_ttc_s in (0.5, 1.0)


def test_nominal_probability_remains_in_range_for_extreme_closing_speed() -> None:
    actor = make_actor("fast-close", longitudinal_m=6.0, longitudinal_speed_mps=-100.0)

    (claim,) = NominalMotionAgent(NominalAgentConfig()).analyze(make_snapshot(actors=(actor,)))

    assert claim.probability is not None
    assert 0.0 <= claim.probability <= 1.0
    assert claim.severity == pytest.approx(claim.probability)
