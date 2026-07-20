import math

from mad_driving.agents.hazard import HazardAgent
from mad_driving.config.models import HazardAgentConfig
from tests.unit.agents.factories import make_actor, make_snapshot


def close_lead_snapshot():
    return make_snapshot(
        ego_speed_mps=10.0,
        actors=(
            make_actor(
                "lead",
                longitudinal_m=10.0,
                longitudinal_speed_mps=2.0,
                same_lane=True,
            ),
        ),
    )


def test_hazard_reports_negative_margin_for_close_braking_lead() -> None:
    snapshot = close_lead_snapshot()

    claim = HazardAgent(HazardAgentConfig()).analyze(snapshot)

    assert claim.agent_id == "hazard"
    assert claim.event_type == "hazard_lead_braking"
    assert claim.target_actor_id == "lead"
    assert claim.stopping_margin_m is not None and claim.stopping_margin_m < 0.0
    assert claim.severity > 0.5
    assert claim.recommended_max_speed_mps < snapshot.ego.speed_mps
    assert claim.hard_stop_required is False
    assert claim.evidence


def test_hazard_positive_margin_is_advisory_and_below_half_severity() -> None:
    actor = make_actor("far-lead", longitudinal_m=100.0, longitudinal_speed_mps=10.0)

    claim = HazardAgent(HazardAgentConfig()).analyze(
        make_snapshot(actors=(actor,))
    )

    assert claim.stopping_margin_m is not None and claim.stopping_margin_m > 0.0
    assert 0.0 <= claim.severity < 0.5
    assert claim.recommended_max_speed_mps == 15.0


def test_hazard_evaluates_crossing_actor_arrival() -> None:
    actor = make_actor(
        "crossing",
        actor_type="crossing_actor",
        longitudinal_m=10.0,
        lateral_m=4.0,
        longitudinal_speed_mps=0.0,
        lateral_speed_mps=-2.0,
        same_lane=False,
    )
    snapshot = make_snapshot(
        actors=(actor,),
        distance_to_conflict_point_m=10.0,
    )

    claim = HazardAgent(HazardAgentConfig()).analyze(snapshot)

    assert claim.event_type == "hazard_crossing"
    assert claim.target_actor_id == "crossing"
    assert claim.min_ttc_s == 1.0
    assert claim.stopping_margin_m is not None and claim.stopping_margin_m < 0.0


def test_hazard_ignores_crossing_actor_that_cannot_reach_conflict() -> None:
    actor = make_actor(
        "far-crossing",
        actor_type="crossing_actor",
        longitudinal_m=10.0,
        lateral_m=100.0,
        same_lane=False,
    )

    claim = HazardAgent(HazardAgentConfig()).analyze(
        make_snapshot(actors=(actor,), distance_to_conflict_point_m=10.0)
    )

    assert claim.event_type == "no_hazard"


def test_hazard_uses_observed_crossing_speed_above_configured_maximum() -> None:
    actor = make_actor(
        "fast-crossing",
        actor_type="crossing_actor",
        longitudinal_m=10.0,
        lateral_m=12.0,
        lateral_speed_mps=-12.0,
        same_lane=False,
    )

    claim = HazardAgent(HazardAgentConfig()).analyze(
        make_snapshot(actors=(actor,), distance_to_conflict_point_m=10.0)
    )

    assert claim.target_actor_id == "fast-crossing"
    assert claim.event_type == "hazard_crossing"


def test_hazard_creates_virtual_occlusion_claim() -> None:
    snapshot = make_snapshot(
        occlusion_present=True,
        distance_to_conflict_point_m=10.0,
    )

    claim = HazardAgent(HazardAgentConfig()).analyze(snapshot)

    assert claim.target_actor_id is None
    assert claim.event_type == "occlusion_hazard"
    assert claim.min_ttc_s == 1.0
    assert claim.stopping_margin_m is not None


def test_occlusion_without_conflict_distance_uses_crawl_speed() -> None:
    claim = HazardAgent(HazardAgentConfig()).analyze(
        make_snapshot(occlusion_present=True, distance_to_conflict_point_m=None)
    )

    assert claim.stopping_margin_m is None
    assert claim.severity == 0.7
    assert claim.recommended_max_speed_mps == 2.0
    assert claim.assumptions == ("conflict_distance_unavailable",)


def test_hazard_returns_neutral_claim_without_candidates() -> None:
    claim = HazardAgent(HazardAgentConfig()).analyze(make_snapshot())

    assert claim.claim_id == "hazard:1:none:no_hazard"
    assert claim.severity == 0.0


def test_hazard_uses_actor_id_as_final_tie_break() -> None:
    actors = (
        make_actor("b", longitudinal_m=10.0, longitudinal_speed_mps=2.0),
        make_actor("a", longitudinal_m=10.0, longitudinal_speed_mps=2.0),
    )

    claim = HazardAgent(HazardAgentConfig()).analyze(
        make_snapshot(actors=actors)
    )

    assert claim.target_actor_id == "a"


def test_hazard_is_exactly_deterministic_and_finite() -> None:
    agent = HazardAgent(HazardAgentConfig())
    snapshot = close_lead_snapshot()

    first = agent.analyze(snapshot)
    second = agent.analyze(snapshot)

    assert first == second
    assert math.isfinite(first.severity)
    assert math.isfinite(first.recommended_max_speed_mps)
    assert first.stopping_margin_m is not None and math.isfinite(first.stopping_margin_m)
