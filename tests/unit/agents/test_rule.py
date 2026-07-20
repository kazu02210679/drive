from mad_driving.agents.rule import RuleAgent
from mad_driving.config.models import RuleAgentConfig
from mad_driving.interfaces import RoadContext
from tests.unit.agents.factories import make_snapshot


def test_rule_hard_stops_for_observable_constraints() -> None:
    (claim,) = RuleAgent(RuleAgentConfig()).analyze(
        make_snapshot(
            road_context=RoadContext(
                stop_required=True,
                distance_to_conflict_point_m=None,
                intersection_entry_prohibited=False,
            )
        )
    )

    assert claim.event_type == "scenario_stop"
    assert claim.hard_stop_required is True
    assert claim.recommended_max_speed_mps == 0.0
    assert claim.probability == 1.0
    assert claim.severity == 1.0


def test_intersection_stop_has_priority_over_scenario_stop() -> None:
    (claim,) = RuleAgent(RuleAgentConfig()).analyze(
        make_snapshot(
            road_context=RoadContext(
                stop_required=True,
                distance_to_conflict_point_m=None,
                intersection_entry_prohibited=True,
            )
        )
    )

    assert claim.event_type == "intersection_stop"


def test_normal_rule_claim_recommends_speed_limit() -> None:
    (claim,) = RuleAgent(RuleAgentConfig()).analyze(
        make_snapshot(ego_speed_mps=10.0, speed_limit_mps=12.0)
    )

    assert claim.event_type == "speed_limit"
    assert claim.recommended_max_speed_mps == 12.0
    assert claim.hard_stop_required is False
    assert claim.probability == 0.0
    assert claim.severity == 0.0


def test_overspeed_severity_is_normalized_and_clipped() -> None:
    agent = RuleAgent(RuleAgentConfig())

    (moderate,) = agent.analyze(make_snapshot(ego_speed_mps=18.0, speed_limit_mps=12.0))
    (extreme,) = agent.analyze(make_snapshot(ego_speed_mps=30.0, speed_limit_mps=10.0))

    assert moderate.probability == 1.0
    assert moderate.severity == 0.5
    assert extreme.severity == 1.0


def test_rule_claim_is_exactly_deterministic() -> None:
    agent = RuleAgent(RuleAgentConfig())
    snapshot = make_snapshot(ego_speed_mps=16.0, speed_limit_mps=12.0)

    assert agent.analyze(snapshot) == agent.analyze(snapshot)
