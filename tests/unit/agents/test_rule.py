import pytest

from mad_driving.agents.rule import RuleAgent
from mad_driving.config.models import RuleAgentConfig
from tests.unit.agents.factories import make_snapshot


@pytest.mark.parametrize(
    ("field", "event_type"),
    [
        ("collision_occurred", "collision_stop"),
        ("off_road", "off_road_stop"),
        ("intersection_entry_prohibited", "intersection_stop"),
        ("stop_required", "scenario_stop"),
    ],
)
def test_rule_hard_stops_for_explicit_constraints(
    field: str, event_type: str
) -> None:
    claim = RuleAgent(RuleAgentConfig()).analyze(make_snapshot(**{field: True}))

    assert claim.event_type == event_type
    assert claim.hard_stop_required is True
    assert claim.recommended_max_speed_mps == 0.0
    assert claim.probability == 1.0
    assert claim.severity == 1.0


def test_collision_has_priority_when_multiple_rules_apply() -> None:
    claim = RuleAgent(RuleAgentConfig()).analyze(
        make_snapshot(
            collision_occurred=True,
            off_road=True,
            intersection_entry_prohibited=True,
            stop_required=True,
        )
    )

    assert claim.event_type == "collision_stop"


def test_rule_uses_exact_fixed_priority_order() -> None:
    agent = RuleAgent(RuleAgentConfig())

    assert agent.analyze(
        make_snapshot(
            off_road=True,
            intersection_entry_prohibited=True,
            stop_required=True,
        )
    ).event_type == "off_road_stop"
    assert agent.analyze(
        make_snapshot(intersection_entry_prohibited=True, stop_required=True)
    ).event_type == "intersection_stop"


def test_normal_rule_claim_recommends_speed_limit() -> None:
    claim = RuleAgent(RuleAgentConfig()).analyze(
        make_snapshot(ego_speed_mps=10.0, speed_limit_mps=12.0)
    )

    assert claim.event_type == "speed_limit"
    assert claim.recommended_max_speed_mps == 12.0
    assert claim.hard_stop_required is False
    assert claim.probability == 0.0
    assert claim.severity == 0.0


def test_overspeed_severity_is_normalized_and_clipped() -> None:
    agent = RuleAgent(RuleAgentConfig())

    moderate = agent.analyze(make_snapshot(ego_speed_mps=18.0, speed_limit_mps=12.0))
    extreme = agent.analyze(make_snapshot(ego_speed_mps=30.0, speed_limit_mps=10.0))

    assert moderate.probability == 1.0
    assert moderate.severity == 0.5
    assert extreme.severity == 1.0


def test_rule_claim_is_exactly_deterministic() -> None:
    agent = RuleAgent(RuleAgentConfig())
    snapshot = make_snapshot(ego_speed_mps=16.0, speed_limit_mps=12.0)

    assert agent.analyze(snapshot) == agent.analyze(snapshot)
