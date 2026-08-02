from mad_driving.agents.claim_factory import neutral_claim
from mad_driving.agents.suite import AgentSuite
from mad_driving.config.models import AgentsConfig
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot
from tests.unit.agents.factories import make_actor, make_snapshot


def test_suite_returns_three_claims_in_fixed_order_and_one_review() -> None:
    claims, review = AgentSuite.from_config(AgentsConfig()).analyze(make_snapshot())

    assert tuple(claim.agent_id for claim in claims) == (
        "nominal",
        "hazard",
        "rule",
    )
    assert isinstance(review, CriticReview)


def test_suite_is_stateless_and_deterministic() -> None:
    suite = AgentSuite.from_config(AgentsConfig())
    snapshot = make_snapshot(
        actors=(
            make_actor(
                "close-lead",
                longitudinal_m=8.0,
                longitudinal_speed_mps=0.0,
            ),
        )
    )

    assert suite.analyze(snapshot) == suite.analyze(snapshot)


class RecordingAgent:
    def __init__(self, agent_id: str, calls: list[str]) -> None:
        self.agent_id = agent_id
        self._calls = calls

    def analyze(self, snapshot: SceneSnapshot) -> RiskClaim:
        self._calls.append(self.agent_id)
        return neutral_claim(self.agent_id, snapshot)


class RecordingCritic:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def review(self, snapshot: SceneSnapshot, claims: tuple[RiskClaim, ...]) -> CriticReview:
        del snapshot, claims
        self._calls.append("critic")
        return CriticReview(
            conflict_score=0.0,
            unresolved_conflict=False,
            max_severity=0.0,
            supported_agent_ids=(),
            challenged_claim_ids=(),
            reasons=(),
        )


def test_suite_calls_each_agent_and_critic_exactly_once_in_order() -> None:
    calls: list[str] = []
    suite = AgentSuite(
        nominal=RecordingAgent("nominal", calls),
        hazard=RecordingAgent("hazard", calls),
        rule=RecordingAgent("rule", calls),
        critic=RecordingCritic(calls),
    )

    suite.analyze(make_snapshot())

    assert calls == ["nominal", "hazard", "rule", "critic"]
