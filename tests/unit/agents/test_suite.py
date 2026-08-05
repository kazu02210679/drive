from dataclasses import FrozenInstanceError

import pytest

from mad_driving.agents import NoOpCritic
from mad_driving.agents.claim_factory import neutral_claim
from mad_driving.agents.suite import AgentAnalysisResult, AgentSuite, analyze_safely
from mad_driving.config.models import AgentsConfig
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot
from mad_driving.methods import build_method_suite
from tests.unit.agents.factories import make_actor, make_analysis, make_claim, make_snapshot


def test_suite_returns_claims_in_fixed_order_and_one_review() -> None:
    result = AgentSuite.from_config(AgentsConfig()).analyze(make_snapshot())

    assert tuple(claim.agent_id for claim in result.claims) == (
        "nominal",
        "hazard",
        "rule",
    )
    assert isinstance(result.review, CriticReview)


def test_from_config_selects_method_through_the_central_registry() -> None:
    suite = AgentSuite.from_config(AgentsConfig(), method_id="b1_nominal")

    result = suite.analyze(make_snapshot())

    assert suite.expected_agent_ids == ("nominal",)
    assert isinstance(suite.critic, NoOpCritic)
    assert result.expected_agent_ids == ("nominal",)
    assert result.review.reasons == ("critic_intentionally_disabled",)


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


def test_profile_built_ablation_omits_specialists_without_recording_a_failure() -> None:
    result = build_method_suite(AgentsConfig(), "proposed_no_hazard").analyze(make_snapshot())

    assert result.expected_agent_ids == ("nominal", "rule")
    assert result.failed_agent_ids == ()


class RecordingAgent:
    def __init__(self, agent_id: str, calls: list[str]) -> None:
        self.agent_id = agent_id
        self._calls = calls

    def analyze(self, snapshot: SceneSnapshot) -> tuple[RiskClaim, ...]:
        self._calls.append(self.agent_id)
        return (neutral_claim(self.agent_id, snapshot),)


class FailingAgent:
    def __init__(self, agent_id: str, error: Exception | str = "unavailable") -> None:
        self.agent_id = agent_id
        self._error = error

    def analyze(self, snapshot: SceneSnapshot) -> tuple[RiskClaim, ...]:
        del snapshot
        if isinstance(self._error, str):
            raise RuntimeError(self._error)
        raise self._error


class UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("string conversion failed")


class RecordingCritic:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.claims: tuple[RiskClaim, ...] | None = None
        self.failed_agent_ids: tuple[str, ...] | None = None

    def review(
        self,
        snapshot: SceneSnapshot,
        claims: tuple[RiskClaim, ...],
        *,
        failed_agent_ids: tuple[str, ...],
    ) -> CriticReview:
        del snapshot
        self._calls.append("critic")
        self.claims = claims
        self.failed_agent_ids = failed_agent_ids
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


def test_one_agent_failure_preserves_surviving_claims_and_sanitizes_error() -> None:
    calls: list[str] = []
    critic = RecordingCritic(calls)
    suite = AgentSuite(
        nominal=FailingAgent("nominal", "sensor unavailable"),
        hazard=RecordingAgent("hazard", calls),
        rule=RecordingAgent("rule", calls),
        critic=critic,
    )

    result = analyze_safely(suite, make_snapshot())

    assert result.failed_agent_ids == ("nominal",)
    assert result.errors == ("nominal:RuntimeError:sensor unavailable",)
    assert tuple(claim.agent_id for claim in result.claims) == ("hazard", "rule")
    assert "agent_analysis_failed:nominal" in result.review.reasons
    assert critic.failed_agent_ids == ("nominal",)
    assert calls == ["hazard", "rule", "critic"]


def test_intentional_ablation_is_absent_not_a_failure() -> None:
    calls: list[str] = []
    suite = AgentSuite(
        nominal=None,
        hazard=RecordingAgent("hazard", calls),
        rule=RecordingAgent("rule", calls),
        critic=RecordingCritic(calls),
    )

    result = analyze_safely(suite, make_snapshot())

    assert result.failed_agent_ids == ()
    assert result.expected_agent_ids == ("hazard", "rule")
    assert result.errors == ()
    assert tuple(claim.agent_id for claim in result.claims) == ("hazard", "rule")


def test_suite_claim_ids_are_unique_and_critic_receives_every_surviving_claim() -> None:
    calls: list[str] = []
    critic = RecordingCritic(calls)
    suite = AgentSuite(
        nominal=RecordingAgent("nominal", calls),
        hazard=RecordingAgent("hazard", calls),
        rule=RecordingAgent("rule", calls),
        critic=critic,
    )

    result = analyze_safely(suite, make_snapshot())

    assert len({claim.claim_id for claim in result.claims}) == len(result.claims)
    assert critic.claims == result.claims
    assert critic.failed_agent_ids == ()


def test_analysis_result_is_frozen_and_defensively_validates_claims() -> None:
    result = AgentAnalysisResult(
        claims=[make_claim()],  # type: ignore[arg-type]
        failed_agent_ids=["hazard"],  # type: ignore[arg-type]
        errors=["hazard:RuntimeError:unavailable"],  # type: ignore[arg-type]
        review=CriticReview(0.0, False, 0.0, (), (), ()),
        expected_agent_ids=("nominal", "hazard", "rule"),
    )

    assert result.claims == (make_claim(),)
    assert result.failed_agent_ids == ("hazard",)
    with pytest.raises(FrozenInstanceError):
        result.errors = ()  # type: ignore[misc]

    invalid = make_claim()
    object.__setattr__(invalid, "severity", float("nan"))
    with pytest.raises(ValueError, match="claims"):
        AgentAnalysisResult(
            (invalid,),
            (),
            (),
            CriticReview(0.0, False, 0.0, (), (), ()),
            ("nominal", "hazard", "rule"),
        )


def test_make_analysis_derives_a_finite_neutral_review_when_omitted() -> None:
    claims = (make_claim("hazard", claim_id="hazard:1:none:test"),)

    result = make_analysis(claims=claims)

    assert result.review == CriticReview(
        conflict_score=0.0,
        unresolved_conflict=False,
        max_severity=0.0,
        supported_agent_ids=("hazard",),
        challenged_claim_ids=(),
        reasons=(),
    )


def test_analysis_result_deeply_freezes_caller_owned_claim_and_review_sequences() -> None:
    evidence = ["observed"]
    assumptions = ["constant_velocity"]
    supported_agent_ids = ["hazard"]
    challenged_claim_ids = ["hazard:1:none:test"]
    reasons = ["reviewed"]
    claim = make_claim(evidence=evidence, assumptions=assumptions)  # type: ignore[arg-type]
    review = CriticReview(
        conflict_score=0.0,
        unresolved_conflict=False,
        max_severity=0.0,
        supported_agent_ids=supported_agent_ids,  # type: ignore[arg-type]
        challenged_claim_ids=challenged_claim_ids,  # type: ignore[arg-type]
        reasons=reasons,  # type: ignore[arg-type]
    )

    result = AgentAnalysisResult(
        (claim,),
        (),
        (),
        review,
        ("nominal", "hazard", "rule"),
    )
    evidence.append("mutated")
    assumptions.append("mutated")
    supported_agent_ids.append("nominal")
    challenged_claim_ids.append("extra")
    reasons.append("mutated")

    assert result.claims[0].evidence == ("observed",)
    assert result.claims[0].assumptions == ("constant_velocity",)
    assert result.review.supported_agent_ids == ("hazard",)
    assert result.review.challenged_claim_ids == ("hazard:1:none:test",)
    assert result.review.reasons == ("reviewed",)


@pytest.mark.parametrize(
    ("failed_agent_ids", "errors"),
    [
        (("nominal",), ()),
        (("nominal",), ("nominal:RuntimeError:first", "nominal:RuntimeError:second")),
        (("nominal",), ("hazard:RuntimeError:unrelated",)),
        (("nominal", "hazard"), ("hazard:RuntimeError:first", "nominal:RuntimeError:second")),
        (("nominal", "hazard"), ("nominal:RuntimeError:same", "nominal:RuntimeError:same")),
    ],
)
def test_analysis_result_rejects_invalid_failed_agent_error_mapping(
    failed_agent_ids: tuple[str, ...], errors: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="errors"):
        AgentAnalysisResult(
            (),
            failed_agent_ids,
            errors,
            CriticReview(0.0, False, 0.0, (), (), ()),
            ("nominal", "hazard", "rule"),
        )


def test_analysis_result_requires_failures_and_claims_to_belong_to_expected_agents() -> None:
    with pytest.raises(ValueError, match="failed_agent_ids"):
        make_analysis(
            failed_agent_ids=("hazard",),
            errors=("hazard:RuntimeError:failed",),
            expected_agent_ids=("nominal",),
        )
    with pytest.raises(ValueError, match="claim agent_ids"):
        make_analysis(
            claims=(make_claim("hazard"),),
            expected_agent_ids=("nominal",),
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            RuntimeError("first\nsecond\rthird\tfourth\x00fifth\u2028sixth"),
            "nominal:RuntimeError:first second third fourth fifth sixth",
        ),
        (UnprintableError(), "nominal:UnprintableError:<unprintable>"),
        (RuntimeError("x" * 300), "nominal:RuntimeError:" + "x" * 256),
    ],
)
def test_specialist_failure_errors_are_sanitized_bounded_and_isolated(
    error: Exception, expected: str
) -> None:
    calls: list[str] = []
    suite = AgentSuite(
        nominal=FailingAgent("nominal", error),
        hazard=RecordingAgent("hazard", calls),
        rule=RecordingAgent("rule", calls),
        critic=RecordingCritic(calls),
    )

    result = analyze_safely(suite, make_snapshot())

    assert result.errors == (expected,)
    assert result.failed_agent_ids == ("nominal",)
    assert tuple(claim.agent_id for claim in result.claims) == ("hazard", "rule")
    assert result.review.reasons == ("agent_analysis_failed:nominal",)
