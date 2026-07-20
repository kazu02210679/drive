import json
from dataclasses import asdict

from mad_driving.agents.claim_factory import claim_id, neutral_claim
from mad_driving.agents.protocol import DrivingAgent
from mad_driving.interfaces import RiskClaim, SceneSnapshot
from tests.unit.agents.factories import make_snapshot


def test_claim_id_is_stable_and_contains_no_uuid() -> None:
    snapshot = make_snapshot(step_index=7)

    first = claim_id("nominal", snapshot, "nominal_collision", "actor-2")
    second = claim_id("nominal", snapshot, "nominal_collision", "actor-2")

    assert first == second == "nominal:7:actor-2:nominal_collision"
    assert claim_id("hazard", snapshot, "no_hazard", None) == "hazard:7:none:no_hazard"


def test_neutral_claim_is_finite_and_valid_for_current_step() -> None:
    snapshot = make_snapshot(step_index=3, speed_limit_mps=13.0)

    claim = neutral_claim("hazard", snapshot)

    assert claim.target_actor_id is None
    assert claim.probability == 0.0
    assert claim.confidence == 1.0
    assert claim.severity == 0.0
    assert claim.time_horizon_s == 0.0
    assert claim.min_ttc_s is None
    assert claim.stopping_margin_m is None
    assert claim.recommended_max_speed_mps == 13.0
    assert claim.hard_stop_required is False
    assert claim.evidence == ("no_applicable_hazard",)
    assert claim.assumptions == ()
    assert claim.valid_until_step == 3
    json.dumps(asdict(claim))


def test_driving_agent_protocol_is_runtime_checkable() -> None:
    class ExampleAgent:
        agent_id = "example"

        def analyze(self, snapshot: SceneSnapshot) -> tuple[RiskClaim, ...]:
            return (neutral_claim(self.agent_id, snapshot),)

    assert isinstance(ExampleAgent(), DrivingAgent)
