"""Method-profile composition contract tests."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from mad_driving.agents import CriticAgent, NoOpCritic
from mad_driving.config.loader import load_config
from mad_driving.config.models import AgentsConfig, MethodConfig
from mad_driving.methods import MethodId, MethodProfile, build_method_suite, get_method_profile
from tests.unit.agents.factories import make_snapshot

EXPECTED_PROFILES = {
    "b0_rule": ("rule", (), False, "enforce"),
    "b1_nominal": ("ppo", ("nominal",), False, "enforce"),
    "b2_multi_no_review": ("ppo", ("nominal", "hazard", "rule"), False, "enforce"),
    "proposed": ("ppo", ("nominal", "hazard", "rule"), True, "enforce"),
    "proposed_no_critic": ("ppo", ("nominal", "hazard", "rule"), False, "enforce"),
    "proposed_no_shield": ("ppo", ("nominal", "hazard", "rule"), True, "off"),
    "proposed_no_hazard": ("ppo", ("nominal", "rule"), True, "enforce"),
}


def test_registry_is_exhaustive_and_profiles_are_immutable() -> None:
    assert set(get_args(MethodId)) == set(EXPECTED_PROFILES)

    for method_id, expected in EXPECTED_PROFILES.items():
        profile = get_method_profile(method_id)  # type: ignore[arg-type]

        assert profile == MethodProfile(method_id, *expected)  # type: ignore[arg-type]
        with pytest.raises(FrozenInstanceError):
            profile.critic_enabled = not profile.critic_enabled  # type: ignore[misc]


@pytest.mark.parametrize("method_id", tuple(EXPECTED_PROFILES))
def test_method_overlays_select_only_their_declared_method_id(method_id: str) -> None:
    config = load_config("configs/base.yaml", f"configs/methods/{method_id}.yaml")

    assert config.method.id == method_id


def test_method_config_defaults_to_proposed_and_rejects_unknown_method_ids_and_keys(
    tmp_path: Path,
) -> None:
    assert MethodConfig().id == "proposed"

    unknown_id = tmp_path / "unknown-id.yaml"
    unknown_id.write_text("method:\n  id: unknown\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="literal"):
        load_config("configs/base.yaml", unknown_id)

    unknown_key = tmp_path / "unknown-key.yaml"
    unknown_key.write_text("method:\n  id: proposed\n  specialists: [nominal]\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config("configs/base.yaml", unknown_key)


def test_b0_builds_no_specialists_and_uses_a_noop_reviewer() -> None:
    suite = build_method_suite(AgentsConfig(), "b0_rule")

    result = suite.analyze(make_snapshot())

    assert (suite.nominal, suite.hazard, suite.rule) == (None, None, None)
    assert isinstance(suite.critic, NoOpCritic)
    assert suite.expected_agent_ids == ()
    assert result.expected_agent_ids == ()
    assert result.failed_agent_ids == ()
    assert result.review.conflict_score == 0.0
    assert result.review.max_severity == 0.0
    assert result.review.unresolved_conflict is False
    assert result.review.supported_agent_ids == ()
    assert result.review.challenged_claim_ids == ()
    assert result.review.reasons == ("critic_intentionally_disabled",)
    with pytest.raises(FrozenInstanceError):
        result.review.reasons = ()  # type: ignore[misc]


def test_profile_builder_represent_intentional_omissions_as_none() -> None:
    b1 = build_method_suite(AgentsConfig(), "b1_nominal")
    b2 = build_method_suite(AgentsConfig(), "b2_multi_no_review")
    proposed = build_method_suite(AgentsConfig(), "proposed")
    no_hazard = build_method_suite(AgentsConfig(), "proposed_no_hazard")

    assert (b1.nominal is not None, b1.hazard, b1.rule) == (True, None, None)
    assert isinstance(b1.critic, NoOpCritic)
    assert all(agent is not None for agent in (b2.nominal, b2.hazard, b2.rule))
    assert isinstance(b2.critic, NoOpCritic)
    assert isinstance(proposed.critic, CriticAgent)
    assert no_hazard.expected_agent_ids == ("nominal", "rule")
    assert (no_hazard.nominal is not None, no_hazard.hazard, no_hazard.rule is not None) == (
        True,
        None,
        True,
    )
