"""Tests for the fixed coordinator observation contract."""

import math
import random
from dataclasses import replace

import numpy as np
import pytest

from mad_driving.config.models import ObservationConfig
from mad_driving.coordinator import ObservationBuilder
from mad_driving.interfaces import CriticReview
from tests.unit.agents.factories import make_claim, make_snapshot

EXPECTED_24_VALUES = np.asarray(
    [
        0.5,
        0.6,
        0.5,
        -0.5,
        0.25,
        0.75,
        0.4,
        0.25,
        0.8,
        0.4,
        0.2,
        -0.5,
        0.7,
        0.9,
        0.2,
        0.0,
        1.0,
        1.0,
        0.25,
        1.0,
        2.0 / 3.0,
        0.9,
        1.0 / 3.0,
        1.0,
    ],
    dtype=np.float32,
)


def make_review(**overrides: object) -> CriticReview:
    values: dict[str, object] = {
        "conflict_score": 0.25,
        "unresolved_conflict": True,
        "max_severity": 0.9,
        "supported_agent_ids": ("nominal", "rule"),
        "challenged_claim_ids": (),
        "reasons": ("test",),
    }
    values.update(overrides)
    return CriticReview(**values)  # type: ignore[arg-type]


def make_layout_snapshot() -> object:
    snapshot = make_snapshot(
        ego_speed_mps=20.0,
        speed_limit_mps=30.0,
        previous_action=1,
        previous_shield_intervention=True,
        intersection_entry_prohibited=True,
    )
    return replace(
        snapshot,
        ego=replace(snapshot.ego, acceleration_mps2=5.0, lane_offset_m=-1.75),
    )


def make_layout_claims() -> tuple[object, ...]:
    return (
        make_claim(
            "nominal",
            min_ttc_s=4.0,
            probability=0.25,
            confidence=0.8,
            recommended_max_speed_mps=16.0,
        ),
        make_claim(
            "hazard",
            min_ttc_s=2.0,
            stopping_margin_m=-25.0,
            severity=0.7,
            confidence=0.9,
            recommended_max_speed_mps=8.0,
        ),
        make_claim(
            "rule",
            recommended_max_speed_mps=0.0,
            hard_stop_required=True,
        ),
    )


def test_observation_has_exact_layout_dtype_and_bounds() -> None:
    obs = ObservationBuilder(ObservationConfig()).build(
        make_layout_snapshot(),  # type: ignore[arg-type]
        make_layout_claims(),  # type: ignore[arg-type]
        make_review(),
    )
    assert obs.shape == (24,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs, EXPECTED_24_VALUES)
    assert np.isfinite(obs).all()
    assert (obs >= -1.0).all() and (obs <= 1.0).all()


def test_missing_claims_become_finite_safe_side_features() -> None:
    obs = ObservationBuilder(ObservationConfig()).build(make_snapshot(), (), make_review())
    assert obs[6:10].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert obs[10:15].tolist() == [0.0, -1.0, 1.0, 0.0, 0.0]
    assert obs[15:18].tolist() == [0.0, 1.0, 1.0]
    assert np.isfinite(obs).all()


def test_normalization_boundaries_and_none_ttc_are_exact() -> None:
    config = ObservationConfig()
    snapshot = make_snapshot(ego_speed_mps=config.max_speed_mps, speed_limit_mps=0.0)
    snapshot = replace(
        snapshot,
        ego=replace(
            snapshot.ego,
            acceleration_mps2=-config.max_abs_acceleration_mps2,
            lane_offset_m=config.max_abs_lane_offset_m,
            route_progress=1.0,
        ),
    )
    claims = (
        make_claim(
            "nominal",
            min_ttc_s=None,
            probability=1.0,
            confidence=1.0,
            recommended_max_speed_mps=config.max_speed_mps,
        ),
        make_claim(
            "hazard",
            min_ttc_s=config.max_ttc_s,
            stopping_margin_m=config.max_abs_stopping_margin_m,
            severity=1.0,
            confidence=1.0,
            recommended_max_speed_mps=config.max_speed_mps,
        ),
        make_claim("rule", recommended_max_speed_mps=config.max_speed_mps),
    )
    obs = ObservationBuilder(config).build(snapshot, claims, make_review())
    np.testing.assert_allclose(
        obs,
        [
            1.0,
            0.0,
            -1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.25,
            1.0,
            2.0 / 3.0,
            0.9,
            0.0,
            0.0,
        ],
    )


def test_huge_valid_values_are_clipped_to_observation_bounds() -> None:
    snapshot = make_snapshot(ego_speed_mps=1_000_000.0, speed_limit_mps=1_000_000.0)
    snapshot = replace(
        snapshot,
        ego=replace(snapshot.ego, acceleration_mps2=1_000_000.0, lane_offset_m=-1_000_000.0),
    )
    claims = (
        make_claim("nominal", min_ttc_s=1_000_000.0, recommended_max_speed_mps=1_000_000.0),
        make_claim(
            "hazard",
            min_ttc_s=1_000_000.0,
            stopping_margin_m=-1_000_000.0,
            recommended_max_speed_mps=1_000_000.0,
        ),
        make_claim("rule", recommended_max_speed_mps=1_000_000.0),
    )
    obs = ObservationBuilder(ObservationConfig()).build(snapshot, claims, make_review())
    assert obs[0] == 1.0
    assert obs[2] == 1.0
    assert obs[3] == -1.0
    assert obs[6] == 1.0
    assert obs[9] == 1.0
    assert obs[10] == 1.0
    assert obs[11] == -1.0
    assert obs[14] == 1.0
    assert obs[15] == 1.0


@pytest.mark.parametrize(
    ("claims", "review"),
    [
        ((make_claim("nominal"), make_claim("nominal", claim_id="nominal:2")), make_review()),
        ((make_claim("nominal"),), make_review(supported_agent_ids=("nominal", "nominal"))),
    ],
)
def test_duplicate_agent_ids_are_rejected(claims: tuple[object, ...], review: CriticReview) -> None:
    with pytest.raises(ValueError, match="duplicate agent_id"):
        ObservationBuilder(ObservationConfig()).build(
            make_snapshot(), claims, review  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("corrupt", ["snapshot", "claim", "review"])
def test_invalid_defensive_input_is_rejected(corrupt: str) -> None:
    snapshot = make_snapshot()
    claims = (make_claim("nominal"),)
    review = make_review()
    if corrupt == "snapshot":
        object.__setattr__(snapshot.ego, "speed_mps", math.nan)
    elif corrupt == "claim":
        object.__setattr__(claims[0], "recommended_max_speed_mps", math.nan)
    else:
        object.__setattr__(review, "max_severity", math.nan)

    with pytest.raises(ValueError, match="invalid"):
        ObservationBuilder(ObservationConfig()).build(snapshot, claims, review)


def test_randomized_valid_inputs_are_finite_bounded_and_deterministic() -> None:
    rng = random.Random(20260720)
    builder = ObservationBuilder(ObservationConfig())
    for _ in range(1_000):
        speed = rng.uniform(0.0, 10_000.0)
        limit = rng.uniform(0.0, 10_000.0)
        snapshot = make_snapshot(
            ego_speed_mps=speed,
            speed_limit_mps=limit,
            previous_action=rng.randrange(4),
            previous_shield_intervention=bool(rng.randrange(2)),
            stop_required=bool(rng.randrange(2)),
            collision_occurred=bool(rng.randrange(2)),
            off_road=bool(rng.randrange(2)),
            intersection_entry_prohibited=bool(rng.randrange(2)),
        )
        snapshot = replace(
            snapshot,
            ego=replace(
                snapshot.ego,
                acceleration_mps2=rng.uniform(-10_000.0, 10_000.0),
                lane_offset_m=rng.uniform(-10_000.0, 10_000.0),
                route_progress=rng.uniform(0.0, 1.0),
            ),
        )
        claims = (
            make_claim(
                "nominal",
                probability=rng.random(),
                confidence=rng.random(),
                min_ttc_s=None if rng.randrange(2) else rng.uniform(0.0, 10_000.0),
                recommended_max_speed_mps=rng.uniform(0.0, 10_000.0),
            ),
            make_claim(
                "hazard",
                severity=rng.random(),
                confidence=rng.random(),
                min_ttc_s=None if rng.randrange(2) else rng.uniform(0.0, 10_000.0),
                stopping_margin_m=rng.uniform(-10_000.0, 10_000.0),
                recommended_max_speed_mps=rng.uniform(0.0, 10_000.0),
            ),
            make_claim(
                "rule",
                recommended_max_speed_mps=rng.uniform(0.0, 10_000.0),
                hard_stop_required=bool(rng.randrange(2)),
            ),
        )
        review = make_review(
            conflict_score=rng.random(),
            unresolved_conflict=bool(rng.randrange(2)),
            max_severity=rng.random(),
            supported_agent_ids=("nominal", "hazard", "rule"),
        )
        first = builder.build(snapshot, claims, review)
        second = builder.build(snapshot, claims, review)
        assert first.shape == (24,)
        assert first.dtype == np.float32
        assert np.isfinite(first).all()
        assert (first >= -1.0).all() and (first <= 1.0).all()
        np.testing.assert_array_equal(first, second)
