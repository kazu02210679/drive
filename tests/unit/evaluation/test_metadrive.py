from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mad_driving.config.models import AppConfig
from mad_driving.evaluation.metadrive import MetaDriveEvaluationRuntime
from mad_driving.evaluation.models import EvaluationRunSpec
from mad_driving.evaluation.policies import PpoPolicyAdapter, VisibleTtcRulePolicy
from mad_driving.evaluation.selection import CheckpointCandidate
from mad_driving.interfaces import SceneObservation


class FakeEvaluationEnvironment:
    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    def set_difficulty_level(self, level: int) -> None:
        del level

    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None:
        del scenario_ids

    def set_evaluation_shield_mode(self, mode: str) -> None:
        del mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], dict[str, object]]:
        del seed, options
        self.steps = 0
        return np.zeros(24, dtype=np.float32), {}

    def current_scene_observation_for_evaluation(self) -> SceneObservation:
        raise AssertionError("not needed by this adapter test")

    def step(
        self, action: int
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]],
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        del action
        self.steps += 1
        return np.zeros(24, dtype=np.float32), 0.0, False, False, {}

    def capture_rgb_frame_for_evaluation(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakePpoModel:
    def predict(self, observation: object, **kwargs: object) -> tuple[np.ndarray[Any, Any], None]:
        del observation, kwargs
        return np.array([0]), None


def _config(method_id: str = "proposed") -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "phase5",
            "decision_steps": 2,
            "fixed_action": [0.0, 0.0],
            "metadrive": {"image_observation": False},
            "scenarios": {
                "selection": "lead_brake",
                "curriculum": {"mode": "fixed", "fixed_level": 1},
            },
            "method": {"id": method_id},
        }
    )


def _spec(method_id: str = "proposed") -> EvaluationRunSpec:
    return EvaluationRunSpec(
        track="system",
        method_id=method_id,  # type: ignore[arg-type]
        policy_seed=None if method_id == "b0_rule" else 42,
        checkpoint_path=None if method_id == "b0_rule" else "checkpoint.zip",
        scenario_cell_id="level1_lead_brake",
        episode_index=0,
        test_seed=20_000,
        shield_mode="enforce",
        is_formal=False,
    )


def test_runtime_captures_canonical_rgb_pngs_and_applies_the_smoke_step_limit() -> None:
    created: list[tuple[AppConfig, FakeEvaluationEnvironment]] = []

    def environment_builder(
        config: AppConfig, *, role: str, worker_index: int
    ) -> FakeEvaluationEnvironment:
        assert role == "test"
        assert worker_index == 0
        environment = FakeEvaluationEnvironment()
        created.append((config, environment))
        return environment

    spec = _spec()
    runtime = MetaDriveEvaluationRuntime(
        capture_episode_keys=("proposed_system_42_level1_lead_brake_20000",),
        max_episode_steps=2,
        environment_builder=environment_builder,
        model_loader=lambda path: FakePpoModel(),
    )

    environment = runtime.environment_factory(spec, _config())
    environment.reset(seed=20_000)
    first = environment.step(0)
    second = environment.step(0)
    frames = runtime.frame_provider(spec, 2)
    environment.close()

    assert created[0][0].metadrive.image_observation is False
    assert first[3] is False
    assert second[3] is True
    assert first[4]["frame_path"] == (
        "episodes/proposed/system/42/level1_lead_brake/episode_20000_frames/000000.png"
    )
    assert second[4]["frame_path"].endswith("/000001.png")
    assert created[0][1].closed is True
    assert len(frames) == 2
    with Image.open(io.BytesIO(frames[0])) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 12)
        assert image.getpixel((0, 0)) == (10, 20, 30)


def test_runtime_uses_visible_rule_policy_or_bound_ppo_loader(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"authenticated checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    candidate = CheckpointCandidate(
        checkpoint,
        digest,
        "e" * 64,
        "proposed",
        42,
        "final",
        1,
        8,
    )
    loaded: list[Path] = []
    runtime = MetaDriveEvaluationRuntime(
        capture_episode_keys=(),
        max_episode_steps=2,
        environment_builder=lambda *args, **kwargs: FakeEvaluationEnvironment(),
        model_loader=lambda path: loaded.append(path) or FakePpoModel(),
    )

    assert isinstance(
        runtime.policy_factory(_spec("b0_rule"), _config("b0_rule"), None),
        VisibleTtcRulePolicy,
    )
    policy = runtime.policy_factory(
        replace(_spec(), checkpoint_path=str(checkpoint)),
        _config(),
        candidate,
    )

    assert isinstance(policy, PpoPolicyAdapter)
    assert loaded == [checkpoint]


@pytest.mark.parametrize("value", (True, 0, -1, 1.5))
def test_runtime_rejects_invalid_episode_limit(value: object) -> None:
    with pytest.raises(ValueError, match="max_episode_steps"):
        MetaDriveEvaluationRuntime(
            capture_episode_keys=(),
            max_episode_steps=value,  # type: ignore[arg-type]
            environment_builder=lambda *args, **kwargs: FakeEvaluationEnvironment(),
        )


@pytest.mark.parametrize(
    "keys",
    ([], ("",), ("same", "same"), ("valid", 1)),
)
def test_runtime_rejects_invalid_capture_keys(keys: object) -> None:
    with pytest.raises(ValueError, match="capture_episode_keys"):
        MetaDriveEvaluationRuntime(
            capture_episode_keys=keys,  # type: ignore[arg-type]
            max_episode_steps=1,
            environment_builder=lambda *args, **kwargs: FakeEvaluationEnvironment(),
        )


def test_runtime_policy_and_frame_factories_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    candidate = CheckpointCandidate(
        checkpoint,
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "e" * 64,
        "proposed",
        42,
        "final",
        1,
        8,
    )
    runtime = MetaDriveEvaluationRuntime(
        capture_episode_keys=("proposed_system_42_level1_lead_brake_20000",),
        max_episode_steps=1,
        environment_builder=lambda *args, **kwargs: FakeEvaluationEnvironment(),
        model_loader=lambda path: FakePpoModel(),
    )

    with pytest.raises(ValueError, match="B0"):
        runtime.policy_factory(_spec("b0_rule"), _config("b0_rule"), candidate)
    with pytest.raises(ValueError, match="requires"):
        runtime.policy_factory(_spec(), _config(), None)
    with pytest.raises(ValueError, match="does not match"):
        runtime.policy_factory(
            replace(_spec(), checkpoint_path=str(checkpoint)),
            _config(),
            replace(candidate, policy_seed=43),
        )
    with pytest.raises(ValueError, match="uncaptured"):
        runtime.frame_provider(_spec("b0_rule"), 0)
    with pytest.raises(ValueError, match="count"):
        runtime.frame_provider(_spec(), 1)
