"""Real MetaDrive and Stable-Baselines3 adapters for Phase 6 evaluation."""

from __future__ import annotations

import io
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from mad_driving.config.models import AppConfig, MethodId
from mad_driving.evaluation.models import EvaluationRunSpec, ShieldMode
from mad_driving.evaluation.policies import (
    EvaluationPolicy,
    PpoPolicyAdapter,
    VisibleTtcRulePolicy,
)
from mad_driving.evaluation.selection import CheckpointCandidate
from mad_driving.interfaces import SceneObservation
from mad_driving.methods import MethodProfileSnapshot


class _CapturingEnvironment(Protocol):
    def set_difficulty_level(self, level: int) -> None: ...

    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None: ...

    def set_evaluation_shield_mode(self, mode: ShieldMode) -> None: ...

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, object]]: ...

    def current_scene_observation_for_evaluation(self) -> SceneObservation: ...

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]: ...

    def capture_rgb_frame_for_evaluation(self) -> NDArray[np.uint8]: ...

    def close(self) -> None: ...


EnvironmentBuilder = Callable[..., _CapturingEnvironment]
ModelLoader = Callable[[Path], object]


def _episode_key_text(spec: EvaluationRunSpec) -> str:
    policy_seed = "rule" if spec.policy_seed is None else str(spec.policy_seed)
    return "_".join(
        (
            spec.method_id,
            spec.track,
            policy_seed,
            spec.scenario_cell_id,
            str(spec.test_seed),
        )
    )


def _frame_path(spec: EvaluationRunSpec, step_index: int) -> str:
    policy_seed = "rule" if spec.policy_seed is None else str(spec.policy_seed)
    return (
        f"episodes/{spec.method_id}/{spec.track}/{policy_seed}/"
        f"{spec.scenario_cell_id}/episode_{spec.test_seed}_frames/{step_index:06d}.png"
    )


def _encode_png(frame: NDArray[np.uint8]) -> bytes:
    output = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(output, format="PNG", optimize=False)
    return output.getvalue()


def _load_ppo(path: Path) -> object:
    from stable_baselines3 import PPO

    return PPO.load(path, device="cpu")


class _EvaluationEnvironmentAdapter:
    def __init__(
        self,
        environment: _CapturingEnvironment,
        *,
        spec: EvaluationRunSpec,
        capture: bool,
        max_episode_steps: int,
        frames: list[bytes],
    ) -> None:
        self._environment = environment
        self._spec = spec
        self._capture = capture
        self._max_episode_steps = max_episode_steps
        self._frames = frames
        self._step_index = 0

    def set_difficulty_level(self, level: int) -> None:
        self._environment.set_difficulty_level(level)

    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None:
        self._environment.set_evaluation_scenario_schedule(scenario_ids)

    def set_evaluation_shield_mode(self, mode: ShieldMode) -> None:
        self._environment.set_evaluation_shield_mode(mode)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        self._step_index = 0
        self._frames.clear()
        return self._environment.reset(seed=seed, options=options)

    def current_scene_observation_for_evaluation(self) -> SceneObservation:
        return self._environment.current_scene_observation_for_evaluation()

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, raw_info = self._environment.step(action)
        info = dict(raw_info)
        if self._capture:
            frame = self._environment.capture_rgb_frame_for_evaluation()
            self._frames.append(_encode_png(frame))
            info["frame_path"] = _frame_path(self._spec, self._step_index)
        self._step_index += 1
        if self._step_index >= self._max_episode_steps and not terminated and not truncated:
            truncated = True
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._environment.close()


class MetaDriveEvaluationRuntime:
    """Stateful production factories shared by one complete online bundle run."""

    _capture_episode_keys: frozenset[str]
    _max_episode_steps: int
    _environment_builder: EnvironmentBuilder
    _model_loader: ModelLoader
    _frames: dict[str, list[bytes]]

    def __init__(
        self,
        *,
        capture_episode_keys: tuple[str, ...],
        max_episode_steps: int,
        environment_builder: EnvironmentBuilder | None = None,
        model_loader: ModelLoader = _load_ppo,
    ) -> None:
        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, Integral)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive non-bool integer")
        if (
            not isinstance(capture_episode_keys, tuple)
            or not all(isinstance(key, str) and key for key in capture_episode_keys)
            or len(capture_episode_keys) != len(set(capture_episode_keys))
        ):
            raise ValueError("capture_episode_keys must contain unique non-empty strings")
        if environment_builder is None:
            from mad_driving.envs import MultiAgentSpeedEnv

            environment_builder = cast(EnvironmentBuilder, MultiAgentSpeedEnv)
        self._capture_episode_keys = frozenset(capture_episode_keys)
        self._max_episode_steps = int(max_episode_steps)
        self._environment_builder = environment_builder
        self._model_loader = model_loader
        self._frames: dict[str, list[bytes]] = {}

    def environment_factory(
        self, spec: EvaluationRunSpec, config: AppConfig
    ) -> _EvaluationEnvironmentAdapter:
        key = _episode_key_text(spec)
        capture = key in self._capture_episode_keys
        frames: list[bytes] = []
        self._frames[key] = frames
        environment = self._environment_builder(
            config,
            role="test",
            worker_index=0,
        )
        return _EvaluationEnvironmentAdapter(
            environment,
            spec=spec,
            capture=capture,
            max_episode_steps=self._max_episode_steps,
            frames=frames,
        )

    def policy_factory(
        self,
        spec: EvaluationRunSpec,
        config: AppConfig,
        candidate: CheckpointCandidate | None,
    ) -> EvaluationPolicy:
        if spec.method_id == "b0_rule":
            if candidate is not None:
                raise ValueError("B0 evaluation must not receive a checkpoint")
            return VisibleTtcRulePolicy()
        if candidate is None:
            raise ValueError("PPO evaluation requires an authenticated checkpoint")
        if (
            candidate.method_id != spec.method_id
            or candidate.policy_seed != spec.policy_seed
            or spec.checkpoint_path is None
            or Path(spec.checkpoint_path) != candidate.path
        ):
            raise ValueError("PPO candidate does not match the evaluation run specification")
        resolved_config = config.model_dump(mode="json")
        checkpoint_path = str(candidate.path)
        metadata: dict[str, object] = {
            "research_contract_version": 7,
            "observation_schema_version": 1,
            "observation_shape": (24,),
            "observation_dtype": "float32",
            "action_schema_version": 1,
            "action_count": 4,
            "action_order": ("KEEP", "SLOW", "PREPARE_STOP", "STOP"),
            "method_profile": MethodProfileSnapshot.from_method_id(config.method.id),
            "resolved_config": resolved_config,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": candidate.sha256,
        }
        model = self._model_loader(candidate.path)
        return PpoPolicyAdapter(
            model,
            method_id=cast(MethodId, spec.method_id),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=candidate.sha256,
            resolved_config=resolved_config,
            checkpoint_metadata=metadata,
        )

    def frame_provider(self, spec: EvaluationRunSpec, record_count: int) -> tuple[bytes, ...]:
        key = _episode_key_text(spec)
        if key not in self._capture_episode_keys:
            raise ValueError("frame provider was requested for an uncaptured episode")
        frames = tuple(self._frames.get(key, ()))
        if len(frames) != record_count:
            raise ValueError("captured RGB frame count does not match persisted steps")
        return frames


__all__ = ["MetaDriveEvaluationRuntime"]
