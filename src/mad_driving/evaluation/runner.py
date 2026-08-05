"""One-environment online evaluation with strict incremental records."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from mad_driving.config.models import AppConfig
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    RECORD_SCHEMA_VERSION,
    RESEARCH_CONTRACT_VERSION,
    EvaluationCase,
    EvaluationEpisodeRecord,
    EvaluationRunSpec,
    EvaluationStepRecord,
    ShieldMode,
    expected_runtime_shield_mode,
)
from mad_driving.evaluation.policies import (
    EvaluationPolicy,
    PpoPolicyAdapter,
    VisibleTtcRulePolicy,
    validate_ppo_checkpoint_identity,
)
from mad_driving.evaluation.serialization import read_jsonl_strict, write_jsonl_strict
from mad_driving.evaluation.workspace import EvaluationWorkspace
from mad_driving.interfaces import DecisionTrace, SceneObservation
from mad_driving.methods import MethodProfileSnapshot

if TYPE_CHECKING:
    from mad_driving.scenarios.seeding import EpisodeSeeds


class EvaluationEnvironment(Protocol):
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

    def close(self) -> None: ...


@dataclass(frozen=True)
class EvaluationRunResult:
    step_records: tuple[EvaluationStepRecord, ...]
    episode_record: EvaluationEpisodeRecord


def run_evaluation_episode(
    spec: EvaluationRunSpec,
    *,
    environment: EvaluationEnvironment,
    policy: EvaluationPolicy,
    config: AppConfig,
    destination: Path,
    checkpoint_sha256: str | None = None,
) -> EvaluationRunResult:
    """Run, persist, and atomically publish one planned test episode."""

    workspace: EvaluationWorkspace | None = None
    result: EvaluationRunResult | None = None
    primary_error: BaseException | None = None
    try:
        from mad_driving.scenarios.seeding import EpisodeSeedAllocator

        profile = _validate_run_binding(spec, policy, config, checkpoint_sha256)
        case = next(case for case in EVALUATION_CASES if case.case_id == spec.scenario_cell_id)
        expected_seeds = EpisodeSeedAllocator("test", config.scenarios.test, 0).allocate(
            spec.test_seed
        )
        workspace = EvaluationWorkspace.stage(destination)
        environment.set_difficulty_level(case.difficulty_level)
        environment.set_evaluation_scenario_schedule((case.scenario_id,))
        environment.set_evaluation_shield_mode(spec.shield_mode)
        policy.reset()
        ppo_observation, reset_info = environment.reset(
            seed=spec.test_seed,
            options={
                "test_seed": spec.test_seed,
                "scenario_cell_id": spec.scenario_cell_id,
            },
        )
        _verify_reset_identity(reset_info, case.scenario_id, case.difficulty_level, expected_seeds)
        steps_path = workspace.path / "steps.jsonl"
        with steps_path.open("xb") as output:
            step_index = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                visible_scene = environment.current_scene_observation_for_evaluation()
                decision_start_ns = time.perf_counter_ns()
                policy_start_ns = decision_start_ns
                if spec.method_id == "b0_rule":
                    action = policy.predict(visible_scene)
                else:
                    action = policy.predict(ppo_observation)
                policy_latency_ms = _latency_ms(policy_start_ns, time.perf_counter_ns())
                ppo_observation, reward, terminated, truncated, info = environment.step(action)
                total_latency_ms = _latency_ms(decision_start_ns, time.perf_counter_ns())
                record = _build_step_record(
                    spec=spec,
                    profile=profile,
                    case=case,
                    expected_seeds=expected_seeds,
                    checkpoint_sha256=checkpoint_sha256,
                    step_index=step_index,
                    visible_scene=visible_scene,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    policy_latency_ms=policy_latency_ms,
                    total_latency_ms=total_latency_ms,
                )
                encoded = (
                    json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                output.write(encoded)
                output.flush()
                step_index += 1
            os.fsync(output.fileno())
        persisted_steps = read_jsonl_strict(steps_path, EvaluationStepRecord)
        episode = _build_episode_record(
            spec=spec,
            profile=profile,
            checkpoint_sha256=checkpoint_sha256,
            reset_info=reset_info,
            steps=persisted_steps,
        )
        write_jsonl_strict(workspace.path / "episode.jsonl", (episode.to_dict(),))
        workspace.write_manifest()
        result = EvaluationRunResult(persisted_steps, episode)
    except BaseException as error:
        primary_error = error
    finally:
        try:
            environment.close()
        except BaseException as close_error:
            if primary_error is None:
                primary_error = close_error
            else:
                primary_error.add_note(f"evaluation environment close also failed: {close_error}")
    if primary_error is not None:
        raise primary_error
    if result is None or workspace is None:
        raise RuntimeError("evaluation episode did not produce a result")
    workspace.publish()
    return result


def _validate_run_binding(
    spec: EvaluationRunSpec,
    policy: EvaluationPolicy,
    config: AppConfig,
    checkpoint_sha256: str | None,
) -> MethodProfileSnapshot:
    _validate_spec_shield_mode(spec)
    if config.method.id != spec.method_id:
        raise ValueError("evaluation config method does not match the run specification")
    profile = MethodProfileSnapshot.from_method_id(spec.method_id)
    if spec.method_id == "b0_rule":
        if not isinstance(policy, VisibleTtcRulePolicy):
            raise ValueError("B0 run requires VisibleTtcRulePolicy")
        if checkpoint_sha256 is not None:
            raise ValueError("B0 must not have checkpoint metadata")
    else:
        if not isinstance(policy, PpoPolicyAdapter):
            raise ValueError("PPO run requires PpoPolicyAdapter")
        validate_ppo_checkpoint_identity(spec.checkpoint_path, checkpoint_sha256)
        validate_ppo_checkpoint_identity(policy.checkpoint_path, policy.checkpoint_sha256)
        if (
            policy.method_profile != profile
            or policy.checkpoint_path != spec.checkpoint_path
            or policy.checkpoint_sha256 != checkpoint_sha256
        ):
            raise ValueError(
                "PPO policy profile or checkpoint does not match the run specification"
            )
        if policy.resolved_config != config.model_dump(mode="json"):
            raise ValueError("PPO policy resolved config does not match evaluation config")
    return profile


def _validate_spec_shield_mode(spec: EvaluationRunSpec) -> None:
    expected: ShieldMode = expected_runtime_shield_mode(spec.track, spec.method_id)
    if spec.shield_mode != expected:
        raise ValueError("evaluation shield mode does not match the fixed plan matrix")


def _verify_reset_identity(
    info: Mapping[str, object],
    scenario_id: str,
    difficulty_level: int,
    seeds: EpisodeSeeds,
) -> None:
    expected: dict[str, object] = {
        "role": "test",
        "scenario_id": scenario_id,
        "difficulty_level": difficulty_level,
        "episode_rng_seed": seeds.episode_rng_seed,
        "metadrive_scenario_index": seeds.metadrive_scenario_index,
        "scenario_selection_seed": seeds.scenario_selection_seed,
        "scenario_parameter_seed": seeds.scenario_parameter_seed,
    }
    for field, value in expected.items():
        if info.get(field) != value:
            raise RuntimeError(f"evaluation reset {field} mismatch")


def _build_step_record(
    *,
    spec: EvaluationRunSpec,
    profile: MethodProfileSnapshot,
    case: EvaluationCase,
    expected_seeds: EpisodeSeeds,
    checkpoint_sha256: str | None,
    step_index: int,
    visible_scene: SceneObservation,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, object],
    policy_latency_ms: float,
    total_latency_ms: float,
) -> EvaluationStepRecord:
    trace = info.get("decision_trace")
    if not isinstance(trace, DecisionTrace):
        raise ValueError("evaluation step info requires a DecisionTrace")
    _verify_step_identity(info, trace, case, expected_seeds)
    total_latency_ms = max(
        total_latency_ms,
        policy_latency_ms,
        trace.analysis_latency_ms,
        trace.shield_latency_ms,
    )
    return EvaluationStepRecord(
        record_schema_version=RECORD_SCHEMA_VERSION,
        research_contract_version=RESEARCH_CONTRACT_VERSION,
        episode_key=spec.episode_key,
        method_profile=profile,
        checkpoint_path=spec.checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        episode_index=spec.episode_index,
        is_formal=spec.is_formal,
        shield_mode=spec.shield_mode,
        step_index=step_index,
        simulation_time_s=cast(float, info["simulation_time_s"]),
        decision_interval_s=cast(float, info["decision_interval_s"]),
        episode_rng_seed=cast(int, info["episode_rng_seed"]),
        metadrive_scenario_index=cast(int, info["metadrive_scenario_index"]),
        scenario_selection_seed=cast(int, info["scenario_selection_seed"]),
        scenario_parameter_seed=cast(int, info["scenario_parameter_seed"]),
        case_id=spec.scenario_cell_id,
        scenario_id=cast(str, info["scenario_id"]),
        difficulty_level=cast(int, info["difficulty_level"]),
        requested_action=trace.raw_action,
        required_action=trace.required_action,
        executed_action=trace.executed_action,
        unsafe_request=trace.intervention_required,
        shield_intervened=trace.shield_intervened,
        shield_reasons=trace.shield_reasons,
        target_speed_mps=trace.target_speed_mps,
        ego_speed_mps=cast(float, info["ego_speed_mps"]),
        ego_speed_limit_mps=visible_scene.ego.speed_limit_mps,
        ego_longitudinal_acceleration_mps2=cast(float, info["ego_longitudinal_acceleration_mps2"]),
        route_completion=cast(float, info["route_completion"]),
        route_progress_m=cast(float, info["route_progress_m"]),
        lane_offset_m=cast(float, info["lane_offset_m"]),
        collision_occurred=cast(bool, info["collision_occurred"]),
        collision_kind=cast(Any, info["collision_kind"]),
        minimum_actual_ttc_s=cast(float | None, info["minimum_actual_ttc_s"]),
        minimum_actual_stopping_margin_m=cast(
            float | None, info["minimum_actual_stopping_margin_m"]
        ),
        pre_step_hard_rule_constraint=cast(bool, info["pre_step_hard_rule_constraint"]),
        post_step_rule_violation_event=cast(bool, info["post_step_rule_violation_event"]),
        scenario_success=cast(bool, info["scenario_success"]),
        scenario_failure=cast(bool, info["scenario_failure"]),
        arrived=cast(bool, info["arrived"]),
        off_road=cast(bool, info["off_road"]),
        terminated=terminated,
        truncated=truncated,
        cumulative_unnecessary_stop_duration_s=cast(float, info["unnecessary_stop_duration_s"]),
        reward_total=float(reward),
        reward_components=trace.reward_components,
        claims=trace.claims,
        review=trace.review,
        expected_agent_ids=trace.expected_agent_ids,
        failed_agent_ids=trace.failed_agent_ids,
        errors=trace.errors,
        policy_inference_latency_ms=policy_latency_ms,
        agent_analysis_latency_ms=trace.analysis_latency_ms,
        shield_latency_ms=trace.shield_latency_ms,
        total_decision_latency_ms=total_latency_ms,
        frame_path=cast(str | None, info.get("frame_path")),
    )


def _verify_step_identity(
    info: Mapping[str, object],
    trace: DecisionTrace,
    case: EvaluationCase,
    seeds: EpisodeSeeds,
) -> None:
    expected: dict[str, object] = {
        "role": "test",
        "episode_rng_seed": seeds.episode_rng_seed,
        "metadrive_scenario_index": seeds.metadrive_scenario_index,
        "scenario_selection_seed": seeds.scenario_selection_seed,
        "scenario_parameter_seed": seeds.scenario_parameter_seed,
        "scenario_id": case.scenario_id,
        "difficulty_level": case.difficulty_level,
    }
    for field, value in expected.items():
        if info.get(field) != value:
            raise RuntimeError(f"evaluation step info {field} mismatch")
        if getattr(trace, field) != value:
            raise RuntimeError(f"evaluation step DecisionTrace {field} mismatch")


def _build_episode_record(
    *,
    spec: EvaluationRunSpec,
    profile: MethodProfileSnapshot,
    checkpoint_sha256: str | None,
    reset_info: Mapping[str, object],
    steps: tuple[EvaluationStepRecord, ...],
) -> EvaluationEpisodeRecord:
    final = steps[-1]
    if any(step.episode_index != spec.episode_index for step in steps):
        raise ValueError("evaluation steps and summary episode_index disagree")
    if any(step.is_formal is not spec.is_formal for step in steps):
        raise ValueError("evaluation steps and summary is_formal disagree")
    if any(step.shield_mode != spec.shield_mode for step in steps):
        raise ValueError("evaluation steps and summary shield_mode disagree")
    parameters = reset_info.get("scenario_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("evaluation reset scenario_parameters must be a mapping")
    collision_steps = tuple(step for step in steps if step.collision_occurred)
    collision_kind = collision_steps[-1].collision_kind if collision_steps else None
    return EvaluationEpisodeRecord(
        record_schema_version=RECORD_SCHEMA_VERSION,
        research_contract_version=RESEARCH_CONTRACT_VERSION,
        episode_key=spec.episode_key,
        method_profile=profile,
        checkpoint_path=spec.checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        episode_index=spec.episode_index,
        is_formal=spec.is_formal,
        shield_mode=spec.shield_mode,
        episode_rng_seed=final.episode_rng_seed,
        metadrive_scenario_index=final.metadrive_scenario_index,
        scenario_selection_seed=final.scenario_selection_seed,
        scenario_parameter_seed=final.scenario_parameter_seed,
        case_id=final.case_id,
        scenario_id=final.scenario_id,
        difficulty_level=final.difficulty_level,
        sampled_scenario_parameters=cast(Mapping[str, object], parameters),
        step_count=len(steps),
        final_step_index=final.step_index,
        simulated_duration_s=sum(step.decision_interval_s for step in steps),
        cumulative_reward=sum(step.reward_total for step in steps),
        collision_occurred=bool(collision_steps),
        collision_kind=collision_kind,
        scenario_success=final.scenario_success,
        scenario_failure=final.scenario_failure,
        arrived=final.arrived,
        off_road=final.off_road,
        terminated=final.terminated,
        truncated=final.truncated,
        complete=True,
    )


def _latency_ms(start_ns: int, end_ns: int) -> float:
    return max((end_ns - start_ns) / 1_000_000.0, 0.0)


__all__ = ["EvaluationRunResult", "run_evaluation_episode"]
