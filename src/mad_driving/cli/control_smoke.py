"""Run the deterministic shielded four-action MetaDrive control loop."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Protocol

from mad_driving.agents.suite import AgentAnalysisResult, AgentSuite, SuiteFactory, analyze_safely
from mad_driving.config.loader import load_config
from mad_driving.config.models import (
    AppConfig,
    CoordinatorConfig,
)
from mad_driving.control import DrivingAction, target_speed_mps
from mad_driving.coordinator import RuleBasedCoordinator
from mad_driving.envs.control_metadrive_env import create_control_metadrive_env
from mad_driving.envs.multi_agent_speed_env import (
    ControlEnvironmentFactory,
    ControlSmokeResult,
    ShieldFactory,
)
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    SceneFrame,
    SceneObservation,
)
from mad_driving.safety import SafetyShield
from mad_driving.scenarios import EpisodeSeeds, NoOpScenarioRuntime, ScenarioStepResult
from mad_driving.world_model import SceneSnapshotBuilder


class Coordinator(Protocol):
    def decide(
        self,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> DrivingAction: ...


class CoordinatorFactory(Protocol):
    def __call__(self, config: CoordinatorConfig) -> Coordinator: ...


def run_control_smoke(
    config: AppConfig,
    env_factory: ControlEnvironmentFactory = create_control_metadrive_env,
    suite_factory: SuiteFactory = AgentSuite.from_config,
    coordinator_factory: CoordinatorFactory = RuleBasedCoordinator,
    shield_factory: ShieldFactory = SafetyShield,
) -> ControlSmokeResult:
    """Run the complete deterministic decision and control pipeline."""

    env = env_factory(config.metadrive_dict(), config.control)
    action_counts = [0, 0, 0, 0]
    intervention_count = 0
    steps_completed = 0
    terminated = False
    truncated = False
    final_trace: DecisionTrace | None = None
    frame: SceneFrame | None = None
    analysis: AgentAnalysisResult | None = None
    seeds = EpisodeSeeds(config.seed, config.seed, config.seed)
    runtime = NoOpScenarioRuntime(config.scenario_id)

    try:
        builder = SceneSnapshotBuilder()
        suite = suite_factory(config.agents)
        coordinator = coordinator_factory(config.coordinator)
        shield = shield_factory(config.shield)
        state = runtime.reset(env, seeds=seeds)
        _, reset_info = env.reset(seed=seeds.metadrive_scenario_index)
        state = runtime.after_simulator_reset(env, state)
        frame = builder.build(
            env,
            step_index=0,
            seeds=seeds,
            context=runtime.observation_context(state),
            scenario_result=ScenarioStepResult(False, False),
            raw_info=reset_info,
            previous_executed_action=int(DrivingAction.KEEP),
            previous_shield_intervention=False,
        )
        analysis = analyze_safely(suite, frame.observation)

        for step_index in range(1, config.decision_steps + 1):
            requested = coordinator.decide(
                frame.observation,
                analysis.claims,
                analysis.review,
            )
            shield_result = shield.filter(requested, frame.observation, analysis.claims)
            executed = shield_result.executed_action
            target = target_speed_mps(
                executed,
                frame.observation.ego.speed_mps,
                frame.observation.ego.speed_limit_mps,
            )
            state = runtime.before_step(env, state, step_index=step_index)
            _, _, terminated_value, truncated_value, raw_info = env.step(int(executed))
            control_fail_safe = raw_info.get("fail_safe", False)
            control_fail_safe_reason = raw_info.get("fail_safe_reason")
            if not isinstance(control_fail_safe, bool):
                raise TypeError("fail_safe must be a boolean")
            if control_fail_safe:
                if not isinstance(control_fail_safe_reason, str) or not control_fail_safe_reason:
                    raise ValueError("fail_safe_reason must identify an active fail-safe")
            elif control_fail_safe_reason is not None:
                raise ValueError("fail_safe_reason must be None when fail_safe is false")
            terminated = bool(terminated_value)
            truncated = bool(truncated_value)
            steps_completed = step_index
            action_counts[int(executed)] += 1
            intervention_count += int(shield_result.intervened)
            final_trace = DecisionTrace(
                step_index=step_index,
                raw_action=int(requested),
                required_action=int(shield_result.required_action),
                executed_action=int(executed),
                intervention_required=shield_result.intervention_required,
                target_speed_mps=target,
                shield_intervened=shield_result.intervened,
                shield_reasons=shield_result.reasons,
                claims=analysis.claims,
                review=analysis.review,
                reward_components={},
                control_fail_safe=control_fail_safe,
                control_fail_safe_reason=control_fail_safe_reason,
                failed_agent_ids=analysis.failed_agent_ids,
                errors=analysis.errors,
                episode_rng_seed=seeds.episode_rng_seed,
                metadrive_scenario_index=seeds.metadrive_scenario_index,
                scenario_parameter_seed=seeds.scenario_parameter_seed,
                role="train",
                worker_index=0,
            )
            transition = runtime.after_step(
                env,
                state,
                step_index=step_index,
                raw_info=raw_info,
            )
            state = transition.state
            frame = builder.build(
                env,
                step_index=step_index,
                seeds=seeds,
                context=runtime.observation_context(state),
                scenario_result=transition.outcome,
                raw_info=raw_info,
                previous_executed_action=int(executed),
                previous_shield_intervention=shield_result.intervened,
            )
            analysis = analyze_safely(suite, frame.observation)
            if terminated or truncated:
                break
    finally:
        env.close()

    if steps_completed == 0 or frame is None or analysis is None or final_trace is None:
        raise RuntimeError("Control smoke completed without a simulator step")
    return ControlSmokeResult(
        steps_completed=steps_completed,
        terminated=terminated,
        truncated=truncated,
        scenario_id=frame.scenario_id,
        seeds=frame.seeds,
        final_snapshot=frame.observation,
        final_claims=analysis.claims,
        final_review=analysis.review,
        final_trace=final_trace,
        action_counts=(
            action_counts[0],
            action_counts[1],
            action_counts[2],
            action_counts[3],
        ),
        shield_intervention_count=intervention_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise, traceback-free operational errors."""

    args = _parser().parse_args(argv)
    try:
        result = run_control_smoke(load_config(args.config))
        output = json.dumps(
            asdict(result),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
    except Exception as exc:
        print(f"control smoke failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
