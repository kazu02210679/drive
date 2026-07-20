"""Run a deterministic, fixed-action MetaDrive smoke episode."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from mad_driving.agents.suite import AgentAnalysisResult, AgentSuite, SuiteFactory, analyze_safely
from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.envs.multi_agent_speed_env import (
    EnvironmentFactory,
    SmokeResult,
    create_metadrive_env,
)
from mad_driving.interfaces import SceneFrame
from mad_driving.scenarios import EpisodeSeeds, NoOpScenarioRuntime
from mad_driving.world_model import SceneSnapshotBuilder


def run_smoke(
    config: AppConfig,
    env_factory: EnvironmentFactory = create_metadrive_env,
    suite_factory: SuiteFactory = AgentSuite.from_config,
) -> SmokeResult:
    """Run fixed controls and return the final typed simulator snapshot."""

    env = env_factory(config.metadrive_dict())
    snapshot_builder = SceneSnapshotBuilder()
    suite = suite_factory(config.agents)
    action = (float(config.fixed_action[0]), float(config.fixed_action[1]))
    terminated = False
    truncated = False
    final_frame: SceneFrame | None = None
    final_analysis: AgentAnalysisResult | None = None
    steps_completed = 0
    seeds = EpisodeSeeds(config.seed, config.seed, config.seed)
    runtime = NoOpScenarioRuntime(config.scenario_id)

    try:
        state = runtime.reset(env, seeds=seeds)
        env.reset(seed=seeds.metadrive_scenario_index)
        runtime.after_simulator_reset(env, state)
        for step_index in range(1, config.decision_steps + 1):
            runtime.before_step(env, state, step_index=step_index)
            _, _, terminated_value, truncated_value, raw_info = env.step(action)
            terminated = bool(terminated_value)
            truncated = bool(truncated_value)
            steps_completed = step_index
            scenario_result = runtime.after_step(
                env,
                state,
                step_index=step_index,
                raw_info=raw_info,
            )
            final_frame = snapshot_builder.build(
                env,
                step_index=step_index,
                seeds=seeds,
                context=runtime.observation_context(state),
                scenario_result=scenario_result,
                raw_info=raw_info,
                previous_executed_action=0,
                previous_shield_intervention=False,
            )
            final_analysis = analyze_safely(suite, final_frame.observation)
            if terminated or truncated:
                break
    finally:
        env.close()

    if final_frame is None or final_analysis is None:
        raise RuntimeError("Smoke run completed without a simulator step")
    return SmokeResult(
        steps_completed=steps_completed,
        terminated=terminated,
        truncated=truncated,
        final_snapshot=final_frame.observation,
        final_claims=final_analysis.claims,
        final_review=final_analysis.review,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise, traceback-free operational errors."""

    args = _parser().parse_args(argv)
    try:
        result = run_smoke(load_config(args.config))
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
