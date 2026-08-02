"""Run a deterministic, fixed-action MetaDrive smoke episode."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from mad_driving.agents.suite import AgentSuite, SuiteFactory
from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.envs.multi_agent_speed_env import (
    EnvironmentFactory,
    SmokeResult,
    create_metadrive_env,
)
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
    final_snapshot = None
    final_claims = None
    final_review = None
    steps_completed = 0

    try:
        env.reset(seed=config.seed)
        for step_index in range(1, config.decision_steps + 1):
            _, _, terminated_value, truncated_value, _ = env.step(action)
            terminated = bool(terminated_value)
            truncated = bool(truncated_value)
            steps_completed = step_index
            final_snapshot = snapshot_builder.build(
                env,
                step_index=step_index,
                scenario_id=config.scenario_id,
                seed=config.seed,
                previous_action=0,
                previous_shield_intervention=False,
            )
            final_claims, final_review = suite.analyze(final_snapshot)
            if terminated or truncated:
                break
    finally:
        env.close()

    if final_snapshot is None or final_claims is None or final_review is None:
        raise RuntimeError("Smoke run completed without a simulator step")
    return SmokeResult(
        steps_completed=steps_completed,
        terminated=terminated,
        truncated=truncated,
        final_snapshot=final_snapshot,
        final_claims=final_claims,
        final_review=final_review,
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
