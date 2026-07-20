"""Run the deterministic shielded four-action MetaDrive control loop."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Protocol

from mad_driving.agents.suite import AgentSuite, AnalysisSuite, SuiteFactory
from mad_driving.config.loader import load_config
from mad_driving.config.models import (
    AppConfig,
    ControlConfig,
    CoordinatorConfig,
    ShieldConfig,
)
from mad_driving.control import DrivingAction, target_speed_mps
from mad_driving.coordinator import RuleBasedCoordinator
from mad_driving.envs.control_metadrive_env import create_control_metadrive_env
from mad_driving.envs.multi_agent_speed_env import (
    ControlSmokeResult,
    DrivingEnvironment,
)
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    SceneSnapshot,
    ShieldResult,
)
from mad_driving.safety import SafetyShield
from mad_driving.world_model import SceneSnapshotBuilder


class ControlEnvironmentFactory(Protocol):
    def __call__(
        self,
        config: dict[str, object],
        control_config: ControlConfig,
    ) -> DrivingEnvironment: ...


class Coordinator(Protocol):
    def decide(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> DrivingAction: ...


class CoordinatorFactory(Protocol):
    def __call__(self, config: CoordinatorConfig) -> Coordinator: ...


class Shield(Protocol):
    def filter(
        self,
        requested_action: DrivingAction | int,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
    ) -> ShieldResult: ...


class ShieldFactory(Protocol):
    def __call__(self, config: ShieldConfig) -> Shield: ...


def _fallback_analysis() -> tuple[tuple[RiskClaim, ...], CriticReview]:
    return (), CriticReview(
        conflict_score=1.0,
        unresolved_conflict=True,
        max_severity=1.0,
        supported_agent_ids=(),
        challenged_claim_ids=(),
        reasons=("agent_analysis_failed",),
    )


def _analyze_safely(
    suite: AnalysisSuite,
    snapshot: SceneSnapshot,
) -> tuple[tuple[RiskClaim, ...], CriticReview]:
    try:
        return suite.analyze(snapshot)
    except Exception:
        return _fallback_analysis()


def run_control_smoke(
    config: AppConfig,
    env_factory: ControlEnvironmentFactory = create_control_metadrive_env,
    suite_factory: SuiteFactory = AgentSuite.from_config,
    coordinator_factory: CoordinatorFactory = RuleBasedCoordinator,
    shield_factory: ShieldFactory = SafetyShield,
) -> ControlSmokeResult:
    """Run the complete deterministic decision and control pipeline."""

    env = env_factory(config.metadrive_dict(), config.control)
    builder = SceneSnapshotBuilder()
    suite = suite_factory(config.agents)
    coordinator = coordinator_factory(config.coordinator)
    shield = shield_factory(config.shield)
    action_counts = [0, 0, 0, 0]
    intervention_count = 0
    steps_completed = 0
    terminated = False
    truncated = False
    final_trace: DecisionTrace | None = None
    snapshot: SceneSnapshot | None = None
    claims: tuple[RiskClaim, ...] | None = None
    review: CriticReview | None = None

    try:
        env.reset(seed=config.seed)
        snapshot = builder.build(
            env,
            step_index=0,
            scenario_id=config.scenario_id,
            seed=config.seed,
            previous_action=int(DrivingAction.KEEP),
            previous_shield_intervention=False,
        )
        claims, review = _analyze_safely(suite, snapshot)

        for step_index in range(1, config.decision_steps + 1):
            requested = coordinator.decide(snapshot, claims, review)
            shield_result = shield.filter(requested, snapshot, claims)
            executed = shield_result.executed_action
            target = target_speed_mps(
                executed,
                snapshot.ego.speed_mps,
                snapshot.ego.speed_limit_mps,
            )
            _, _, terminated_value, truncated_value, _ = env.step(int(executed))
            terminated = bool(terminated_value)
            truncated = bool(truncated_value)
            steps_completed = step_index
            action_counts[int(executed)] += 1
            intervention_count += int(shield_result.intervened)
            final_trace = DecisionTrace(
                step_index=step_index,
                raw_action=int(requested),
                executed_action=int(executed),
                target_speed_mps=target,
                shield_intervened=shield_result.intervened,
                shield_reasons=shield_result.reasons,
                claims=claims,
                review=review,
                reward_components={},
            )
            snapshot = builder.build(
                env,
                step_index=step_index,
                scenario_id=config.scenario_id,
                seed=config.seed,
                previous_action=int(executed),
                previous_shield_intervention=shield_result.intervened,
            )
            claims, review = _analyze_safely(suite, snapshot)
            if terminated or truncated:
                break
    finally:
        env.close()

    if (
        steps_completed == 0
        or snapshot is None
        or claims is None
        or review is None
        or final_trace is None
    ):
        raise RuntimeError("Control smoke completed without a simulator step")
    return ControlSmokeResult(
        steps_completed=steps_completed,
        terminated=terminated,
        truncated=truncated,
        final_snapshot=snapshot,
        final_claims=claims,
        final_review=review,
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
