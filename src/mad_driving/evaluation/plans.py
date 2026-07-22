"""Pure deterministic builders for the fixed Phase 6 evaluation matrices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from mad_driving.config.models import MethodId
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    TEST_SEED_STOP,
    EvaluationPlanConfig,
    EvaluationRunSpec,
    EvaluationTrack,
    ShieldMode,
)

FORMAL_POLICY_SEEDS = (42, 43, 44, 45, 46)

_TRACK_METHODS: tuple[tuple[EvaluationTrack, tuple[MethodId, ...]], ...] = (
    ("decision", ("b1_nominal", "b2_multi_no_review", "proposed")),
    ("system", ("b0_rule", "b1_nominal", "b2_multi_no_review", "proposed")),
    (
        "ablation",
        (
            "proposed",
            "proposed_no_critic",
            "proposed_no_shield",
            "proposed_no_hazard",
        ),
    ),
)
_PPO_METHODS = frozenset(
    method_id
    for _, method_ids in _TRACK_METHODS
    for method_id in method_ids
    if method_id != "b0_rule"
)
_CheckpointKey = tuple[MethodId, int]


def build_smoke_plan(
    config: EvaluationPlanConfig,
    checkpoint_paths: Mapping[tuple[str, int], str],
) -> tuple[EvaluationRunSpec, ...]:
    """Build the one-common-PPO-seed non-formal matrix without filesystem access."""

    if config.plan_kind != "phase6_smoke":
        raise ValueError("smoke plan builder requires plan_kind phase6_smoke")
    seeds_by_method = _binding_seeds_by_method(config)
    common_seeds = {seeds for seeds in seeds_by_method.values()}
    if len(common_seeds) != 1:
        raise ValueError("smoke plan requires exactly one common PPO policy seed")
    policy_seeds = next(iter(common_seeds))
    if len(policy_seeds) != 1:
        raise ValueError("smoke plan requires exactly one common PPO policy seed")
    checkpoints = _validate_checkpoints(checkpoint_paths, policy_seeds)
    return _build_plan(config, policy_seeds, checkpoints, is_formal=False)


def build_formal_plan(
    config: EvaluationPlanConfig,
    checkpoint_paths: Mapping[tuple[str, int], str],
) -> tuple[EvaluationRunSpec, ...]:
    """Build the exact five-seed formal matrix without filesystem access."""

    if config.plan_kind != "phase6_formal":
        raise ValueError("formal plan builder requires plan_kind phase6_formal")
    seeds_by_method = _binding_seeds_by_method(config)
    if any(seeds != FORMAL_POLICY_SEEDS for seeds in seeds_by_method.values()):
        raise ValueError("formal plan requires policy seeds (42, 43, 44, 45, 46)")
    checkpoints = _validate_checkpoints(checkpoint_paths, FORMAL_POLICY_SEEDS)
    return _build_plan(config, FORMAL_POLICY_SEEDS, checkpoints, is_formal=True)


def _binding_seeds_by_method(
    config: EvaluationPlanConfig,
) -> dict[MethodId, tuple[int, ...]]:
    seeds: dict[MethodId, list[int]] = {method_id: [] for method_id in _PPO_METHODS}
    for binding in config.ppo_run_bindings:
        if binding.method_id not in _PPO_METHODS:
            raise ValueError("plan contains a binding for a method outside the evaluation matrix")
        seeds[binding.method_id].append(binding.policy_seed)
    if any(not method_seeds for method_seeds in seeds.values()):
        raise ValueError("plan requires a policy seed binding for every PPO method")
    return {
        method_id: tuple(sorted(method_seeds)) for method_id, method_seeds in seeds.items()
    }


def _validate_checkpoints(
    values: Mapping[tuple[str, int], str], policy_seeds: tuple[int, ...]
) -> dict[_CheckpointKey, str]:
    expected = {
        (method_id, policy_seed)
        for method_id in _PPO_METHODS
        for policy_seed in policy_seeds
    }
    normalized: dict[_CheckpointKey, str] = {}
    for raw_key, path in values.items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
            or raw_key[0] not in _PPO_METHODS
            or type(raw_key[1]) is not int
            or not isinstance(path, str)
            or not path
        ):
            raise ValueError("checkpoint mapping contains a malformed key or path")
        key = (cast(MethodId, raw_key[0]), raw_key[1])
        normalized[key] = path
    if set(normalized) != expected:
        raise ValueError("checkpoint mapping must exactly match all PPO method/policy seed keys")
    return normalized


def _build_plan(
    config: EvaluationPlanConfig,
    policy_seeds: tuple[int, ...],
    checkpoints: Mapping[_CheckpointKey, str],
    *,
    is_formal: bool,
) -> tuple[EvaluationRunSpec, ...]:
    required_seed_count = len(EVALUATION_CASES) * config.episodes_per_case
    if config.test_seed_start + required_seed_count > TEST_SEED_STOP:
        raise ValueError("evaluation test seeds must stay in [20000, 21000)")
    rows: list[EvaluationRunSpec] = []
    for track, method_ids in _TRACK_METHODS:
        for method_id in method_ids:
            method_policy_seeds: tuple[int | None, ...] = (
                (None,) if method_id == "b0_rule" else policy_seeds
            )
            for policy_seed in method_policy_seeds:
                for case_index, case in enumerate(EVALUATION_CASES):
                    for episode_index in range(config.episodes_per_case):
                        test_seed = (
                            config.test_seed_start
                            + case_index * config.episodes_per_case
                            + episode_index
                        )
                        checkpoint = (
                            None
                            if policy_seed is None
                            else checkpoints[(method_id, policy_seed)]
                        )
                        rows.append(
                            EvaluationRunSpec(
                                track=track,
                                method_id=method_id,
                                policy_seed=policy_seed,
                                checkpoint_path=checkpoint,
                                scenario_cell_id=case.case_id,
                                episode_index=episode_index,
                                test_seed=test_seed,
                                shield_mode=_shield_mode(track, method_id),
                                is_formal=is_formal,
                            )
                        )
    return tuple(rows)


def _shield_mode(track: EvaluationTrack, method_id: MethodId) -> ShieldMode:
    if track == "decision":
        return "monitor"
    if track == "ablation" and method_id == "proposed_no_shield":
        return "off"
    return "enforce"


__all__ = ["FORMAL_POLICY_SEEDS", "build_formal_plan", "build_smoke_plan"]
