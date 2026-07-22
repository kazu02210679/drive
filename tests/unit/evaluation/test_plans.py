from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    EvaluationPlanConfig,
    EvaluationRunSpec,
    PpoRunBinding,
)
from mad_driving.evaluation.plans import build_formal_plan, build_smoke_plan
from mad_driving.evaluation.serialization import load_evaluation_plan

PPO_METHODS = (
    "b1_nominal",
    "b2_multi_no_review",
    "proposed",
    "proposed_no_critic",
    "proposed_no_shield",
    "proposed_no_hazard",
)
FORMAL_SEEDS = (42, 43, 44, 45, 46)


def make_config(kind: str, seeds: tuple[int, ...], *, episodes: int = 2) -> EvaluationPlanConfig:
    return EvaluationPlanConfig(
        plan_kind=kind,
        evaluation_id=f"eval-{kind}",
        app_config_path=Path("configs/base.yaml"),
        episodes_per_case=episodes,
        test_seed_start=20_000,
        ppo_run_bindings=tuple(
            PpoRunBinding(
                method_id=method_id,
                policy_seed=seed,
                training_run_dir=Path(f"runs/{method_id}/seed-{seed}"),
            )
            for method_id in PPO_METHODS
            for seed in seeds
        ),
        capture_episode_keys=(),
    )


def checkpoints(seeds: tuple[int, ...]) -> dict[tuple[str, int], str]:
    return {
        (method_id, seed): f"runs/{method_id}/seed-{seed}/best_model.zip"
        for method_id in PPO_METHODS
        for seed in seeds
    }


def write_plan_yaml(
    path: Path,
    *,
    episodes_per_case: str = "1",
    test_seed_start: str = "20000",
    policy_seed: str = "42",
    app_config_path: str = "configs/base.yaml",
    training_run_dir: str = "runs/proposed/seed-42",
) -> None:
    path.write_text(
        "\n".join(
            (
                "plan_kind: phase6_smoke",
                "evaluation_id: strict-plan",
                f"app_config_path: {app_config_path}",
                f"episodes_per_case: {episodes_per_case}",
                f"test_seed_start: {test_seed_start}",
                "ppo_run_bindings:",
                "  - method_id: proposed",
                f"    policy_seed: {policy_seed}",
                f"    training_run_dir: {training_run_dir}",
                "capture_episode_keys: []",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_smoke_plan_has_exact_tracks_shields_cells_seeds_and_order() -> None:
    config = make_config("phase6_smoke", (77,))
    plan = build_smoke_plan(config, checkpoints((77,)))

    expected_methods = {
        "decision": ("b1_nominal", "b2_multi_no_review", "proposed"),
        "system": ("b0_rule", "b1_nominal", "b2_multi_no_review", "proposed"),
        "ablation": (
            "proposed",
            "proposed_no_critic",
            "proposed_no_shield",
            "proposed_no_hazard",
        ),
    }
    assert len(plan) == 11 * len(EVALUATION_CASES) * 2
    assert {row.track for row in plan} == set(expected_methods)
    for track, methods in expected_methods.items():
        assert tuple(dict.fromkeys(row.method_id for row in plan if row.track == track)) == methods
    assert {row.scenario_cell_id for row in plan} == {case.case_id for case in EVALUATION_CASES}
    assert {row.shield_mode for row in plan if row.track == "decision"} == {"monitor"}
    assert {row.shield_mode for row in plan if row.track == "system"} == {"enforce"}
    assert {
        row.method_id: row.shield_mode for row in plan if row.track == "ablation"
    } == {
        "proposed": "enforce",
        "proposed_no_critic": "enforce",
        "proposed_no_shield": "off",
        "proposed_no_hazard": "enforce",
    }
    assert all(row.is_formal is False for row in plan)

    b0_rows = [row for row in plan if row.method_id == "b0_rule"]
    assert all(row.policy_seed is None and row.checkpoint_path is None for row in b0_rows)
    assert all(
        row.policy_seed == 77 and row.checkpoint_path is not None
        for row in plan
        if row.method_id != "b0_rule"
    )

    physical_cells: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in plan:
        physical_cells[(row.scenario_cell_id, row.episode_index)].add(row.test_seed)
    assert all(len(seeds) == 1 for seeds in physical_cells.values())
    assert sorted({row.test_seed for row in plan}) == list(range(20_000, 20_010))

    def sort_key(row: EvaluationRunSpec) -> tuple[int, int, int, int, int]:
        return (
            ("decision", "system", "ablation").index(row.track),
            expected_methods[row.track].index(row.method_id),
            -1 if row.policy_seed is None else row.policy_seed,
            tuple(case.case_id for case in EVALUATION_CASES).index(row.scenario_cell_id),
            row.episode_index,
        )

    assert list(plan) == sorted(plan, key=sort_key)


def test_formal_plan_uses_exact_five_policy_seeds_for_every_ppo_method() -> None:
    config = make_config("phase6_formal", FORMAL_SEEDS, episodes=1)
    plan = build_formal_plan(config, checkpoints(FORMAL_SEEDS))

    assert all(row.is_formal is True for row in plan)
    for method_id in PPO_METHODS:
        assert tuple(sorted({row.policy_seed for row in plan if row.method_id == method_id})) == (
            FORMAL_SEEDS
        )


@pytest.mark.parametrize(
    ("builder", "kind", "seeds"),
    [
        (build_smoke_plan, "phase6_smoke", (76, 77)),
        (build_formal_plan, "phase6_formal", (42, 43, 44, 45)),
    ],
)
def test_plan_builders_reject_wrong_policy_seed_sets(
    builder: object, kind: str, seeds: tuple[int, ...]
) -> None:
    config = make_config(kind, seeds)

    with pytest.raises(ValueError, match="policy seed"):
        builder(config, checkpoints(seeds))  # type: ignore[operator]


def test_plan_builder_rejects_missing_extra_or_b0_checkpoint_bindings() -> None:
    config = make_config("phase6_smoke", (77,))
    complete = checkpoints((77,))
    missing = dict(complete)
    missing.pop(("proposed", 77))
    with pytest.raises(ValueError, match="checkpoint"):
        build_smoke_plan(config, missing)

    with pytest.raises(ValueError, match="checkpoint"):
        build_smoke_plan(config, complete | {("b0_rule", 77): "forbidden.zip"})


def test_plan_config_rejects_duplicate_keys_unknown_fields_and_seed_overflow(
    tmp_path: Path,
) -> None:
    binding = PpoRunBinding(
        method_id="proposed",
        policy_seed=77,
        training_run_dir=Path("runs/proposed/seed-77"),
    )
    with pytest.raises(ValidationError, match="duplicate"):
        EvaluationPlanConfig(
            plan_kind="phase6_smoke",
            evaluation_id="dup",
            app_config_path=Path("configs/base.yaml"),
            episodes_per_case=1,
            test_seed_start=20_000,
            ppo_run_bindings=(binding, binding),
            capture_episode_keys=(),
        )

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        "\n".join(
            (
                "plan_kind: phase6_smoke",
                "evaluation_id: unknown",
                "app_config_path: configs/base.yaml",
                "episodes_per_case: 1",
                "test_seed_start: 20000",
                "ppo_run_bindings: []",
                "capture_episode_keys: []",
                "unexpected: true",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unexpected"):
        load_evaluation_plan(unknown)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "plan_kind: phase6_smoke\nevaluation_id: one\nevaluation_id: two\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_evaluation_plan(duplicate)

    overflow = make_config("phase6_smoke", (77,), episodes=201)
    with pytest.raises(ValueError, match=r"\[20000, 21000\)"):
        build_smoke_plan(overflow, checkpoints((77,)))


def test_plan_config_rejects_duplicate_capture_episode_keys() -> None:
    config = make_config("phase6_smoke", (77,))
    payload = config.model_dump(mode="python")
    payload["capture_episode_keys"] = ("decision/proposed/one", "decision/proposed/one")

    with pytest.raises(ValidationError, match="capture_episode_keys"):
        EvaluationPlanConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "yaml_value"),
    (
        ("episodes_per_case", "1.0"),
        ("episodes_per_case", "'1'"),
        ("episodes_per_case", "true"),
        ("test_seed_start", "20000.0"),
        ("test_seed_start", "'20000'"),
        ("test_seed_start", "true"),
        ("policy_seed", "42.0"),
        ("policy_seed", "'42'"),
        ("policy_seed", "true"),
    ),
)
def test_plan_yaml_rejects_coerced_integer_fields(
    tmp_path: Path, field: str, yaml_value: str
) -> None:
    path = tmp_path / f"invalid-{field}.yaml"
    overrides = {field: yaml_value}
    write_plan_yaml(path, **overrides)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match=field):
        load_evaluation_plan(path)


@pytest.mark.parametrize("yaml_value", ('""', '"   "'))
@pytest.mark.parametrize("field", ("app_config_path", "training_run_dir"))
def test_plan_yaml_rejects_empty_or_whitespace_paths(
    tmp_path: Path, field: str, yaml_value: str
) -> None:
    path = tmp_path / f"invalid-{field}.yaml"
    overrides = {field: yaml_value}
    write_plan_yaml(path, **overrides)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match=field):
        load_evaluation_plan(path)


def test_plan_yaml_converts_valid_string_paths_without_numeric_coercion(tmp_path: Path) -> None:
    path = tmp_path / "valid.yaml"
    write_plan_yaml(path)

    config = load_evaluation_plan(path)

    assert config.app_config_path == Path("configs/base.yaml")
    assert config.ppo_run_bindings[0].training_run_dir == Path("runs/proposed/seed-42")
