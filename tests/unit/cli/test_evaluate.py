from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from mad_driving.cli import evaluate as evaluate_module
from mad_driving.cli.evaluate import main
from mad_driving.evaluation.models import (
    EvaluationPlanConfig,
    Phase6PublicationPlan,
    PpoRunBinding,
)
from mad_driving.evaluation.plans import build_smoke_plan
from mad_driving.evaluation.selection import CheckpointCandidate
from mad_driving.evaluation.serialization import load_evaluation_plan
from mad_driving.visualization import SMOKE_RESULT_LABEL

PPO_METHODS = (
    "b1_nominal",
    "b2_multi_no_review",
    "proposed",
    "proposed_no_critic",
    "proposed_no_shield",
    "proposed_no_hazard",
)
METHOD_OVERLAYS = (
    "b0_rule",
    "b1_nominal",
    "b2_multi_no_review",
    "proposed",
    "proposed_no_critic",
    "proposed_no_shield",
    "proposed_no_hazard",
)


def _candidate(binding: PpoRunBinding) -> CheckpointCandidate:
    assert binding.checkpoint_path is not None
    return CheckpointCandidate(
        path=binding.checkpoint_path.resolve(),
        sha256=hashlib.sha256(binding.checkpoint_path.read_bytes()).hexdigest(),
        method_id=binding.method_id,
        policy_seed=binding.policy_seed,
        checkpoint_kind="final",
        curriculum_level=3,
        training_timestep=32,
    )


def _install_authenticated_discovery(
    monkeypatch: pytest.MonkeyPatch,
    plan: EvaluationPlanConfig,
) -> tuple[CheckpointCandidate, ...]:
    candidates = tuple(_candidate(binding) for binding in plan.ppo_run_bindings)
    by_run = {
        binding.training_run_dir.resolve(): candidate
        for binding, candidate in zip(plan.ppo_run_bindings, candidates, strict=True)
    }
    monkeypatch.setattr(
        evaluate_module,
        "discover_checkpoint_candidates",
        lambda run_dir: (by_run[run_dir.resolve()],),
    )
    return candidates


def _plan(tmp_path: Path) -> Phase6PublicationPlan:
    app_config = tmp_path / "base.yaml"
    repository = Path(__file__).resolve().parents[3]
    app_config.write_bytes(repository.joinpath("configs/base.yaml").read_bytes())
    overlays = tuple(tmp_path / f"{method}.yaml" for method in METHOD_OVERLAYS)
    for method, overlay in zip(METHOD_OVERLAYS, overlays, strict=True):
        overlay.write_text(f"method:\n  id: {method}\n", encoding="utf-8")
    bindings = []
    for method in PPO_METHODS:
        run_dir = tmp_path / "runs" / method
        checkpoint = run_dir / "checkpoints" / "final_model.zip"
        checkpoint.parent.mkdir(parents=True)
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("data", '{"num_timesteps":32}')
        bindings.append(
            PpoRunBinding(
                method_id=method,
                policy_seed=42,
                training_run_dir=run_dir,
                checkpoint_path=checkpoint,
            )
        )
    legacy = EvaluationPlanConfig(
        plan_kind="phase6_smoke",
        evaluation_id="cli-smoke",
        is_formal=False,
        result_label=SMOKE_RESULT_LABEL,
        app_config_path=app_config,
        method_overlays=overlays,
        max_episode_steps=32,
        episodes_per_case=1,
        test_seed_start=20_000,
        ppo_run_bindings=tuple(bindings),
        capture_episode_keys=(),
    )
    return Phase6PublicationPlan.model_validate(legacy.model_dump(mode="python"))


def test_help_lists_evaluation_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--plan" in output
    assert "--output" in output
    assert "--overlay" in output
    assert "--smoke" in output


def test_plan_and_output_are_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--plan" in captured.err
    assert "--output" in captured.err
    assert "Traceback" not in captured.err


def test_ordered_plan_and_cli_overlays_are_validated_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("training:\n  eval_episodes: 7\n", encoding="utf-8")
    second.write_text("training:\n  eval_episodes: 9\n", encoding="utf-8")
    plan = _plan(tmp_path)
    destination = tmp_path / "evaluation"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    candidates = _install_authenticated_discovery(monkeypatch, plan)

    def capture(**kwargs: object) -> Path:
        calls.append(kwargs)
        return destination.resolve()

    monkeypatch.setattr(evaluate_module, "run_evaluation_bundle", capture)

    assert (
        main(
            [
                "--plan",
                str(plan_path),
                "--output",
                str(destination),
                "--overlay",
                str(first),
                "--overlay",
                str(second),
                "--smoke",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    run_plan = calls[0]["run_plan"]
    assert isinstance(run_plan, tuple)
    assert len(run_plan) == 55
    assert {row.test_seed for row in run_plan} == {20_000}
    assert calls[0]["authenticated_checkpoints"] == candidates
    method_configs = calls[0]["method_configs"]
    assert tuple(config.method.id for config in method_configs) == METHOD_OVERLAYS
    assert all(config.training.eval_episodes == 9 for config in method_configs)
    assert calls[0]["cli_overlays"] == (first.resolve(), second.resolve())
    assert calls[0]["destination"] == destination.resolve()
    assert json.loads(capsys.readouterr().out) == {
        "output": str(destination.resolve()),
        "result_label": SMOKE_RESULT_LABEL,
    }


def test_missing_checkpoint_fails_before_execution_without_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    missing = plan.ppo_run_bindings[0].checkpoint_path
    assert missing is not None
    missing.unlink()
    destination = tmp_path / "evaluation"
    called = False
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)

    def unexpected(**kwargs: object) -> Path:
        nonlocal called
        called = True
        return destination

    monkeypatch.setattr(evaluate_module, "run_evaluation_bundle", unexpected)

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "checkpoint" in captured.err.lower()
    assert "not found" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert called is False
    assert not destination.exists()


def test_checkpoint_not_in_authenticated_candidate_inventory_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    binding = plan.ppo_run_bindings[0]
    copied = binding.training_run_dir / "checkpoints" / "copied_model.zip"
    copied.write_bytes(binding.checkpoint_path.read_bytes())
    rebound = binding.model_copy(update={"checkpoint_path": copied})
    plan = plan.model_copy(update={"ppo_run_bindings": (rebound, *plan.ppo_run_bindings[1:])})
    destination = tmp_path / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    original_candidate = _candidate(binding)
    monkeypatch.setattr(
        evaluate_module,
        "discover_checkpoint_candidates",
        lambda run_dir: (original_candidate,),
    )
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "checkpoint" in captured.err.lower()
    assert "authenticated" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not destination.exists()


def test_checkpoint_outside_bound_training_run_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    original = plan.ppo_run_bindings[0]
    assert original.checkpoint_path is not None
    external = tmp_path / "foreign" / "final_model.zip"
    external.parent.mkdir()
    external.write_bytes(original.checkpoint_path.read_bytes())
    rebound = original.model_copy(update={"checkpoint_path": external})
    plan = plan.model_copy(update={"ppo_run_bindings": (rebound, *plan.ppo_run_bindings[1:])})
    destination = tmp_path / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "checkpoint" in captured.err.lower()
    assert "training run" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not destination.exists()


def test_misordered_method_overlay_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    plan.method_overlays[0].write_text("method:\n  id: proposed\n", encoding="utf-8")
    destination = tmp_path / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "overlay" in captured.err.lower()
    assert "b0_rule" in captured.err
    assert "Traceback" not in captured.err
    assert not destination.exists()


@pytest.mark.parametrize(
    "payload",
    (
        "method:\n  id: b0_rule\n",
        "unexpected_phase6_field: true\n",
        "training: 7\n",
        "training:\n  eval_episodes: 7\ntraining:\n  eval_episodes: 9\n",
    ),
)
def test_cli_overlay_malformed_conflicting_or_method_override_fails_before_execution(
    tmp_path: Path,
    payload: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    overlay = tmp_path / "invalid.yaml"
    overlay.write_text(payload, encoding="utf-8")
    destination = tmp_path / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert (
        main(
            [
                "--plan",
                str(plan_path),
                "--output",
                str(destination),
                "--overlay",
                str(overlay),
                "--smoke",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert not destination.exists()


def test_duplicate_cli_overlay_paths_fail_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    overlay = tmp_path / "duplicate.yaml"
    overlay.write_text("training:\n  eval_episodes: 7\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert (
        main(
            [
                "--plan",
                str(plan_path),
                "--output",
                str(tmp_path / "evaluation"),
                "--overlay",
                str(overlay),
                "--overlay",
                str(overlay),
                "--smoke",
            ]
        )
        == 2
    )


def test_output_inside_authenticated_training_run_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    destination = plan.ppo_run_bindings[0].training_run_dir / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    _install_authenticated_discovery(monkeypatch, plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2
    captured = capsys.readouterr()
    assert "source" in captured.err.lower() or "training run" in captured.err.lower()
    assert not destination.exists()


@pytest.mark.parametrize("link_kind", ("dangling", "parent"))
def test_output_link_or_aliased_parent_fails_before_execution(
    tmp_path: Path,
    link_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    if link_kind == "dangling":
        destination = tmp_path / "dangling"
        target = tmp_path / "missing-target"
        try:
            os.symlink(target, destination, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")
    else:
        target = tmp_path / "real-parent"
        target.mkdir()
        linked_parent = tmp_path / "linked-parent"
        try:
            os.symlink(target, linked_parent, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")
        destination = linked_parent / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    _install_authenticated_discovery(monkeypatch, plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2


def test_existing_output_and_malformed_plan_fail_concisely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("not: [valid\n", encoding="utf-8")
    destination = tmp_path / "occupied"
    destination.mkdir()
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "already exists" in captured.err.lower()
    assert "Traceback" not in captured.err

    destination.rmdir()
    assert main(["--plan", str(plan_path), "--output", str(destination)]) == 2
    captured = capsys.readouterr()
    assert "plan" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not destination.exists()


def test_plan_validation_reports_only_the_first_stable_pydantic_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "incomplete.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    destination = tmp_path / "evaluation"

    assert main(["--plan", str(plan_path), "--output", str(destination)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "evaluation failed: evaluation_id: Field required\n"
    assert not destination.exists()


def test_orchestration_error_is_traceback_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("plan_kind: phase6_smoke\n", encoding="utf-8")
    plan = _plan(tmp_path)
    destination = tmp_path / "evaluation"
    monkeypatch.setattr(evaluate_module, "load_phase6_publication_plan", lambda path: plan)
    _install_authenticated_discovery(monkeypatch, plan)
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation_bundle",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulator exploded\ninternal details\r\nmore details")
        ),
    )

    assert main(["--plan", str(plan_path), "--output", str(destination), "--smoke"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("evaluation failed: simulator exploded internal details more details\n")
    assert "Traceback" not in captured.err
    assert not destination.exists()


def test_smoke_runtime_delegates_without_fabricating_checkpoint_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    destination = tmp_path / "evaluation"
    captured: list[dict[str, object]] = []

    def implementation(**kwargs: object) -> Path:
        captured.append(kwargs)
        return destination

    monkeypatch.setattr("mad_driving.evaluation.bundle.run_evaluation_bundle", implementation)

    published = evaluate_module.run_evaluation_bundle(
        evaluation_config=plan,
        plan_path=tmp_path / "plan.yaml",
        run_plan=(),
        method_configs=(),
        method_profiles=(),
        cli_overlays=(),
        authenticated_checkpoints=(),
        destination=destination,
        smoke=True,
        environment_factory=lambda spec, config: pytest.fail("must not execute"),
        policy_factory=lambda spec, config, candidate: pytest.fail("must not execute"),
        frame_provider=lambda spec, count: (),
    )

    assert published == destination
    assert len(captured) == 1
    assert captured[0]["selection_scores"] is None


def test_phase6_smoke_config_is_explicit_short_and_exact() -> None:
    repository = Path(__file__).resolve().parents[3]
    config = load_evaluation_plan(repository / "configs" / "evaluation" / "phase6_smoke.yaml")

    assert config.plan_kind == "phase6_smoke"
    assert config.is_formal is False
    assert config.result_label == SMOKE_RESULT_LABEL
    assert config.episodes_per_case == 1
    assert config.max_episode_steps is not None
    assert 0 < config.max_episode_steps <= 64
    assert config.test_seed_start == 20_000
    assert tuple(path.as_posix() for path in config.method_overlays) == tuple(
        f"configs/methods/{method}.yaml" for method in METHOD_OVERLAYS
    )
    assert {(binding.method_id, binding.policy_seed) for binding in config.ppo_run_bindings} == {
        (method, 42) for method in PPO_METHODS
    }
    assert all(binding.checkpoint_path is not None for binding in config.ppo_run_bindings)

    checkpoint_paths = {
        (binding.method_id, binding.policy_seed): str(binding.checkpoint_path)
        for binding in config.ppo_run_bindings
        if binding.checkpoint_path is not None
    }
    rows = build_smoke_plan(config, checkpoint_paths)
    assert len(rows) == 55
    assert all(row.is_formal is False for row in rows)
    assert all(row.policy_seed is None for row in rows if row.method_id == "b0_rule")
    assert all(row.checkpoint_path is None for row in rows if row.method_id == "b0_rule")
    assert {row.test_seed for row in rows} == {20_000}
    assert config.capture_episode_keys == ("proposed_system_42_level1_lead_brake_20000",)
    assert {(row.track, row.method_id, row.shield_mode) for row in rows} == {
        ("decision", "b1_nominal", "monitor"),
        ("decision", "b2_multi_no_review", "monitor"),
        ("decision", "proposed", "monitor"),
        ("system", "b0_rule", "enforce"),
        ("system", "b1_nominal", "enforce"),
        ("system", "b2_multi_no_review", "enforce"),
        ("system", "proposed", "enforce"),
        ("ablation", "proposed", "enforce"),
        ("ablation", "proposed_no_critic", "enforce"),
        ("ablation", "proposed_no_shield", "off"),
        ("ablation", "proposed_no_hazard", "enforce"),
    }
