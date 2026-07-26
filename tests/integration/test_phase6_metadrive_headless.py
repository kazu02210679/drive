from __future__ import annotations

import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from mad_driving.cli.evaluate import run_evaluation_bundle
from mad_driving.cli.train import main as train_main
from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.evaluation.metadrive import MetaDriveEvaluationRuntime
from mad_driving.evaluation.models import EvaluationRunSpec, Phase6PublicationPlan
from mad_driving.evaluation.plans import build_smoke_plan
from mad_driving.evaluation.runner import run_evaluation_episode
from mad_driving.evaluation.selection import (
    CheckpointCandidate,
    discover_checkpoint_candidates,
    validate_ppo_checkpoint_archive,
)
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.scenarios import EpisodeSeedAllocator
from mad_driving.training.metadata import RESEARCH_CONTRACT_VERSION
from mad_driving.visualization import METHOD_ORDER, _verify_bundle

PPO_METHODS = tuple(method_id for method_id in METHOD_ORDER if method_id != "b0_rule")


@dataclass(frozen=True)
class ProposedSmokeCheckpoint:
    config: AppConfig
    candidate: CheckpointCandidate
    run_dir: Path


@pytest.fixture(scope="module")
def phase6_smoke_checkpoints(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, ProposedSmokeCheckpoint]:
    root = tmp_path_factory.mktemp("phase6-ppo-smoke")
    overlay = root / "training-smoke.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "metadrive": {"horizon": 120},
                "scenarios": {
                    "lead_brake": {
                        "trigger_s": {"minimum": 0.1, "maximum": 0.1},
                        "survival_s": 0.1,
                    }
                },
                "training": {
                    "n_steps": 8,
                    "batch_size": 8,
                    "n_epochs": 1,
                    "total_timesteps": 8,
                    "smoke_timesteps": 8,
                    "checkpoint_interval_steps": 8,
                    "eval_interval_steps": 8,
                    "eval_episodes": 1,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    values: dict[str, ProposedSmokeCheckpoint] = {}
    for method_id in PPO_METHODS:
        run_dir = root / method_id
        method_overlay = f"configs/methods/{method_id}.yaml"
        args = [
            "--config",
            "configs/base.yaml",
            "--overlay",
            "configs/scenarios/lead_brake.yaml",
            "--overlay",
            method_overlay,
            "--overlay",
            str(overlay),
            "--smoke",
            "--run-dir",
            str(run_dir),
        ]
        assert train_main(args) == 0
        config = load_config(
            "configs/base.yaml",
            "configs/scenarios/lead_brake.yaml",
            method_overlay,
            overlay,
        )
        candidates = tuple(
            candidate
            for candidate in discover_checkpoint_candidates(run_dir)
            if candidate.path.name == "final_model.zip"
        )
        assert len(candidates) == 1
        candidate = candidates[0]
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["research_contract_version"] == RESEARCH_CONTRACT_VERSION == 7
        assert metadata["observation_schema_version"] == 1
        assert metadata["observation_shape"] == [24]
        assert metadata["action_schema_version"] == 1
        assert metadata["method_profile"]["method_id"] == method_id
        assert candidate.sha256 == validate_ppo_checkpoint_archive(candidate.path)
        assert candidate.training_timestep == 8
        values[method_id] = ProposedSmokeCheckpoint(config, candidate, run_dir)
    return values


@pytest.fixture(scope="module")
def proposed_smoke_checkpoint(
    phase6_smoke_checkpoints: dict[str, ProposedSmokeCheckpoint],
) -> ProposedSmokeCheckpoint:
    return phase6_smoke_checkpoints["proposed"]


@pytest.mark.integration
def test_real_b0_lead_brake_is_finite_captured_and_reproducible(tmp_path: Path) -> None:
    config = load_config(
        "configs/base.yaml",
        "configs/scenarios/lead_brake.yaml",
        "configs/methods/b0_rule.yaml",
    )
    spec = EvaluationRunSpec(
        track="system",
        method_id="b0_rule",
        policy_seed=None,
        checkpoint_path=None,
        scenario_cell_id="level1_lead_brake",
        episode_index=0,
        test_seed=20_000,
        shield_mode="enforce",
        is_formal=False,
    )
    episode_key = "b0_rule_system_rule_level1_lead_brake_20000"
    expected_seeds = EpisodeSeedAllocator("test", config.scenarios.test, 0).allocate(20_000)

    def run_once(
        name: str,
    ) -> tuple[tuple[tuple[int, int, int, bool, bool], ...], tuple[bytes, ...]]:
        runtime = MetaDriveEvaluationRuntime(
            capture_episode_keys=(episode_key,),
            max_episode_steps=16,
        )
        result = run_evaluation_episode(
            spec,
            environment=runtime.environment_factory(spec, config),
            policy=runtime.policy_factory(spec, config, None),
            config=config,
            destination=tmp_path / name,
        )
        frames = runtime.frame_provider(spec, len(result.step_records))
        scientific_trace = tuple(
            (
                record.step_index,
                record.requested_action,
                record.executed_action,
                record.terminated,
                record.truncated,
            )
            for record in result.step_records
        )
        assert result.episode_record.scenario_id == "lead_brake"
        assert result.episode_record.episode_rng_seed == 20_000
        assert (
            result.episode_record.metadrive_scenario_index
            == expected_seeds.metadrive_scenario_index
        )
        assert all(np.isfinite(record.reward_total) for record in result.step_records)
        assert len(frames) == len(result.step_records)
        with Image.open(io.BytesIO(frames[0])) as image:
            assert image.mode == "RGB"
            assert image.width > 0
            assert image.height > 0
        return scientific_trace, frames

    first_trace, first_frames = run_once("first")
    second_trace, second_frames = run_once("second")

    assert first_trace == second_trace
    assert len(first_frames) == len(second_frames)


@pytest.mark.integration
def test_real_proposed_checkpoint_runs_occluded_crossing_with_24d_observation(
    tmp_path: Path,
    proposed_smoke_checkpoint: ProposedSmokeCheckpoint,
) -> None:
    config = proposed_smoke_checkpoint.config
    candidate = proposed_smoke_checkpoint.candidate
    spec = EvaluationRunSpec(
        track="system",
        method_id="proposed",
        policy_seed=42,
        checkpoint_path=str(candidate.path),
        scenario_cell_id="level3_occluded_crossing",
        episode_index=0,
        test_seed=20_000,
        shield_mode="enforce",
        is_formal=False,
    )
    runtime = MetaDriveEvaluationRuntime(
        capture_episode_keys=(),
        max_episode_steps=16,
    )

    result = run_evaluation_episode(
        spec,
        environment=runtime.environment_factory(spec, config),
        policy=runtime.policy_factory(spec, config, candidate),
        config=config,
        destination=tmp_path / "proposed-occluded",
        checkpoint_sha256=candidate.sha256,
    )

    assert result.episode_record.scenario_id == "occluded_crossing"
    assert result.episode_record.checkpoint_sha256 == candidate.sha256
    assert result.episode_record.step_count == 16
    assert result.episode_record.truncated is True
    assert all(record.method_profile.method_id == "proposed" for record in result.step_records)
    assert all(np.isfinite(record.reward_total) for record in result.step_records)


@pytest.mark.integration
def test_real_task10_smoke_bundle_publishes_all_methods_and_documented_render(
    tmp_path: Path,
    phase6_smoke_checkpoints: dict[str, ProposedSmokeCheckpoint],
) -> None:
    proposed_config = phase6_smoke_checkpoints["proposed"].config
    method_configs = tuple(
        (
            proposed_config.model_copy(
                update={"method": proposed_config.method.model_copy(update={"id": method_id})}
            )
            if method_id == "b0_rule"
            else phase6_smoke_checkpoints[method_id].config
        )
        for method_id in METHOD_ORDER
    )
    candidates = tuple(phase6_smoke_checkpoints[method_id].candidate for method_id in PPO_METHODS)
    plan = Phase6PublicationPlan.model_validate(
        {
            "plan_kind": "phase6_smoke",
            "evaluation_id": "phase6_real_metadrive_smoke",
            "is_formal": False,
            "result_label": "SMOKE - NOT A RESEARCH RESULT",
            "app_config_path": "configs/base.yaml",
            "method_overlays": [f"configs/methods/{method_id}.yaml" for method_id in METHOD_ORDER],
            "max_episode_steps": 2,
            "episodes_per_case": 1,
            "test_seed_start": 20_000,
            "ppo_run_bindings": [
                {
                    "method_id": candidate.method_id,
                    "policy_seed": candidate.policy_seed,
                    "training_run_dir": str(phase6_smoke_checkpoints[candidate.method_id].run_dir),
                    "checkpoint_path": str(candidate.path),
                }
                for candidate in candidates
            ],
            "capture_episode_keys": ["proposed_system_42_level1_lead_brake_20000"],
        }
    )
    plan_path = tmp_path / "phase6-real-smoke.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_paths = {
        (candidate.method_id, candidate.policy_seed): str(candidate.path)
        for candidate in candidates
    }
    run_plan = build_smoke_plan(plan, checkpoint_paths)
    runtime = MetaDriveEvaluationRuntime(
        capture_episode_keys=plan.capture_episode_keys,
        max_episode_steps=plan.max_episode_steps,
    )

    published = run_evaluation_bundle(
        evaluation_config=plan,
        plan_path=plan_path,
        run_plan=run_plan,
        method_configs=method_configs,
        method_profiles=tuple(
            MethodProfileSnapshot.from_method_id(method_id) for method_id in METHOD_ORDER
        ),
        cli_overlays=(),
        authenticated_checkpoints=candidates,
        destination=tmp_path / "phase6-real-bundle",
        smoke=True,
        environment_factory=runtime.environment_factory,
        policy_factory=runtime.policy_factory,
        frame_provider=runtime.frame_provider,
        selection_scores=None,
    )

    verified = _verify_bundle(published)
    traces = tuple(published.glob("episodes/**/*_trace.jsonl"))
    documented_trace = (
        published
        / "episodes"
        / "proposed"
        / "system"
        / "42"
        / "level1_lead_brake"
        / "episode_20000_trace.jsonl"
    )
    renders = tuple(published.glob("renders/*.gif"))

    assert verified.root == published.resolve()
    assert len(run_plan) == len(traces) == 55
    assert documented_trace.is_file()
    selection = json.loads((published / "selected_checkpoints.json").read_text(encoding="utf-8"))
    assert selection["selected_checkpoints"] == []
    assert len(renders) == 1
    with Image.open(renders[0]) as image:
        assert image.format == "GIF"
        assert image.n_frames == 2
        assert image.width > 0
        assert image.height > 0

    comparison = tmp_path / "offline-comparison"
    render = tmp_path / "offline-render"
    script = f"""
import importlib.abc
import sys

FORBIDDEN = ("metadrive", "stable_baselines3", "tensorboard")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise AssertionError(f"forbidden offline import: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())
from mad_driving.cli import compare, render_episode

assert compare.main([
    "--evaluation", {str(published)!r},
    "--output", {str(comparison)!r},
]) == 0
assert render_episode.main([
    "--evaluation", {str(published)!r},
    "--episode-key", "proposed_system_42_level1_lead_brake_20000",
    "--output", {str(render)!r},
]) == 0
assert not any(
    module == name or module.startswith(name + ".")
    for module in sys.modules
    for name in FORBIDDEN
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    _verify_bundle(comparison)
    _verify_bundle(render)
    with Image.open(render / "render.gif") as image:
        assert image.n_frames == 2
