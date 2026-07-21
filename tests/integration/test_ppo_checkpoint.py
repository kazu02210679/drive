import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from mad_driving.config.models import AppConfig
from mad_driving.scenarios import EnvironmentRole
from mad_driving.training import run_training
from mad_driving.training.metadata import RESEARCH_CONTRACT_VERSION


class TinyDeterministicEnv(gym.Env[NDArray[np.float32], int]):
    """Fast deterministic 24D/Discrete(4) environment for real PPO integration."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def __init__(self, *, role: EnvironmentRole, worker_index: int) -> None:
        self.steps = 0
        self.closed = False
        self.role = role
        self.worker_index = worker_index
        self.reset_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.steps = 0
        role_offset = 100 if self.role == "train" else 10_000
        episode_seed = role_offset + self.worker_index * 100 + self.reset_count
        self.reset_count += 1
        return np.zeros(24, dtype=np.float32), {
            "episode_rng_seed": episode_seed,
            "metadrive_scenario_index": episode_seed + 20_000,
            "scenario_parameter_seed": episode_seed + 30_000,
        }

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self.steps += 1
        observation = np.full(24, self.steps / 4.0, dtype=np.float32)
        reward = float(action == self.steps % 4)
        return observation, reward, self.steps == 4, False, {}

    def close(self) -> None:
        self.closed = True


class SeedAwareTinyEnv(gym.Env[NDArray[np.float32], int]):
    """One-step environment that reports its explicit or derived Gymnasium seed."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        episode_seed = (
            seed if seed is not None else int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )
        return np.zeros(24, dtype=np.float32), {
            "episode_rng_seed": episode_seed,
            "metadrive_scenario_index": episode_seed + 20_000,
            "scenario_parameter_seed": episode_seed + 30_000,
        }

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        return np.zeros(24, dtype=np.float32), 0.0, True, False, {}


class CapturingSubprocVecEnv(SubprocVecEnv):
    created: ClassVar[list["CapturingSubprocVecEnv"]] = []

    def __init__(self, env_fns: list[Any]) -> None:
        self.identity_events: list[str] = []
        super().__init__(env_fns)
        type(self).created.append(self)

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        if attr_name == "episode_seed_artifact_descriptor":
            self.identity_events.append(f"identity:closed={self.closed}")
        return super().get_attr(attr_name, indices)


def tiny_env_factory(
    received_config: AppConfig,
    *,
    role: EnvironmentRole,
    worker_index: int,
) -> TinyDeterministicEnv:
    del received_config
    return TinyDeterministicEnv(role=role, worker_index=worker_index)


def seed_aware_tiny_env_factory(
    received_config: AppConfig,
    *,
    role: EnvironmentRole,
    worker_index: int,
) -> SeedAwareTinyEnv:
    del received_config, role, worker_index
    return SeedAwareTinyEnv()


def make_real_ppo_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 7,
            "scenario_id": "tiny-ppo-artifacts",
            "decision_steps": 4,
            "fixed_action": [0.0, 0.0],
            "metadrive": {"use_render": False},
            "training": {
                "n_steps": 8,
                "batch_size": 8,
                "total_timesteps": 16,
                "checkpoint_interval_steps": 8,
                "eval_interval_steps": 8,
                "eval_episodes": 1,
            },
        }
    )


def load_and_predict_policy(checkpoint: Path) -> PPO:
    model = PPO.load(checkpoint, device="cpu")
    action, _ = model.predict(np.zeros(24, dtype=np.float32), deterministic=True)
    predicted_action = int(np.asarray(action).item())
    assert 0 <= predicted_action <= 3
    return model


def checkpoint_hash(checkpoint: Path) -> str:
    with checkpoint.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


def episode_seed_records(run_dir: Path, role: EnvironmentRole) -> list[int]:
    artifact = run_dir / "episode_seeds" / f"{role}-worker-000.jsonl"
    return [
        int(record["episode_rng_seed"])
        for record in (
            json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()[1:]
        )
    ]


@pytest.mark.integration
def test_validation_reuses_the_same_episode_seed_sequence_for_fresh_and_resumed_runs(
    tmp_path: Path,
) -> None:
    config = make_real_ppo_config()
    first_run = tmp_path / "seed-sequence-first"
    first_result = run_training(
        config,
        smoke=False,
        run_dir=first_run,
        env_factory=seed_aware_tiny_env_factory,
    )
    continued_run = tmp_path / "seed-sequence-resumed"
    resumed_config = config.model_copy(
        update={"training": config.training.model_copy(update={"seed": 123})}
    )

    run_training(
        resumed_config,
        smoke=False,
        run_dir=continued_run,
        resume_from=first_result.final_checkpoint,
        env_factory=seed_aware_tiny_env_factory,
    )

    first_validation_seeds = episode_seed_records(first_run, "validation")
    resumed_validation_seeds = episode_seed_records(continued_run, "validation")
    assert len(first_validation_seeds) == 4
    assert first_validation_seeds[0] == config.seed
    assert first_validation_seeds[:2] == first_validation_seeds[2:]
    assert resumed_validation_seeds == first_validation_seeds


@pytest.mark.integration
def test_parent_collects_seed_identity_through_real_subproc_control_channel(
    tmp_path: Path,
) -> None:
    base_config = make_real_ppo_config()
    config = base_config.model_copy(
        update={
            "training": base_config.training.model_copy(update={"num_envs": 2}),
        }
    )
    run_dir = tmp_path / "subproc-identities"
    CapturingSubprocVecEnv.created = []

    run_training(
        config,
        smoke=False,
        run_dir=run_dir,
        env_factory=tiny_env_factory,
        subproc_vec_env_factory=CapturingSubprocVecEnv,
    )

    assert len(CapturingSubprocVecEnv.created) == 1
    vector_env = CapturingSubprocVecEnv.created[0]
    assert vector_env.identity_events == ["identity:closed=False"]
    assert vector_env.closed is True
    assert [process.exitcode for process in vector_env.processes] == [0, 0]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summaries = metadata["episode_seed_artifacts"]
    assert [(item["role"], item["worker_index"]) for item in summaries] == [
        ("train", 0),
        ("train", 1),
        ("validation", 0),
    ]
    for summary in summaries:
        artifact = run_dir / summary["path"]
        stat_result = artifact.stat()
        assert summary["file_identity"] == {
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
        }


@pytest.mark.integration
def test_real_ppo_writes_artifacts_and_resumes_transactionally(tmp_path: Path) -> None:
    config = make_real_ppo_config()
    run_dir = tmp_path / "run"
    environments: list[TinyDeterministicEnv] = []

    def env_factory(
        received_config: AppConfig,
        *,
        role: EnvironmentRole,
        worker_index: int,
    ) -> TinyDeterministicEnv:
        assert received_config == config
        assert received_config is not config
        assert role in {"train", "validation"}
        assert worker_index == 0
        environment = TinyDeterministicEnv(role=role, worker_index=worker_index)
        environments.append(environment)
        return environment

    first_result = run_training(
        config,
        smoke=False,
        run_dir=run_dir,
        env_factory=env_factory,
    )

    checkpoints_dir = run_dir / "checkpoints"
    periodic_checkpoints = {
        path.name: path for path in checkpoints_dir.glob("ppo_checkpoint_*_steps.zip")
    }
    expected_initial_timesteps = {
        "ppo_checkpoint_8_steps.zip": 8,
        "ppo_checkpoint_16_steps.zip": 16,
    }
    assert periodic_checkpoints.keys() == expected_initial_timesteps.keys()
    all_checkpoints = [
        *periodic_checkpoints.values(),
        first_result.best_checkpoint,
        first_result.final_checkpoint,
    ]
    assert first_result.timesteps == 16
    assert all(path.is_file() and zipfile.is_zipfile(path) for path in all_checkpoints)
    for name, expected_timesteps in expected_initial_timesteps.items():
        assert (
            load_and_predict_policy(periodic_checkpoints[name]).num_timesteps == expected_timesteps
        )
    initial_best_hash = checkpoint_hash(first_result.best_checkpoint)
    initial_best = load_and_predict_policy(first_result.best_checkpoint)
    assert initial_best.num_timesteps in {8, 16}
    initial_final_hash = checkpoint_hash(first_result.final_checkpoint)
    initial_final = load_and_predict_policy(first_result.final_checkpoint)
    assert initial_final.num_timesteps == 16
    assert list((run_dir / "tensorboard").rglob("events.out.tfevents.*"))

    seed_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert seed_metadata["research_contract_version"] == RESEARCH_CONTRACT_VERSION
    assert seed_metadata["observation_schema_version"] == 1
    seed_summaries = seed_metadata["episode_seed_artifacts"]
    assert [(summary["role"], summary["worker_index"]) for summary in seed_summaries] == [
        ("train", 0),
        ("validation", 0),
    ]
    seed_records: dict[str, list[dict[str, object]]] = {}
    for summary in seed_summaries:
        artifact = run_dir / summary["path"]
        values = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
        header, *records = values
        seed_records[summary["role"]] = records
        assert summary["record_count"] == len(records)
        assert summary["sha256"] == checkpoint_hash(artifact)
        assert summary["schema_version"] == 2
        assert summary["file_identity"] == header["file_identity"]
    train_seeds = [record["episode_rng_seed"] for record in seed_records["train"]]
    validation_seeds = [record["episode_rng_seed"] for record in seed_records["validation"]]
    assert len(set(train_seeds)) >= 2
    assert all(seed < 10_000 for seed in train_seeds)
    assert all(seed >= 10_000 for seed in validation_seeds)
    assert all(
        set(record)
        == {
            "episode_rng_seed",
            "metadrive_scenario_index",
            "role",
            "scenario_parameter_seed",
            "worker_index",
        }
        for records in seed_records.values()
        for record in records
    )

    resolved_config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved_config == config.model_dump(mode="json")
    assert {
        key: resolved_config["training"][key]
        for key in (
            "n_steps",
            "batch_size",
            "total_timesteps",
            "checkpoint_interval_steps",
            "eval_interval_steps",
        )
    } == {
        "n_steps": 8,
        "batch_size": 8,
        "total_timesteps": 16,
        "checkpoint_interval_steps": 8,
        "eval_interval_steps": 8,
    }
    # The validation environment is constructed in its subprocess and is not shared back.
    assert len(environments) == 1
    assert all(environment.closed for environment in environments)
    source_files = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    continued_dir = tmp_path / "continued"

    resumed_result = run_training(
        config,
        smoke=False,
        run_dir=continued_dir,
        resume_from=first_result.final_checkpoint,
        env_factory=env_factory,
    )

    assert resumed_result.final_checkpoint == continued_dir / "checkpoints" / "final_model.zip"
    assert resumed_result.timesteps == 16
    resumed_checkpoints_dir = continued_dir / "checkpoints"
    resumed_periodic_checkpoints = {
        path.name: path for path in resumed_checkpoints_dir.glob("ppo_checkpoint_*_steps.zip")
    }
    expected_resumed_timesteps = {
        "ppo_checkpoint_24_steps.zip": 24,
        "ppo_checkpoint_32_steps.zip": 32,
    }
    assert resumed_periodic_checkpoints.keys() == expected_resumed_timesteps.keys()
    assert all(
        zipfile.is_zipfile(path)
        for path in [
            *resumed_periodic_checkpoints.values(),
            resumed_result.best_checkpoint,
            resumed_result.final_checkpoint,
        ]
    )
    for name, expected_timesteps in expected_resumed_timesteps.items():
        resumed_periodic = load_and_predict_policy(resumed_periodic_checkpoints[name])
        assert resumed_periodic.num_timesteps == expected_timesteps
    resumed_best = load_and_predict_policy(resumed_result.best_checkpoint)
    assert resumed_best.num_timesteps in {24, 32}
    assert checkpoint_hash(resumed_result.best_checkpoint) != initial_best_hash
    resumed_final = load_and_predict_policy(resumed_result.final_checkpoint)
    assert resumed_final.num_timesteps == 32
    assert checkpoint_hash(resumed_result.final_checkpoint) != initial_final_hash
    assert not list(resumed_checkpoints_dir.glob(".training-*"))
    assert {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    } == source_files
    resume_metadata = json.loads((continued_dir / "run_metadata.json").read_text(encoding="utf-8"))[
        "resume"
    ]
    assert resume_metadata["parent_checkpoint_sha256"] == initial_final_hash
    assert resume_metadata["parent_checkpoint_path"] == str(first_result.final_checkpoint.resolve())
    assert resume_metadata["parent_run_dir"] == str(run_dir.resolve())
    assert resume_metadata["start_num_timesteps"] == 16
    assert resume_metadata["config_diff"] == {}
    assert len(environments) == 2
    assert all(environment.closed for environment in environments)
