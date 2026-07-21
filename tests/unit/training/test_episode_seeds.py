import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from mad_driving.training.episode_seeds import (
    EpisodeSeedRecordingWrapper,
    summarize_episode_seed_artifacts,
)


class ResetInfoEnv(gym.Env[NDArray[np.float32], int]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def __init__(self, reset_infos: list[dict[str, Any]]) -> None:
        self.reset_infos = reset_infos
        self.reset_calls = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        info = self.reset_infos[self.reset_calls]
        self.reset_calls += 1
        return np.zeros(24, dtype=np.float32), info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(24, dtype=np.float32), 0.0, False, False, {}


def seed_info(episode: int, road: int, parameters: int) -> dict[str, int]:
    return {
        "episode_rng_seed": episode,
        "metadrive_scenario_index": road,
        "scenario_parameter_seed": parameters,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_each_actual_reset_info_in_order_and_summarizes_it(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    environment = ResetInfoEnv([seed_info(41, 101, 201), seed_info(42, 102, 202)])
    wrapped = EpisodeSeedRecordingWrapper(
        environment,
        workspace=workspace,
        role="train",
        worker_index=0,
    )

    wrapped.reset(seed=999)
    wrapped.reset()

    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    assert read_jsonl(artifact) == [
        {
            "episode_rng_seed": 41,
            "metadrive_scenario_index": 101,
            "role": "train",
            "scenario_parameter_seed": 201,
            "worker_index": 0,
        },
        {
            "episode_rng_seed": 42,
            "metadrive_scenario_index": 102,
            "role": "train",
            "scenario_parameter_seed": 202,
            "worker_index": 0,
        },
    ]
    summaries = summarize_episode_seed_artifacts(
        workspace,
        expected_identities=(("train", 0),),
    )
    assert summaries == (
        {
            "path": "episode_seeds/train-worker-000.jsonl",
            "record_count": 2,
            "role": "train",
            "schema_version": 1,
            "sha256": summaries[0]["sha256"],
            "worker_index": 0,
        },
    )
    assert len(summaries[0]["sha256"]) == 64


def test_role_and_worker_artifacts_never_collide(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    identities = (("train", 0), ("train", 1), ("validation", 0))

    for ordinal, (role, worker_index) in enumerate(identities):
        wrapped = EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(ordinal, 100 + ordinal, 200 + ordinal)]),
            workspace=workspace,
            role=role,
            worker_index=worker_index,
        )
        wrapped.reset()

    summaries = summarize_episode_seed_artifacts(
        workspace,
        expected_identities=identities,
    )
    assert [summary["path"] for summary in summaries] == [
        "episode_seeds/train-worker-000.jsonl",
        "episode_seeds/train-worker-001.jsonl",
        "episode_seeds/validation-worker-000.jsonl",
    ]
    assert [read_jsonl(workspace / str(summary["path"]))[0]["role"] for summary in summaries] == [
        "train",
        "train",
        "validation",
    ]


@pytest.mark.parametrize(
    "malformed_info",
    [
        {"metadrive_scenario_index": 1, "scenario_parameter_seed": 2},
        seed_info(True, 1, 2),
        seed_info(-1, 1, 2),
        seed_info(1, 1.5, 2),
        seed_info(1, 2, math.nan),
    ],
)
def test_malformed_reset_seed_info_fails_closed_without_a_record(
    tmp_path: Path,
    malformed_info: dict[str, Any],
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([malformed_info]),
        workspace=workspace,
        role="validation",
        worker_index=0,
    )

    with pytest.raises(ValueError, match="reset seed info"):
        wrapped.reset()

    artifact = workspace / "episode_seeds" / "validation-worker-000.jsonl"
    assert artifact.read_bytes() == b""


def test_summary_rejects_malformed_or_unowned_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    artifact_dir = workspace / "episode_seeds"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "train-worker-000.jsonl"
    artifact.write_text('{"role":"train","worker_index":0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="seed artifact"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_identities=(("train", 0),),
        )

    outside = tmp_path / "outside.jsonl"
    outside.write_text("foreign", encoding="utf-8")
    assert outside.read_text(encoding="utf-8") == "foreign"


@pytest.mark.parametrize(
    ("role", "worker_index", "message"),
    [
        ("unknown", 0, "role"),
        ("train", True, "worker_index"),
        ("train", -1, "worker_index"),
    ],
)
def test_recorder_rejects_invalid_environment_identity(
    tmp_path: Path,
    role: Any,
    worker_index: Any,
    message: str,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match=message):
        EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(1, 2, 3)]),
            workspace=workspace,
            role=role,
            worker_index=worker_index,
        )

    assert list(workspace.iterdir()) == []


def test_recorder_requires_an_existing_directory_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "not-a-directory"
    workspace.write_text("owned file", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="workspace"):
        EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(1, 2, 3)]),
            workspace=workspace,
            role="train",
            worker_index=0,
        )

    assert workspace.read_text(encoding="utf-8") == "owned file"


def test_reset_forwards_nonempty_options_and_records_actual_info(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )

    wrapped.reset(options={"mode": "evaluation"})

    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    assert read_jsonl(artifact)[0]["episode_rng_seed"] == 1


def test_summary_rejects_an_extra_file_in_owned_artifact_inventory(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    wrapped.reset()
    extra = workspace / "episode_seeds" / "foreign.jsonl"
    extra.write_text("foreign", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_identities=(("train", 0),),
        )

    assert extra.read_text(encoding="utf-8") == "foreign"
