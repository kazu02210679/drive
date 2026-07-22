import hashlib
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from mad_driving.config.models import CurriculumConfig
from mad_driving.training.curriculum import (
    CurriculumController,
    CurriculumEpisodeResult,
    CurriculumState,
    checkpoint_curriculum_sidecar_path,
    read_checkpoint_curriculum_state,
    read_curriculum_state,
    write_checkpoint_curriculum_state,
    write_curriculum_state,
)


def episode_results(
    *,
    successes: int,
    collisions: int,
    episodes: int,
    scenario_ids: tuple[str, ...] = ("nominal",),
    unnecessary_stop_duration_s: float = 0.0,
) -> tuple[CurriculumEpisodeResult, ...]:
    return tuple(
        CurriculumEpisodeResult(
            scenario_id=scenario_ids[index % len(scenario_ids)],
            success=index < successes,
            collision=index < collisions,
            route_progress=0.25,
            unnecessary_stop_duration_s=unnecessary_stop_duration_s,
        )
        for index in range(episodes)
    )


def automatic_config(**overrides: Any) -> CurriculumConfig:
    values = {
        "mode": "automatic",
        "initial_level": 0,
        "success_rate_threshold": 0.8,
        "collision_rate_threshold": 0.05,
        "consecutive_evaluations": 2,
    }
    values.update(overrides)
    return CurriculumConfig.model_validate(values)


def fixed_config(level: int) -> CurriculumConfig:
    return CurriculumConfig(mode="fixed", fixed_level=level)


def test_curriculum_state_is_immutable_and_strictly_validated() -> None:
    state = CurriculumState(level=1, consecutive_passes=2, evaluations=3)

    with pytest.raises(FrozenInstanceError):
        state.level = 2  # type: ignore[misc]

    for field, value in (
        ("level", True),
        ("level", 4),
        ("consecutive_passes", -1),
        ("consecutive_passes", 1.0),
        ("evaluations", -1),
        ("evaluations", False),
    ):
        values = {"level": 0, "consecutive_passes": 0, "evaluations": 0}
        values[field] = value
        with pytest.raises(ValueError, match=field):
            CurriculumState(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="consecutive_passes"):
        CurriculumState(level=0, consecutive_passes=2, evaluations=1)


def test_automatic_curriculum_advances_after_two_passing_validations() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    first = controller.observe("validation", episode_results(successes=4, collisions=0, episodes=5))
    second = controller.observe(
        "validation", episode_results(successes=5, collisions=0, episodes=5)
    )

    assert first == CurriculumState(level=0, consecutive_passes=1, evaluations=1)
    assert second == CurriculumState(level=1, consecutive_passes=0, evaluations=2)
    assert controller.state == second


def test_automatic_curriculum_requires_both_thresholds_and_resets_failed_streak() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(1, 1, 4))

    collision_failure = controller.observe(
        "validation",
        episode_results(
            successes=5,
            collisions=1,
            episodes=5,
            scenario_ids=("lead_brake",),
        ),
    )
    success_failure = controller.observe(
        "validation",
        episode_results(
            successes=3,
            collisions=0,
            episodes=5,
            scenario_ids=("lead_brake",),
        ),
    )

    assert collision_failure == CurriculumState(1, 0, 5)
    assert success_failure == CurriculumState(1, 0, 6)


def test_automatic_curriculum_advances_exactly_one_level_and_never_regresses() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(1, 1, 7))

    advanced = controller.observe(
        "validation",
        episode_results(
            successes=5,
            collisions=0,
            episodes=5,
            scenario_ids=("lead_brake",),
        ),
    )
    failed = controller.observe(
        "validation",
        episode_results(
            successes=0,
            collisions=0,
            episodes=5,
            scenario_ids=("lead_brake", "cut_in"),
        ),
    )

    assert advanced == CurriculumState(2, 0, 8)
    assert failed == CurriculumState(2, 0, 9)


def test_automatic_curriculum_caps_at_level_three() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(3, 1, 7))

    state = controller.observe(
        "validation",
        episode_results(
            successes=5,
            collisions=0,
            episodes=5,
            scenario_ids=("occluded_crossing",),
        ),
    )

    assert state.level == 3
    assert state.evaluations == 8


def test_test_role_cannot_drive_curriculum() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    with pytest.raises(ValueError, match="test role"):
        controller.observe("test", episode_results(successes=5, collisions=0, episodes=5))


def test_automatic_curriculum_rejects_training_observations() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    with pytest.raises(ValueError, match="validation"):
        controller.observe("train", episode_results(successes=5, collisions=0, episodes=5))


@pytest.mark.parametrize("results", [(), (object(),), "invalid"])
def test_curriculum_observations_require_non_empty_typed_results(results: object) -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    with pytest.raises((TypeError, ValueError), match="results"):
        controller.observe("validation", results)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_id", ""),
        ("success", 1),
        ("collision", 0),
        ("route_progress", -0.1),
        ("route_progress", 1.1),
        ("unnecessary_stop_duration_s", -0.1),
    ],
)
def test_curriculum_episode_result_is_strictly_validated(field: str, value: object) -> None:
    values: dict[str, object] = {
        "scenario_id": "nominal",
        "success": True,
        "collision": False,
        "route_progress": 0.25,
        "unnecessary_stop_duration_s": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        CurriculumEpisodeResult(**values)  # type: ignore[arg-type]


def test_automatic_curriculum_rejects_successful_episodes_that_stop_too_long() -> None:
    controller = CurriculumController(
        automatic_config(consecutive_evaluations=1, maximum_unnecessary_stop_duration_s=1.0),
        CurriculumState(0, 0, 0),
    )

    state = controller.observe(
        "validation",
        episode_results(
            successes=5,
            collisions=0,
            episodes=5,
            unnecessary_stop_duration_s=1.1,
        ),
    )

    assert state == CurriculumState(level=0, consecutive_passes=0, evaluations=1)


def test_level_two_requires_each_scenario_to_meet_the_success_threshold() -> None:
    controller = CurriculumController(
        automatic_config(consecutive_evaluations=1),
        CurriculumState(2, 0, 4),
    )
    results = (
        *episode_results(
            successes=4,
            collisions=0,
            episodes=4,
            scenario_ids=("lead_brake",),
        ),
        *episode_results(
            successes=0,
            collisions=0,
            episodes=1,
            scenario_ids=("cut_in",),
        ),
    )

    state = controller.observe("validation", results)

    assert state == CurriculumState(level=2, consecutive_passes=0, evaluations=5)


def test_level_two_cannot_advance_when_cut_in_was_not_evaluated() -> None:
    controller = CurriculumController(
        automatic_config(consecutive_evaluations=1),
        CurriculumState(2, 0, 4),
    )

    state = controller.observe(
        "validation",
        episode_results(
            successes=5,
            collisions=0,
            episodes=5,
            scenario_ids=("lead_brake",),
        ),
    )

    assert state == CurriculumState(level=2, consecutive_passes=0, evaluations=5)


def test_fixed_curriculum_never_advances() -> None:
    controller = CurriculumController(fixed_config(level=2), CurriculumState(2, 0, 0))

    state = controller.observe(
        "validation",
        episode_results(
            successes=5,
            collisions=0,
            episodes=5,
            scenario_ids=("lead_brake", "cut_in"),
        ),
    )

    assert state == CurriculumState(level=2, consecutive_passes=0, evaluations=1)


def test_curriculum_controller_rejects_invalid_configuration_and_state_pairs() -> None:
    with pytest.raises(TypeError, match="config"):
        CurriculumController(object(), CurriculumState(0, 0, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state"):
        CurriculumController(fixed_config(0), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fixed_level"):
        CurriculumController(fixed_config(2), CurriculumState(1, 0, 0))
    with pytest.raises(ValueError, match="initial_level"):
        CurriculumController(
            automatic_config(initial_level=2),
            CurriculumState(1, 0, 0),
        )


def test_curriculum_state_round_trips_through_atomic_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "curriculum_state.yaml"
    state = CurriculumState(level=2, consecutive_passes=1, evaluations=4)

    write_curriculum_state(state, destination)

    assert read_curriculum_state(destination) == state
    assert not list(tmp_path.glob(".curriculum_state.yaml.*.tmp"))


def atomic_writer_case(
    tmp_path: Path,
    artifact_kind: str,
) -> tuple[Path, Callable[[], None]]:
    state = CurriculumState(1, 0, 1)
    if artifact_kind == "curriculum":
        destination = tmp_path / "curriculum_state.yaml"

        def writer() -> None:
            write_curriculum_state(state, destination)

    else:
        checkpoint = tmp_path / "model.zip"
        checkpoint.write_bytes(b"checkpoint")
        destination = checkpoint_curriculum_sidecar_path(checkpoint)

        def writer() -> None:
            write_checkpoint_curriculum_state(state, checkpoint)

    destination.write_bytes(b"old-state\n")
    return destination, writer


@pytest.mark.parametrize("artifact_kind", ["curriculum", "checkpoint-sidecar"])
def test_curriculum_atomic_write_preserves_old_state_when_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    destination, writer = atomic_writer_case(tmp_path, artifact_kind)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("mad_driving.training.curriculum.os.fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        writer()

    assert destination.read_bytes() == b"old-state\n"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("artifact_kind", ["curriculum", "checkpoint-sidecar"])
def test_curriculum_atomic_write_preserves_old_state_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    destination, writer = atomic_writer_case(tmp_path, artifact_kind)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("mad_driving.training.curriculum.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        writer()

    assert destination.read_bytes() == b"old-state\n"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("artifact_kind", ["curriculum", "checkpoint-sidecar"])
def test_curriculum_atomic_write_preserves_primary_and_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    destination, writer = atomic_writer_case(tmp_path, artifact_kind)
    cleanup_attempts: list[Path] = []

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("primary replace failure")

    def fail_unlink(path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        cleanup_attempts.append(path)
        raise OSError("secondary unlink failure")

    monkeypatch.setattr("mad_driving.training.curriculum.os.replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="primary replace failure") as caught:
        writer()

    assert destination.read_bytes() == b"old-state\n"
    assert len(cleanup_attempts) == 1
    assert "secondary unlink failure" in "\n".join(caught.value.__notes__)
    assert cleanup_attempts[0].is_file()


def test_curriculum_state_artifact_rejects_non_state_and_malformed_yaml(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "curriculum_state.yaml"
    with pytest.raises(TypeError, match="state"):
        write_curriculum_state(object(), destination)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="malformed"):
        read_curriculum_state(destination)

    destination.write_text("level: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_curriculum_state(destination)

    destination.write_text(
        "level: zero\nconsecutive_passes: 0\nevaluations: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="values"):
        read_curriculum_state(destination)


def test_curriculum_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    state_path = tmp_path / "curriculum_state.yaml"
    state_path.write_text(
        "level: 0\nlevel: 1\nconsecutive_passes: 0\nevaluations: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        read_curriculum_state(state_path)


def test_checkpoint_curriculum_read_rejects_path_replacement_during_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint-v1")
    sidecar = write_checkpoint_curriculum_state(CurriculumState(1, 0, 1), checkpoint)
    expected_bytes = sidecar.read_bytes()
    expected_digest = hashlib.sha256(expected_bytes).hexdigest()
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text(
        "schema_version: 1\n"
        f"checkpoint_sha256: {'0' * 64}\n"
        "level: 3\nconsecutive_passes: 0\nevaluations: 9\n",
        encoding="utf-8",
    )
    real_stat = Path.stat

    def replacement_identity(path: Path, *args: object, **kwargs: object):
        if path == sidecar:
            return real_stat(replacement, *args, **kwargs)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", replacement_identity)

    with pytest.raises(ValueError, match="replaced|changed|identity"):
        read_checkpoint_curriculum_state(
            sidecar,
            expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            expected_sha256=expected_digest,
        )


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_checkpoint_curriculum_schema_version_requires_an_exact_integer(
    tmp_path: Path,
    schema_version: object,
) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = write_checkpoint_curriculum_state(CurriculumState(1, 0, 1), checkpoint)
    payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    sidecar.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_checkpoint_curriculum_state(
            sidecar,
            expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize(
    ("config", "state", "message"),
    [
        (fixed_config(1), CurriculumState(1, 1, 1), "streak"),
        (
            automatic_config(consecutive_evaluations=2),
            CurriculumState(1, 2, 4),
            "streak",
        ),
        (
            automatic_config(consecutive_evaluations=2),
            CurriculumState(2, 0, 3),
            "evaluations",
        ),
        (
            automatic_config(consecutive_evaluations=2),
            CurriculumState(1, 1, 2),
            "evaluations",
        ),
    ],
)
def test_curriculum_controller_rejects_unreachable_config_dependent_state(
    config: CurriculumConfig,
    state: CurriculumState,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CurriculumController(config, state)


def test_automatic_level_three_retains_valid_capped_streak() -> None:
    controller = CurriculumController(
        automatic_config(consecutive_evaluations=2),
        CurriculumState(level=3, consecutive_passes=5, evaluations=11),
    )

    assert controller.state == CurriculumState(3, 5, 11)
