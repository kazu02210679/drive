import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from mad_driving.config.models import CurriculumConfig
from mad_driving.training.curriculum import (
    CurriculumController,
    CurriculumState,
    read_checkpoint_curriculum_state,
    read_curriculum_state,
    write_checkpoint_curriculum_state,
    write_curriculum_state,
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

    first = controller.observe("validation", successes=4, collisions=0, episodes=5)
    second = controller.observe("validation", successes=5, collisions=0, episodes=5)

    assert first == CurriculumState(level=0, consecutive_passes=1, evaluations=1)
    assert second == CurriculumState(level=1, consecutive_passes=0, evaluations=2)
    assert controller.state == second


def test_automatic_curriculum_requires_both_thresholds_and_resets_failed_streak() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(1, 1, 4))

    collision_failure = controller.observe(
        "validation", successes=5, collisions=1, episodes=5
    )
    success_failure = controller.observe(
        "validation", successes=3, collisions=0, episodes=5
    )

    assert collision_failure == CurriculumState(1, 0, 5)
    assert success_failure == CurriculumState(1, 0, 6)


def test_automatic_curriculum_advances_exactly_one_level_and_never_regresses() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(1, 1, 7))

    advanced = controller.observe("validation", successes=5, collisions=0, episodes=5)
    failed = controller.observe("validation", successes=0, collisions=0, episodes=5)

    assert advanced == CurriculumState(2, 0, 8)
    assert failed == CurriculumState(2, 0, 9)


def test_automatic_curriculum_caps_at_level_three() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(3, 1, 7))

    state = controller.observe("validation", successes=5, collisions=0, episodes=5)

    assert state.level == 3
    assert state.evaluations == 8


def test_test_role_cannot_drive_curriculum() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    with pytest.raises(ValueError, match="test role"):
        controller.observe("test", successes=5, collisions=0, episodes=5)


def test_automatic_curriculum_rejects_training_observations() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    with pytest.raises(ValueError, match="validation"):
        controller.observe("train", successes=5, collisions=0, episodes=5)


@pytest.mark.parametrize(
    ("successes", "collisions", "episodes", "message"),
    [
        (True, 0, 5, "successes"),
        (4.0, 0, 5, "successes"),
        (-1, 0, 5, "successes"),
        (6, 0, 5, "successes"),
        (4, False, 5, "collisions"),
        (4, 1.0, 5, "collisions"),
        (4, -1, 5, "collisions"),
        (4, 6, 5, "collisions"),
        (4, 0, True, "episodes"),
        (4, 0, 5.0, "episodes"),
        (0, 0, 0, "episodes"),
    ],
)
def test_curriculum_observations_require_bounded_integer_counts(
    successes: object,
    collisions: object,
    episodes: object,
    message: str,
) -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))

    with pytest.raises(ValueError, match=message):
        controller.observe(  # type: ignore[arg-type]
            "validation",
            successes=successes,
            collisions=collisions,
            episodes=episodes,
        )


def test_fixed_curriculum_never_advances() -> None:
    controller = CurriculumController(fixed_config(level=2), CurriculumState(2, 0, 0))

    state = controller.observe("validation", successes=5, collisions=0, episodes=5)

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


def test_curriculum_atomic_write_preserves_old_state_when_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "curriculum_state.yaml"
    destination.write_bytes(b"old-state\n")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("mad_driving.training.curriculum.os.fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        write_curriculum_state(CurriculumState(1, 0, 1), destination)

    assert destination.read_bytes() == b"old-state\n"
    assert not list(tmp_path.glob(".curriculum_state.yaml.*.tmp"))


def test_curriculum_atomic_write_preserves_old_state_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "curriculum_state.yaml"
    destination.write_bytes(b"old-state\n")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("mad_driving.training.curriculum.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_curriculum_state(CurriculumState(1, 0, 1), destination)

    assert destination.read_bytes() == b"old-state\n"
    assert not list(tmp_path.glob(".curriculum_state.yaml.*.tmp"))


def test_curriculum_atomic_write_preserves_primary_and_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "curriculum_state.yaml"
    destination.write_bytes(b"old-state\n")
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
        write_curriculum_state(CurriculumState(1, 0, 1), destination)

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
