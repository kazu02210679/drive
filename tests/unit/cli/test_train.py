import json
from pathlib import Path

import pytest

from mad_driving.cli import train as train_module
from mad_driving.cli.train import main
from mad_driving.config.models import AppConfig
from mad_driving.training.train import TrainingResult


def make_config(*, run_root: str = "runs") -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "cli-training-test",
            "decision_steps": 2,
            "fixed_action": [0.0, 0.0],
            "metadrive": {"use_render": False},
            "training": {"run_root": run_root},
        }
    )


def test_help_lists_all_training_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--config" in output
    assert "--overlay" in output
    assert "--smoke" in output
    assert "--run-dir" in output
    assert "--resume-from" in output


def test_ordered_overlays_are_validated_and_forwarded_to_config_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "base.yaml"
    first_overlay = tmp_path / "lead.yaml"
    second_overlay = tmp_path / "automatic.yaml"
    for path in (config_path, first_overlay, second_overlay):
        path.write_text("placeholder: true\n", encoding="utf-8")
    config = make_config()
    calls: list[tuple[Path, tuple[Path, ...]]] = []

    def capture_load(path: Path, *overlays: Path) -> AppConfig:
        calls.append((path, overlays))
        return config

    monkeypatch.setattr(train_module, "load_config", capture_load)
    monkeypatch.setattr(
        train_module,
        "run_training",
        lambda received, **kwargs: TrainingResult(
            run_dir=kwargs["run_dir"],
            final_checkpoint=kwargs["run_dir"] / "checkpoints" / "final_model.zip",
            best_checkpoint=kwargs["run_dir"] / "checkpoints" / "best_model.zip",
            timesteps=8,
        ),
    )

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--overlay",
                str(first_overlay),
                "--overlay",
                str(second_overlay),
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert calls == [(config_path, (first_overlay, second_overlay))]


def test_config_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--config" in captured.err
    assert "required" in captured.err
    assert "Traceback" not in captured.err


def test_smoke_and_resume_are_forwarded_and_success_is_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    resume_path = tmp_path / "resume.zip"
    resume_path.write_bytes(b"checkpoint")
    run_dir = tmp_path / "run"
    config = make_config()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(train_module, "load_config", lambda path: config)

    def fake_run_training(
        received_config: AppConfig,
        *,
        smoke: bool,
        run_dir: Path,
        resume_from: Path | None,
        require_absent_run_dir: bool,
    ) -> TrainingResult:
        calls.append(
            {
                "config": received_config,
                "smoke": smoke,
                "run_dir": run_dir,
                "resume_from": resume_from,
                "require_absent_run_dir": require_absent_run_dir,
            }
        )
        return TrainingResult(
            run_dir=run_dir,
            final_checkpoint=run_dir / "checkpoints" / "final_model.zip",
            best_checkpoint=run_dir / "checkpoints" / "best_model.zip",
            timesteps=5_000,
        )

    monkeypatch.setattr(train_module, "run_training", fake_run_training)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--smoke",
            "--run-dir",
            str(run_dir),
            "--resume-from",
            str(resume_path),
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "config": config,
            "smoke": True,
            "run_dir": run_dir,
            "resume_from": resume_path,
            "require_absent_run_dir": False,
        }
    ]
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "best_checkpoint": str(run_dir / "checkpoints" / "best_model.zip"),
        "final_checkpoint": str(run_dir / "checkpoints" / "final_model.zip"),
        "run_dir": str(run_dir),
        "timesteps": 5_000,
    }


def test_omitted_run_dir_allocates_fresh_directory_under_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    run_root = tmp_path / "configured-runs"
    config = make_config(run_root=str(run_root))
    calls: list[Path] = []
    monkeypatch.setattr(train_module, "load_config", lambda path: config)

    def capture_training(
        received_config: AppConfig,
        *,
        smoke: bool,
        run_dir: Path,
        resume_from: Path | None,
        require_absent_run_dir: bool,
    ) -> TrainingResult:
        assert received_config is config
        assert smoke is True
        assert resume_from is None
        assert require_absent_run_dir is True
        assert not run_dir.exists()
        calls.append(run_dir)
        return TrainingResult(
            run_dir=run_dir,
            final_checkpoint=run_dir / "checkpoints" / "final_model.zip",
            best_checkpoint=run_dir / "checkpoints" / "best_model.zip",
            timesteps=8,
        )

    monkeypatch.setattr(train_module, "run_training", capture_training)

    assert main(["--config", str(config_path), "--smoke"]) == 0

    assert len(calls) == 1
    assert calls[0].parent == run_root
    assert not calls[0].exists()
    output = json.loads(capsys.readouterr().out)
    assert Path(output["run_dir"]) == calls[0]


def test_implicit_run_directory_skips_existing_name_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "configured-runs"
    occupied = run_root / "occupied"
    occupied.mkdir(parents=True)
    marker = occupied / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    names = iter(["occupied", "fresh"])
    monkeypatch.setattr(
        train_module,
        "_run_directory_name",
        lambda *, smoke: next(names),
    )

    allocated = train_module._fresh_run_directory(run_root, smoke=True)

    assert allocated == run_root / "fresh"
    assert not allocated.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_implicit_preownership_failure_never_precreates_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    reserved = run_root / "reserved"
    monkeypatch.setattr(
        train_module,
        "load_config",
        lambda path: make_config(run_root=str(run_root)),
    )
    monkeypatch.setattr(train_module, "_run_directory_name", lambda *, smoke: "reserved")

    def fail_before_ownership(*args: object, **kwargs: object) -> None:
        del args
        assert kwargs["require_absent_run_dir"] is True
        run_dir = kwargs["run_dir"]
        assert isinstance(run_dir, Path)
        assert not run_dir.exists()
        raise ValueError("malformed resume")

    monkeypatch.setattr(train_module, "run_training", fail_before_ownership)

    assert main(["--config", str(config_path), "--smoke"]) == 2

    assert not reserved.exists()
    assert "malformed resume" in capsys.readouterr().err


def test_malformed_resume_before_training_ownership_leaves_candidate_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    resume_path = tmp_path / "orphan.zip"
    resume_path.write_bytes(b"not-a-run-checkpoint")
    run_root = tmp_path / "runs"
    reserved = run_root / "malformed-resume"
    monkeypatch.setattr(
        train_module,
        "load_config",
        lambda path: make_config(run_root=str(run_root)),
    )
    monkeypatch.setattr(
        train_module,
        "_run_directory_name",
        lambda *, smoke: "malformed-resume",
    )

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--resume-from",
                str(resume_path),
            ]
        )
        == 2
    )

    assert not reserved.exists()


def test_explicit_empty_run_directory_is_not_cleaned_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    run_dir = tmp_path / "explicit"
    run_dir.mkdir()
    monkeypatch.setattr(train_module, "load_config", lambda path: make_config())
    monkeypatch.setattr(
        train_module,
        "run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    assert main(["--config", str(config_path), "--run-dir", str(run_dir)]) == 2
    assert run_dir.is_dir()
    assert list(run_dir.iterdir()) == []


@pytest.mark.parametrize("option", ["--config", "--resume-from"])
def test_missing_input_path_is_rejected_without_calling_training(
    tmp_path: Path,
    option: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    args = ["--config", str(config_path)]
    args.extend(["--run-dir", str(tmp_path / "run")])
    missing = tmp_path / "missing.file"
    if option == "--config":
        args = ["--config", str(missing), "--run-dir", str(tmp_path / "run")]
    else:
        args.extend(["--resume-from", str(missing)])
    called = False

    def unexpected_training(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(train_module, "run_training", unexpected_training)

    assert main(args) == 2
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert "not found" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert called is False


def test_existing_file_cannot_be_used_as_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    run_file = tmp_path / "not-a-directory"
    run_file.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(train_module, "load_config", lambda path: make_config())

    assert main(["--config", str(config_path), "--run-dir", str(run_file)]) == 2
    captured = capsys.readouterr()
    assert "run directory" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_nonempty_run_directory_is_rejected_without_calling_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    run_dir = tmp_path / "occupied"
    run_dir.mkdir()
    marker = run_dir / "keep.bin"
    marker.write_bytes(b"preserve-me\x00")
    monkeypatch.setattr(train_module, "load_config", lambda path: make_config())
    called = False

    def unexpected_training(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(train_module, "run_training", unexpected_training)

    assert main(["--config", str(config_path), "--run-dir", str(run_dir)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"training failed: Run directory is non-empty: {run_dir}" in captured.err
    assert "Traceback" not in captured.err
    assert marker.read_bytes() == b"preserve-me\x00"
    assert list(run_dir.iterdir()) == [marker]
    assert called is False


def test_operational_error_is_concise_and_traceback_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "load_config",
        lambda path: make_config(run_root=str(tmp_path / "failed-run")),
    )

    def failing_training(*args: object, **kwargs: object) -> None:
        raise RuntimeError("training exploded")

    monkeypatch.setattr(train_module, "run_training", failing_training)

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "failed-run"),
                "--smoke",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "training failed: training exploded" in captured.err
    assert "Traceback" not in captured.err
