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
    assert "--smoke" in output
    assert "--run-dir" in output
    assert "--resume-from" in output


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
    ) -> TrainingResult:
        calls.append(
            {
                "config": received_config,
                "smoke": smoke,
                "run_dir": run_dir,
                "resume_from": resume_from,
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
        }
    ]
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "best_checkpoint": str(run_dir / "checkpoints" / "best_model.zip"),
        "final_checkpoint": str(run_dir / "checkpoints" / "final_model.zip"),
        "run_dir": str(run_dir),
        "timesteps": 5_000,
    }


def test_omitted_run_dir_uses_configured_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    configured_run_root = tmp_path / "configured-runs"
    config = make_config(run_root=str(configured_run_root))
    run_dirs: list[Path] = []
    monkeypatch.setattr(train_module, "load_config", lambda path: config)

    def fake_run_training(
        received_config: AppConfig,
        *,
        smoke: bool,
        run_dir: Path,
        resume_from: Path | None,
    ) -> TrainingResult:
        del received_config, smoke, resume_from
        run_dirs.append(run_dir)
        return TrainingResult(run_dir, run_dir / "final.zip", run_dir / "best.zip", 500_000)

    monkeypatch.setattr(train_module, "run_training", fake_run_training)

    assert main(["--config", str(config_path)]) == 0
    assert run_dirs == [configured_run_root]


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
    missing = tmp_path / "missing.file"
    if option == "--config":
        args = ["--config", str(missing)]
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


def test_operational_error_is_concise_and_traceback_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(train_module, "load_config", lambda path: make_config())

    def failing_training(*args: object, **kwargs: object) -> None:
        raise RuntimeError("training exploded")

    monkeypatch.setattr(train_module, "run_training", failing_training)

    assert main(["--config", str(config_path), "--smoke"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "training failed: training exploded" in captured.err
    assert "Traceback" not in captured.err
