from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mad_driving.cli import render_episode as render_module
from mad_driving.cli.render_episode import main

EPISODE_KEY = "proposed_system_42_level1_lead_brake_20000"
TRACE = "episodes/proposed/system/42/level1_lead_brake/episode_20000_trace.jsonl"
FRAME = "episodes/proposed/system/42/level1_lead_brake/episode_20000_frames/frame_000000.png"


def _bundle(root: Path, artifacts: dict[str, bytes]) -> Path:
    inventory = []
    for relative, payload in sorted(artifacts.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    root.joinpath("evaluation_manifest.json").write_bytes(
        (
            json.dumps(
                {"artifacts": inventory, "schema_version": 1},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    return root


def test_help_lists_render_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--evaluation" in output
    assert "--episode-key" in output
    assert "--output" in output


def test_evaluation_and_episode_key_are_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--evaluation" in captured.err
    assert "--episode-key" in captured.err
    assert "Traceback" not in captured.err


def test_unique_persisted_trace_and_frame_set_is_forwarded_without_mutating_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {TRACE: b"{}\n", FRAME: b"png"})
    destination = tmp_path / "render"
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    calls: list[dict[str, object]] = []

    def capture(**kwargs: object) -> Path:
        calls.append(kwargs)
        return destination.resolve()

    monkeypatch.setattr(render_module, "run_render_bundle", capture)

    assert main(
        [
            "--evaluation",
            str(source),
            "--episode-key",
            EPISODE_KEY,
            "--output",
            str(destination),
        ]
    ) == 0

    assert calls == [
        {
            "evaluation": source.resolve(),
            "episode_key": EPISODE_KEY,
            "step_jsonl": source.resolve() / TRACE,
            "frames_dir": source.resolve() / Path(FRAME).parent,
            "destination": destination.resolve(),
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"output": str(destination.resolve())}
    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not destination.exists()


def test_omitted_output_uses_episode_specific_absent_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {TRACE: b"{}\n", FRAME: b"png"})
    expected = tmp_path / f"evaluation-render-{EPISODE_KEY}"
    monkeypatch.setattr(
        render_module,
        "run_render_bundle",
        lambda **kwargs: kwargs["destination"],
    )

    assert main(["--evaluation", str(source), "--episode-key", EPISODE_KEY]) == 0

    assert json.loads(capsys.readouterr().out) == {"output": str(expected.resolve())}


@pytest.mark.parametrize(
    ("episode_key", "artifacts"),
    [
        ("../escape", {TRACE: b"{}\n", FRAME: b"png"}),
        ("proposed_system_42_level1_lead_brake_20001", {TRACE: b"{}\n", FRAME: b"png"}),
        (EPISODE_KEY, {TRACE: b"{}\n"}),
        (EPISODE_KEY, {FRAME: b"png"}),
    ],
)
def test_invalid_or_non_unique_episode_selection_fails_before_execution(
    tmp_path: Path,
    episode_key: str,
    artifacts: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", artifacts)
    called = False

    def unexpected(**kwargs: object) -> Path:
        nonlocal called
        called = True
        return tmp_path / "render"

    monkeypatch.setattr(render_module, "run_render_bundle", unexpected)

    assert main(["--evaluation", str(source), "--episode-key", episode_key]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "episode key" in captured.err.lower() or "trace/frame" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert called is False


def test_destination_inside_source_and_executor_error_are_concise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {TRACE: b"{}\n", FRAME: b"png"})
    inside = source / "renders"
    monkeypatch.setattr(
        render_module,
        "run_render_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )
    assert main(
        [
            "--evaluation",
            str(source),
            "--episode-key",
            EPISODE_KEY,
            "--output",
            str(inside),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert "inside" in captured.err.lower()
    assert "Traceback" not in captured.err

    monkeypatch.setattr(
        render_module,
        "run_render_bundle",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("malformed trace")),
    )
    assert main(["--evaluation", str(source), "--episode-key", EPISODE_KEY]) == 2
    captured = capsys.readouterr()
    assert "render failed: malformed trace" in captured.err
    assert "Traceback" not in captured.err


def test_compare_and_render_import_and_patched_execution_remain_offline() -> None:
    script = r'''
import hashlib
import importlib.abc
import json
import sys
import tempfile
from pathlib import Path

FORBIDDEN = ("metadrive", "stable_baselines3", "tensorboard")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise AssertionError(f"forbidden offline import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from mad_driving.cli import compare, render_episode

root = Path(tempfile.mkdtemp()) / "evaluation"
trace_name = "episodes/proposed/system/42/level1_lead_brake/episode_20000_trace.jsonl"
frame_name = "episodes/proposed/system/42/level1_lead_brake/episode_20000_frames/frame.png"
artifacts = {trace_name: b"{}\n", frame_name: b"png"}
inventory = []
for relative, payload in sorted(artifacts.items()):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    inventory.append({
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    })
root.joinpath("evaluation_manifest.json").write_bytes(
    (
        json.dumps(
            {"artifacts": inventory, "schema_version": 1},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
)
compare.run_comparison_bundle = lambda **kwargs: kwargs["destination"]
render_episode.run_render_bundle = lambda **kwargs: kwargs["destination"]
assert compare.main(["--evaluation", str(root)]) == 0
assert render_episode.main([
    "--evaluation", str(root),
    "--episode-key", "proposed_system_42_level1_lead_brake_20000",
]) == 0
assert not any(
    module == name or module.startswith(name + ".")
    for module in sys.modules
    for name in FORBIDDEN
)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
