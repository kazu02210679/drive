from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mad_driving.cli import compare as compare_module
from mad_driving.cli.compare import main


def _bundle(root: Path, artifacts: dict[str, bytes] | None = None) -> Path:
    inventory = []
    for relative, payload in sorted((artifacts or {}).items()):
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
    root.mkdir(parents=True, exist_ok=True)
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


def test_help_lists_compare_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--evaluation" in output
    assert "--output" in output


def test_evaluation_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--evaluation" in captured.err
    assert "Traceback" not in captured.err


def test_verified_source_delegates_to_absent_explicit_destination_and_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "source")
    destination = tmp_path / "comparison"
    before = source.joinpath("evaluation_manifest.json").read_bytes()
    calls: list[tuple[Path, Path]] = []

    def capture(*, evaluation: Path, destination: Path) -> Path:
        calls.append((evaluation, destination))
        return destination

    monkeypatch.setattr(compare_module, "run_comparison_bundle", capture)

    assert main(
        ["--evaluation", str(source), "--output", str(destination)]
    ) == 0
    assert calls == [(source.resolve(), destination.resolve())]
    assert json.loads(capsys.readouterr().out) == {"output": str(destination.resolve())}
    assert source.joinpath("evaluation_manifest.json").read_bytes() == before
    assert not destination.exists()


def test_omitted_output_uses_deterministic_absent_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation")
    expected = tmp_path / "evaluation-comparison"
    monkeypatch.setattr(
        compare_module,
        "run_comparison_bundle",
        lambda *, evaluation, destination: destination,
    )

    assert main(["--evaluation", str(source)]) == 0

    assert json.loads(capsys.readouterr().out) == {"output": str(expected.resolve())}
    assert not expected.exists()


@pytest.mark.parametrize("destination_kind", ["existing", "inside"])
def test_invalid_destination_fails_before_execution(
    tmp_path: Path,
    destination_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation")
    if destination_kind == "existing":
        destination = tmp_path / "existing"
        destination.mkdir()
    else:
        destination = source / "generated"
    called = False

    def unexpected(**kwargs: object) -> Path:
        nonlocal called
        called = True
        return destination

    monkeypatch.setattr(compare_module, "run_comparison_bundle", unexpected)

    assert main(
        ["--evaluation", str(source), "--output", str(destination)]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert called is False


def test_malformed_or_replaced_bundle_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {"metrics/eval_metrics.csv": b"valid\n"})
    source.joinpath("metrics/eval_metrics.csv").write_bytes(b"changed\n")
    monkeypatch.setattr(
        compare_module,
        "run_comparison_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--evaluation", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "sha-256" in captured.err.lower() or "size" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_operational_error_is_concise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation")
    monkeypatch.setattr(
        compare_module,
        "run_comparison_bundle",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad comparison rows")),
    )

    assert main(["--evaluation", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "comparison failed: bad comparison rows" in captured.err
    assert "Traceback" not in captured.err
