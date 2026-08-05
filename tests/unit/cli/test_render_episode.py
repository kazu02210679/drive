from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mad_driving.cli import render_episode as render_module
from mad_driving.cli.render_episode import main
from mad_driving.evaluation.models import (
    REWARD_COMPONENT_KEYS,
    EvaluationEpisodeKey,
    EvaluationStepRecord,
)
from mad_driving.interfaces import CriticReview, RiskClaim
from mad_driving.methods import MethodProfileSnapshot

EPISODE_KEY = "proposed_system_42_level1_lead_brake_20000"
TRACE = "episodes/proposed/system/42/level1_lead_brake/episode_20000_trace.jsonl"
FRAMES_DIR = "episodes/proposed/system/42/level1_lead_brake/episode_20000_frames"
FRAME = f"{FRAMES_DIR}/000000.png"


def _claim(agent_id: str, severity: float, recommended_speed: float) -> RiskClaim:
    return RiskClaim(
        claim_id=f"{agent_id}:aggregate",
        agent_id=agent_id,
        event_type="aggregate",
        target_actor_id="lead",
        probability=severity,
        confidence=0.9,
        severity=severity,
        time_horizon_s=3.0,
        min_ttc_s=2.5,
        stopping_margin_m=4.0,
        recommended_max_speed_mps=recommended_speed,
        hard_stop_required=False,
        evidence=("persisted evidence",),
        assumptions=("persisted assumption",),
        valid_until_step=3,
    )


def _step(
    *, step_index: int, frame_path: str, episode_rng_seed: int = 20_000
) -> EvaluationStepRecord:
    components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
    components["progress_reward"] = 0.5
    claims = (
        _claim("nominal", 0.2, 12.0),
        _claim("hazard", 0.8, 4.0),
        _claim("rule", 0.5, 6.0),
    )
    return EvaluationStepRecord(
        record_schema_version=1,
        research_contract_version=7,
        episode_key=EvaluationEpisodeKey(
            method_id="proposed",
            track="system",
            role="test",
            policy_seed=42,
            case_id="level1_lead_brake",
            episode_rng_seed=episode_rng_seed,
        ),
        method_profile=MethodProfileSnapshot.from_method_id("proposed"),
        checkpoint_path="runs/proposed/42.zip",
        checkpoint_sha256="a" * 64,
        episode_index=0,
        is_formal=False,
        shield_mode="enforce",
        step_index=step_index,
        simulation_time_s=step_index * 0.1,
        decision_interval_s=0.1,
        episode_rng_seed=episode_rng_seed,
        metadrive_scenario_index=17,
        scenario_selection_seed=31,
        scenario_parameter_seed=37,
        case_id="level1_lead_brake",
        scenario_id="lead_brake",
        difficulty_level=1,
        requested_action=1,
        required_action=2,
        executed_action=2,
        unsafe_request=True,
        shield_intervened=True,
        shield_reasons=("minimum_ttc",),
        target_speed_mps=8.0,
        ego_speed_mps=10.0,
        ego_speed_limit_mps=13.0,
        ego_longitudinal_acceleration_mps2=-1.0,
        route_completion=0.2,
        route_progress_m=12.5,
        lane_offset_m=-0.1,
        collision_occurred=False,
        collision_kind=None,
        minimum_actual_ttc_s=2.4,
        minimum_actual_stopping_margin_m=3.5,
        pre_step_hard_rule_constraint=False,
        post_step_rule_violation_event=False,
        scenario_success=False,
        scenario_failure=False,
        arrived=False,
        off_road=False,
        terminated=False,
        truncated=False,
        cumulative_unnecessary_stop_duration_s=0.0,
        reward_total=0.5,
        reward_components=components,
        claims=claims,
        review=CriticReview(
            conflict_score=0.65,
            unresolved_conflict=False,
            max_severity=0.8,
            supported_agent_ids=("hazard", "rule"),
            challenged_claim_ids=("nominal:aggregate",),
            reasons=("persisted disagreement",),
        ),
        expected_agent_ids=("nominal", "hazard", "rule"),
        failed_agent_ids=(),
        errors=(),
        policy_inference_latency_ms=1.0,
        agent_analysis_latency_ms=2.0,
        shield_latency_ms=0.5,
        total_decision_latency_ms=4.0,
        frame_path=frame_path,
    )


def _trace_payload(*frame_paths: str, episode_rng_seed: int = 20_000) -> bytes:
    return b"".join(
        (
            json.dumps(
                _step(
                    step_index=index,
                    frame_path=frame_path,
                    episode_rng_seed=episode_rng_seed,
                ).to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for index, frame_path in enumerate(frame_paths)
    )


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
    trace_payload = _trace_payload(FRAME)
    source = _bundle(tmp_path / "evaluation", {TRACE: trace_payload, FRAME: b"png"})
    destination = tmp_path / "render"
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    calls: list[dict[str, object]] = []

    def capture(**kwargs: object) -> Path:
        calls.append(kwargs)
        return destination.resolve()

    monkeypatch.setattr(render_module, "run_render_bundle", capture)

    assert (
        main(
            [
                "--evaluation",
                str(source),
                "--episode-key",
                EPISODE_KEY,
                "--output",
                str(destination),
            ]
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["evaluation"] == source.resolve()
    assert calls[0]["destination"] == destination.resolve()
    render_inputs = calls[0]["render_inputs"]
    assert render_inputs.episode_key == _step(step_index=0, frame_path=FRAME).episode_key
    assert render_inputs.trace.relative_path == TRACE
    assert render_inputs.trace.payload == trace_payload
    assert tuple(frame.relative_path for frame in render_inputs.frames) == (FRAME,)
    assert tuple(frame.payload for frame in render_inputs.frames) == (b"png",)
    assert json.loads(capsys.readouterr().out) == {"output": str(destination.resolve())}
    after = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    assert after == before
    assert not destination.exists()


def test_omitted_output_uses_episode_specific_absent_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {TRACE: _trace_payload(FRAME), FRAME: b"png"})
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
        ("../escape", {TRACE: _trace_payload(FRAME), FRAME: b"png"}),
        (
            "proposed_system_42_level1_lead_brake_20001",
            {TRACE: _trace_payload(FRAME), FRAME: b"png"},
        ),
        (EPISODE_KEY, {TRACE: _trace_payload(FRAME)}),
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
    assert any(
        phrase in captured.err.lower()
        for phrase in ("episode key", "trace/frame", "frame inventory")
    )
    assert "Traceback" not in captured.err
    assert called is False


@pytest.mark.parametrize(
    "artifacts",
    [
        {
            TRACE: _trace_payload(FRAME),
            FRAME: b"png",
            f"{FRAMES_DIR}/README.txt": b"not a frame",
        },
        {
            TRACE: _trace_payload(FRAME),
            FRAME: b"png",
            f"{FRAMES_DIR}/000001.png": b"extra",
        },
        {
            TRACE: _trace_payload(f"{FRAMES_DIR}/000001.png"),
            FRAME: b"png",
        },
    ],
)
def test_frame_inventory_must_match_trace_steps_exactly(
    tmp_path: Path,
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

    assert main(["--evaluation", str(source), "--episode-key", EPISODE_KEY]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "frame" in captured.err.lower()
    assert called is False


@pytest.mark.parametrize(
    "trace_payload",
    [
        b"{}\n",
        _trace_payload(FRAME, episode_rng_seed=20_001),
    ],
)
def test_trace_must_parse_strictly_and_match_its_canonical_path(
    tmp_path: Path,
    trace_payload: bytes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {TRACE: trace_payload, FRAME: b"png"})
    monkeypatch.setattr(
        render_module,
        "run_render_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    assert main(["--evaluation", str(source), "--episode-key", EPISODE_KEY]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "trace" in captured.err.lower() or "jsonl" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_destination_inside_source_and_executor_error_are_concise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _bundle(tmp_path / "evaluation", {TRACE: _trace_payload(FRAME), FRAME: b"png"})
    inside = source / "renders"
    monkeypatch.setattr(
        render_module,
        "run_render_bundle",
        lambda **kwargs: pytest.fail("execution must not start"),
    )
    assert (
        main(
            [
                "--evaluation",
                str(source),
                "--episode-key",
                EPISODE_KEY,
                "--output",
                str(inside),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "inside" in captured.err.lower()
    assert "Traceback" not in captured.err

    monkeypatch.setattr(
        render_module,
        "run_render_bundle",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("malformed trace\nprivate detail")),
    )
    assert main(["--evaluation", str(source), "--episode-key", EPISODE_KEY]) == 2
    captured = capsys.readouterr()
    assert captured.err == "render failed: malformed trace private detail\n"
    assert "Traceback" not in captured.err


def test_compare_and_render_import_and_patched_execution_remain_offline() -> None:
    trace_payload = _trace_payload(FRAME)
    script = r"""
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
frame_name = "__FRAME_PATH__"
artifacts = {trace_name: __TRACE_PAYLOAD__, frame_name: b"png"}
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
"""
    script = script.replace("__FRAME_PATH__", FRAME).replace(
        "__TRACE_PAYLOAD__", repr(trace_payload)
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
