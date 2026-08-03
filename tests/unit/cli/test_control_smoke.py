import json
from typing import Any

import pytest

from mad_driving.cli import control_smoke as control_smoke_module
from mad_driving.cli.control_smoke import main, run_control_smoke
from mad_driving.config.models import AppConfig, ControlConfig
from mad_driving.interfaces import CriticReview, SceneSnapshot
from tests.unit.agents.factories import make_claim


class FakeLane:
    speed_limit = 36.0
    index = ("A", "B", 0)

    @staticmethod
    def local_coordinates(position: tuple[float, float]) -> tuple[float, float]:
        return position


class FakeNavigation:
    current_lane = FakeLane()
    route_completion = 0.1


class FakeVehicle:
    name = "ego"
    position = (0.0, 0.0)
    velocity = (1.0, 0.0)
    last_velocity = (1.0, 0.0)
    heading_theta = 0.0
    navigation = FakeNavigation()
    lane_index = FakeLane.index
    max_speed_m_s = 20.0
    speed = 1.0


class FakeEngine:
    def __init__(self, vehicle: FakeVehicle) -> None:
        self.vehicle = vehicle

    def get_objects(self) -> dict[str, FakeVehicle]:
        return {"ego": self.vehicle}


class FakeControlEnv:
    def __init__(
        self,
        options: dict[str, object],
        control_config: ControlConfig | None = None,
        *,
        fail_on_step: bool = False,
        truncate: bool = False,
    ) -> None:
        del control_config
        self.options = options
        self.fail_on_step = fail_on_step
        self.truncate = truncate
        self.vehicle = FakeVehicle()
        self.agent = self.vehicle
        self.engine = FakeEngine(self.vehicle)
        self.action_space = None
        self.config: dict[str, Any] = {
            **options,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
        }
        self.reset_seeds: list[int | None] = []
        self.actions: list[int] = []
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        return {}, {}

    def step(self, action: int):
        self.actions.append(action)
        if self.fail_on_step:
            raise RuntimeError("step failed")
        done = len(self.actions) == 2
        return {}, 0.0, done and not self.truncate, done and self.truncate, {}

    def close(self) -> None:
        self.closed = True


def make_config(
    *,
    decision_steps: int = 4,
    shield_mode: str = "enforce",
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "unit_control_smoke",
            "decision_steps": decision_steps,
            "fixed_action": [0.0, 0.25],
            "metadrive": {"use_render": False},
            "shield": {"mode": shield_mode},
        }
    )


def neutral_review() -> CriticReview:
    return CriticReview(
        conflict_score=0.0,
        unresolved_conflict=False,
        max_severity=0.0,
        supported_agent_ids=(),
        challenged_claim_ids=(),
        reasons=(),
    )


class RecordingSuite:
    def __init__(self, *, min_ttc_s: float | None = None) -> None:
        self.min_ttc_s = min_ttc_s
        self.snapshots: list[SceneSnapshot] = []

    def analyze(self, snapshot: SceneSnapshot) -> tuple[tuple[Any, ...], CriticReview]:
        self.snapshots.append(snapshot)
        claims = (
            make_claim("nominal"),
            make_claim("hazard", min_ttc_s=self.min_ttc_s),
            make_claim("rule"),
        )
        return claims, neutral_review()


class FailingSuite:
    def analyze(self, snapshot: SceneSnapshot):
        del snapshot
        raise RuntimeError("analysis failed")


def test_control_smoke_runs_decision_pipeline_and_closes() -> None:
    created: list[FakeControlEnv] = []

    def factory(options: dict[str, object], control: ControlConfig) -> FakeControlEnv:
        del control
        env = FakeControlEnv(options)
        created.append(env)
        return env

    result = run_control_smoke(make_config(decision_steps=4), env_factory=factory)

    env = created[0]
    assert env.reset_seeds == [42]
    assert env.actions == [0, 0]
    assert env.closed is True
    assert result.steps_completed == 2
    assert result.terminated is True
    assert result.truncated is False
    assert sum(result.action_counts) == result.steps_completed
    assert result.action_counts == (2, 0, 0, 0)
    assert result.final_trace.executed_action == 0
    assert result.final_trace.target_speed_mps == 10.0


def test_analysis_failure_executes_stop_and_still_closes() -> None:
    env = FakeControlEnv({})
    result = run_control_smoke(
        make_config(),
        env_factory=lambda options, control: env,
        suite_factory=lambda config: FailingSuite(),
    )

    assert env.actions[0] == 3
    assert env.closed is True
    assert result.final_trace.executed_action == 3
    assert "multiple_agents_missing" in result.final_trace.shield_reasons
    assert result.final_review.reasons == ("agent_analysis_failed",)


def test_step_failure_always_closes() -> None:
    env = FakeControlEnv({}, fail_on_step=True)
    with pytest.raises(RuntimeError, match="step failed"):
        run_control_smoke(
            make_config(),
            env_factory=lambda options, control: env,
        )
    assert env.closed is True


def test_suite_factory_failure_always_closes() -> None:
    env = FakeControlEnv({})

    def raising_suite_factory(config: object) -> object:
        del config
        raise RuntimeError("suite construction failed")

    with pytest.raises(RuntimeError, match="suite construction failed"):
        run_control_smoke(
            make_config(),
            env_factory=lambda options, control: env,
            suite_factory=raising_suite_factory,  # type: ignore[arg-type]
        )
    assert env.closed is True


def test_previous_action_and_intervention_are_propagated() -> None:
    suite = RecordingSuite(min_ttc_s=0.5)
    result = run_control_smoke(
        make_config(),
        env_factory=FakeControlEnv,
        suite_factory=lambda config: suite,
    )

    assert suite.snapshots[0].step_index == 0
    assert suite.snapshots[1].previous_action == 3
    assert suite.snapshots[1].previous_shield_intervention is True
    assert result.final_snapshot.previous_action == 3
    assert result.final_trace.target_speed_mps == 0.0
    assert result.shield_intervention_count == result.steps_completed


def test_monitor_mode_counts_no_actual_intervention() -> None:
    suite = RecordingSuite(min_ttc_s=0.5)
    result = run_control_smoke(
        make_config(shield_mode="monitor"),
        env_factory=FakeControlEnv,
        suite_factory=lambda config: suite,
    )

    assert result.action_counts == (2, 0, 0, 0)
    assert result.shield_intervention_count == 0
    assert result.final_trace.shield_intervened is False
    assert "imminent_ttc" in result.final_trace.shield_reasons


def test_truncation_stops_the_loop() -> None:
    env = FakeControlEnv({}, truncate=True)
    result = run_control_smoke(
        make_config(),
        env_factory=lambda options, control: env,
    )
    assert result.steps_completed == 2
    assert result.terminated is False
    assert result.truncated is True


def test_repeated_fake_runs_are_exactly_deterministic() -> None:
    first = run_control_smoke(make_config(), env_factory=FakeControlEnv)
    second = run_control_smoke(make_config(), env_factory=FakeControlEnv)
    assert first == second


def test_main_serializes_finite_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_control_smoke(make_config(), env_factory=FakeControlEnv)
    monkeypatch.setattr(control_smoke_module, "load_config", lambda path: make_config())
    monkeypatch.setattr(control_smoke_module, "run_control_smoke", lambda config: result)

    exit_code = main(["--config", "ignored.yaml"])

    document = json.loads(
        capsys.readouterr().out,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")),
    )
    assert exit_code == 0
    assert document["action_counts"] == [2, 0, 0, 0]
    assert document["final_trace"]["target_speed_mps"] == 10.0


def test_main_reports_operational_error_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--config", "definitely-missing.yaml"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration file not found" in captured.err
    assert "Traceback" not in captured.err
