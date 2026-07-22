import json
from types import SimpleNamespace
from typing import Any

import pytest

from mad_driving.agents.suite import AgentAnalysisResult
from mad_driving.cli import control_smoke as control_smoke_module
from mad_driving.cli.control_smoke import main, run_control_smoke
from mad_driving.config.models import AppConfig, ControlConfig
from mad_driving.interfaces import CriticReview, SceneObservation
from mad_driving.world_model import SceneSnapshotBuilder
from tests.unit.agents.factories import make_analysis, make_claim


class FakeLane:
    speed_limit = 36.0
    index = ("A", "B", 0)
    width = 3.5

    @staticmethod
    def local_coordinates(position: tuple[float, float]) -> tuple[float, float]:
        return position


class FakeNavigation:
    current_lane = FakeLane()
    route_completion = 0.1


class FakeVehicle:
    LENGTH = 4.5
    WIDTH = 1.8
    name = "ego"
    position = (0.0, 0.0)
    velocity = (1.0, 0.0)
    last_velocity = (1.0, 0.0)
    heading_theta = 0.0
    navigation = FakeNavigation()
    lane_index = FakeLane.index
    max_speed_m_s = 20.0
    speed = 1.0
    on_lane = True
    crash_vehicle = False
    crash_human = False
    crash_object = False
    crash_sidewalk = False
    crash_building = False


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
        self.current_seed = 0

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        assert seed is not None
        self.current_seed = seed
        return {}, {"env_seed": seed}

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
        self.observations: list[SceneObservation] = []

    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        self.observations.append(observation)
        claims = (
            make_claim("nominal"),
            make_claim("hazard", min_ttc_s=self.min_ttc_s),
            make_claim("rule"),
        )
        return make_analysis(claims=claims, review=neutral_review())


class FailedAgentSuite:
    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        del observation
        return AgentAnalysisResult(
            claims=(),
            failed_agent_ids=("nominal", "hazard", "rule"),
            errors=(
                "nominal:RuntimeError:analysis failed",
                "hazard:RuntimeError:analysis failed",
                "rule:RuntimeError:analysis failed",
            ),
            review=CriticReview(
                conflict_score=1.0,
                unresolved_conflict=True,
                max_severity=1.0,
                supported_agent_ids=(),
                challenged_claim_ids=(),
                reasons=("agent_analysis_failed",),
            ),
            expected_agent_ids=("nominal", "hazard", "rule"),
        )


class StepIdentifiedSuite:
    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        step_index = observation.step_index
        return make_analysis(
            claims=(
                make_claim(
                    "nominal",
                    claim_id=f"nominal:{step_index}:none:test",
                    valid_until_step=step_index,
                ),
            ),
            expected_agent_ids=("nominal",),
        )


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


def test_control_smoke_trace_uses_pre_step_analysis_and_current_shield_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter(
        (
            0,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            9_000_000,
            10_000_000,
            12_500_000,
            20_000_000,
            29_000_000,
        )
    )
    monkeypatch.setattr(
        control_smoke_module,
        "time",
        SimpleNamespace(perf_counter_ns=lambda: next(clock_values)),
        raising=False,
    )

    result = run_control_smoke(
        make_config(),
        env_factory=FakeControlEnv,
        suite_factory=lambda config: StepIdentifiedSuite(),
    )

    assert result.final_trace.claims[0].claim_id == "nominal:1:none:test"
    assert result.final_claims[0].claim_id == "nominal:2:none:test"
    assert result.final_trace.analysis_latency_ms == pytest.approx(4.0)
    assert result.final_trace.shield_latency_ms == pytest.approx(2.5)


def test_default_builder_uses_resolved_hazard_oracle_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_values = make_config().model_dump(mode="python")
    hazard_values = config_values["agents"]["hazard"]  # type: ignore[index]
    assert isinstance(hazard_values, dict)
    hazard_values.update(
        {
            "reaction_delay_s": 0.25,
            "ego_max_safe_deceleration_mps2": -2.0,
        }
    )
    config = AppConfig.model_validate(config_values)
    real_builder = SceneSnapshotBuilder
    margins: list[float | None] = []

    class CapturingSceneSnapshotBuilder:
        def __init__(
            self,
            *,
            reaction_delay_s: float,
            safe_deceleration_mps2: float,
        ) -> None:
            self._delegate = real_builder(
                reaction_delay_s=reaction_delay_s,
                safe_deceleration_mps2=safe_deceleration_mps2,
            )

        def build(self, *args: Any, **kwargs: Any):
            frame = self._delegate.build(*args, **kwargs)
            margins.append(frame.privileged.minimum_actual_stopping_margin_m)
            return frame

    class LeadVehicle(FakeVehicle):
        name = "lead"
        position = (6.0, 0.0)
        velocity = (0.0, 0.0)
        last_velocity = (0.0, 0.0)
        speed = 0.0

    env = FakeControlEnv({})
    lead = LeadVehicle()
    env.engine.get_objects = lambda: {"ego": env.vehicle, "lead": lead}  # type: ignore[method-assign]
    monkeypatch.setattr(
        control_smoke_module,
        "SceneSnapshotBuilder",
        CapturingSceneSnapshotBuilder,
    )

    result = run_control_smoke(
        config,
        env_factory=lambda options, control: env,
    )

    assert result.steps_completed == 2
    assert margins
    assert all(margin == pytest.approx(1.0) for margin in margins)


def test_injected_builder_factory_remains_no_argument() -> None:
    calls = 0

    def builder_factory() -> SceneSnapshotBuilder:
        nonlocal calls
        calls += 1
        return SceneSnapshotBuilder(reaction_delay_s=0.1, safe_deceleration_mps2=3.0)

    result = run_control_smoke(
        make_config(),
        env_factory=FakeControlEnv,
        builder_factory=builder_factory,
    )

    assert result.steps_completed == 2
    assert calls == 1


def test_analysis_failure_executes_stop_and_still_closes() -> None:
    env = FakeControlEnv({})
    result = run_control_smoke(
        make_config(),
        env_factory=lambda options, control: env,
        suite_factory=lambda config: FailedAgentSuite(),
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

    assert suite.observations[0].step_index == 0
    assert suite.observations[1].previous_executed_action == 3
    assert suite.observations[1].previous_shield_intervention is True
    assert result.final_snapshot.previous_executed_action == 3
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
    assert result.final_trace.required_action == 3
    assert result.final_trace.intervention_required is True
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
