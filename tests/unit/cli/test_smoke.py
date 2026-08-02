from typing import Any

import pytest

from mad_driving.cli.smoke import main, run_smoke
from mad_driving.config.models import AppConfig


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


class FakeEnv:
    def __init__(self, options: dict[str, object], *, fail_on_step: bool = False) -> None:
        self.options = options
        self.fail_on_step = fail_on_step
        self.vehicle = FakeVehicle()
        self.engine = FakeEngine(self.vehicle)
        self.config: dict[str, Any] = {
            **options,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
        }
        self.reset_seeds: list[int | None] = []
        self.actions: list[tuple[float, float]] = []
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        return {}, {}

    def step(self, action: tuple[float, float]):
        self.actions.append(action)
        if self.fail_on_step:
            raise RuntimeError("step failed")
        terminated = len(self.actions) == 2
        return {}, 0.0, terminated, False, {}

    def close(self) -> None:
        self.closed = True


def make_config(*, decision_steps: int = 4) -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "unit_smoke",
            "decision_steps": decision_steps,
            "fixed_action": [0.0, 0.25],
            "metadrive": {"use_render": False},
        }
    )


def test_run_smoke_resets_steps_until_done_and_always_closes() -> None:
    created: list[FakeEnv] = []

    def factory(options: dict[str, object]) -> FakeEnv:
        env = FakeEnv(options)
        created.append(env)
        return env

    result = run_smoke(make_config(), env_factory=factory)

    env = created[0]
    assert env.reset_seeds == [42]
    assert env.actions == [(0.0, 0.25), (0.0, 0.25)]
    assert env.closed is True
    assert result.steps_completed == 2
    assert result.terminated is True
    assert result.truncated is False
    assert result.final_snapshot.step_index == 2
    assert result.final_snapshot.scenario_id == "unit_smoke"


def test_run_smoke_closes_when_step_raises() -> None:
    created: list[FakeEnv] = []

    def factory(options: dict[str, object]) -> FakeEnv:
        env = FakeEnv(options, fail_on_step=True)
        created.append(env)
        return env

    with pytest.raises(RuntimeError, match="step failed"):
        run_smoke(make_config(), env_factory=factory)

    assert created[0].closed is True


def test_main_reports_missing_config_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--config", "definitely-missing.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration file not found" in captured.err
    assert "Traceback" not in captured.err
