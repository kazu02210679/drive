from pathlib import Path

import pytest
from pydantic import ValidationError

from mad_driving.config.loader import load_config

VALID_CONFIG = """\
seed: 42
scenario_id: phase1_smoke
decision_steps: 100
fixed_action: [0.0, 0.25]
metadrive:
  use_render: false
  image_observation: false
  num_scenarios: 1
  start_seed: 42
  traffic_density: 0.1
  horizon: 200
"""


def write_config(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_config_and_exports_metadrive_values(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.seed == 42
    assert config.fixed_action == (0.0, 0.25)
    assert config.metadrive_dict() == {
        "use_render": False,
        "image_observation": False,
        "num_scenarios": 1,
        "start_seed": 42,
        "traffic_density": 0.1,
        "horizon": 200,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "map_config": {"lane_width": 3.5},
    }


def test_missing_config_file_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(missing)


@pytest.mark.parametrize(
    "text",
    [
        VALID_CONFIG + "extra: true\n",
        VALID_CONFIG.replace("  horizon: 200\n", "  horizon: 200\n  bogus: 1\n"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize("decision_steps", [0, -1])
def test_decision_steps_must_be_positive(tmp_path: Path, decision_steps: int) -> None:
    text = VALID_CONFIG.replace("decision_steps: 100", f"decision_steps: {decision_steps}")

    with pytest.raises(ValidationError, match="decision_steps"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize("fixed_action", ["[0.0]", "[0.0, 0.25, 0.5]"])
def test_fixed_action_requires_two_values(tmp_path: Path, fixed_action: str) -> None:
    text = VALID_CONFIG.replace("[0.0, 0.25]", fixed_action)

    with pytest.raises(ValidationError, match="fixed_action"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize("bad_value", [".nan", ".inf", "-.inf"])
def test_fixed_action_rejects_non_finite_values(tmp_path: Path, bad_value: str) -> None:
    text = VALID_CONFIG.replace("[0.0, 0.25]", f"[{bad_value}, 0.25]")

    with pytest.raises(ValidationError, match="fixed_action"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("num_scenarios", 0), ("traffic_density", -0.1), ("traffic_density", 1.1), ("horizon", 0)],
)
def test_metadrive_ranges_are_validated(tmp_path: Path, field: str, bad_value: float) -> None:
    original = {
        "num_scenarios": 1,
        "traffic_density": 0.1,
        "horizon": 200,
    }[field]
    text = VALID_CONFIG.replace(f"  {field}: {original}", f"  {field}: {bad_value}")

    with pytest.raises(ValidationError, match=field):
        load_config(write_config(tmp_path, text))
