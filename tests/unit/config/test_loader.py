from pathlib import Path

import pytest
from pydantic import ValidationError

from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig

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


def test_phase5_defaults_are_strict() -> None:
    config = load_config("configs/base.yaml")

    assert config.scenarios.curriculum.mode == "fixed"
    assert config.scenarios.curriculum.fixed_level == 0
    assert config.scenarios.lead_brake.initial_gap_m.minimum == 35.0
    assert config.scenarios.lead_brake.initial_gap_m.maximum == 55.0


def test_overlay_selects_fixed_lead_brake() -> None:
    config = load_config("configs/base.yaml", "configs/scenarios/lead_brake.yaml")

    assert config.scenarios.curriculum.fixed_level == 1
    assert config.scenarios.selection == "lead_brake"


def test_overlay_selects_fixed_cut_in() -> None:
    config = load_config("configs/base.yaml", "configs/scenarios/cut_in.yaml")

    assert config.scenarios.selection == "cut_in"
    assert config.scenarios.curriculum.fixed_level == 2
    assert config.scenarios.cut_in.initial_gap_m.minimum == 20.0
    assert config.scenarios.cut_in.initial_gap_m.maximum == 40.0


def test_overlay_selects_fixed_occluded_crossing() -> None:
    config = load_config("configs/base.yaml", "configs/scenarios/occluded_crossing.yaml")

    assert config.scenarios.selection == "occluded_crossing"
    assert config.scenarios.curriculum.fixed_level == 3
    assert config.scenarios.occluded_crossing.conflict_distance_m.minimum == 20.0
    assert config.scenarios.occluded_crossing.conflict_distance_m.maximum == 40.0


def test_overlay_rejects_mapping_scalar_conflict(tmp_path: Path) -> None:
    overlay = tmp_path / "bad.yaml"
    overlay.write_text("scenarios:\n  lead_brake: 4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping conflict"):
        load_config("configs/base.yaml", overlay)


def test_overlay_merges_nested_scenario_settings(tmp_path: Path) -> None:
    overlay = tmp_path / "lead-gap.yaml"
    overlay.write_text(
        "scenarios:\n  lead_brake:\n    initial_gap_m:\n      minimum: 40.0\n",
        encoding="utf-8",
    )

    config = load_config("configs/base.yaml", overlay)

    assert config.scenarios.lead_brake.initial_gap_m.minimum == 40.0
    assert config.scenarios.lead_brake.initial_gap_m.maximum == 55.0


def test_missing_config_file_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(missing)


def test_yaml_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, VALID_CONFIG + "seed: 43\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_config(config_path)


def test_yaml_unhashable_mapping_keys_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "unhashable-key.yaml"
    config_path.write_text("? [invalid, key]\n: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unhashable"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("mode", "level", "selection"),
    [
        ("automatic", 0, "lead_brake"),
        ("fixed", 0, "auto"),
        ("fixed", 1, "nominal"),
        ("fixed", 3, "cut_in"),
    ],
)
def test_scenario_selection_must_match_curriculum_mode_and_level(
    tmp_path: Path,
    mode: str,
    level: int,
    selection: str,
) -> None:
    overlay = tmp_path / f"invalid-{mode}-{level}-{selection}.yaml"
    level_field = "initial_level" if mode == "automatic" else "fixed_level"
    overlay.write_text(
        "scenarios:\n"
        f"  selection: {selection}\n"
        "  curriculum:\n"
        f"    mode: {mode}\n"
        f"    {level_field}: {level}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="selection|curriculum"):
        load_config("configs/base.yaml", overlay)


@pytest.mark.parametrize("selection", ["lead_brake", "cut_in", "auto"])
def test_fixed_level_two_accepts_each_designed_selection(
    tmp_path: Path,
    selection: str,
) -> None:
    overlay = tmp_path / f"level-two-{selection}.yaml"
    overlay.write_text(
        "scenarios:\n"
        f"  selection: {selection}\n"
        "  curriculum:\n"
        "    mode: fixed\n"
        "    fixed_level: 2\n",
        encoding="utf-8",
    )

    config = load_config("configs/base.yaml", overlay)

    assert config.scenarios.selection == selection
    assert config.scenarios.curriculum.fixed_level == 2


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


@pytest.mark.parametrize(
    ("horizon", "scenario"),
    [
        (60, "lead_brake"),
        (80, "cut_in"),
        (100, "occluded_crossing"),
    ],
)
def test_scenario_worst_case_duration_must_fit_metadrive_horizon(
    horizon: int,
    scenario: str,
) -> None:
    payload = load_config("configs/base.yaml").model_dump(mode="python")
    payload["metadrive"]["horizon"] = horizon

    with pytest.raises(ValidationError, match=scenario):
        AppConfig.model_validate(payload)


def test_scenario_worst_case_duration_accepts_the_exact_configured_capacity() -> None:
    payload = load_config("configs/base.yaml").model_dump(mode="python")
    payload["metadrive"]["horizon"] = 120

    config = AppConfig.model_validate(payload)

    assert config.metadrive.horizon == 120


@pytest.mark.parametrize(
    ("scenario", "field", "maximum"),
    [
        ("lead_brake", "initial_gap_m", 55.0),
        ("lead_brake", "speed_fraction", 1.0),
        ("lead_brake", "trigger_s", 3.0),
        ("lead_brake", "mild_deceleration_mps2", 4.0),
        ("lead_brake", "severe_deceleration_mps2", 8.0),
        ("cut_in", "initial_gap_m", 40.0),
        ("cut_in", "trigger_s", 3.0),
        ("cut_in", "merge_duration_s", 3.0),
        ("cut_in", "speed_fraction", 1.05),
        ("occluded_crossing", "conflict_distance_m", 40.0),
        ("occluded_crossing", "crossing_start_offset_m", 12.0),
        ("occluded_crossing", "crossing_speed_mps", 6.0),
        ("occluded_crossing", "trigger_s", 3.0),
        ("occluded_crossing", "secondary_lead_gap_m", 55.0),
        ("occluded_crossing", "secondary_lead_speed_fraction", 1.0),
    ],
)
@pytest.mark.parametrize("bad_minimum", [0.0, -0.1])
def test_scenario_physical_ranges_require_strictly_positive_minimums(
    tmp_path: Path,
    scenario: str,
    field: str,
    maximum: float,
    bad_minimum: float,
) -> None:
    overlay = tmp_path / f"invalid-{scenario}-{field}.yaml"
    overlay.write_text(
        "scenarios:\n"
        f"  {scenario}:\n"
        f"    {field}:\n"
        f"      minimum: {bad_minimum}\n"
        f"      maximum: {maximum}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=field):
        load_config("configs/base.yaml", overlay)


@pytest.mark.parametrize("scenario", ["lead_brake", "cut_in", "occluded_crossing"])
@pytest.mark.parametrize("survival_s", [0.0, -0.1])
def test_scenario_survival_duration_must_be_strictly_positive(
    tmp_path: Path,
    scenario: str,
    survival_s: float,
) -> None:
    overlay = tmp_path / f"invalid-{scenario}-survival.yaml"
    overlay.write_text(
        f"scenarios:\n  {scenario}:\n    survival_s: {survival_s}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="survival_s"):
        load_config("configs/base.yaml", overlay)


@pytest.mark.parametrize("crossing_start_minimum", [3.0, 2.5])
def test_crossing_start_minimum_must_begin_outside_the_reveal_boundary(
    tmp_path: Path,
    crossing_start_minimum: float,
) -> None:
    overlay = tmp_path / "visible-at-reset.yaml"
    overlay.write_text(
        "scenarios:\n"
        "  occluded_crossing:\n"
        "    crossing_start_offset_m:\n"
        f"      minimum: {crossing_start_minimum}\n"
        "      maximum: 12.0\n"
        "    reveal_lateral_m: 3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="crossing_start_offset_m|reveal_lateral_m"):
        load_config("configs/base.yaml", overlay)


def test_crossing_start_minimum_may_be_just_outside_the_reveal_boundary(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "hidden-at-reset.yaml"
    overlay.write_text(
        "scenarios:\n"
        "  occluded_crossing:\n"
        "    crossing_start_offset_m:\n"
        "      minimum: 3.0001\n"
        "      maximum: 12.0\n"
        "    reveal_lateral_m: 3.0\n",
        encoding="utf-8",
    )

    config = load_config("configs/base.yaml", overlay)

    crossing = config.scenarios.occluded_crossing
    assert crossing.crossing_start_offset_m.minimum == 3.0001
    assert crossing.reveal_lateral_m == 3.0
