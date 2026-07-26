from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mad_driving.evaluation import models
from mad_driving.evaluation import selection as selection_module
from mad_driving.evaluation.models import (
    EvaluationCase,
    EvaluationEpisodeKey,
    EvaluationRunSpec,
    PpoRunBinding,
)
from mad_driving.evaluation.selection import (
    CheckpointCandidate,
    write_unselected_smoke_checkpoint_artifacts,
)
from mad_driving.evaluation.training_metrics import (
    TensorBoardEventSource,
    TrainingMetricPoint,
    extract_training_metrics,
    extract_training_metrics_from_event_sources,
    write_training_metrics_csv,
)
from mad_driving.methods import MethodProfileSnapshot
from tests.unit.evaluation.test_models import make_claim, make_episode, make_step
from tests.unit.evaluation.test_selection import score


@pytest.mark.parametrize(
    "values",
    (
        ("unknown", 0, "nominal"),
        ("level0_nominal", True, "nominal"),
        ("level0_nominal", 0, ""),
    ),
)
def test_evaluation_case_rejects_every_invalid_identity(values: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        EvaluationCase(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    (
        {"method_id": "unknown"},
        {"track": "unknown"},
        {"role": "unknown"},
        {"case_id": "unknown"},
        {"policy_seed": True},
        {"episode_rng_seed": 19_999},
        {"episode_rng_seed": 21_000},
    ),
)
def test_episode_key_rejects_invalid_closed_contract(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "method_id": "proposed",
        "track": "system",
        "role": "test",
        "policy_seed": 42,
        "case_id": "level0_nominal",
        "episode_rng_seed": 20_000,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        EvaluationEpisodeKey(**values)  # type: ignore[arg-type]


def test_episode_key_rejects_validation_seed_outside_validation_range() -> None:
    with pytest.raises(ValueError, match="10000"):
        EvaluationEpisodeKey(
            "proposed",
            "system",
            "validation",
            42,
            "level0_nominal",
            20_000,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"track": "unknown"},
        {"method_id": "unknown"},
        {"scenario_cell_id": "unknown"},
        {"shield_mode": "unknown"},
        {"episode_index": True},
        {"test_seed": 19_999},
        {"test_seed": 21_000},
        {"is_formal": 1},
        {"policy_seed": None},
        {"checkpoint_path": ""},
    ),
)
def test_run_spec_rejects_invalid_plan_identity(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "track": "system",
        "method_id": "proposed",
        "policy_seed": 42,
        "checkpoint_path": "checkpoint.zip",
        "scenario_cell_id": "level0_nominal",
        "episode_index": 0,
        "test_seed": 20_000,
        "shield_mode": "enforce",
        "is_formal": False,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        EvaluationRunSpec(**values)  # type: ignore[arg-type]


def test_run_spec_rejects_checkpoint_on_rule_baseline() -> None:
    with pytest.raises(ValueError, match="B0"):
        EvaluationRunSpec(
            "system",
            "b0_rule",
            None,
            "checkpoint.zip",
            "level0_nominal",
            0,
            20_000,
            "enforce",
            False,
        )


@pytest.mark.parametrize(
    "payload",
    (
        {
            "method_id": "b0_rule",
            "policy_seed": 42,
            "training_run_dir": "run",
        },
        {
            "method_id": "proposed",
            "policy_seed": True,
            "training_run_dir": "run",
        },
        {
            "method_id": "proposed",
            "policy_seed": 42,
            "training_run_dir": ".",
        },
        {
            "method_id": "proposed",
            "policy_seed": 42,
            "training_run_dir": 3,
        },
        {
            "method_id": "proposed",
            "policy_seed": 42,
            "training_run_dir": "run",
            "checkpoint_path": "",
        },
    ),
)
def test_ppo_binding_rejects_invalid_paths_and_policy_identity(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PpoRunBinding.model_validate(payload)


@pytest.mark.parametrize(
    ("function", "args"),
    (
        (models._require_mapping, ([], "value")),
        (models._require_mapping, ({1: "bad"}, "value")),
        (models._require_sequence, ("bad", "value")),
        (models._require_sequence, (object(), "value")),
        (models._require_finite, ("value", True)),
        (models._require_finite, ("value", float("inf"))),
        (models._require_probability, ("value", 2.0)),
        (models._require_non_empty_string, ("value", "")),
        (models._strict_non_empty_path, (3, "value")),
        (models._strict_non_empty_path, (" ", "value")),
    ),
)
def test_model_scalar_helpers_fail_closed(function: Any, args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize(
    ("values", "unique", "non_empty"),
    (
        ("text", False, False),
        (object(), False, False),
        ((1,), False, False),
        (("",), False, True),
        (("same", "same"), True, False),
    ),
)
def test_canonical_strings_reject_malformed_sequences(
    values: object,
    unique: bool,
    non_empty: bool,
) -> None:
    with pytest.raises(ValueError):
        models._canonical_strings(
            "values",
            values,
            unique=unique,
            non_empty=non_empty,
        )


def test_strict_record_helpers_reject_wrong_runtime_types() -> None:
    with pytest.raises(ValueError, match="record_schema_version"):
        models._validate_versions(True, 7)
    with pytest.raises(ValueError, match="research_contract_version"):
        models._validate_versions(1, True)
    with pytest.raises(ValueError, match="episode_key"):
        models._canonical_episode_key({})
    with pytest.raises(ValueError, match="method_profile"):
        models._canonical_profile({}, "proposed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must match"):
        models._canonical_profile(
            MethodProfileSnapshot.from_method_id("b1_nominal"),
            "proposed",
        )


@pytest.mark.parametrize(
    ("occurred", "kind"),
    ((1, None), (False, "unknown"), (False, "vehicle"), (True, None)),
)
def test_collision_contract_rejects_inconsistent_values(occurred: object, kind: object) -> None:
    with pytest.raises(ValueError):
        models._validate_collision(occurred, kind)


def test_case_and_checkpoint_helpers_reject_mismatched_identity() -> None:
    with pytest.raises(ValueError, match="case_id"):
        models._validate_case_identity("unknown", "nominal", 0)
    with pytest.raises(ValueError, match="inconsistent"):
        models._validate_case_identity("level0_nominal", "lead_brake", 0)
    with pytest.raises(ValueError, match="B0"):
        models._validate_checkpoint_identity("b0_rule", "bad.zip", None)
    with pytest.raises(ValueError, match="SHA-256"):
        models._validate_checkpoint_identity("proposed", "model.zip", "bad")


def test_nested_json_and_immutable_mappings_reject_mutation() -> None:
    frozen = models._freeze_json_object({"nested": [1, {"ok": True}]}, "payload")
    with pytest.raises(TypeError, match="immutable"):
        frozen |= {"new": 1}
    with pytest.raises(ValueError, match="JSON object"):
        models._freeze_json_object([], "payload")
    with pytest.raises(ValueError, match="finite"):
        models._freeze_json(float("nan"), "payload")
    with pytest.raises(ValueError, match="JSON values"):
        models._freeze_json({1, 2}, "payload")


def test_claim_and_error_mapping_helpers_reject_ambiguous_records() -> None:
    with pytest.raises(ValueError, match="RiskClaim"):
        models._canonical_claims("claim")
    claim = make_claim()
    duplicate = replace(claim, agent_id="hazard")
    with pytest.raises(ValueError, match="unique"):
        models._canonical_claims((claim, duplicate))
    with pytest.raises(ValueError, match="one-to-one"):
        models._validate_error_mapping(("nominal",), ())
    for error in (
        "hazard:RuntimeError:bad",
        "nominal:bad",
        "nominal:invalid-name!:bad",
        "nominal:RuntimeError:" + "x" * 257,
        "nominal:RuntimeError:bad\nmessage",
    ):
        with pytest.raises(ValueError):
            models._validate_error_mapping(("nominal",), (error,))


@pytest.mark.parametrize(
    "overrides",
    (
        {"episode_rng_seed": 20_002},
        {"case_id": "level0_nominal"},
        {"metadrive_scenario_index": True},
        {"scenario_selection_seed": -1},
        {"scenario_parameter_seed": -1},
    ),
)
def test_step_seed_identity_rejects_mismatch(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_step(**overrides)


@pytest.mark.parametrize(
    "overrides",
    (
        {"episode_rng_seed": 20_002},
        {"case_id": "level0_nominal"},
        {"metadrive_scenario_index": True},
        {"scenario_selection_seed": -1},
        {"scenario_parameter_seed": -1},
    ),
)
def test_episode_seed_identity_rejects_mismatch(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_episode(**overrides)


@pytest.mark.parametrize(
    "overrides",
    (
        {"run_id": ""},
        {"method_id": ""},
        {"policy_seed": True},
        {"timestep": True},
        {"metric": "unknown"},
        {"value": True},
        {"value": float("inf")},
    ),
)
def test_training_metric_point_rejects_invalid_values(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "run_id": "run",
        "method_id": "proposed",
        "policy_seed": 42,
        "timestep": 1,
        "metric": "rollout/ep_rew_mean",
        "value": 1.0,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        TrainingMetricPoint(**values)  # type: ignore[arg-type]


def _event_source(**overrides: object) -> TensorBoardEventSource:
    payload = b"event"
    values: dict[str, object] = {
        "run_id": "run",
        "method_id": "proposed",
        "policy_seed": 42,
        "event_relative_path": "events.out.tfevents.test",
        "payload": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    values.update(overrides)
    return TensorBoardEventSource(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    (
        {"run_id": ""},
        {"method_id": ""},
        {"policy_seed": True},
        {"event_relative_path": ""},
        {"event_relative_path": "../events.out.tfevents.test"},
        {"event_relative_path": "not-an-event"},
        {"payload": "event"},
        {"sha256": "A" * 64},
        {"sha256": "a" * 64},
    ),
)
def test_event_source_rejects_invalid_provenance(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _event_source(**overrides)


def test_training_metric_entry_points_reject_invalid_boundaries(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="smoke"):
        extract_training_metrics((tmp_path,), smoke=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        extract_training_metrics((), smoke=True)
    with pytest.raises(TypeError, match="smoke"):
        extract_training_metrics_from_event_sources((), smoke=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        extract_training_metrics_from_event_sources((), smoke=True)
    with pytest.raises(TypeError, match="TensorBoardEventSource"):
        extract_training_metrics_from_event_sources((object(),), smoke=True)  # type: ignore[arg-type]


def test_training_metric_csv_rejects_invalid_mode_points_and_destination(
    tmp_path: Path,
) -> None:
    point = TrainingMetricPoint("run", "proposed", 42, 0, "policy_entropy", None)
    with pytest.raises(TypeError, match="smoke"):
        write_training_metrics_csv(tmp_path / "metrics.csv", (point,), smoke=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TrainingMetricPoint"):
        write_training_metrics_csv(
            tmp_path / "metrics.csv",
            (object(),),
            smoke=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unavailable"):
        write_training_metrics_csv(tmp_path / "metrics.csv", (point,), smoke=False)
    destination = tmp_path / "exists.csv"
    destination.touch()
    with pytest.raises(FileExistsError):
        write_training_metrics_csv(destination, (), smoke=True)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"path": Path("model.txt")},
        {"sha256": "A" * 64},
        {"method_id": "b0_rule"},
        {"policy_seed": True},
        {"checkpoint_kind": "unknown"},
        {"curriculum_level": 4},
        {"training_timestep": -1},
    ),
)
def test_checkpoint_candidate_rejects_invalid_identity(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "path": Path("model.zip"),
        "sha256": "a" * 64,
        "method_id": "proposed",
        "policy_seed": 42,
        "checkpoint_kind": "final",
        "curriculum_level": 3,
        "training_timestep": 10,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        CheckpointCandidate(**values)  # type: ignore[arg-type]


def test_unselected_smoke_writer_rejects_empty_wrong_and_duplicate_candidates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="non-empty"):
        write_unselected_smoke_checkpoint_artifacts(output, ())
    with pytest.raises(TypeError, match="CheckpointCandidate"):
        write_unselected_smoke_checkpoint_artifacts(output, (object(),))  # type: ignore[arg-type]
    candidate = CheckpointCandidate(
        Path("model.zip"),
        "a" * 64,
        "proposed",
        42,
        "final",
        3,
        10,
    )
    with pytest.raises(ValueError, match="duplicate"):
        write_unselected_smoke_checkpoint_artifacts(output, (candidate, candidate))


def test_selection_filesystem_helpers_reject_missing_and_wrong_kinds(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    regular = tmp_path / "regular"
    regular.write_text("data", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(ValueError, match="unavailable"):
        selection_module._validated_directory(missing, "directory")
    with pytest.raises(ValueError, match="directory"):
        selection_module._validated_directory(regular, "directory")
    with pytest.raises(ValueError, match="unavailable"):
        selection_module._validated_regular_file(missing, "file")
    with pytest.raises(ValueError, match="regular file"):
        selection_module._validated_regular_file(directory, "file")
    with pytest.raises(ValueError, match="unavailable"):
        selection_module._path_regular_file_signature(missing)
    with pytest.raises(ValueError, match="stable regular file"):
        selection_module._path_regular_file_signature(directory)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        ({"a": 1}, [1], False),
        ([1], {"a": 1}, False),
        ({"a": 1}, {"b": 1}, False),
        ([1], "one", False),
        ("one", [1], False),
        ([1], [1, 2], False),
        ({"a": [1]}, {"a": [1]}, True),
        (1, True, False),
    ),
)
def test_json_equivalence_is_type_and_shape_strict(
    left: object, right: object, expected: bool
) -> None:
    assert selection_module._json_equivalent(left, right) is expected


def test_selection_json_and_integer_helpers_reject_ambiguity() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        selection_module._strict_json_object([("key", 1), ("key", 2)])
    with pytest.raises(ValueError, match="integer"):
        selection_module._strict_json_integer(True, "value")
    with pytest.raises(ValueError, match="integer"):
        selection_module._strict_json_integer(-1, "value")


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"num_timesteps": True},
        {"num_timesteps": -1},
        {"num_timesteps": float("nan")},
    ),
)
def test_checkpoint_timestep_rejects_noncanonical_json(tmp_path: Path, payload: object) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr(
            "data",
            json.dumps(payload, allow_nan=True, separators=(",", ":")),
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        selection_module._checkpoint_training_timestep(checkpoint, digest)


def test_checkpoint_timestep_rejects_malformed_digest_and_missing_data(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        selection_module._checkpoint_training_timestep(tmp_path / "missing.zip", "bad")
    checkpoint = tmp_path / "missing-data.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("other", b"value")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="data member"):
        selection_module._checkpoint_training_timestep(checkpoint, digest)


@pytest.mark.parametrize(
    "config",
    (
        {},
        {"method": {"id": "proposed", "extra": 1}, "training": {"seed": 42}},
        {"method": {"id": 3}, "training": {"seed": 42}},
        {"method": {"id": "proposed"}},
        {"method": {"id": "proposed"}, "training": {"seed": True}},
        {"method": {"id": "proposed"}, "training": {"seed": -1}},
    ),
)
def test_resolved_method_seed_rejects_malformed_config(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        selection_module._resolved_method_and_seed(config)


def test_checkpoint_directory_inventory_rejects_unsupported_entry(
    tmp_path: Path,
) -> None:
    (tmp_path / "unsupported.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        selection_module._validate_checkpoint_directory_entries(tmp_path)


def test_selection_entry_points_reject_empty_wrong_and_cross_group_scores(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="non-empty"):
        selection_module.select_checkpoint(())
    with pytest.raises(TypeError, match="CheckpointScore"):
        selection_module.select_checkpoint((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        selection_module.write_selection_artifacts(output, ())
    with pytest.raises(TypeError, match="CheckpointScore"):
        selection_module.write_selection_artifacts(output, (object(),))  # type: ignore[arg-type]

    first = score(
        tmp_path,
        digest="1" * 64,
        name="first.zip",
        timestep=10,
        reward=1.0,
        collisions=0,
        successes=1,
        route_completion=0.5,
    )
    second = score(
        tmp_path,
        digest="2" * 64,
        name="second.zip",
        timestep=10,
        reward=1.0,
        collisions=0,
        successes=1,
        route_completion=0.5,
        policy_seed=43,
    )
    with pytest.raises(ValueError, match="method/policy seed"):
        selection_module.select_checkpoint((first, second))


def test_selection_writers_refuse_existing_artifacts_and_duplicate_smoke_keys(
    tmp_path: Path,
) -> None:
    first = CheckpointCandidate(
        Path("first.zip"),
        "1" * 64,
        "proposed",
        42,
        "final",
        3,
        10,
    )
    duplicate_key = CheckpointCandidate(
        Path("second.zip"),
        "2" * 64,
        "proposed",
        42,
        "periodic",
        2,
        5,
    )
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="method/policy seed"):
        write_unselected_smoke_checkpoint_artifacts(output, (first, duplicate_key))

    (output / "model_selection.csv").touch()
    with pytest.raises(FileExistsError):
        write_unselected_smoke_checkpoint_artifacts(output, (first,))
