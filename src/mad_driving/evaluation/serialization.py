"""Strict YAML and JSONL boundaries for evaluation plans and records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, TypeVar, cast

import yaml

from mad_driving.config.parsing import load_unique_yaml
from mad_driving.evaluation.models import (
    EvaluationPlanConfig,
    EvaluationStepRecord,
    Phase6PublicationPlan,
)


class _StrictRecord(Protocol):
    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> _StrictRecord: ...


T = TypeVar("T", bound=_StrictRecord)


def _load_plan_payload(path: Path) -> dict[str, object]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        payload = load_unique_yaml(text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"evaluation plan YAML is unreadable: {path}") from error
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ValueError("evaluation plan YAML root must be an object with string keys")
    return dict(payload)


def load_evaluation_plan(path: Path) -> EvaluationPlanConfig:
    """Load one duplicate-safe legacy-compatible frozen plan model."""

    return EvaluationPlanConfig.model_validate(_load_plan_payload(path))


def load_phase6_publication_plan(path: Path) -> Phase6PublicationPlan:
    """Load a complete duplicate-safe Phase 6 publication plan."""

    return Phase6PublicationPlan.model_validate(_load_plan_payload(path))


def write_jsonl_strict(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Create one compact sorted UTF-8 JSON object per line, never overwriting."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    encoded_rows: list[bytes] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("JSONL rows must be mappings")
        encoded_rows.append(
            (
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        for encoded in encoded_rows:
            output.write(encoded)


def read_jsonl_strict(path: Path, model: type[T]) -> tuple[T, ...]:
    """Read a path once and parse it through the shared strict bytes boundary."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"JSONL file is unreadable: {source}") from error
    return parse_jsonl_bytes_strict(payload, model)


def parse_jsonl_bytes_strict(payload: bytes, model: type[T]) -> tuple[T, ...]:
    """Parse strict UTF-8/LF JSONL records from caller-supplied immutable bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("JSONL payload must be bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("JSONL bytes are not valid UTF-8") from error
    if payload and not payload.endswith(b"\n"):
        raise ValueError("JSONL file must end with a trailing newline")
    if not payload:
        if model is EvaluationStepRecord:
            raise ValueError("step JSONL file must not be empty")
        return ()
    lines = text.splitlines()
    blank_line = next((index for index, line in enumerate(lines, start=1) if not line), None)
    if blank_line is not None:
        raise ValueError(f"JSONL file contains a blank record at line {blank_line}")
    records: list[T] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL record is malformed at line {line_number}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL record must be an object at line {line_number}")
        try:
            record = cast(T, model.from_dict(cast(Mapping[str, object], value)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"JSONL record is invalid at line {line_number}: {error}") from error
        records.append(record)
    result = tuple(records)
    if model is EvaluationStepRecord:
        _validate_step_stream(cast(tuple[EvaluationStepRecord, ...], result))
    return result


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(constant: str) -> object:
    raise ValueError(f"non-finite JSON number: {constant}")


def _validate_step_stream(records: tuple[EvaluationStepRecord, ...]) -> None:
    if not records:
        raise ValueError("step JSONL file must not be empty")
    episode_key = records[0].episode_key
    if any(record.episode_key != episode_key for record in records[1:]):
        raise ValueError("step JSONL file contains more than one episode key")
    episode_index = records[0].episode_index
    if any(record.episode_index != episode_index for record in records[1:]):
        raise ValueError("step JSONL file contains more than one episode_index")
    is_formal = records[0].is_formal
    if any(record.is_formal is not is_formal for record in records[1:]):
        raise ValueError("step JSONL file contains more than one is_formal value")
    shield_mode = records[0].shield_mode
    if any(record.shield_mode != shield_mode for record in records[1:]):
        raise ValueError("step JSONL file contains more than one shield_mode")
    if any(record.step_index != expected for expected, record in enumerate(records)):
        raise ValueError("step indices must be contiguous and zero-based")


__all__ = [
    "load_evaluation_plan",
    "load_phase6_publication_plan",
    "parse_jsonl_bytes_strict",
    "read_jsonl_strict",
    "write_jsonl_strict",
]
