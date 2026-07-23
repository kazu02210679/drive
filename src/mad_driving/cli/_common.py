"""Shared strict command-line operational boundaries."""

from __future__ import annotations

from unicodedata import category

from pydantic import ValidationError

from mad_driving.evaluation.paths import validate_absent_destination

_MAX_OPERATIONAL_ERROR_LENGTH = 240


def concise_operational_error(error: Exception) -> str:
    """Return one stable, sanitized line without leaking nested validation input."""

    if isinstance(error, ValidationError):
        details = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        if details:
            first = details[0]
            location = ".".join(str(part) for part in first["loc"])
            message = str(first["msg"])
            raw = f"{location}: {message}" if location else message
        else:
            raw = "configuration validation failed"
    else:
        raw = str(error) or type(error).__name__
    sanitized = "".join(
        " " if character.isspace() or category(character)[0] in {"C", "Z"} else character
        for character in raw
    )
    concise = " ".join(sanitized.split())
    if not concise:
        concise = type(error).__name__
    if len(concise) > _MAX_OPERATIONAL_ERROR_LENGTH:
        concise = f"{concise[: _MAX_OPERATIONAL_ERROR_LENGTH - 3].rstrip()}..."
    return concise


__all__ = ["concise_operational_error", "validate_absent_destination"]
