"""Shared strict command-line operational boundaries."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path
from unicodedata import category

from pydantic import ValidationError

_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
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


def validate_absent_destination(
    path: Path,
    *,
    source_roots: Sequence[Path] = (),
) -> Path:
    """Prove an absent destination has only regular directory ancestors."""

    candidate = Path(os.path.abspath(path))
    current = candidate
    missing_parts: list[str] = []
    while True:
        try:
            current.lstat()
            existing_ancestor = current
            break
        except FileNotFoundError as error:
            if current.parent == current:
                raise ValueError(f"output parent is unavailable: {candidate}") from error
            missing_parts.append(current.name)
            current = current.parent
        except OSError as error:
            raise ValueError(f"output path is unreadable: {candidate}") from error

    if existing_ancestor == candidate:
        raise FileExistsError(f"Output already exists: {candidate}")
    for ancestor in (existing_ancestor, *existing_ancestor.parents):
        try:
            metadata = ancestor.lstat()
        except OSError as error:
            raise ValueError(f"output parent is unreadable: {ancestor}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"output parent cannot be a symbolic link: {ancestor}")
        if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"output parent cannot be a reparse point: {ancestor}")
    if not existing_ancestor.is_dir():
        raise ValueError(f"output parent must be a regular directory: {existing_ancestor}")
    try:
        resolved_ancestor = existing_ancestor.resolve(strict=True)
    except OSError as error:
        raise ValueError("output containment could not be established") from error
    resolved_output = resolved_ancestor.joinpath(*reversed(missing_parts))
    for source_root in source_roots:
        try:
            resolved_source = Path(source_root).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"authenticated source is unavailable: {source_root}") from error
        try:
            resolved_output.relative_to(resolved_source)
        except ValueError:
            continue
        raise ValueError(
            f"output cannot resolve inside an authenticated training-run source: {candidate}"
        )
    return candidate


__all__ = ["concise_operational_error", "validate_absent_destination"]
