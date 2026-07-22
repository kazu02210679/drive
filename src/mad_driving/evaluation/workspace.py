"""Atomic sibling staging for one online evaluation episode."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Self

from mad_driving.atomic import rename_no_replace

_MANIFEST_NAME = "evaluation_manifest.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass
class EvaluationWorkspace:
    """A private staging directory that may be atomically published once."""

    destination: Path
    path: Path

    @classmethod
    def stage(cls, destination: Path) -> Self:
        final = Path(destination)
        if final.exists():
            raise FileExistsError(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
        return cls(destination=final, path=staging)

    def publish(self) -> Path:
        if self.destination.exists():
            raise FileExistsError(self.destination)
        if not self.path.is_dir():
            raise RuntimeError("evaluation staging workspace is unavailable")
        self._validate_manifest()
        self._fsync_files()
        rename_no_replace(self.path, self.destination)
        return self.destination

    def write_manifest(self) -> Path:
        """Write the canonical inventory after every other staged artifact."""

        manifest_path = self.path / _MANIFEST_NAME
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        artifacts = [
            {
                "path": path.relative_to(self.path).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in self._artifact_paths()
        ]
        payload = {"artifacts": artifacts, "schema_version": 1}
        encoded = _canonical_json(payload)
        with manifest_path.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        return manifest_path

    def _validate_manifest(self) -> None:
        manifest_path = self.path / _MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("evaluation manifest is missing")
        encoded = manifest_path.read_bytes()
        try:
            decoded = encoded.decode("utf-8")
            payload = json.loads(decoded, object_pairs_hook=_unique_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("evaluation manifest is not strict JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"artifacts", "schema_version"}:
            raise ValueError("evaluation manifest fields are invalid")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("evaluation manifest schema_version must be 1")
        raw_artifacts = payload["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ValueError("evaluation manifest artifacts must be a list")
        inventory: dict[str, tuple[int, str]] = {}
        for item in raw_artifacts:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
                raise ValueError("evaluation manifest artifact fields are invalid")
            path = item["path"]
            size = item["size_bytes"]
            digest = item["sha256"]
            if not _safe_manifest_path(path):
                raise ValueError("evaluation manifest contains an unsafe path")
            if path in inventory:
                raise ValueError("evaluation manifest contains a duplicate path")
            if type(size) is not int or size < 0:
                raise ValueError("evaluation manifest size is invalid")
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError("evaluation manifest SHA-256 is invalid")
            inventory[path] = (size, digest)
        if list(inventory) != sorted(inventory):
            raise ValueError("evaluation manifest inventory is not sorted")
        if encoded != _canonical_json(payload):
            raise ValueError("evaluation manifest encoding is not canonical")
        actual_paths = self._artifact_paths()
        actual = {path.relative_to(self.path).as_posix(): path for path in actual_paths}
        if set(inventory) != set(actual):
            raise ValueError("evaluation manifest inventory does not match staged files")
        for relative, path in actual.items():
            expected_size, expected_digest = inventory[relative]
            if path.stat().st_size != expected_size or _sha256_file(path) != expected_digest:
                raise ValueError("evaluation manifest artifact verification failed")

    def _artifact_paths(self) -> list[Path]:
        manifest_path = self.path / _MANIFEST_NAME
        paths = sorted(
            (
                candidate
                for candidate in self.path.rglob("*")
                if candidate.is_file() and candidate != manifest_path
            ),
            key=lambda path: path.relative_to(self.path).as_posix(),
        )
        if any(path.is_symlink() for path in paths):
            raise ValueError("evaluation manifest cannot inventory symbolic links")
        return paths

    def _fsync_files(self) -> None:
        for path in sorted(candidate for candidate in self.path.rglob("*") if candidate.is_file()):
            with path.open("ab") as output:
                output.flush()
                os.fsync(output.fileno())


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_manifest_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        value != _MANIFEST_NAME
        and posix.as_posix() == value
        and not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and "." not in posix.parts
        and ".." not in posix.parts
    )


__all__ = ["EvaluationWorkspace"]
