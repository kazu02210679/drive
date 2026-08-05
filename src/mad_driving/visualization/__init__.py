"""Strict offline visualization entry points for verified evaluation bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final, Literal

_MANIFEST_NAME = "evaluation_manifest.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_EntryKind = Literal["file", "directory"]
SMOKE_RESULT_LABEL: Final = "SMOKE - NOT A RESEARCH RESULT"
METHOD_ORDER: Final = (
    "b0_rule",
    "b1_nominal",
    "b2_multi_no_review",
    "proposed",
    "proposed_no_critic",
    "proposed_no_shield",
    "proposed_no_hazard",
)
PLOT_INVENTORY: Final = (
    "learning_curve.png",
    "collision_rate.png",
    "success_route_completion.png",
    "unnecessary_braking.png",
    "comfort.png",
    "agent_disagreement.png",
)


@dataclass(frozen=True)
class _ArtifactIdentity:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    size_bytes: int
    modified_ns: int
    file_attributes: int


@dataclass(frozen=True)
class _VerifiedBundle:
    root: Path
    artifacts: dict[str, _ArtifactIdentity]

    def relative_artifact(self, path: Path) -> str:
        candidate = Path(os.path.abspath(path))
        root = Path(os.path.abspath(self.root))
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"artifact input is outside the verified bundle: {path}") from error
        if relative not in self.artifacts:
            raise ValueError(f"artifact input is undeclared in the evaluation manifest: {relative}")
        return relative

    def read_bytes(self, path: Path) -> bytes:
        relative = self.relative_artifact(path)
        source = self.root / PurePosixPath(relative)
        expected = self.artifacts[relative]
        return _read_stable_regular_file(source, expected=expected, label=relative)

    def read_text(self, path: Path) -> str:
        try:
            return self.read_bytes(path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"verified artifact is not UTF-8: {path}") from error


def _find_and_verify_bundle(input_path: Path) -> _VerifiedBundle:
    candidate = Path(os.path.abspath(input_path))
    current = candidate if candidate.is_dir() else candidate.parent
    for parent in (current, *current.parents):
        if parent.joinpath(_MANIFEST_NAME).exists():
            bundle = _verify_bundle(parent)
            bundle.relative_artifact(candidate)
            return bundle
    raise ValueError(f"artifact input is not inside a manifested evaluation bundle: {input_path}")


def _verify_bundle(bundle_dir: Path) -> _VerifiedBundle:
    root = Path(os.path.abspath(bundle_dir))
    _require_regular_directory(root, "evaluation bundle")
    files = _bundle_files(root)
    manifest = root / _MANIFEST_NAME
    if manifest not in files:
        raise ValueError("evaluation manifest is missing")
    encoded = _read_stable_regular_file(manifest, label=_MANIFEST_NAME)
    try:
        payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("evaluation manifest is not strict JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"artifacts", "schema_version"}:
        raise ValueError("evaluation manifest fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("evaluation manifest schema_version must be 1")
    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ValueError("evaluation manifest artifacts must be a list")
    inventory: dict[str, _ArtifactIdentity] = {}
    for item in raw_artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("evaluation manifest artifact fields are invalid")
        relative = item["path"]
        size = item["size_bytes"]
        digest = item["sha256"]
        if not _safe_manifest_path(relative):
            raise ValueError("evaluation manifest contains an unsafe path")
        if relative in inventory:
            raise ValueError("evaluation manifest contains a duplicate path")
        if type(size) is not int or size < 0:
            raise ValueError("evaluation manifest size is invalid")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("evaluation manifest SHA-256 is invalid")
        inventory[relative] = _ArtifactIdentity(size_bytes=size, sha256=digest)
    if list(inventory) != sorted(inventory):
        raise ValueError("evaluation manifest inventory is not sorted")
    if encoded != _canonical_json(payload):
        raise ValueError("evaluation manifest encoding is not canonical")
    actual = {path.relative_to(root).as_posix(): path for path in files if path != manifest}
    if set(inventory) != set(actual):
        raise ValueError("evaluation manifest inventory does not match bundle files")
    for relative, identity in inventory.items():
        path = actual[relative]
        _read_stable_regular_file(path, expected=identity, label=relative)
    return _VerifiedBundle(root=root, artifacts=inventory)


def _bundle_files(root: Path) -> list[Path]:
    directories = [root]
    files: list[Path] = []
    while directories:
        directory = directories.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"evaluation bundle directory is unreadable: {directory}") from error
        for entry in entries:
            kind = _validated_entry_kind(entry)
            path = Path(entry.path)
            if kind == "directory":
                directories.append(path)
            else:
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _validated_entry_kind(entry: os.DirEntry[str]) -> _EntryKind:
    if entry.is_symlink():
        raise ValueError(f"evaluation bundle entry is a symbolic link: {entry.name}")
    try:
        metadata = entry.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"evaluation bundle entry is unreadable: {entry.name}") from error
    if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"evaluation bundle entry is a reparse point: {entry.name}")
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    raise ValueError(f"evaluation bundle entry is not a regular file or directory: {entry.name}")


def _require_regular_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} cannot be a symbolic link")
    if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"{label} cannot be a reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular directory")


def _require_regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"artifact is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"artifact cannot be a symbolic link: {path}")
    if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"artifact cannot be a reparse point: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact must be a regular file: {path}")


def _checked_file_identity(metadata: os.stat_result | Any, label: str) -> _FileIdentity:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"verified file cannot be a symbolic link: {label}")
    attributes = getattr(metadata, "st_file_attributes", 0)
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"verified file cannot be a reparse point: {label}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"verified file must be regular: {label}")
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size_bytes=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        file_attributes=attributes,
    )


def _read_stable_regular_file(
    path: Path,
    *,
    expected: _ArtifactIdentity | None = None,
    label: str,
) -> bytes:
    try:
        path_before = _checked_file_identity(path.lstat(), label)
        with path.open("rb") as source:
            handle_before = _checked_file_identity(os.fstat(source.fileno()), label)
            if handle_before != path_before:
                raise ValueError(f"verified file identity changed before reading: {label}")
            payload = source.read()
            handle_after = _checked_file_identity(os.fstat(source.fileno()), label)
        path_after = _checked_file_identity(path.lstat(), label)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"verified file is unreadable: {label}") from error
    if handle_after != handle_before or path_after != handle_after:
        raise ValueError(f"verified file identity changed while reading: {label}")
    if len(payload) != handle_after.size_bytes:
        raise ValueError(f"verified file size changed while reading: {label}")
    if expected is not None:
        if len(payload) != expected.size_bytes:
            raise ValueError(f"evaluation manifest size verification failed: {label}")
        if hashlib.sha256(payload).hexdigest() != expected.sha256:
            raise ValueError(f"evaluation manifest SHA-256 verification failed: {label}")
    return payload


def _reject_output_in_source_bundle(bundle_root: Path, output_path: Path) -> None:
    candidate = Path(os.path.abspath(output_path))
    current = candidate
    missing_parts: list[str] = []
    while True:
        try:
            current.lstat()
            existing_ancestor = current
            break
        except FileNotFoundError as error:
            if current.parent == current:
                raise ValueError(f"output parent is unavailable: {output_path}") from error
            missing_parts.append(current.name)
            current = current.parent
        except OSError as error:
            raise ValueError(f"output path is unreadable: {output_path}") from error

    for ancestor in (existing_ancestor, *existing_ancestor.parents):
        try:
            metadata = ancestor.lstat()
        except OSError as error:
            raise ValueError(f"output parent is unreadable: {ancestor}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"output parent cannot be a symbolic link: {ancestor}")
        if getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"output parent cannot be a reparse point: {ancestor}")

    try:
        resolved_ancestor = existing_ancestor.resolve(strict=True)
        resolved_root = bundle_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("output containment could not be established") from error
    resolved_output = resolved_ancestor.joinpath(*reversed(missing_parts))
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    raise ValueError(f"output cannot resolve inside the verified source bundle: {output_path}")


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


def write_learning_curve(train_metrics_csv: Path, output_png: Path) -> None:
    from mad_driving.visualization.plots import write_learning_curve as implementation

    implementation(train_metrics_csv, output_png)


def write_safety_efficiency_plots(eval_metrics_csv: Path, output_dir: Path) -> tuple[Path, ...]:
    from mad_driving.visualization.plots import write_safety_efficiency_plots as implementation

    return implementation(eval_metrics_csv, output_dir)


def write_episode_gif(step_jsonl: Path, frames_dir: Path, output_gif: Path) -> None:
    from mad_driving.visualization.overlay import write_episode_gif as implementation

    implementation(step_jsonl, frames_dir, output_gif)


def write_markdown_report(bundle_dir: Path, output_md: Path) -> None:
    from mad_driving.visualization.report import write_markdown_report as implementation

    implementation(bundle_dir, output_md)


__all__ = [
    "METHOD_ORDER",
    "PLOT_INVENTORY",
    "SMOKE_RESULT_LABEL",
    "write_episode_gif",
    "write_learning_curve",
    "write_markdown_report",
    "write_safety_efficiency_plots",
]
