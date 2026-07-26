from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from mad_driving import visualization


def _write_manifest(directory: Path, payload: object, *, canonical: bool = True) -> None:
    if canonical:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    else:
        encoded = json.dumps(payload, indent=2)
    (directory / "evaluation_manifest.json").write_bytes((encoded + "\n").encode("utf-8"))


def test_verified_bundle_rejects_outside_undeclared_and_non_utf8_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"\xff")
    identity = visualization._ArtifactIdentity(
        size_bytes=1,
        sha256=hashlib.sha256(b"\xff").hexdigest(),
    )
    bundle = visualization._VerifiedBundle(tmp_path, {"artifact.bin": identity})

    with pytest.raises(ValueError, match="outside"):
        bundle.relative_artifact(tmp_path.parent / "outside.bin")
    with pytest.raises(ValueError, match="undeclared"):
        bundle.relative_artifact(tmp_path / "other.bin")
    with pytest.raises(ValueError, match="UTF-8"):
        bundle.read_text(artifact)


def test_find_bundle_rejects_path_without_manifest(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.txt"
    path.parent.mkdir()
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="not inside"):
        visualization._find_and_verify_bundle(path)


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"schema_version": 1},
        {"schema_version": True, "artifacts": []},
        {"schema_version": 1, "artifacts": {}},
        {"schema_version": 1, "artifacts": [1]},
        {
            "schema_version": 1,
            "artifacts": [{"path": "../bad", "size_bytes": 0, "sha256": "a" * 64}],
        },
        {
            "schema_version": 1,
            "artifacts": [{"path": "a", "size_bytes": True, "sha256": "a" * 64}],
        },
        {
            "schema_version": 1,
            "artifacts": [{"path": "a", "size_bytes": 0, "sha256": "bad"}],
        },
    ),
)
def test_manifest_verifier_rejects_invalid_schema(tmp_path: Path, payload: object) -> None:
    _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError):
        visualization._verify_bundle(tmp_path)


def test_manifest_verifier_rejects_duplicate_unsorted_noncanonical_and_inventory(
    tmp_path: Path,
) -> None:
    duplicate = {
        "schema_version": 1,
        "artifacts": [
            {"path": "a", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            {"path": "a", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        ],
    }
    _write_manifest(tmp_path, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        visualization._verify_bundle(tmp_path)

    (tmp_path / "evaluation_manifest.json").unlink()
    unsorted = {
        "schema_version": 1,
        "artifacts": [
            {"path": "b", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            {"path": "a", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        ],
    }
    _write_manifest(tmp_path, unsorted)
    with pytest.raises(ValueError, match="sorted"):
        visualization._verify_bundle(tmp_path)

    (tmp_path / "evaluation_manifest.json").unlink()
    _write_manifest(tmp_path, {"schema_version": 1, "artifacts": []}, canonical=False)
    with pytest.raises(ValueError, match="canonical"):
        visualization._verify_bundle(tmp_path)

    (tmp_path / "evaluation_manifest.json").unlink()
    inventory = {
        "schema_version": 1,
        "artifacts": [
            {"path": "missing", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        ],
    }
    _write_manifest(tmp_path, inventory)
    with pytest.raises(ValueError, match="inventory"):
        visualization._verify_bundle(tmp_path)


def test_manifest_verifier_accepts_empty_canonical_bundle(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"schema_version": 1, "artifacts": []})
    bundle = visualization._verify_bundle(tmp_path)
    assert bundle.root == tmp_path.resolve()
    assert bundle.artifacts == {}


@pytest.mark.parametrize(
    "value",
    (
        None,
        "",
        "evaluation_manifest.json",
        "absolute/path/../bad",
        "../bad",
        "/absolute",
        "C:/drive",
        "back\\slash",
        "./dot",
    ),
)
def test_manifest_path_validator_rejects_unsafe_forms(value: object) -> None:
    assert visualization._safe_manifest_path(value) is False


def test_manifest_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        visualization._unique_json_object([("key", 1), ("key", 2)])


def test_regular_path_helpers_reject_missing_and_wrong_kind(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    regular = tmp_path / "regular"
    regular.write_text("data", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(ValueError, match="unavailable"):
        visualization._require_regular_directory(missing, "directory")
    with pytest.raises(ValueError, match="directory"):
        visualization._require_regular_directory(regular, "directory")
    with pytest.raises(ValueError, match="unavailable"):
        visualization._require_regular_file(missing)
    with pytest.raises(ValueError, match="regular file"):
        visualization._require_regular_file(directory)


def test_checked_identity_rejects_symbolic_nonregular_and_reparse_modes() -> None:
    base = {
        "st_dev": 1,
        "st_ino": 2,
        "st_size": 3,
        "st_mtime_ns": 4,
    }
    with pytest.raises(ValueError, match="symbolic"):
        visualization._checked_file_identity(
            SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0, **base),
            "file",
        )
    with pytest.raises(ValueError, match="reparse"):
        visualization._checked_file_identity(
            SimpleNamespace(
                st_mode=stat.S_IFREG,
                st_file_attributes=visualization._FILE_ATTRIBUTE_REPARSE_POINT,
                **base,
            ),
            "file",
        )
    with pytest.raises(ValueError, match="regular"):
        visualization._checked_file_identity(
            SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0, **base),
            "file",
        )


def test_stable_reader_rejects_expected_size_and_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="size verification"):
        visualization._read_stable_regular_file(
            source,
            expected=visualization._ArtifactIdentity(1, hashlib.sha256(b"payload").hexdigest()),
            label="source",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        visualization._read_stable_regular_file(
            source,
            expected=visualization._ArtifactIdentity(len(b"payload"), "a" * 64),
            label="source",
        )
