"""Atomic, cross-platform publication of one training run directory."""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

_OWNERSHIP_MARKER = ".training-owner"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4


def _raise_rename_error(error_number: int, destination: Path) -> NoReturn:
    raise OSError(error_number, os.strerror(error_number), destination)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when destination does not exist."""

    if os.name == "nt":
        os.rename(source, destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    result = -1
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            _raise_rename_error(errno.ENOTSUP, destination)
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            _raise_rename_error(errno.ENOTSUP, destination)
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, _RENAME_EXCL)
    else:
        _raise_rename_error(errno.ENOTSUP, destination)
    if result != 0:
        _raise_rename_error(ctypes.get_errno(), destination)


def _private_sibling(path: Path, purpose: str) -> Path:
    """Return an unpredictable sibling path that does not currently exist."""

    while True:
        candidate = path.parent / f".{path.name}.{purpose}-{secrets.token_hex(16)}"
        if not candidate.exists():
            return candidate


def _publication_error(path: Path) -> FileExistsError:
    return FileExistsError(f"Run directory ownership changed before publication: {path}")


@dataclass
class RunDirectoryOwnership:
    """Private workspace plus an atomically installed destination claim.

    Claim and failed-workspace directories are intentionally retained as explicitly named
    recovery artifacts. No path is unlinked after an identity check, so a concurrent writer's
    replacement can never be deleted by this transaction.
    """

    path: Path
    workspace: Path
    claim_recovery: Path
    marker_stat: os.stat_result
    token: bytes
    claim_retired: bool = False
    published: bool = False

    @classmethod
    def acquire(
        cls,
        path: Path,
        *,
        require_absent: bool = False,
    ) -> RunDirectoryOwnership:
        """Claim a preflighted absent/empty destination before training side effects."""

        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        if existed:
            if require_absent:
                raise FileExistsError(f"Run directory already exists before ownership: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"Run directory is not a directory: {path}")
            if any(path.iterdir()):
                raise FileExistsError(f"Run directory is non-empty: {path}")

        claim_recovery = _private_sibling(path, "ownership-recovery")
        claim_recovery.mkdir(mode=0o700)
        marker = claim_recovery / _OWNERSHIP_MARKER
        token = secrets.token_hex(32).encode("ascii")
        descriptor = os.open(
            marker,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            os.write(descriptor, token)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        marker_stat = marker.stat()

        workspace = _private_sibling(path, "training")
        if existed:
            try:
                _rename_no_replace(path, workspace)
            except OSError:
                raise
            if any(workspace.iterdir()):
                cls._restore_unclaimed_directory(workspace, path)
                raise FileExistsError(f"Run directory is non-empty: {path}")
        else:
            workspace.mkdir(mode=0o700)

        try:
            _rename_no_replace(claim_recovery, path)
        except OSError as exc:
            if existed and not path.exists():
                cls._restore_unclaimed_directory(workspace, path)
            if path.exists():
                raise FileExistsError(f"Run directory is non-empty: {path}") from None
            raise exc

        return cls(
            path=path,
            workspace=workspace,
            claim_recovery=claim_recovery,
            marker_stat=marker_stat,
            token=token,
        )

    @staticmethod
    def _restore_unclaimed_directory(source: Path, destination: Path) -> None:
        """Restore a moved directory only when the destination is still absent."""

        try:
            _rename_no_replace(source, destination)
        except OSError:
            # Preserve both the recovery tree and anything that won the destination race.
            pass

    def _owned_marker(self, marker: Path) -> bool:
        try:
            return (
                os.path.samestat(self.marker_stat, marker.stat())
                and marker.read_bytes() == self.token
            )
        except OSError:
            return False

    def _retire_claim(self) -> bool:
        """Move the complete claim away atomically, then validate what was moved."""

        if self.claim_retired:
            return self._claim_is_clean()
        try:
            _rename_no_replace(self.path, self.claim_recovery)
        except FileNotFoundError:
            return False
        except OSError as exc:
            if self.path.exists():
                raise _publication_error(self.path) from exc
            raise
        self.claim_retired = True
        return self._claim_is_clean()

    def _claim_is_clean(self) -> bool:
        try:
            entries = list(self.claim_recovery.iterdir())
        except OSError:
            return False
        marker = self.claim_recovery / _OWNERSHIP_MARKER
        return entries == [marker] and self._owned_marker(marker)

    def _restore_foreign_entries(self) -> None:
        """Return foreign entries to the destination without deleting the owned claim."""

        try:
            entries = list(self.claim_recovery.iterdir())
        except OSError:
            return
        foreign = [
            entry
            for entry in entries
            if entry.name != _OWNERSHIP_MARKER or not self._owned_marker(entry)
        ]
        if not foreign:
            return
        try:
            self.path.mkdir()
        except FileExistsError:
            return
        except OSError:
            return
        for entry in foreign:
            target = self.path / entry.name
            if target.exists():
                continue
            try:
                _rename_no_replace(entry, target)
            except OSError:
                # The entry remains recoverable in claim_recovery.
                continue

    def publish(self) -> None:
        """Atomically replace the clean claim with the complete private workspace."""

        if self.published:
            return
        if not self._retire_claim():
            self._restore_foreign_entries()
            raise _publication_error(self.path)
        try:
            _rename_no_replace(self.workspace, self.path)
        except OSError as exc:
            if self.path.exists():
                raise _publication_error(self.path) from exc
            raise
        self.published = True

    def release(self) -> None:
        """Retire an unpublished claim without unlinking any filesystem path."""

        if self.published or self.claim_retired:
            return
        if not self._retire_claim():
            self._restore_foreign_entries()


__all__ = ["RunDirectoryOwnership"]
