"""Atomic, cross-platform ownership of one training destination."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

_OWNERSHIP_MARKER = ".training-owner"


@dataclass
class RunDirectoryOwnership:
    """Exclusive marker held for the full training artifact transaction."""

    path: Path
    marker: Path
    descriptor: int
    token: bytes
    created_directory: bool
    _released: bool = False

    @classmethod
    def acquire(cls, path: Path) -> RunDirectoryOwnership:
        """Atomically claim an absent or currently empty destination."""

        path.parent.mkdir(parents=True, exist_ok=True)
        created_directory = False
        try:
            path.mkdir()
            created_directory = True
        except FileExistsError:
            if not path.is_dir():
                raise NotADirectoryError(f"Run directory is not a directory: {path}") from None
            if any(path.iterdir()):
                raise FileExistsError(f"Run directory is non-empty: {path}") from None

        marker = path / _OWNERSHIP_MARKER
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(marker, flags, 0o600)
        except BaseException as exc:
            cls._remove_empty_created_directory(path, created_directory)
            if isinstance(exc, FileExistsError):
                raise FileExistsError(f"Run directory is non-empty: {path}") from None
            raise

        ownership = cls(
            path=path,
            marker=marker,
            descriptor=descriptor,
            token=secrets.token_hex(32).encode("ascii"),
            created_directory=created_directory,
        )
        try:
            os.write(descriptor, ownership.token)
            os.fsync(descriptor)
            entries = list(path.iterdir())
            if entries != [marker]:
                raise FileExistsError(f"Run directory is non-empty: {path}")
        except BaseException as primary_error:
            try:
                ownership.release()
            except Exception as cleanup_error:
                primary_error.add_note(f"Ownership cleanup also failed: {cleanup_error}")
            raise
        return ownership

    def release(self) -> None:
        """Remove only the marker/directory still proven to belong to this owner."""

        if self._released:
            return
        self._released = True
        cleanup_errors: list[Exception] = []
        marker_owned = False
        marker_stat: os.stat_result | None = None
        try:
            marker_stat = os.fstat(self.descriptor)
            marker_owned = os.path.samestat(marker_stat, self.marker.stat())
        except (FileNotFoundError, OSError) as exc:
            cleanup_errors.append(exc)
            marker_owned = False
        try:
            os.close(self.descriptor)
        except OSError as exc:
            cleanup_errors.append(exc)
            marker_owned = False

        if marker_owned and marker_stat is not None:
            try:
                if os.path.samestat(marker_stat, self.marker.stat()):
                    self.marker.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(exc)

        self._remove_empty_created_directory(self.path, self.created_directory)
        if cleanup_errors:
            details = "; ".join(str(error) for error in cleanup_errors)
            raise RuntimeError(f"Run ownership cleanup failed: {details}") from cleanup_errors[0]

    @staticmethod
    def _remove_empty_created_directory(path: Path, created_directory: bool) -> None:
        if not created_directory:
            return
        try:
            path.rmdir()
        except OSError:
            # A non-empty directory may contain competitor or published artifact data.
            pass


__all__ = ["RunDirectoryOwnership"]
