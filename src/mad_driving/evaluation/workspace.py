"""Atomic sibling staging for one online evaluation episode."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self


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
        self._fsync_files()
        self.path.replace(self.destination)
        return self.destination

    def _fsync_files(self) -> None:
        for path in sorted(candidate for candidate in self.path.rglob("*") if candidate.is_file()):
            with path.open("ab") as output:
                output.flush()
                os.fsync(output.fileno())


__all__ = ["EvaluationWorkspace"]
