"""Cross-platform atomic filesystem operations."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path
from typing import NoReturn

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4


def _raise_rename_error(error_number: int, destination: Path) -> NoReturn:
    raise OSError(error_number, os.strerror(error_number), destination)


def rename_no_replace(source: Path, destination: Path) -> None:
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


__all__ = ["rename_no_replace"]
