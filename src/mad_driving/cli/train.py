"""Train or resume the configured PPO speed policy."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mad_driving.config.loader import load_config
from mad_driving.training.train import require_empty_run_directory, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Ordered YAML overlay applied after --config; repeat for multiple overlays",
    )
    parser.add_argument("--smoke", action="store_true", help="Train for smoke_timesteps")
    parser.add_argument(
        "--run-dir",
        help="Fresh artifact directory; defaults to a unique directory under training.run_root",
    )
    parser.add_argument("--resume-from", help="Existing PPO checkpoint to resume")
    return parser


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _run_directory_name(*, smoke: bool) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    purpose = "smoke" if smoke else "train"
    return f"phase5-{purpose}-{timestamp}-{secrets.token_hex(4)}"


@dataclass(frozen=True)
class _ImplicitRunDirectoryReservation:
    path: Path
    device: int
    inode: int
    created_ns: int


def _fresh_run_directory(
    run_root: Path,
    *,
    smoke: bool,
) -> _ImplicitRunDirectoryReservation:
    """Atomically reserve a collision-free directory beneath the configured root."""

    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = root / _run_directory_name(smoke=smoke)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        identity = candidate.stat()
        return _ImplicitRunDirectoryReservation(
            path=candidate,
            device=int(identity.st_dev),
            inode=int(identity.st_ino),
            created_ns=int(identity.st_ctime_ns),
        )


def _cleanup_implicit_run_directory(
    reservation: _ImplicitRunDirectoryReservation,
) -> bool:
    """Remove only the unchanged, still-empty directory reserved by this process."""

    try:
        current = reservation.path.stat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or int(current.st_dev) != reservation.device
            or int(current.st_ino) != reservation.inode
            or int(current.st_ctime_ns) != reservation.created_ns
            or any(reservation.path.iterdir())
        ):
            return False
        confirmed = reservation.path.stat()
        if not os.path.samestat(current, confirmed):
            return False
        reservation.path.rmdir()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise, traceback-free operational errors."""

    args = _parser().parse_args(argv)
    implicit_reservation: _ImplicitRunDirectoryReservation | None = None
    try:
        config_path = Path(args.config)
        _require_file(config_path, "Configuration file")
        overlay_paths = tuple(Path(path) for path in args.overlay)
        for overlay_path in overlay_paths:
            _require_file(overlay_path, "Configuration overlay")
        resume_from = None if args.resume_from is None else Path(args.resume_from)
        if resume_from is not None:
            _require_file(resume_from, "Resume checkpoint")

        config = load_config(config_path, *overlay_paths)
        if args.run_dir is not None:
            run_dir = Path(args.run_dir)
        else:
            implicit_reservation = _fresh_run_directory(
                Path(config.training.run_root),
                smoke=args.smoke,
            )
            run_dir = implicit_reservation.path
        require_empty_run_directory(run_dir)

        result = run_training(
            config,
            smoke=args.smoke,
            run_dir=run_dir,
            resume_from=resume_from,
        )
        output = {
            "run_dir": str(result.run_dir),
            "final_checkpoint": str(result.final_checkpoint),
            "best_checkpoint": str(result.best_checkpoint),
            "timesteps": result.timesteps,
        }
    except Exception as exc:
        if implicit_reservation is not None:
            _cleanup_implicit_run_directory(implicit_reservation)
        print(f"training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
