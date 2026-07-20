"""Train or resume the configured PPO speed policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mad_driving.config.loader import load_config
from mad_driving.training.train import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--smoke", action="store_true", help="Train for smoke_timesteps")
    parser.add_argument("--run-dir", help="Artifact directory (defaults to training.run_root)")
    parser.add_argument("--resume-from", help="Existing PPO checkpoint to resume")
    return parser


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise, traceback-free operational errors."""

    args = _parser().parse_args(argv)
    try:
        config_path = Path(args.config)
        _require_file(config_path, "Configuration file")
        resume_from = None if args.resume_from is None else Path(args.resume_from)
        if resume_from is not None:
            _require_file(resume_from, "Resume checkpoint")

        config = load_config(config_path)
        run_dir = Path(args.run_dir) if args.run_dir is not None else Path(config.training.run_root)
        if run_dir.exists() and not run_dir.is_dir():
            raise NotADirectoryError(f"Run directory is not a directory: {run_dir}")

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
        print(f"training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
