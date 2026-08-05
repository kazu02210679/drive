# Phase 6 Implementation Log

## Scope and research boundary

Phase 6 implements the reproducible evaluation infrastructure for the seven approved
method profiles and the decision, system, and ablation tracks. It preserves observation
schema v1 with shape `(24,)`, keeps Reward on method-independent privileged oracle state,
distinguishes intentional ablation from runtime Agent failure, and retains the Safety
Shield as the final enforcement authority where the selected track enables it.

The completed runs documented here are infrastructure smoke runs. They are not research
results. The formal benchmark over policy seeds `42, 43, 44, 45, 46`, fixed-validation
checkpoint selection, and independent test seeds has not been run.

## Environment and commands

The verified environment uses Python 3.11, MetaDrive 0.4.3, Gymnasium 1.3.0,
Stable-Baselines3 2.9.0, and the locked dependencies in `uv.lock`. Training and evaluation
dependencies are installed together:

```powershell
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training --extra evaluation
```

The six PPO smoke profiles were trained independently with policy seed 42 and fresh run
directories:

```powershell
$methods = @(
  "b1_nominal",
  "b2_multi_no_review",
  "proposed",
  "proposed_no_critic",
  "proposed_no_shield",
  "proposed_no_hazard"
)
foreach ($method in $methods) {
  .venv\Scripts\python.exe -m mad_driving.cli.train `
    --config configs/base.yaml `
    --overlay "configs/methods/$method.yaml" `
    --smoke `
    --run-dir "runs/phase6_smoke/${method}_seed42"
}
```

Each PPO run completed 6,144 actual transitions at the full rollout boundary. The
user-facing evaluation, offline comparison, and representative rendering commands were:

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.evaluate `
  --plan configs/evaluation/phase6_smoke.yaml `
  --output evaluations/phase6_smoke `
  --smoke

.venv\Scripts\python.exe -m mad_driving.cli.compare `
  --evaluation evaluations/phase6_smoke `
  --output evaluations/phase6_smoke_comparison

.venv\Scripts\python.exe -m mad_driving.cli.render_episode `
  --evaluation evaluations/phase6_smoke `
  --episode-key proposed_system_42_level1_lead_brake_20000 `
  --output evaluations/phase6_smoke_render
```

These commands use exclusive destinations and do not overwrite an existing run or
publish incomplete staging directories.

## Smoke artifacts inspected

The real MetaDrive smoke evaluation completed 55 episodes. The three Git-ignored local
output directories were:

```text
evaluations/phase6_smoke
evaluations/phase6_smoke_comparison
evaluations/phase6_smoke_render
```

Inspection covered four CSV files, 114 JSON/JSONL files, 44 PNG files, and two 32-frame
GIFs (the bundle render and its offline regenerated copy).
The bundle included the resolved plan/config, checkpoint-selection provenance, episode
records, reduced metrics, six fixed plots, representative frames/GIF, Markdown report,
and a SHA-256 manifest. PNG and GIF files decoded with finite non-empty dimensions. The
wall-clock duration of the historical six training commands and 55-episode smoke run was
not retained; this log does not invent elapsed times for them.

Every smoke artifact and report is labelled `SMOKE - NOT A RESEARCH RESULT`. These
ignored local outputs are reproducibility evidence for the pipeline only and must not be
reported as a scientific comparison.

## Final verification evidence

After the horizon-rounding regression fix, the local full gate was:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check --no-cache .
.venv\Scripts\ruff.exe format --check --no-cache .
.venv\Scripts\mypy.exe --cache-dir C:\tmp\drive-phase6-horizon-mypy src
```

The final documentation-remediation rerun completed in 292.05 seconds with 1,696 passed,
29 warnings, and 90.56% branch-aware
coverage. Ruff and format passed, and mypy reported no issues in 90 source files.

GitHub Actions run [`30969252711`](https://github.com/kazu02210679/drive/actions/runs/30969252711)
verified commit
`383d373114581c41be805dbcf5ce3524a098c0a9`. The pytest step completed in 250.98
seconds with 1,696 passed, 30 warnings, and 90.79% coverage; the complete quality job
finished in 5 minutes 41 seconds. Ruff, format, and mypy also passed.

## Known warnings and environment notes

- Stable-Baselines3 warns that the training and evaluation VecEnv concrete types differ
  and that the evaluation environment is not wrapped in `Monitor`; the project consumes
  its own typed evaluation records and does not suppress these warnings.
- On the original Windows checkout under a Japanese user path, reinstalling editable mode
  can write a mojibake path into `_editable_impl_mad_driving.pth`. Child-process tests may
  then fail to import the package. Using an ASCII-path worktree is preferred; setting
  `PYTHONPATH` to the resolved `src` directory is a limited local workaround.
- Earlier Phase 6 CI attempts lacked the evaluation extra and later duplicated source and
  site-packages coverage. Commits `955775d` and `fe6f88d` corrected those CI-only issues.

## Deferred formal gate

Formal execution remains fail-closed until fixed-validation `selection_scores` are
generated and connected to the evaluation CLI/orchestration. The selection types and
bundle injection boundary exist, but the formal five-seed benchmark has not been
executed. The next stage must connect this path with TDD, select checkpoints using only
the fixed validation matrix, and then evaluate on the independent test seeds.

## Architecture-map follow-up

The separate remote branch `feat-architecture-map-flow-cards` contains the Phase 6 flow
card update at commit `ef70118055e271a64c75f4ebf78445b595e2b792`. It remains separate
from the production-code PR; how that commit is integrated into `main` is still pending.
