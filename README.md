# Multi-Agent Driving

MetaDrive上の1台の自車内部に、複数の決定論的な専門Agentと、速度判断を統合するCoordinatorを配置する研究用MVPです。複数車両を学習対象にするMARLではありません。

最上位要件は [`docs/multi_agent_driving_mvp_spec.md`](docs/multi_agent_driving_mvp_spec.md) です。

## Architecture map

実装済みのPhase 1〜6と、正式評価までに残る境界を1枚で確認できます。

- [`architecture-map.html`](architecture-map.html): Flowと依存関係を探索する画面
- [`architecture-map.json`](architecture-map.json): 根拠とcoverage gapを含む機械可読データ

ローカルではrepository rootで`python -m http.server 4173`を実行し、
`http://localhost:4173/architecture-map.html`を開いてください。現在のsmoke成果物は
評価基盤の確認用であり、正式な5-policy-seed研究結果ではありません。

## Phase 6 evaluation workflow

Phase 6の評価パイプラインは実装済みです。実MetaDriveで固定比較表を走行し、
JSONL・CSV・PNG・GIF・Markdownレポートを、改変検知用manifest付きの新規directoryへ
まとめます。評価用RGB画像はheadless top-down rendererから保存するだけで、Policy入力には
使いません。Coordinator Observationは引き続きshape `(24,)`の`float32`です。

比較方式は次のとおりです。

| track | methods | Shield |
|---|---|---|
| decision | B1 Nominal、B2 Multi-no-review、Proposed | 全方式`monitor` |
| system | B0 Rule、B1、B2、Proposed | 全方式`enforce` |
| ablation | Proposed、no-Critic、no-Shield、no-Hazard | no-Shieldだけ`off`、他は`enforce` |

各方式は同じ5セル（Level 0 Nominal、Level 1 Lead Brake、Level 2 Lead Brake、
Level 2 Cut-in、Level 3 Occluded Crossing）で比較します。train、validation、testの
scenario rangeはそれぞれ`[0, 10000)`、`[10000, 11000)`、`[20000, 21000)`です。
test seedは学習、Curriculum進行、checkpoint選択には使いません。

正式評価では、各PPO候補を固定all-level validationへ通し、平均reward、collision rate、
success rate、route completion、学習step、SHA-256の順でcheckpointを選びます。
`--smoke`は配線確認なので、completed-run metadataとSHA-256で認証したfinal checkpointを
未選択のまま使い、`SMOKE - NOT A RESEARCH RESULT`と明記します。架空のvalidation scoreは
生成しません。

まず、smoke用の6方式をpolicy seed 42で短時間学習します。出力先は評価planに固定されて
いるため、既に存在する場合は別名へ変えず、不要な旧smokeを退避してから実行してください。

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

次の3コマンドで、実走行bundle、オフライン再集計、1 episodeのGIFを作成します。
どのコマンドも既存の出力先を上書きしません。

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

中心となる成果物は次の構成です。`compare`と`render_episode`は検証済みbundleだけを読み、
MetaDrive、Stable-Baselines3、TensorBoardを起動せずに再生成します。

```text
evaluations/phase6_smoke/
├─ evaluation_plan.yaml
├─ config_resolved.yaml
├─ evaluation_manifest.json
├─ model_selection.csv
├─ selected_checkpoints.json
├─ sources/<method>/<policy-seed>/tensorboard/...
├─ episodes/<method>/<track>/<policy-seed>/<case>/...
├─ metrics/
│  ├─ train_metrics.csv
│  ├─ eval_metrics.csv
│  └─ comparison.csv
├─ plots/
│  ├─ learning_curve.png
│  ├─ collision_rate.png
│  ├─ success_route_completion.png
│  ├─ unnecessary_braking.png
│  ├─ comfort.png
│  └─ agent_disagreement.png
├─ renders/proposed_42_level1_lead_brake_20000.gif
└─ comparison_report.md
```

正式な5 policy seed（42～46）比較は計算量の大きい次の実験として未実施です。
smoke結果を研究結果として扱ってはいけません。

## Phase 5 scenarios and curriculum

Phase 5 adds deterministic Lead Brake, Cut-in, and Occluded Crossing hazards plus
a validation-driven Levels 0--3 curriculum. The Coordinator contract is unchanged:
the Agent still receives exactly one 24-dimensional `float32` observation and
chooses from `KEEP=0`, `SLOW=1`, `PREPARE_STOP=2`, and `STOP=3`. Scenario identity,
scenario parameters, and seed identities are provenance only; they are never added
to the Agent observation.

Configuration is loaded as one base YAML followed by ordered recursive overlays.
The training CLI accepts repeatable `--overlay` arguments:

```powershell
# Fixed Level 1: Lead Brake
python -m mad_driving.cli.train --config configs/base.yaml --overlay configs/scenarios/lead_brake.yaml --smoke --run-dir runs/phase5_lead_brake_smoke

# Fixed Level 2: Cut-in
python -m mad_driving.cli.train --config configs/base.yaml --overlay configs/scenarios/cut_in.yaml --smoke --run-dir runs/phase5_cut_in_smoke

# Fixed Level 3: Occluded Crossing with its seeded secondary lead vehicle
python -m mad_driving.cli.train --config configs/base.yaml --overlay configs/scenarios/occluded_crossing.yaml --smoke --run-dir runs/phase5_occluded_crossing_smoke
```

The same merge is available from Python as
`load_config("configs/base.yaml", "configs/scenarios/lead_brake.yaml")`.
Overlay order is significant, mapping/scalar conflicts are rejected, and only the
fully resolved configuration is written to `config_resolved.yaml`.

Curriculum levels have a stable mapping:

- Level 0: `nominal`.
- Level 1: `lead_brake`.
- Level 2: either a concrete fixed `lead_brake`/`cut_in` selection, or `auto` for a
  uniform seeded choice between them. The dedicated Cut-in overlay remains concrete.
- Level 3: `occluded_crossing` plus its secondary lead vehicle.

`fixed` mode holds `fixed_level`. `automatic` mode starts at `initial_level` and
advances exactly one level only after the configured number of consecutive
scheduled validations satisfy both the success-rate and collision-rate thresholds.
Only validation episodes can update it; test episodes are rejected. Level changes
are queued in training and validation environments and activate on the next reset.
For example, an automatic overlay can contain:

```yaml
scenario_id: phase5
scenarios:
  selection: auto
  curriculum:
    mode: automatic
    initial_level: 0
    success_rate_threshold: 0.80
    collision_rate_threshold: 0.05
    maximum_unnecessary_stop_duration_s: 1.0
    consecutive_evaluations: 2
```

Every actual reset is appended and `fsync`ed to a schema-v4 per-worker JSONL file.
Each record has exactly `role`, `worker_index`, `environment_seed`,
`scenario_selection_seed`, `scenario_parameter_seed`, `scenario_id`,
`difficulty_level`, and recursively finite JSON-safe `scenario_parameters`.
`environment_seed` is the Gymnasium episode RNG identity;
`scenario_selection_seed` independently drives the Phase 5 scenario choice; and
`scenario_parameter_seed` independently drives concrete parameter sampling. The
MetaDrive road index is derived from a third independent child seed. Train
`[0, 10000)`, validation `[10000, 11000)`, and test
`[20000, 21000)` scenario identities remain disjoint. Test seeds are never used for
training, validation, checkpoint selection, or curriculum progression.

Automatic curriculum validation uses typed per-episode records. Safe STOP commands
longer than `maximum_unnecessary_stop_duration_s` do not count as successful, and
Level 2 validation alternates Lead Brake and Cut-in so each scenario must meet the
success threshold independently. Best reward comparison resets at level changes and
archives `best_model_level_<level>.zip`; Phase 6 performs final model selection on a
fixed all-level validation suite.

`curriculum_state.yaml` is atomically replaced with flush/`fsync` semantics. Every
periodic, best, and final `*.zip` also has an adjacent `*.zip.curriculum.yaml`
sidecar containing the exact curriculum state at that checkpoint's lifecycle point
and the checkpoint SHA-256. Research contract v7 inventories both checkpoint and
sidecar hashes in `run_metadata.json`. Resume resolves the sidecar bound to the
selected checkpoint, reads and hashes one immutable byte snapshot, rejects path
replacement races and malformed/duplicate-key data, and restores that exact state
after validating curriculum compatibility. It never substitutes the run-final state
for an earlier periodic or best checkpoint.

`--run-dir` remains available for an explicit fresh destination. When omitted, the
CLI atomically reserves a collision-free directory below configured
`training.run_root`, so the exact nominal smoke command is safe to run directly.

Useful commands:

```powershell
python -m mad_driving.cli.train --help
python -m mad_driving.cli.train --config configs/base.yaml --smoke
python -m mad_driving.cli.train --config configs/base.yaml --smoke --run-dir runs/phase5_nominal_smoke
python -m pytest tests/integration/test_phase5_metadrive_headless.py -m integration -q
```

## Current status

Phase 5 scenario generation, curriculum progression, provenance, and exact resume are
implemented. The Phase 4.2 control and PPO behavior described below remains preserved.

Phase 4.2（comparison-validity remediation）まで実装済みです。Phase 1の厳密な設定検証とMetaDrive境界、Phase 2の専門Agent、Phase 3の安全制御、Phase 4のGymnasium/PPO経路に加え、次を含みます。

- Nominal Agentによる等加速度運動予測
- Hazard Agentによる先行車急制動・横断Actor・遮蔽の最悪ケース評価
- Rule Agentによる衝突・路外・進入禁止・停止要求・速度制限の判定
- Critic Agentによる固定8規則の相互レビュー。Criticは4番目の専門Agentではありません
- Nominal / Hazard / Ruleの各専門Agentが1〜3件の`RiskClaim`を返し、1つの`CriticReview`が全Claimを検査
- `RuleBasedCoordinator`によるAgent提案の決定論的な統合
- `SafetyShield`によるTTC、停止余裕、Agent欠落、hard stopの最終判定
- anti-windup付きPIDによる車線追従と高レベル速度Actionの実車両入力への変換
- `SceneFrame(metadata + SceneObservation + PrivilegedWorldState) → AgentSuite → Coordinator → SafetyShield → MetaDrive`のheadless control smoke
- 複数Claimをfield-wiseで保守集約する有限な24次元`float32` Observationと、action-selection/transition境界を分けた10成分Reward
- `physics_dt_s=0.02`、`decision_repeat=5`、`decision_dt_s=0.10`の明示timing
- `ScenarioRuntime` lifecycle、role別seed identity、Gymnasium暗黙resetで進む再現可能なepisode seed列
- Agent可視構造から遮蔽Actorの運動学を除外し、全Actor truthから計算する固定oracle TTC、Rule制約、衝突・路外・到着・scenario outcomeをReward専用privileged stateへ分離
- 構成上無効なAgentと実行時失敗したAgentを区別するablation-aware Safety Shield
- CPU上の標準PPO学習、best/final/periodic checkpoint、resolved config、TensorBoard、strict resume provenance

速度Actionは安全側へ単調な`KEEP=0`、`SLOW=1`、`PREPARE_STOP=2`、`STOP=3`です。Safety Shieldには診断も介入もしない`off`、診断だけ行う`monitor`、要求より安全側のActionを強制する`enforce`があります。既定値は`enforce`です。

Phase 6では意思決定性能のB1・B2・Proposedをすべて`monitor`で比較し、実行可能システムは全方式を`enforce`で比較します。最終test seedsはmodel選択やCurriculumへ使いません。Rewardは比較対象AgentのClaimを入力にせず、全方式共通のprivileged oracle transitionだけから計算します。Rule violationとUnnecessary-brakeはpre-step oracle、Near-missはpost-step oracle TTCを使用し、将来lookaheadは使いません。

座標はMetaDrive world XYを使い、headingは反時計回りが正です。Actor相対座標は自車body frameで前方・左方が正、`lane_offset_m`はMetaDrive lane-local lateral signを保持します。`same_lane`はlane index一致だけでなくlane幅内の位置も要求します。

train、validation、testのscenario rangeはそれぞれ`[0, 10000)`、`[10000, 11000)`、`[20000, 21000)`です。正式training comparisonは`training.seed`をpolicy/RNG seed `42, 43, 44, 45, 46`へ設定した5本の独立runで行い、validation episode列を決めるroot `seed`は固定します。自動multi-seed sweepは実装していません。validationは`EvalCallback`とbest-checkpoint選択だけに使い、testはPhase 6の最終比較まで隔離します。

専用シナリオ生成とcurriculumはPhase 5、評価artifact・baseline・ablation・可視化はPhase 6です。`ttc_valid`、`claim_valid`、`agent_failed`、`target_actor_present`のObservation featureは未実装で、現在の24次元schemaには存在しません。追加時はschema versionを上げて再学習します。LLM/VLM、画像認識、学習による操舵、車線変更、実車接続はMVP対象外です。

## Setup

Python 3.11を使用します。Windowsの非ASCIIパスでも再現できるよう、通常wheelとして同期します。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install uv==0.8.0
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training --extra evaluation
```

### Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install uv==0.8.0
.venv/bin/uv sync --no-editable --group dev --extra training --extra evaluation
```

## Run

### Fixed-action smoke

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.smoke --config configs/base.yaml
```

Phase 1・2互換の経路です。画面を開かず、固定された低レベルActionで100 decision steps（10秒）を走行します。Agent解析は操作を変更しません。

標準出力のJSONには、`final_snapshot`、Nominal / Hazard / Ruleの順に並ぶ1〜3件/Agentの`final_claims`、1件の`final_review`、完了step数と終了状態が入ります。

### Shielded control smoke

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.control_smoke --config configs/base.yaml
```

Phase 3の経路です。毎decisionで3 AgentとCriticを実行し、Coordinatorの要求をSafety Shieldへ通した整数ActionだけをMetaDriveへ渡します。車線追従と加減速はカスタムPolicyが担当します。

JSONには最終Snapshot・Claims・Reviewに加え、`final_trace`、4 Actionの`action_counts`、実際にActionを書き換えた回数`shield_intervention_count`、完了step数と終了状態が入ります。seed 42のcanonical runは100 decision steps（10秒）を完走します。

初回だけMetaDrive 0.4.3の公式assetsをダウンロードします。

### PPO training

Windows PowerShell:

```powershell
# Headless CPU smoke (allocates a fresh directory under training.run_root)
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/base.yaml --smoke

# Standard 500,000-timestep run
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/base.yaml --run-dir runs/phase4_standard_seed42

# Resume into a new empty destination with parent provenance
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/base.yaml --run-dir runs/phase4_standard_seed43_continued --resume-from runs/phase4_standard_seed42/checkpoints/final_model.zip
```

Linux:

```bash
# Headless CPU smoke (allocates a fresh directory under training.run_root)
.venv/bin/python -m mad_driving.cli.train --config configs/base.yaml --smoke

# Standard 500,000-timestep run
.venv/bin/python -m mad_driving.cli.train --config configs/base.yaml --run-dir runs/phase4_standard_seed42

# Resume into a new empty destination with parent provenance
.venv/bin/python -m mad_driving.cli.train --config configs/base.yaml --run-dir runs/phase4_standard_seed43_continued --resume-from runs/phase4_standard_seed42/checkpoints/final_model.zip
```

`configs/base.yaml`がsmokeとtrainingの共通canonical configです。`metadrive.start_seed`と`metadrive.num_scenarios`はsmoke用の既定値で、PPO環境はroleごとに`scenarios.<role>`の範囲へ上書きします。

`--run-dir` is optional for new training and resume runs. If omitted, the CLI reserves
a unique empty destination under `training.run_root`; if supplied, it must still be
absent or empty and is never overwritten. Resume authenticates one immutable checkpoint
byte snapshot, restores the curriculum sidecar bound to that checkpoint, and records the
parent checkpoint/config/diff/start step in the new run. Current runs use
`research_contract_version=7` and `observation_schema_version=1`. Per-worker schema-v4
episode seed JSONL artifacts retain descriptor-bound identity, exact counts, and SHA-256
digests in `run_metadata.json`.

`training.num_envs=1`ではtrainをparent processの`DummyVecEnv`、validationを1 workerの`SubprocVecEnv`にします。`num_envs>1`ではtrainを`SubprocVecEnv`、validation worker 0をparent processの`DummyVecEnv`にします。この相補構成でMetaDrive engineを1 processに1つに保ちつつ、学習中にperiodic evaluationとbest-checkpoint選択を行います。各評価直前にvalidation VecEnvをAppConfigのroot `seed`で再seedし、比較するcheckpointごとに同じepisode-seed列を再生します。

標準Stable-Baselines3 PPOは`n_steps * num_envs`単位でrolloutを完了するため、実step数は要求値を超える場合があります。更新をエピソード境界へ同期しません。出力構造は次のとおりです。periodic checkpointは設定したintervalに到達した場合だけ生成されます。

```text
<run-dir>/
├── config_resolved.yaml
├── run_metadata.json              # research contract v7 and artifact digests
├── curriculum_state.yaml          # final run curriculum state
├── episode_seeds/
│   ├── train-worker-000.jsonl
│   └── validation-worker-000.jsonl
├── checkpoints/
│   ├── best_model.zip
│   ├── best_model.zip.curriculum.yaml
│   ├── final_model.zip
│   ├── final_model.zip.curriculum.yaml
│   ├── ppo_checkpoint_<steps>_steps.zip  # interval到達時のみ
│   └── ppo_checkpoint_<steps>_steps.zip.curriculum.yaml
└── tensorboard/
    └── PPO_<n>/
        └── events.out.tfevents.*
```

旧Task 11の再生成seed列と、path occupantをidentity trust sourceにした後続runは実run証拠として廃止しました。`runs/phase4_1_worker_identity_final_smoke_20260721_a`と`runs/phase4_1_worker_identity_final_smoke_20260721_b`は旧research contract v2、`runs/phase4_2_review_fix_v3_smoke_20260721_c`と`runs/phase4_2_review_fix_v3_smoke_20260721_d`は旧v3の履歴証拠です。Phase 4のhistorical v4 evidenceは`runs/phase4_2_oracle_v4_smoke_20260721_g`と`runs/phase4_2_oracle_v4_smoke_20260721_h`です。両runは5,000 step時点で同一の5-episode periodic validationを行い、その後6,144 stepまで学習しました。両final checkpointは6,144 stepで再読込でき、各100 decision stepsのObservationとRewardはすべて有限かつ同一でした。詳細は [`docs/phase4_implementation_log.md`](docs/phase4_implementation_log.md) にあります。

## Verify

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
```

Phase 1の実装判断は [`docs/implementation_plan.md`](docs/implementation_plan.md)、Phase 2の検証結果は [`docs/phase2_implementation_log.md`](docs/phase2_implementation_log.md)、Phase 3のMetaDrive API差異と実測結果は [`docs/phase3_implementation_log.md`](docs/phase3_implementation_log.md)、Phase 4の学習・checkpoint検証は [`docs/phase4_implementation_log.md`](docs/phase4_implementation_log.md) にあります。
