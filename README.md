# Multi-Agent Driving

MetaDrive上の1台の自車内部に、複数の決定論的な専門Agentと、速度判断を統合するCoordinatorを配置する研究用MVPです。複数車両を学習対象にするMARLではありません。

最上位要件は [`docs/multi_agent_driving_mvp_spec.md`](docs/multi_agent_driving_mvp_spec.md) です。

## Current status

Phase 4.1（research-validity hardening）まで実装済みです。Phase 1の厳密な設定検証とMetaDrive境界、Phase 2の専門Agent、Phase 3の安全制御、Phase 4のGymnasium/PPO経路に加え、次を含みます。

- Nominal Agentによる等加速度運動予測
- Hazard Agentによる先行車急制動・横断Actor・遮蔽の最悪ケース評価
- Rule Agentによる衝突・路外・進入禁止・停止要求・速度制限の判定
- Critic Agentによる固定8規則の相互レビュー。Criticは4番目の専門Agentではありません
- Nominal / Hazard / Ruleの各専門Agentが1〜3件の`RiskClaim`を返し、1つの`CriticReview`が全Claimを検査
- `RuleBasedCoordinator`によるAgent提案の決定論的な統合
- `SafetyShield`によるTTC、停止余裕、Agent欠落、hard stopの最終判定
- anti-windup付きPIDによる車線追従と高レベル速度Actionの実車両入力への変換
- `SceneFrame(SceneObservation + PrivilegedWorldState) → AgentSuite → Coordinator → SafetyShield → MetaDrive`のheadless control smoke
- 複数Claimをfield-wiseで保守集約する有限な24次元`float32` Observationと、現在状態だけを使う10成分Reward
- `physics_dt_s=0.02`、`decision_repeat=5`、`decision_dt_s=0.10`の明示timing
- `ScenarioRuntime` lifecycle、role別seed identity、Gymnasium暗黙resetで進む再現可能なepisode seed列
- Agent可視構造から遮蔽Actorの運動学を除外し、衝突・路外・到着・scenario outcomeをReward専用privileged stateへ分離
- CPU上の標準PPO学習、best/final/periodic checkpoint、resolved config、TensorBoard、strict resume provenance

速度Actionは安全側へ単調な`KEEP=0`、`SLOW=1`、`PREPARE_STOP=2`、`STOP=3`です。Safety Shieldには診断も介入もしない`off`、診断だけ行う`monitor`、要求より安全側のActionを強制する`enforce`があります。既定値は`enforce`です。

Phase 6では意思決定性能のB1・B2・Proposedをすべて`monitor`で比較し、実行可能システムは全方式を`enforce`で比較します。最終test seedsはmodel選択やCurriculumへ使いません。Unnecessary-brake Rewardは現在のpost-step状態だけで即時判定し、将来lookaheadを使いません。

座標はMetaDrive world XYを使い、headingは反時計回りが正です。Actor相対座標は自車body frameで前方・左方が正、`lane_offset_m`はMetaDrive lane-local lateral signを保持します。`same_lane`はlane index一致だけでなくlane幅内の位置も要求します。

train、validation、testのscenario rangeはそれぞれ`[0, 10000)`、`[10000, 11000)`、`[20000, 21000)`です。正式training comparisonはpolicy/RNG seed `42, 43, 44, 45, 46`を使います。validationは`EvalCallback`とbest-checkpoint選択だけに使い、testはPhase 6の最終比較まで隔離します。

専用シナリオ生成とcurriculumはPhase 5、評価artifact・baseline・ablation・可視化はPhase 6です。`ttc_valid`、`claim_valid`、`agent_failed`、`target_actor_present`のObservation featureは未実装で、現在の24次元schemaには存在しません。追加時はschema versionを上げて再学習します。LLM/VLM、画像認識、学習による操舵、車線変更、実車接続はMVP対象外です。

## Setup

Python 3.11を使用します。Windowsの非ASCIIパスでも再現できるよう、通常wheelとして同期します。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install uv==0.8.0
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training
```

### Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install uv==0.8.0
.venv/bin/uv sync --no-editable --group dev --extra training
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
# Headless CPU smoke (replace <UNIQUE_RUN_ID>; destination must be fresh)
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42_<UNIQUE_RUN_ID>

# Standard 500,000-timestep run
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed42

# Resume into a new empty destination with parent provenance
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed43_continued --resume-from runs/phase4_standard_seed42/checkpoints/final_model.zip
```

Linux:

```bash
# Headless CPU smoke (replace <UNIQUE_RUN_ID>; destination must be fresh)
.venv/bin/python -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42_<UNIQUE_RUN_ID>

# Standard 500,000-timestep run
.venv/bin/python -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed42

# Resume into a new empty destination with parent provenance
.venv/bin/python -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed43_continued --resume-from runs/phase4_standard_seed42/checkpoints/final_model.zip
```

新規学習とresumeは、存在しない、または空の`--run-dir`だけを受け付けます。`<UNIQUE_RUN_ID>`は毎回新しい識別子へ置換してください。非空directoryを上書きしません。Resume sourceはread-onlyとして扱い、checkpoint SHA-256、親run/config、config差分、開始step、Observation/Action schemaを新しいrunの`run_metadata.json`へ記録します。全runは`research_contract_version=2`、`observation_schema_version=1`です。各train/validation環境の実際のreset情報は`episode_seeds/<role>-worker-<index>.jsonl`へ耐久書き込みし、metadataの`episode_seed_artifacts`が相対path、role、worker、件数、schema version、SHA-256を示します。

標準Stable-Baselines3 PPOは`n_steps * num_envs`単位でrolloutを完了するため、実step数は要求値を超える場合があります。更新をエピソード境界へ同期しません。出力構造は次のとおりです。periodic checkpointは設定したintervalに到達した場合だけ生成されます。

```text
<run-dir>/
├── config_resolved.yaml
├── run_metadata.json
├── episode_seeds/
│   ├── train-worker-000.jsonl
│   └── validation-worker-000.jsonl
├── checkpoints/
│   ├── best_model.zip
│   ├── final_model.zip
│   └── ppo_checkpoint_<steps>_steps.zip  # interval到達時のみ
└── tensorboard/
    └── PPO_<n>/
        └── events.out.tfevents.*
```

`runs/phase4_1_smoke_seed42_a`と`runs/phase4_1_smoke_seed42_b`の旧Task 11 seed列は学習後に再生成したため、実run証拠としては廃止しました。後続のfresh smokeは`runs/phase4_1_seed_artifact_smoke_20260721_a`と`runs/phase4_1_seed_artifact_smoke_20260721_b`です。両runは5,000 requested / 6,144 actual stepsを完了し、実reset JSONLの全31 train recordsと全6 validation recordsが一致しました。両final checkpointは6,144 stepで再読込でき、各100 decision stepsのObservationとRewardは全て有限でした。詳細は [`docs/phase4_implementation_log.md`](docs/phase4_implementation_log.md) にあります。

## Verify

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
```

Phase 1の実装判断は [`docs/implementation_plan.md`](docs/implementation_plan.md)、Phase 2の検証結果は [`docs/phase2_implementation_log.md`](docs/phase2_implementation_log.md)、Phase 3のMetaDrive API差異と実測結果は [`docs/phase3_implementation_log.md`](docs/phase3_implementation_log.md)、Phase 4の学習・checkpoint検証は [`docs/phase4_implementation_log.md`](docs/phase4_implementation_log.md) にあります。
