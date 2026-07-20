# Multi-Agent Driving

MetaDrive上の1台の自車内部に、複数の決定論的な専門Agentと、速度判断を統合するCoordinatorを配置する研究用MVPです。複数車両を学習対象にするMARLではありません。

最上位要件は [`docs/multi_agent_driving_mvp_spec.md`](docs/multi_agent_driving_mvp_spec.md) です。

## Current status

Phase 4（Gymnasium学習環境とPPO学習）まで実装済みです。Phase 1の厳密な設定検証とMetaDrive境界、Phase 2の専門Agent、Phase 3の安全制御に加え、次を含みます。

- Nominal Agentによる等加速度運動予測
- Hazard Agentによる先行車急制動・横断Actor・遮蔽の最悪ケース評価
- Rule Agentによる衝突・路外・進入禁止・停止要求・速度制限の判定
- Critic Agentによる固定8規則の相互レビュー
- 3つの`RiskClaim`と1つの`CriticReview`を毎decision stepで受動計算
- `RuleBasedCoordinator`によるAgent提案の決定論的な統合
- `SafetyShield`によるTTC、停止余裕、Agent欠落、hard stopの最終判定
- anti-windup付きPIDによる車線追従と高レベル速度Actionの実車両入力への変換
- `SceneSnapshot → AgentSuite → Coordinator → SafetyShield → MetaDrive`のheadless control smoke
- 有限な24次元`float32` Observationと10成分のReward
- Gymnasium準拠の`MultiAgentSpeedEnv`とseed再現性
- CPU上のPPO学習・再開、best/final/periodic checkpoint、resolved config、TensorBoard出力

速度Actionは安全側へ単調な`KEEP=0`、`SLOW=1`、`PREPARE_STOP=2`、`STOP=3`です。Safety Shieldには診断も介入もしない`off`、診断だけ行う`monitor`、要求より安全側のActionを強制する`enforce`があります。既定値は`enforce`です。

専用シナリオ生成とcurriculumはPhase 5、評価artifact・baseline・ablation・可視化はPhase 6です。LLM/VLM、画像認識、学習による操舵、車線変更、実車接続はMVP対象外です。

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

標準出力のJSONには、`final_snapshot`、Nominal / Hazard / Ruleの順に並ぶ3件の`final_claims`、1件の`final_review`、完了step数と終了状態が入ります。

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
# Canonical headless CPU smoke (5,000 requested timesteps)
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42

# Standard 500,000-timestep run
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed42

# Resume the same run transactionally from its canonical final checkpoint
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed42 --resume-from runs/phase4_standard_seed42/checkpoints/final_model.zip
```

Linux:

```bash
# Canonical headless CPU smoke (5,000 requested timesteps)
.venv/bin/python -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42

# Standard 500,000-timestep run
.venv/bin/python -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed42

# Resume the same run transactionally from its canonical final checkpoint
.venv/bin/python -m mad_driving.cli.train --config configs/train.yaml --run-dir runs/phase4_standard_seed42 --resume-from runs/phase4_standard_seed42/checkpoints/final_model.zip
```

`n_steps * num_envs`単位でrolloutを完了するため、実step数は要求値を超える場合があります。出力構造は次のとおりです。periodic checkpointは設定したintervalに到達した場合だけ生成されます。

```text
<run-dir>/
├── config_resolved.yaml
├── checkpoints/
│   ├── best_model.zip
│   ├── final_model.zip
│   └── ppo_checkpoint_<steps>_steps.zip  # interval到達時のみ
└── tensorboard/
    └── PPO_<n>/
        └── events.out.tfevents.*
```

Phase 4の最終実測`runs/phase4_smoke_seed42_postreview1`では、5,000 requested / 6,144 actual training stepsを37.1秒で実行し、1,000 evaluation steps（5 episodes × 200、全てhorizon truncation）、best step 5,000、final step 6,144を確認しました。`EvalCallback`はbest checkpoint選択に使いますが、Phase 6で定義する評価artifactは保存しません。final checkpointのfresh環境への再読込後、決定論的100 stepsでObservationとRewardが全て有限であることも確認しています。詳細は [`docs/phase4_implementation_log.md`](docs/phase4_implementation_log.md) にあります。

## Verify

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
```

Phase 1の実装判断は [`docs/implementation_plan.md`](docs/implementation_plan.md)、Phase 2の検証結果は [`docs/phase2_implementation_log.md`](docs/phase2_implementation_log.md)、Phase 3のMetaDrive API差異と実測結果は [`docs/phase3_implementation_log.md`](docs/phase3_implementation_log.md)、Phase 4の学習・checkpoint検証は [`docs/phase4_implementation_log.md`](docs/phase4_implementation_log.md) にあります。
