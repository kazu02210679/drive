# Multi-Agent Driving

MetaDrive上の1台の自車内部に、複数の決定論的な専門Agentと、速度判断を統合するCoordinatorを配置する研究用MVPです。複数車両を学習対象にするMARLではありません。

最上位要件は [`docs/multi_agent_driving_mvp_spec.md`](docs/multi_agent_driving_mvp_spec.md) です。

## Current status

Phase 2（決定論的Agent）まで実装済みです。Phase 1の厳密な設定検証、型付きインターフェース、MetaDrive状態からの`SceneSnapshot`生成、固定Actionによるheadless smoke走行に加え、次を含みます。

- Nominal Agentによる等加速度運動予測
- Hazard Agentによる先行車急制動・横断Actor・遮蔽の最悪ケース評価
- Rule Agentによる衝突・路外・進入禁止・停止要求・速度制限の判定
- Critic Agentによる固定8規則の相互レビュー
- 3つの`RiskClaim`と1つの`CriticReview`を毎decision stepで受動計算

Agentの出力はまだ車両制御へ戻しません。Coordinator、Safety Shield、4行動への変換、24次元Observation、PPO、急制動・割り込み・遮蔽横断のシナリオ生成、比較実験はPhase 3以降です。LLM/VLM、画像認識、学習による操舵、車線変更、実車接続はMVP対象外です。

## Setup

Python 3.11を使用します。Windowsの非ASCIIパスでも再現できるよう、通常wheelとして同期します。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install uv==0.8.0
.venv\Scripts\uv.exe sync --no-editable --group dev
```

### Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install uv==0.8.0
.venv/bin/uv sync --no-editable --group dev
```

## Run

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.smoke --config configs/base.yaml
```

初回だけMetaDrive 0.4.3の公式assetsをダウンロードします。`configs/base.yaml`では画面を開かず、固定Actionで100 decision steps（10秒）を走行します。Agent解析は操作を変更しません。

標準出力のJSONには、`final_snapshot`、Nominal / Hazard / Ruleの順に並ぶ3件の`final_claims`、1件の`final_review`、完了step数と終了状態が入ります。

## Verify

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
```

Phase 1の実装判断は [`docs/implementation_plan.md`](docs/implementation_plan.md)、Phase 2のMetaDrive API確認と検証結果は [`docs/phase2_implementation_log.md`](docs/phase2_implementation_log.md) にあります。
