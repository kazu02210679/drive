# Multi-Agent Driving

MetaDrive上の1台の自車内部に、複数の決定論的な専門Agentと、速度判断を統合するCoordinatorを配置する研究用MVPです。複数車両を学習対象にするMARLではありません。

最上位要件は [`docs/multi_agent_driving_mvp_spec.md`](docs/multi_agent_driving_mvp_spec.md) です。

## Current status

Phase 1（基盤）を実装済みです。厳密な設定検証、型付きインターフェース、MetaDrive状態からの`SceneSnapshot`生成、固定Actionによるheadless smoke走行を含みます。

Nominal / Hazard / Rule / Critic、Coordinator、Safety Shield、24次元Observation、PPO、3つのシナリオはPhase 2以降で実装します。LLM/VLM、画像認識、学習による操舵、車線変更、実車接続はMVP対象外です。

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

初回だけMetaDrive 0.4.3の公式assetsをダウンロードします。`configs/base.yaml`では画面を開かず、固定Actionで100 decision stepsを走行し、最終SnapshotをJSON出力します。

## Verify

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
```

Phase 1の実装判断と検証記録は [`docs/implementation_plan.md`](docs/implementation_plan.md) にあります。
