# Phase 2 実装・検証記録

## 対象

Phase 2 は、Nominal / Hazard / Rule の決定論的な `RiskClaim` と、Critic
による1回の相互レビューを実装する。既存の固定操作によるMetaDrive走行へ
受動的に接続し、Agent出力は車両制御へ戻さない。

## MetaDrive 0.4.3 API確認

2026-07-20に、インストール済み
`.venv/Lib/site-packages/metadrive` を `rg` と読み取り専用の
`Get-Content` で確認した。

- `BaseVehicleState.init_state_info()` は `crash_vehicle`、`crash_human`、
  `crash_object`、`crash_sidewalk`、`crash_building` を真偽値として初期化する。
- `BaseVehicle` の衝突検出は、車両、建物、交通物体、人・自転車、境界・歩道・
  ガードレールとの接触を上記の各属性へ記録する。
- `NodeNetworkNavigation._update_current_lane()` はレイによる車線位置判定結果を
  `ego_vehicle.on_lane` へ設定する。
- `MetaDriveEnv._is_out_of_road()` の基準は `not vehicle.on_lane` であり、設定に
  応じて経路外、連続線、破線、歩道接触を追加する。本MVPの `off_road` は
  「走行可能な車線面の外」を表すため、安定した車両属性 `on_lane is False`
  だけを使用する。
- 汎用のPhase 2 smokeにはScenarioManagerがないため、
  `intersection_entry_prohibited` はbuilderの明示入力がない限り `False` のまま。

この確認に基づき、`collision_occurred` は5つの `crash_*` 属性の論理和、
`off_road` は `on_lane is False` とする。互換性のため、欠けた属性は
`getattr(..., False)` 相当で非発生扱いにする。車線幅が不明な場合に
`lane_offset_m` から路外を推測しない。

## 固定依存関係

- Python 3.11.9
- MetaDrive 0.4.3
- Gymnasium 1.3.0
- NumPy 1.26.4
- Pydantic 2.11.7
- pytest 8.4.1
- Ruff 0.12.4
- mypy 1.16.1

## 検証結果

2026-07-20に次を実行した。

- `pytest --cov=mad_driving --cov-report=term-missing -q`:
  **145 passed**、branchを含む総合coverage **96.23%**（必須80%以上）。
- 実MetaDrive integration: reset、1 step、snapshot、3 claim、1 review、closeが成功。
- `ruff check .`: 成功。
- `ruff format --check .`: 41 files formatted、差分なし。
- `mypy src`: strict設定で27 source files、問題なし。
- `git diff --check`: 問題なし。
- headless smoke: **100 decision steps / 10.0 simulated seconds**、
  `terminated=false`、`truncated=false`。画面を開かず、最終JSONへ
  Nominal / Hazard / Ruleの3 claimとCritic reviewを出力した。

全テストで14件の警告が出る。すべてMetaDriveが読み込むMatplotlib内部の
Pyparsing非推奨API（`oneOf`、`parseString`、`resetCache`、
`enablePackrat`）であり、本プロジェクトのコードからは発生していない。

## 不整合と最小対応

- 確認したMetaDrive 0.4.3 APIと設計の不整合はなかった。衝突状態は
  `crash_vehicle` / `crash_human` / `crash_object` / `crash_sidewalk` /
  `crash_building`、路外状態は`on_lane`から取得できた。
- `uv sync --no-editable`は、同一バージョン`0.1.0`の旧ローカルwheelを
  再利用したため、初回smokeでPhase 2の`agents`設定を認識しなかった。
  ソース変更や依存関係変更は行わず、
  `uv sync --no-editable --group dev --reinstall-package mad-driving`で現在の
  ブランチを明示的に再インストールした。その後の100-step smokeは成功した。
- Coordinator、Safety Shield、学習、シナリオ生成はPhase 3以降のため、
  Phase 2では実装していない。Agent解析結果は固定Actionへ接続していない。
