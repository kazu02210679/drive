# MAD Driving MVP

MetaDrive上の単一自車を、複数の決定論的Agent、PPO Coordinator、Safety Shieldで制御するMVPの仕様リポジトリです。現時点では実装前の仕様段階で、[`docs/multi_agent_driving_mvp_spec.md`](docs/multi_agent_driving_mvp_spec.md) が最上位要件です。

## Architecture map

仕様書から、予定モジュール、データ契約、意思決定フロー、実装フェーズを1枚にまとめています。

- [`architecture-map.html`](architecture-map.html): 人が探索するインタラクティブマップ
- [`architecture-map.json`](architecture-map.json): 次の実装Agentが利用できる機械可読コンテキスト

ローカルで開くには、リポジトリのルートで次を実行します。

```bash
python -m http.server 4173
```

その後、`http://localhost:4173/architecture-map.html` を開いてください。マップ上のノード選択、検索、意思決定・学習・評価フローの切り替え、ズーム、パンができます。ノードの色分けはマップ上部に表示されます。右ペインには説明付きの大きなFlow選択カードが4行1列で並びます。Flowを選ぶと対象経路へ自動で寄り、選択肢の下へ、目的、開始条件、結果、段階、裏側の処理、安全条件、仕様根拠をまとめたFlow inspectorが表示されます。選択後はFlow詳細の先頭まで自動スクロールし、上へ戻ると4つのFlow選択肢を再利用できます。

Flow inspectorは、[`joeyvansommeren/journey-mapper`](https://github.com/joeyvansommeren/journey-mapper) の「1つのシナリオを段階とservice blueprintで読む」考え方を、システム制御Flow向けに調整して参照しています。現リポジトリは仕様段階のため、実装コードや実測データが存在しない項目はcoverage noteで明示します。

JSONの参照整合性は標準ライブラリだけで検証できます。

```bash
python tools/validate_architecture_map.py
```

## Status semantics

`architecture-map.json` の各ノードは現在 `planned` です。仕様に記載された予定ソースパスを示しており、ファイルが実装済みであることを意味しません。実装後は、対応ノードの `status`、責務、入出力、パスをコードに合わせて更新してください。
