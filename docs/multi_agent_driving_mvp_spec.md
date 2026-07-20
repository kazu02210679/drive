# 車内マルチエージェント運転シミュレーション MVP 仕様書

## 0. Codexへの実装指示

この仕様書を最上位要件として扱い、記載のない機能を独断で追加しないこと。

実装前に、採用するMetaDrive・Gymnasium・Stable-Baselines3のAPI互換性を公式ドキュメントとインストール済みコードで確認すること。依存関係は再現可能な形で固定し、`uv.lock`または同等のロックファイルをコミットすること。

実装はテスト駆動で進める。各モジュールについて、正常系だけでなく、境界値、タイムアウト、NaN・無限値、不正なActor状態、乱数seedの再現性をテストすること。

MVPでは外部LLM API、VLM、自然言語による自由討論を使用しない。「Agent」は独立した認識・予測・検証ロジックを持つPythonモジュールを意味する。Agent間の通信は構造化データのみとする。

学習中も、1エピソードの途中ではモデルの重みを更新しない。走行データを一定ステップ収集した後、PPOのrollout単位でCoordinatorを更新する。

最初からステアリング学習、画像認識、経路探索、複数自車、実車接続を実装しない。MVPの学習対象は速度判断を統合するCoordinatorだけとする。

---

## 1. 目的

MetaDrive上の1台の自車に、複数の専門Agentを持つ判断系を実装する。

各Agentは同じScene Snapshotを異なる観点から独立に分析し、危険仮説、信頼度、根拠、推奨速度制約を提出する。Critic AgentがAgent間の矛盾や見落としを検査し、強化学習で訓練されたCoordinatorが最終的な速度行動を選択する。

最終行動は決定論的なSafety Shieldを通し、物理的に危険な行動は強制的に安全側へ上書きする。

### 検証する研究仮説

> 単一の予測器だけで速度を決める方式よりも、異なる根拠を持つ複数Agentの独立分析と相互検証を行う方式の方が、危険見逃しを減らしつつ、過剰な停止を抑えられる。

---

## 2. MVPの範囲

### 2.1 実装対象

- MetaDrive上の単一自車
- 手続き生成された仮想道路
- 周囲の交通車両と障害物
- Simulatorの正解状態から作成するScene Snapshot
- 車線追従用の固定・決定論的な横方向制御
- 4段階の速度判断
- 4つの専門Agent
- Agent間の矛盾検出
- PPOで学習するCoordinator
- Safety Shield
- 学習・評価・可視化・ログ出力
- 単一Agent方式、対話なし方式、提案方式の比較

### 2.2 非対象

- 実車、ROS、Autowareとの接続
- 生のカメラ画像、LiDAR点群からの物体認識
- LLM・VLM・外部API
- 自然言語によるAgent討論
- ステアリングの強化学習
- 車線変更、追い越し、右左折の学習
- クラウド経路探索、渋滞予測
- V2X通信
- 複数の学習対象車両
- エピソード途中のニューラルネットワーク更新
- 実車安全規格への適合証明

### 2.3 MVPシナリオ

1. **Lead Brake**  
   同一車線の前方車が、ランダムな時刻と減速度で急制動する。

2. **Cut-in**  
   隣接車線の車両が、自車前方へ割り込む。

3. **Occluded Crossing Actor**  
   遮蔽物の背後から、交差方向へ交通Actorが進入する。MVPでは手続き生成環境で制御しやすい車両または小型Actorを使用する。歩行者Actorは拡張フェーズで追加する。

各シナリオは、距離、速度、発生時刻、減速度、遮蔽位置をseedに基づいて変化させる。

---

## 3. 技術スタック

- Python 3.11
- MetaDrive
- Gymnasium API
- Stable-Baselines3 PPO
- PyTorch
- NumPy
- Pydantic v2 または標準dataclass
- PyYAML
- TensorBoard
- pandas
- matplotlib
- imageio
- pytest
- pytest-cov
- Ruff
- mypy
- uv

### 実行環境

- Windows 11およびUbuntu系Linuxを想定
- 学習はheadless実行を標準とする
- CPUのみでもsmoke testが完了すること
- GPUは任意
- RGB画像を学習入力に使用しない

---

## 4. システム構成

```text
MetaDrive Simulation
        │
        ▼
SceneSnapshotBuilder
        │
        ├───────────────┬────────────────┬───────────────┐
        ▼               ▼                ▼               ▼
NominalMotionAgent  HazardAgent      RuleAgent       CriticAgent
        │               │                │               ▲
        └───────────────┴────────────────┘               │
                        │                                 │
                        └──────── RiskClaim ──────────────┘
                                      │
                                      ▼
                           Coordinator Observation
                                      │
                                      ▼
                             PPO Coordinator
                                      │
                                      ▼
                      KEEP / SLOW / PREPARE_STOP / STOP
                                      │
                                      ▼
                              Safety Shield
                                      │
                                      ▼
                      LaneKeepingLongitudinalPolicy
                                      │
                                      ▼
                                MetaDrive.step
```

### 更新周期

- Simulator step: MetaDrive設定に従う
- Agent分析: 各decision stepで1回
- Coordinator推論: 各decision stepで1回
- PPO更新: rollout収集後のみ
- Agentは同一Snapshotを読み、互いの内部状態を直接変更しない

---

## 5. データモデル

データモデルは`src/mad_driving/interfaces/`に定義する。すべて型注釈を付け、JSON化できること。

### 5.1 ActorState

```python
@dataclass(frozen=True)
class ActorState:
    actor_id: str
    actor_type: Literal["vehicle", "crossing_actor", "obstacle"]
    position_xy_m: tuple[float, float]
    velocity_xy_mps: tuple[float, float]
    acceleration_xy_mps2: tuple[float, float]
    heading_rad: float
    length_m: float
    width_m: float
    relative_longitudinal_m: float
    relative_lateral_m: float
    same_lane: bool
    visible: bool
    occluded: bool
```

### 5.2 EgoState

```python
@dataclass(frozen=True)
class EgoState:
    position_xy_m: tuple[float, float]
    speed_mps: float
    acceleration_mps2: float
    heading_rad: float
    lane_offset_m: float
    route_progress: float
    speed_limit_mps: float
```

### 5.3 SceneSnapshot

```python
@dataclass(frozen=True)
class SceneSnapshot:
    step_index: int
    sim_time_s: float
    scenario_id: str
    seed: int
    ego: EgoState
    actors: tuple[ActorState, ...]
    stop_required: bool
    occlusion_present: bool
    distance_to_conflict_point_m: float | None
    previous_action: int
    previous_shield_intervention: bool
```

### 5.4 RiskClaim

```python
@dataclass(frozen=True)
class RiskClaim:
    claim_id: str
    agent_id: str
    event_type: str
    target_actor_id: str | None
    probability: float | None
    confidence: float
    severity: float
    time_horizon_s: float
    min_ttc_s: float | None
    stopping_margin_m: float | None
    recommended_max_speed_mps: float
    hard_stop_required: bool
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    valid_until_step: int
```

値域：

- `probability`: `None`または0.0–1.0
- `confidence`: 0.0–1.0
- `severity`: 0.0–1.0
- `recommended_max_speed_mps`: 0以上
- 非有限値は生成時に例外とする

### 5.5 CriticReview

```python
@dataclass(frozen=True)
class CriticReview:
    conflict_score: float
    unresolved_conflict: bool
    max_severity: float
    supported_agent_ids: tuple[str, ...]
    challenged_claim_ids: tuple[str, ...]
    reasons: tuple[str, ...]
```

### 5.6 DecisionTrace

```python
@dataclass(frozen=True)
class DecisionTrace:
    step_index: int
    raw_action: int
    executed_action: int
    target_speed_mps: float
    shield_intervened: bool
    shield_reasons: tuple[str, ...]
    claims: tuple[RiskClaim, ...]
    review: CriticReview
    reward_components: dict[str, float]
```

---

## 6. Agent仕様

Agentは共通インターフェースを実装する。

```python
class DrivingAgent(Protocol):
    agent_id: str

    def analyze(self, snapshot: SceneSnapshot) -> RiskClaim:
        ...
```

Agentは副作用を持たない。同じSnapshotを与えた場合、同じ設定とseedで同じRiskClaimを返すこと。

### 6.1 NominalMotionAgent

目的：観測済みActorの起こりやすい運動を予測する。

MVP実装：

- 5秒先まで予測
- 時間刻み0.25秒
- constant-velocityまたはconstant-acceleration model
- 同一車線の前方車、割り込み候補、交差Actorを評価
- 予測最小距離とTTCを算出
- TTCと相対速度から連続的なcollision probability heuristicを算出
- 最も危険なActorに対する1件のRiskClaimを返す

禁止事項：

- 遮蔽Actorを存在しないものとして補完しない
- Safety Shieldと同一の最悪ケース計算を複製しない

### 6.2 HazardAgent

目的：低確率でも重大な最悪ケースを評価する。

MVP実装：

- 前方車最大制動：設定値、初期値 `-8.0 m/s²`
- 交差Actor最大進入速度：設定値
- 反応遅れ：設定値、初期値 `0.5 s`
- 自車最大安全減速度：設定値、初期値 `-6.0 m/s²`
- worst-case TTCとstopping marginを算出
- 遮蔽が存在する場合、仮想Actorが遮蔽境界から出現するケースを評価
- stopping marginが負の場合、severityを高くし、推奨最高速度を低下させる

`stopping_margin_m`は次の意味とする。

```text
停止可能距離との差 = 利用可能距離 - 必要停止距離
正: 余裕あり
負: 現在条件では停止余裕なし
```

### 6.3 RuleAgent

目的：学習に任せるべきでない交通制約を提示する。

MVP実装：

- speed limit
- scenario-defined stop requirement
- collision後、off-road時のhard stop
- 交差点進入禁止条件

ルール違反が予測される場合は`hard_stop_required=True`または`recommended_max_speed_mps`を制限する。

RuleAgentは機械学習を使用しない。

### 6.4 CriticAgent

CriticAgentはRiskClaimを生成せず、Snapshotと他AgentのRiskClaimを受け取りCriticReviewを返す。

```python
class CriticAgent:
    def review(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
    ) -> CriticReview:
        ...
```

最低限、次の検査を実装する。

1. Nominal riskが低いがHazard stopping marginが負
2. 遮蔽ありだがNominalが低リスク判定
3. RuleAgentがhard stopを要求しているが他Agentが進行推奨
4. Agent間の推奨最高速度差が設定閾値を超える
5. claimの有効期限切れ
6. confidenceが低いのに断定的な推奨
7. evidenceが空
8. NaN・無限値・範囲外値

Criticは他Agentを再実行しない。MVPでは1回のCross Reviewだけとする。

---

## 7. Coordinator

### 7.1 学習対象

学習するのはCoordinatorのみ。各AgentとSafety Shieldは固定する。

### 7.2 Observation

CoordinatorのObservationは固定長24次元の`float32`ベクトルとする。

#### Ego: 6

1. normalized speed
2. normalized target/speed-limit speed
3. normalized acceleration
4. normalized lane offset
5. route progress
6. normalized speed limit

#### NominalMotionAgent: 4

7. normalized min TTC
8. collision probability
9. confidence
10. normalized recommended max speed

#### HazardAgent: 5

11. normalized worst TTC
12. normalized stopping margin
13. severity
14. confidence
15. normalized recommended max speed

#### RuleAgent: 3

16. normalized recommended max speed
17. hard stop required
18. predicted hard violation

#### Critic: 4

19. conflict score
20. unresolved conflict
21. supported agent ratio
22. max severity

#### History: 2

23. previous action normalized by 3
24. previous shield intervention

正規化後は原則`[-1, 1]`、確率・フラグは`[0, 1]`とする。TTCは上限値でclipする。欠損TTCは「危険なし」を表す上限値へ変換する。

### 7.3 Action Space

`gymnasium.spaces.Discrete(4)`とする。

| Action | 名称 | 目標速度 |
|---:|---|---|
| 0 | KEEP | speed limit |
| 1 | SLOW | `min(current speed, 0.60 × speed limit)` |
| 2 | PREPARE_STOP | `min(current speed, 0.25 × speed limit)` |
| 3 | STOP | `0 m/s` |

目標速度は即時適用せず、低レベル速度PIDで加減速度上限を守って追従する。

### 7.4 PPO初期設定

初期値として以下をYAMLに置き、コードへ直書きしない。

```yaml
algorithm: PPO
policy: MlpPolicy
learning_rate: 0.0003
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
seed: 42
```

Smoke trainingは`5_000` timesteps、標準trainingは`500_000` timestepsを初期値とする。

複数環境並列化は設定で切り替える。Windowsで問題がある場合に単一環境へフォールバックできること。

---

## 8. LaneKeepingLongitudinalPolicy

目的：横方向は決定論的に車線追従し、Coordinatorの4段階Actionを縦方向制御へ変換する。

要件：

- MetaDriveのPolicy APIを使用する
- route/navigation情報を用いたlane keeping steering
- target speedに対するPID throttle/brake
- steering、throttle、brakeをMetaDriveの正規化Actionへ変換
- 最大加速、通常減速、緊急減速を設定可能にする
- Controllerの状態はエピソードreset時に初期化する
- 速度振動を防ぐためanti-windupを実装する

Coordinatorはsteering値を直接出力しない。

---

## 9. Safety Shield

Safety Shieldは学習モデルとは独立した決定論的モジュールとする。

```python
class SafetyShield:
    def filter(
        self,
        requested_action: int,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
    ) -> ShieldResult:
        ...
```

### 原則

- 実行Actionを同じか、より安全側へだけ変更できる
- Action indexは`KEEP < SLOW < PREPARE_STOP < STOP`の安全順序とする
- STOPをKEEPへ戻すような緩和は禁止
- intervention理由を必ず記録する

### 強制介入条件

- RuleAgentの`hard_stop_required=True`
- stopping marginが設定閾値未満
- imminent collisionが予測される
- off-roadまたは衝突後
- SnapshotまたはClaimに非有限値、不整合がある
- Agent処理失敗・タイムアウトで安全性を判定できない

### 動作モード

- `off`: 比較用。介入しない
- `monitor`: 介入候補を記録するがActionを変更しない
- `enforce`: Actionを変更する

標準評価は`enforce`。学習は設定可能とし、初期値は`enforce`＋介入ペナルティとする。

---

## 10. 環境Wrapper

`MultiAgentSpeedEnv(gymnasium.Env)`を実装する。

### reset

1. scenarioとseedを決定
2. MetaDriveをreset
3. ScenarioManagerがActor条件を設定
4. Agent、Critic、PID、ログ状態をreset
5. SceneSnapshotを作成
6. RiskClaimsとCriticReviewを生成
7. 24次元Observationを返す

### step

1. PPO CoordinatorからActionを受け取る
2. Safety ShieldでActionをfilter
3. 低レベルPolicyへtarget speedを設定
4. MetaDriveを1 decision step進める
5. 新Snapshotを作成
6. Agent分析とCriticReviewを実行
7. rewardを計算
8. terminated/truncatedを判定
9. DecisionTraceを記録
10. 新Observation、reward、terminated、truncated、infoを返す

### Gymnasium準拠

```python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
```

`gymnasium.utils.env_checker.check_env()`を通すこと。

---

## 11. Reward設計

Rewardは各成分を`info["reward_components"]`へ出力する。

```text
reward =
    progress_reward
  + arrival_reward
  - collision_penalty
  - near_miss_penalty
  - offroad_penalty
  - rule_violation_penalty
  - jerk_penalty
  - unnecessary_brake_penalty
  - standstill_penalty
  - shield_intervention_penalty
```

初期値：

```yaml
reward:
  progress_per_meter: 0.10
  arrival: 100.0
  collision_vehicle: 200.0
  collision_crossing_actor: 500.0
  near_miss_max: 50.0
  offroad: 100.0
  hard_rule_violation: 100.0
  jerk_scale: 0.05
  unnecessary_brake_scale: 0.20
  standstill_per_second: 0.50
  shield_intervention: 2.0
```

### Near-miss

TTCに応じた連続ペナルティとする。閾値を跨いだ瞬間だけの不連続な報酬にしない。

### Unnecessary brake

以下をすべて満たす場合に発生する。

- ActionがSLOW以上
- HazardAgent severityが低い
- RuleAgentに制約なし
- 近傍ActorとのTTCに十分な余裕
- 直後の数stepで危険イベントが発生しない

将来情報を直接Observationへ入れてはならない。Reward計算だけに使用する。

### 終了条件

`terminated=True`：

- 目的地到達
- 衝突
- off-road
- scenario-defined failure

`truncated=True`：

- 最大step数到達
- シミュレーター異常

---

## 12. ScenarioManager

シナリオ生成とtrain/eval分離を担当する。

### 共通要件

- seedで完全再現可能
- train seedsとeval seedsを重複させない
- Actor生成条件をJSONLへ記録
- difficulty levelを持つ

### Curriculum

#### Level 0

- 交通密度が低い
- 危険イベントなし、または遠距離
- lane keepingと速度維持の確認

#### Level 1

- Lead Brakeのみ
- 十分な初期間隔
- 低い減速度

#### Level 2

- 強い急制動
- Cut-in
- 短いTTC

#### Level 3

- 遮蔽Crossing Actor
- 複数危険候補
- Agent間で意見が割れやすい条件

Curriculumの進行は、直近評価のsuccess rateとcollision rateに基づく。進行条件はYAMLで設定する。自動進行を無効化してLevel固定でも実行可能にする。

---

## 13. 比較実験

以下のモードを同じeval seedsで比較する。

### B0: Rule Baseline

- 簡易TTCルールのみ
- PPOなし
- 複数Agentなし

### B1: Single Agent

- NominalMotionAgentのみ
- PPO Coordinator
- Criticなし

### B2: Multi-Agent No Review

- Nominal、Hazard、Rule
- PPO Coordinator
- Critic出力をObservationに含めない

### P: Proposed

- Nominal、Hazard、Rule、Critic
- PPO Coordinator
- Safety Shield

### Ablation

- Proposed without Critic
- Proposed without Shield
- Proposed without Hazard Agent

比較結果をCSVとMarkdownレポートへ出力する。

---

## 14. 評価指標

### 安全性

- collision rate
- crossing-actor collision rate
- near-miss rate
- minimum TTC
- negative stopping-margin rate
- rule violation rate
- shield intervention rate

### 走行効率

- success rate
- route completion
- average speed
- travel time
- unnecessary braking count
- unnecessary stop duration

### 快適性

- longitudinal acceleration RMS
- maximum deceleration
- jerk RMS

### マルチエージェント固有

- Agent disagreement rate
- Critic challenge rate
- Criticが危険見逃しを発見した割合
- Critic false challenge rate
- Agent failure時のfallback回数
- Decision latency p50/p95/p99

### 学習

- episode reward
- reward component推移
- policy entropy
- value loss
- explained variance

---

## 15. ログと可視化

### ファイル出力

```text
runs/<run_id>/
├─ config_resolved.yaml
├─ metadata.json
├─ checkpoints/
├─ tensorboard/
├─ episodes/
│  ├─ episode_<id>_trace.jsonl
│  └─ episode_<id>_summary.json
├─ metrics/
│  ├─ train_metrics.csv
│  ├─ eval_metrics.csv
│  └─ comparison.csv
├─ plots/
│  ├─ learning_curve.png
│  ├─ collision_rate.png
│  ├─ unnecessary_braking.png
│  └─ agent_disagreement.png
└─ renders/
   └─ episode_<id>.gif
```

### GIF Overlay

評価GIFには最低限、次を表示する。

- scenario、seed、step
- 自車速度、speed limit、target speed
- 各Agentのseverityと推奨最高速度
- Critic conflict score
- requested action
- executed action
- Safety Shield intervention
- 累積reward

### Decision Trace

各stepで全RiskClaim、CriticReview、requested/executed action、reward componentsをJSONLへ保存する。

---

## 16. CLI

最低限、次のCLIを実装する。

```bash
# 環境のsmoke test
python -m mad_driving.cli.smoke --config configs/base.yaml

# 学習
python -m mad_driving.cli.train --config configs/train.yaml

# 評価
python -m mad_driving.cli.evaluate \
  --config configs/eval.yaml \
  --checkpoint runs/<run_id>/checkpoints/best_model.zip

# 1エピソードを可視化
python -m mad_driving.cli.render_episode \
  --config configs/eval.yaml \
  --checkpoint runs/<run_id>/checkpoints/best_model.zip \
  --scenario lead_brake \
  --seed 10001

# 比較実験
python -m mad_driving.cli.compare \
  --config configs/compare.yaml
```

すべてのCLIは`--help`を持ち、不正なパス・設定値に明確なエラーを返すこと。

---

## 17. 設定ファイル

```text
configs/
├─ base.yaml
├─ train.yaml
├─ eval.yaml
├─ compare.yaml
├─ agents.yaml
├─ reward.yaml
├─ shield.yaml
└─ scenarios/
   ├─ lead_brake.yaml
   ├─ cut_in.yaml
   └─ occluded_crossing.yaml
```

設定は起動時に統合し、`config_resolved.yaml`として保存する。

未知の設定キーを黙って無視しない。型・範囲を検証し、異常時は起動を停止する。

---

## 18. リポジトリ構成

```text
multi-agent-driving/
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ configs/
├─ src/
│  └─ mad_driving/
│     ├─ __init__.py
│     ├─ interfaces/
│     │  ├─ actor_state.py
│     │  ├─ scene_snapshot.py
│     │  ├─ risk_claim.py
│     │  ├─ critic_review.py
│     │  └─ decision_trace.py
│     ├─ world_model/
│     │  ├─ snapshot_builder.py
│     │  ├─ normalization.py
│     │  └─ validation.py
│     ├─ agents/
│     │  ├─ base.py
│     │  ├─ nominal_motion.py
│     │  ├─ hazard.py
│     │  ├─ rule.py
│     │  └─ critic.py
│     ├─ coordination/
│     │  ├─ observation_builder.py
│     │  ├─ action_mapping.py
│     │  └─ fallback.py
│     ├─ safety/
│     │  ├─ shield.py
│     │  ├─ stopping_distance.py
│     │  └─ collision_checks.py
│     ├─ control/
│     │  ├─ lane_keeping_policy.py
│     │  └─ speed_pid.py
│     ├─ envs/
│     │  ├─ multi_agent_speed_env.py
│     │  ├─ reward.py
│     │  └─ registration.py
│     ├─ scenarios/
│     │  ├─ base.py
│     │  ├─ manager.py
│     │  ├─ lead_brake.py
│     │  ├─ cut_in.py
│     │  └─ occluded_crossing.py
│     ├─ training/
│     │  ├─ train.py
│     │  ├─ callbacks.py
│     │  └─ curriculum.py
│     ├─ evaluation/
│     │  ├─ evaluate.py
│     │  ├─ metrics.py
│     │  ├─ compare.py
│     │  └─ report.py
│     ├─ visualization/
│     │  ├─ overlay.py
│     │  ├─ render_gif.py
│     │  └─ plots.py
│     ├─ logging/
│     │  ├─ trace_writer.py
│     │  └─ run_directory.py
│     ├─ config/
│     │  ├─ models.py
│     │  └─ loader.py
│     └─ cli/
│        ├─ smoke.py
│        ├─ train.py
│        ├─ evaluate.py
│        ├─ render_episode.py
│        └─ compare.py
└─ tests/
   ├─ unit/
   ├─ integration/
   └─ smoke/
```

1ファイルへ複数責務を詰め込まない。各Agent、Reward、Shield、Observation Builderを独立してテスト可能にする。

---

## 19. テスト要件

### Unit Test

- stopping distance計算
- TTC計算
- RiskClaim値域検証
- Observationが24次元で有限値
- Action mapping
- Safety Shieldの単調性
- Criticの各conflict rule
- Reward各成分
- seed再現性
- PID resetとanti-windup

### Property Test相当

外部ライブラリ追加は必須ではないが、ループによるランダム入力で次を検証する。

- 距離が短くなるほどHazard推奨速度が上がらない
- severityが高くなるほどSafety Shieldが危険側へ緩和しない
- 同一入力から異なる出力が発生しない
- ObservationにNaN・infが入らない

### Integration Test

- Gymnasium env checker通過
- reset→100 step→closeが完了
- collisionでterminated
- horizonでtruncated
- checkpoint保存・読込
- eval metrics生成
- trace JSONLがschema準拠

### Smoke Test

- headlessで5,000 timesteps学習が完了
- best modelが保存される
- 10 episode評価が完了
- GIFが1本生成される
- Windows/Linuxでパス処理に依存しない

### CI

- Ruff
- mypy
- pytest
- coverage
- headless smoke testは実行時間が長い場合、通常CIとnightlyを分離する

目標coverageはコアロジック80%以上。MetaDrive内部コードをcoverage対象に含めない。

---

## 20. エラー処理とFallback

- Agentが例外を出した場合、そのAgentのClaimを無効として記録し、Coordinatorへ安全側の欠損値を渡す
- 複数Agentが失敗した場合はPREPARE_STOPまたはSTOP
- ObservationにNaN・infがある場合はSTOP
- MetaDriveが例外を出した場合、episodeをtruncatedとして保存し、次episodeへ進む
- 学習checkpointは一定間隔で保存し、中断後に再開可能にする
- 設定不正は起動時に停止する
- ログ書込失敗で運転判断を停止させない。ただしstderrへ明示する

---

## 21. 実装フェーズ

### Phase 1: 基盤

- pyproject、設定loader、interfaces
- MetaDrive環境の起動・終了
- SceneSnapshot生成
- 固定Actionで走行
- headless smoke test

### Phase 2: 決定論Agent

- NominalMotionAgent
- HazardAgent
- RuleAgent
- CriticAgent
- Decision Trace

### Phase 3: 制御とShield

- lane keeping
- speed PID
- 4 Action mapping
- Safety Shield
- ルールベースCoordinatorでend-to-end確認

### Phase 4: RL環境

- 24次元Observation
- Reward
- Gymnasium checker
- PPO training
- checkpoint・TensorBoard

### Phase 5: シナリオ

- Lead Brake
- Cut-in
- Occluded Crossing Actor
- Curriculum
- train/eval seed分離

### Phase 6: 評価

- baselines・ablations
- metrics
- plots
- GIF overlay
- Markdown comparison report

各Phase終了時にテストを実行し、独立したcommitを作ること。

---

## 22. 受け入れ基準

### 必須

1. `uv sync`相当で環境を再現できる
2. headless環境でsmoke commandが成功する
3. Gymnasium APIに準拠する
4. Observationが常に24次元で有限値
5. Action spaceがDiscrete(4)
6. 4つのAgentとSafety Shieldが独立モジュールとして存在する
7. Coordinatorだけが学習対象である
8. 同一seedでscenario初期条件とAgent出力が再現する
9. 5,000 timestepsのsmoke trainingが完了する
10. checkpointを読み込み、評価できる
11. JSONL trace、CSV metrics、PNG plots、GIFが生成される
12. B0、B1、B2、Proposedを同じeval seedsで比較できる
13. Safety Shieldが危険側へActionを変更しないことをテストで保証する
14. 全unit/integration testsが通る
15. READMEにセットアップ、学習、評価、可視化のコマンドを記載する

### 実験上の成功目標

以下は機能完成条件ではなく、提案方式の有効性判断に使用する。

- ProposedがSingle Agentよりcollision rateを低下させる
- Proposedのunnecessary stop durationがSingle Agentの2倍以内
- Criticあり方式がCriticなし方式よりnegative stopping-margin rateを低下させる
- Decision latency p95が設定したdecision periodを超えない

目標未達の場合も結果を隠さず、比較レポートに原因候補と失敗シナリオを記載する。

---

## 23. READMEに記載する内容

- 目的と研究仮説
- 「複数車両のMARL」ではなく「1台内部の複数判断Agent」であること
- MVPではLLMを使わない理由
- Architecture図
- インストール
- smoke test
- training
- evaluation
- rendering
- config説明
- baselines・ablations
- 出力ディレクトリ
- 既知の制約
- 次段階の拡張候補

---

## 24. 次段階の拡張候補

MVP完成後にのみ検討する。

1. 手続き生成環境への歩行者・自転車Policy追加
2. カメラ・LiDAR観測と認識誤差
3. Agentごとに異なるセンサー入力
4. Learned Motion Predictor
5. Criticの学習
6. 複数Roundの限定的Cross Review
7. 車線変更Action
8. Waymax・実データシナリオでの評価
9. CARLAへの移植
10. VLMによる臨時標識・工事員などの意味理解

自然言語Agentを追加する場合も、出力は必ずRiskClaim形式へ変換し、Safety Shieldを迂回させない。

---

## 25. Codexが最初に行う作業

1. 空のリポジトリの場合は、上記構成の最小骨格を作る
2. MetaDriveを最小コードで起動し、headless reset/step/closeを確認する
3. 動作確認済みの依存バージョンを固定する
4. `docs/implementation_plan.md`にPhase単位の実装計画を作成する
5. Phase 1のテストから実装する
6. 各Phaseでテスト結果と変更点をREADMEまたは作業ログへ残す

仕様に矛盾またはMetaDrive API上の実装不能点が見つかった場合、勝手に大幅変更せず、問題、原因、最小の代替案を文書化してから修正すること
