# 車内マルチエージェント運転シミュレーション MVP 仕様書

## 0. Codexへの実装指示

この仕様書を最上位要件として扱い、記載のない機能を独断で追加しないこと。

実装前に、採用するMetaDrive・Gymnasium・Stable-Baselines3のAPI互換性を公式ドキュメントとインストール済みコードで確認すること。依存関係は再現可能な形で固定し、`uv.lock`または同等のロックファイルをコミットすること。

実装はテスト駆動で進める。各モジュールについて、正常系だけでなく、境界値、タイムアウト、NaN・無限値、不正なActor状態、乱数seedの再現性をテストすること。

MVPでは外部LLM API、VLM、自然言語による自由討論を使用しない。「Agent」は独立した認識・予測・検証ロジックを持つPythonモジュールを意味する。Agent間の通信は構造化データのみとする。

学習にはStable-Baselines3の標準PPOを使用する。`n_steps * num_envs`件のtransitionを収集するたびにCoordinatorを更新し、更新時点をエピソード境界へ同期しない。

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
- 3つの専門Agentと1つのCritic
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
- 実車安全規格への適合証明

### 2.3 MVPシナリオ

1. **Lead Brake**  
   同一車線の前方車が、ランダムな時刻と減速度で急制動する。

2. **Cut-in**  
   隣接車線の車両が、自車前方へ割り込む。

3. **Occluded Crossing Actor**  
   遮蔽物の背後から、交差方向へ交通Actorが進入する。MVPでは手続き生成環境で制御しやすい車両または小型Actorを使用する。歩行者Actorは拡張フェーズで追加する。

各シナリオは、距離、速度、発生時刻、減速度、遮蔽位置をseedに基づいて変化させる。Phase 4.1は`ScenarioRuntime`境界とseed分離だけを実装し、これら3種類の専用ActorとCurriculumはPhase 5へ残す。

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
        ▼
SceneFrame = scenario metadata + SceneObservation + PrivilegedWorldState
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

- physics step: `physics_dt_s=0.02 s`
- decision repeat: `5`
- decision step: `decision_dt_s=0.10 s`。`decision_dt_s == physics_dt_s * decision_repeat`を起動時とruntime境界で検証する
- Agent分析: 各decision stepで1回
- Coordinator推論: 各decision stepで1回
- PPO更新: 標準Stable-Baselines3 PPOに従い、`n_steps * num_envs`のrollout収集後に実行する。エピソード境界へ同期しない
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

### 5.3 SceneObservation、PrivilegedWorldState、SceneFrame

```python
@dataclass(frozen=True)
class SceneObservation:
    step_index: int
    sim_time_s: float
    ego: EgoState
    visible_actors: tuple[ActorState, ...]
    occlusion_regions: tuple[OcclusionRegion, ...]
    road_context: RoadContext
    previous_executed_action: int
    previous_shield_intervention: bool


@dataclass(frozen=True)
class PrivilegedWorldState:
    all_actors: tuple[ActorState, ...]
    collision_occurred: bool
    collision_kind: Literal[
        "vehicle", "crossing_actor", "object", "sidewalk", "building"
    ] | None
    off_road: bool
    arrived: bool
    scenario_success: bool
    scenario_failure: bool


@dataclass(frozen=True)
class SceneFrame:
    scenario_id: str
    seeds: EpisodeSeeds
    observation: SceneObservation
    privileged: PrivilegedWorldState
```

Nominal、Hazard、Rule、Critic、Coordinator、Safety Shieldが受け取れるのは`SceneObservation`だけとする。`scenario_id`と`EpisodeSeeds`はAgent入力ではなく`SceneFrame`の運用metadataとする。遮蔽されたActorは`visible_actors`へ入れず、`visible=False`のActorを運動学情報付きでAgent可視構造へ残してはならない。`PrivilegedWorldState.all_actors`では可視Actorを`visible=True, occluded=False`、非可視Actorを`visible=False, occluded=True`としてtruth上の可視性を保持する。`PrivilegedWorldState`はReward、評価、debugログだけが使用できる。

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
    required_action: int
    executed_action: int
    target_speed_mps: float
    intervention_required: bool
    shield_intervened: bool
    shield_reasons: tuple[str, ...]
    control_fail_safe: bool
    control_fail_safe_reason: str | None
    claims: tuple[RiskClaim, ...]
    review: CriticReview
    reward_components: dict[str, float]
    failed_agent_ids: tuple[str, ...]
    errors: tuple[str, ...]
    episode_rng_seed: int
    metadrive_scenario_index: int
    scenario_parameter_seed: int
    role: Literal["train", "validation", "test"]
    worker_index: int
```

### 5.7 座標系

- `position_xy_m`と`velocity_xy_mps`はMetaDrive world XY座標を使う
- `heading_rad`はworld headingを`[-pi, pi)`へ正規化し、反時計回りを正とする
- `relative_longitudinal_m`と`relative_lateral_m`は自車body frameを使い、前方と左方を正とする
- `lane_offset_m`は現在laneのlocal座標とMetaDriveのlateral signを保持する
- `same_lane`はcanonical lane indexの一致とlane幅内のActor位置を両方要求する
- scenarioのconflict pointは各`ScenarioRuntime`がgeometryから計算し、自車path上の距離をm単位で渡す

---

## 6. Agent仕様

Agentは共通インターフェースを実装する。

```python
class DrivingAgent(Protocol):
    agent_id: str

    def analyze(self, observation: SceneObservation) -> tuple[RiskClaim, ...]:
        ...
```

Agentは副作用を持たない。同じObservationを与えた場合、同じ設定とseedで同じ1〜3件のRiskClaimを同じ安全順序で返すこと。1つの専門Agentが失敗しても他のClaimを保持し、失敗したAgent IDとsanitize済みerrorをCritic、Trace、Shieldへ渡す。

### 6.1 NominalMotionAgent

目的：観測済みActorの起こりやすい運動を予測する。

MVP実装：

- 5秒先まで予測
- 時間刻み0.25秒
- constant-velocityまたはconstant-acceleration model
- 同一車線の前方車、割り込み候補、交差Actorを評価
- 予測最小距離とTTCを算出
- TTCと相対速度から連続的なcollision probability heuristicを算出
- hard stop、severity、有限TTC、停止余裕、推奨速度、Actor IDの安全順で最大3件のRiskClaimを返す

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
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        *,
        failed_agent_ids: Sequence[str],
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

各専門Agentの複数Claimは、既存slot位置を変えずにfield-wiseで安全側へ集約する。有限TTCと停止余裕は最小、severityとprobabilityは最大、推奨最高速度とconfidenceは最小、hard stopは論理和を使う。正規化後は原則`[-1, 1]`、確率・フラグは`[0, 1]`とする。TTCは上限値でclipする。

Observationは引き続きshape `(24,)`、dtype `numpy.float32`、有限、`[-1, 1]`内とする。`ttc_valid`、`claim_valid`、`agent_failed`、`target_actor_present`の明示featureは未実装であり、この24次元へ存在しない。これらはObservation schema versionを上げて再学習する別Phaseで追加する。

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

正式比較では`training.seed`をpolicy/RNG seed `42, 43, 44, 45, 46`へ設定した5本の独立runを使用し、validation episode列を決めるAppConfigのroot `seed`は固定する。自動multi-seed sweepは要件としない。PPOは標準Stable-Baselines3実装を使い、`n_steps * num_envs`のrollout境界で更新する。エピソード途中にrollout境界が来ることを禁止せず、独自collectorを追加しない。

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
        observation: SceneObservation,
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

学習は設定可能とし、初期値は`enforce`＋介入ペナルティとする。比較実験では、意思決定性能比較のB1・B2・Proposedをすべて`monitor`にそろえ、実行可能システム比較の全方式を`enforce`にそろえる。`Proposed without Shield`は主baselineではなくablationとして扱う。

`monitor`でも`required_action`と`intervention_required`を`info`と`DecisionTrace`へ必ず記録する。Shieldへは構成上存在する`expected_agent_ids`と、そのうち実行時に失敗した`failed_agent_ids`を別々に渡す。expectedに含まれないAgentは意図的ablationとして欠落数へ含めず、expectedなのにClaimがないAgentだけへ安全floorを適用する。低レベル制御がfail-safeへ入った場合は内部故障としてresourceをcloseし、通常のAction遷移をPPO bufferへ返さない。

---

## 10. 環境Wrapper

`MultiAgentSpeedEnv(gymnasium.Env)`を実装する。

### reset

1. `episode_rng_seed`からrole-scoped allocatorで`metadrive_scenario_index`と`scenario_parameter_seed`を決定する
2. `ScenarioRuntime.reset`を呼ぶ
3. 選択したMetaDrive scenario indexでSimulatorをresetする
4. `ScenarioRuntime.after_simulator_reset`を呼ぶ
5. `SceneFrame`を作成し、Agent可視ObservationだけからRiskClaimsとCriticReviewを生成する
6. 24次元Coordinator Observationを返す

### step

1. PPO CoordinatorからActionを受け取る
2. Safety ShieldでActionをfilter
3. 低レベルPolicyへtarget speedを設定
4. `ScenarioRuntime.before_step`を呼び、MetaDriveを1 decision step進め、authoritative decision intervalを再検証してから`ScenarioRuntime.after_step`を呼ぶ
5. `after_step`が返す`ScenarioTransition(state, outcome)`のstateを次のlifecycleへ引き継ぐ。observation context取得後にdecision intervalを再検証し、直後に新SceneFrameを作成する。不一致時はsnapshot、Agent分析、Reward、Trace、next stateへ進まずcloseして元の例外を送出する
6. Agent可視ObservationからAgent分析とCriticReviewを実行
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

自然なMDP結果はGymnasium semanticsに従う。typed privileged stateの衝突、off-road、到着、scenario success/failureだけから`terminated=True`を導出し、設定horizonのraw truncationは`truncated=True`として保持する。Simulatorのraw terminationがtrueなら、少なくとも1つのtyped privileged termination outcomeとの一致を必須とし、不一致は内部consistency errorとする。typed outcomeがあるvalidな場合だけboth-trueを許す。Simulator、ScenarioRuntime、snapshot、reward、observationなどの内部errorは所有resourceをcloseして元の例外を送出し、zero Observationや偽のtruncationへ変換しない。

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

### 比較方式に依存しないReward oracle

Reward APIはAgent Claim、CriticReview、Agent失敗状態を入力に取らない。全方式共通の`PrivilegedWorldState.minimum_actual_ttc_s`は、可視・遮蔽を問わず全Simulator truth Actorについて、現在速度を固定した自車座標系の矩形衝突包絡への最初の進入時刻として決定論的に計算する。`hard_rule_constraint`もprivileged stateへ固定する。Agent ClaimはObservation、Shield、診断だけに使用する。

### Near-miss

遷移後の`minimum_actual_ttc_s`に応じた連続ペナルティとする。閾値を跨いだ瞬間だけの不連続な報酬にしない。

### Unnecessary brake

以下をすべて満たす場合に発生する。

- ActionがSLOW以上
- Action選択時のoracle TTCが欠損して衝突course上のActorがいない、または最小TTCが安全閾値以上
- Action選択時のoracle Rule制約がない
- 遷移後の衝突、路外、Shield介入がない

Actionの妥当性はAction選択時のprevious privileged oracleで判定し、`-scale * executed_action`を即時に適用する。遷移後の衝突、路外、Shield介入は安全側のActionを誤罰しないための抑止条件としてだけ使う。将来lookahead、safe-brake streakを使用しない。「後から危険eventが発生しなかった」は評価metricであり、学習Reward入力ではない。`vehicle`、`object`、`sidewalk`、`building`への衝突はvehicle collision penalty、`crossing_actor`への衝突は専用のより大きいpenaltyを適用する。

### 終了条件

`terminated=True`：

- 目的地到達
- 衝突
- off-road
- scenario-defined failure

`truncated=True`は最大step数到達だけに使用する。内部errorはclose後に送出する。

---

## 12. ScenarioManager

Phase 4.1ではSimulator非依存の`ScenarioRuntime` lifecycleとrole別seed allocationを実装する。Lead Brake、Cut-in、Occluded CrossingのActor生成とCurriculumはPhase 5でこの境界へ接続する。

### 共通要件

- seedで完全再現可能
- `episode_rng_seed`、`metadrive_scenario_index`、`scenario_parameter_seed`を別identityとしてreset infoとDecisionTraceへ記録する。学習ではrole/worker別JSONLをexclusive createし、wrapper lifetime中は同じdescriptorだけでappend/fsyncする。headerのplatform file identityとpathのidentityを各append前後に照合する。VecEnv close後は各fileを1回だけopen/readし、同一byte列でstrict JSONL parse、件数、SHA-256を確定する。置換、history injection、parse/hash raceはfail closedとし、`run_metadata.json`の`episode_seed_artifacts`にfile identityも記録する
- `train=[0, 10000)`、`validation=[10000, 11000)`、`test=[20000, 21000)`を使い、空rangeと重複rangeを拒否する
- train workerはroleとworker indexを明示する。`num_envs=1`ではtrainをparent processの`DummyVecEnv`、validationを1 workerの`SubprocVecEnv`にする。`num_envs>1`ではtrainを`SubprocVecEnv`、validation worker 0をparent processの`DummyVecEnv`にする。この相補構成で学習中のperiodic evaluationとbest-checkpoint選択を必ず行う
- 各scheduled evaluation直前にvalidation VecEnvをAppConfigのroot `seed`で再seedし、すべてのcheckpoint比較とresumeで同一のvalidation episode-seed列を使う
- test rangeはPhase 6の最終比較専用とし、学習、checkpoint選択、Curriculum進行に使用しない
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

比較対象は同じ最終test scenario seedsを使う。意思決定性能と実行可能システム性能を混同しない。

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

### Shield比較契約

1. 意思決定性能比較ではB1、B2、Proposedをすべて`shield=monitor`にし、診断だけを記録してActionを変更しない
2. 実行可能システム比較では全方式を`shield=enforce`にし、collision rate、raw unsafe-request rate、intervention rate、post-Shield outcomeを報告する

`Proposed without Shield`はablationであり、主baselineではない。比較結果をCSVとMarkdownレポートへ出力する。

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
├─ run_metadata.json
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

新規学習は必須の`--run-dir`で、存在しない、または空のrun destinationを明示した場合だけ受け付ける。`training.run_root`へのgeneric fallback、非空directoryの再利用、overwrite optionは設けない。Resumeもsource runとは別の新しい空destinationへ書き、source checkpointのSHA-256、source hostで正規化したpath、親run/config、current configとの差分、開始`num_timesteps`、Observation/Action schemaを記録する。current resume sourceはcurrent hostでcanonicalize/dereferenceするが、既存metadata内のhistorical parent pathはcross-host provenance文字列として保持し、current hostのPath flavorで再解釈しない。全runは`research_contract_version=4`と`observation_schema_version=1`を持つ。旧contractのcheckpointを正式比較へ混在させない。

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
python -m mad_driving.cli.train \
  --config configs/base.yaml \
  --run-dir runs/phase4_seed42_<unique_run_id>

# 別runへresume
python -m mad_driving.cli.train \
  --config configs/base.yaml \
  --run-dir runs/<new_run_id> \
  --resume-from runs/<parent_run_id>/checkpoints/final_model.zip

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
- MetaDrive、ScenarioRuntime、snapshot、Reward、Observationの内部例外は所有resourceをcloseして元の例外を送出する。zero Observationや`truncated=True`へ変換しない
- primary exceptionがないcleanup/close failureはrun failureとして送出する。primary exceptionがある場合はcleanup failureをnoteへ付加してprimaryを保持する。train/validation closeとseed artifact finalizationが成功するまでrun destinationをsuccess artifactとしてpublishしない
- 学習checkpointは一定間隔で保存し、別の新規destinationへ厳密なprovenance付きで再開可能にする
- 設定不正は起動時に停止する
- best-effort運転ログの書込失敗はstderrへ明示する。研究provenanceであるseed artifact、metadata、checkpointの書込・identity検証失敗はfail closedとする

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

- 24次元のCoordinator Observation
- Reward
- Gymnasium checker
- PPO training
- checkpoint・TensorBoard

### Phase 4.1: Research-validity hardening

- 3専門Agentの1〜3 Claimと1 Critic、24-slot保守集約
- role別seed、`ScenarioRuntime`、Agent可視/privileged state分離
- action-selection/transition境界を分けたReward、内部errorのfail-fast、標準PPO rollout更新
- timing/座標契約、fresh destination、resume provenance version 3
- explicit validity featureは未実装のまま24次元を維持

### Phase 5: シナリオ

- Lead Brake
- Cut-in
- Occluded Crossing Actor
- Curriculum
- Phase 4.1の`ScenarioRuntime`へ専用Actorを接続

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
6. Nominal、Hazard、Ruleの3専門Agent、1 Critic、Safety Shieldが独立モジュールとして存在する
7. Coordinatorだけが学習対象である
8. 同一seedでscenario初期条件とAgent出力が再現する
9. 5,000 timestepsのsmoke trainingが完了する
10. checkpointを読み込み、評価できる
11. JSONL trace、CSV metrics、PNG plots、GIFが生成される
12. B0、B1、B2、Proposedを同じ最終test seedsで比較し、意思決定性能はall-monitor、実行可能システムはall-enforceにできる
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
