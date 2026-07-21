# Phase 3 制御・Safety Shield設計

## Status and authority

> **Status: historical Phase 3 design.** This document preserves the original control/Shield delivery. Phase 4.1 supersedes its all-purpose snapshot boundary and defines the current comparison modes; the four-action order and monotone Shield remain active.

本設計は、`docs/multi_agent_driving_mvp_spec.md` のPhase 3「制御とShield」を
実装可能な粒度へ固定する。最上位要件はMVP仕様書であり、本設計はそこにない
機能を追加しない。

ユーザーは次の範囲を承認した。

- Phase 2の決定論的Agentを使う。
- RuleBasedCoordinator、SafetyShield、4 Action mapping、lane keeping、速度PIDを
  独立モジュールとして実装する。
- 既存の固定Action smokeは残し、新しいcontrol smokeでend-to-end制御を確認する。
- 24次元Observation、Reward、Gymnasium wrapper、PPOはPhase 4へ残す。

Phase 3は「判断が車両を動かす」最初の段階である。ただし、学習はまだ始めない。

## Goals

Phase 3の目的は次の五つである。

1. Phase 2のclaimsとreviewから、決定論的に4段階Actionを要求する。
2. SafetyShieldが要求Actionを同じか安全側にだけ変更する。
3. Actionを目標速度へ変換し、PIDでMetaDriveの正規化操作へ変える。
4. route laneを決定論的に追従し、Coordinatorから操舵を分離する。
5. 実MetaDriveで100 decision stepsのheadless制御走行を再現する。

## Non-goals

Phase 3では次を実装しない。

- 24次元Coordinator Observation
- `MultiAgentSpeedEnv(gymnasium.Env)`
- Reward計算
- PPO、checkpoint、TensorBoard
- ScenarioManagerと3つの危険シナリオ
- JSONL trace、CSV metrics、PNG、GIF
- curriculum、baseline、ablation、学習評価
- 車線変更、経路探索、画像認識、LLM/VLM、実車接続

`DecisionTrace`はメモリ上のcontrol smoke結果に使う。全stepの永続化は行わない。

## Architecture

責務は次の境界に分ける。

```text
SceneSnapshot
  -> AgentSuite
  -> claims + CriticReview
  -> RuleBasedCoordinator
  -> requested Action
  -> SafetyShield
  -> executed Action
  -> ActionMapper
  -> target speed
  -> LaneKeepingLongitudinalPolicy
  -> [steering, throttle/brake]
  -> MetaDrive
  -> next SceneSnapshot
```

追加する主要パッケージは次のとおりである。

```text
src/mad_driving/
├─ control/
│  ├─ actions.py
│  ├─ action_mapper.py
│  ├─ pid.py
│  └─ lane_keeping_policy.py
├─ coordinator/
│  └─ rule_based.py
├─ safety/
│  └─ shield.py
└─ cli/
   └─ control_smoke.py
```

既存の`AgentSuite`、`SceneSnapshotBuilder`、`RiskClaim`、`CriticReview`、
`DecisionTrace`を再利用する。CoordinatorとShieldはMetaDrive objectを受け取らない。
PolicyだけがMetaDriveの車両・navigation APIへ接続する。

## Configuration

`AppConfig`へ`coordinator`、`shield`、`control`を追加する。すべてstrictかつfrozenとし、
未知キー、非有限値、不正な範囲を起動時に拒否する。既存のPhase 1/2設定との互換性を
保つため、各セクションにはdefault factoryを持たせる。

### CoordinatorConfig

```yaml
coordinator:
  conflict_min_action: 1
  severe_min_action: 2
  severe_threshold: 0.75
```

- `conflict_min_action`は`CriticReview.unresolved_conflict=True`時の最低Actionである。
- `severe_min_action`は`max_severity >= severe_threshold`時の最低Actionである。
- Action値は0から3の範囲だけを許可する。

### ShieldConfig

```yaml
shield:
  mode: enforce
  imminent_ttc_s: 1.0
  caution_ttc_s: 3.0
  emergency_margin_m: 0.0
  caution_margin_m: 5.0
  missing_agent_action: 2
  multiple_missing_action: 3
```

設定検証は次の関係を保証する。

- `0 < imminent_ttc_s <= caution_ttc_s`
- `emergency_margin_m <= caution_margin_m`
- missing actionは0から3
- modeは`off`、`monitor`、`enforce`のいずれか

### ControlConfig

```yaml
control:
  speed:
    kp: 0.50
    ki: 0.05
    kd: 0.10
    integral_limit: 10.0
    max_acceleration_mps2: 2.5
    normal_deceleration_mps2: -3.0
    emergency_deceleration_mps2: -6.0
  steering:
    heading_kp: 1.7
    heading_ki: 0.01
    heading_kd: 3.5
    lateral_kp: 0.3
    lateral_ki: 0.002
    lateral_kd: 0.05
    integral_limit: 5.0
    lookahead_m: 1.0
```

加速度は正、減速度は負とする。検証は
`emergency_deceleration_mps2 <= normal_deceleration_mps2 < 0 < max_acceleration_mps2`
を保証する。PID gainは有限かつ非負、integral limitとlookaheadは正でなければならない。

## Action model

`DrivingAction`は安全順序を持つ`IntEnum`とする。

| Index | Name | Target speed |
|---:|---|---|
| 0 | KEEP | speed limit |
| 1 | SLOW | `min(current speed, 0.60 * speed limit)` |
| 2 | PREPARE_STOP | `min(current speed, 0.25 * speed limit)` |
| 3 | STOP | `0.0 m/s` |

`ActionMapper.target_speed(action, current_speed_mps, speed_limit_mps)`は純粋関数である。
入力を有限・非負として検証し、有限・非負の目標速度を返す。不正Actionは暗黙に丸めず
例外にする。Policy境界では、その例外をSTOPの最大brakeへ変換する。

Claimの推奨速度をActionへ変換するときは、速度制限に対する比率を使う。

- 推奨速度が0以下ならSTOP
- 推奨速度が速度制限の25%以下ならPREPARE_STOP
- 推奨速度が速度制限の60%以下ならSLOW
- それより高ければKEEP

速度制限が0の場合はSTOPとする。比較は境界を含む。

## RuleBasedCoordinator

`RuleBasedCoordinator.decide(snapshot, claims, review) -> DrivingAction`は、入力だけで
決まる純粋な決定を返す。状態、乱数、MetaDrive参照を持たない。

候補Actionを次の固定順で作り、最大値を返す。

1. 基本候補はKEEP。
2. 各valid claimの`recommended_max_speed_mps`をActionへ変換する。
3. `hard_stop_required=True`のclaimがあればSTOPを候補にする。
4. `review.unresolved_conflict=True`なら`conflict_min_action`を候補にする。
5. `review.max_severity >= severe_threshold`なら`severe_min_action`を候補にする。

CoordinatorはSafetyShieldのTTC・stopping margin判定を複製しない。Agentの提出した
制約とCriticの集約値をActionへ変換するだけである。

claimsが空、必要Agentが欠ける、または入力が壊れている場合、Coordinator単体は
PREPARE_STOPを返す。最終的なSTOP判断はShieldが行う。

## SafetyShield

### Interface

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

`ShieldResult`はfrozen dataclassとする。

```python
requested_action: DrivingAction
required_action: DrivingAction
executed_action: DrivingAction
intervention_required: bool
intervened: bool
reasons: tuple[str, ...]
```

`intervention_required`は`required_action > requested_action`を表す。
`intervened`は実行Actionが要求Actionと異なることを表す。monitor modeでは前者がtrue、
後者がfalseになり得る。

### Validation

Shieldは算術の前にsnapshotとclaimsを防御的に検証する。dataclass constructorの通常検証を
通った値でも、外部deserializationや`object.__setattr__`による破損を想定する。

- Snapshotのfloatはすべて有限で、速度は非負でなければならない。
- Claimのprobability、confidence、severityは既定範囲内でなければならない。
- TTCは`None`または有限・非負とする。
- stopping marginは`None`または有限とする。
- recommended speedは有限・非負とする。
- 必須Agent IDは`nominal`、`hazard`、`rule`である。

invalid inputは他の算術へ入れず、required actionをSTOPにする。

### Fixed rule order

理由は次の固定順で評価し、重複を除く。

1. `invalid_input`: snapshotまたはclaimが不正
2. `collision_occurred`: snapshotが衝突後
3. `off_road`: snapshotが走行可能lane外
4. `hard_stop_required`: いずれかのclaimがhard stop
5. `multiple_agents_missing`: 必須Agentが2つ以上欠損
6. `agent_missing`: 必須Agentが1つ欠損
7. `imminent_ttc`: 最小の有限TTCがimminent閾値以下
8. `negative_stopping_margin`: 最小marginがemergency閾値未満
9. `caution_ttc`: 最小の有限TTCがcaution閾値以下
10. `low_stopping_margin`: 最小marginがcaution閾値未満
11. `claim_speed_limit`: Claim推奨速度から変換したActionが要求Actionより安全側

必要Actionは次のように決める。

- 規則1から5、7、8はSTOP
- 規則6、9、10は少なくともPREPARE_STOPまたは設定されたmissing action
- 規則11はAction変換結果
- すべての候補の最大値を`required_action`とする

TTCの境界は`<=`、marginの境界は`<`を使う。marginがちょうど0.0 mなら
emergencyではなくcaution条件に入る。この差をテストで固定する。

### Modes and monotonicity

- `off`: required actionの診断を行わず、reasonsを空にして要求Actionを実行する。
- `monitor`: required actionとreasonsを計算するが、要求Actionを実行する。
- `enforce`: `max(requested_action, required_action)`を実行する。

enforce modeでは常に`executed_action >= requested_action`である。同じ要求Actionに対して
TTCが短くなる、marginが小さくなる、欠損Agentが増える、hard stopが追加される、severityが
高いClaimの速度制約が厳しくなる場合、実行Actionは小さいindexへ戻らない。

## PID controller

`BoundedPID`はMetaDrive非依存の状態付きクラスとする。入力はerrorと`dt_s`、出力は指定範囲の
有限floatである。

```text
derivative = (error - previous_error) / dt
candidate_integral = clip(integral + error * dt, -integral_limit, integral_limit)
raw = kp * error + ki * candidate_integral + kd * derivative
output = clip(raw, lower, upper)
```

anti-windupはconditional integrationで実装する。

- rawが範囲内ならcandidate integralを採用する。
- 上限飽和中にerrorが正なら積分しない。
- 下限飽和中にerrorが負なら積分しない。
- 飽和から戻す向きのerrorなら積分を許可する。

初回はderivativeを0とする。`reset()`はintegral、previous error、初回フラグを初期状態へ戻す。
非有限入力、`dt_s <= 0`、不正な上下限は例外にする。

## LaneKeepingLongitudinalPolicy

### MetaDrive boundary

Policyはインストール済みMetaDrive 0.4.3の`BasePolicy`を継承する。
`get_input_space()`は`gymnasium.spaces.Discrete(4)`を返す。MetaDriveのagent managerは
各stepで`act(agent_id)`を呼び、その戻り値`[steering, throttle_brake]`を車両へ適用する。

control smoke用environment factoryはMetaDrive configへruntime値として
`agent_policy=LaneKeepingLongitudinalPolicy`を追加する。Python classをYAMLへ直列化しない。

### Lateral control

Policyはlane changeを行わない。対象laneは次の順で取得する。

1. `vehicle.navigation.current_lane`
2. `vehicle.lane`

laneがある場合、現在位置をlane座標へ変換し、`longitudinal + lookahead_m`のlane headingを
取得する。heading errorは`[-pi, pi]`へwrapし、lateral offsetと別々のPIDへ入れる。

```text
steering = clip(heading_command + lateral_command, -1, 1)
```

符号はMetaDrive IDM policyの実装と一致させ、テストでは左右両側のoffsetとheading errorが
中心へ戻す向きになることを確認する。

### Longitudinal control

ActionMapperがSI単位のtarget speedを返す。

```text
speed_error = target_speed_mps - vehicle.speed
desired_acceleration = speed_pid(speed_error, dt)
```

KEEP、SLOW、PREPARE_STOPではdesired accelerationを
`[normal_deceleration_mps2, max_acceleration_mps2]`へclipする。STOPでは下限を
`emergency_deceleration_mps2`にする。

MetaDrive操作への正規化は次のとおりである。

```text
if desired_acceleration >= 0:
    throttle_brake = desired_acceleration / max_acceleration_mps2
else:
    throttle_brake = desired_acceleration / abs(active_deceleration_limit_mps2)
```

結果を`[-1, 1]`へclipする。これはMetaDrive actuatorへのcommand上限であり、タイヤ・質量・
路面を含む実加速度が常に設定値と完全一致するという意味ではない。

### Reset and fail-safe

MetaDriveがPolicyをresetすると、speed、heading、lateralの全PIDをresetする。

次の場合は通常計算を中止し、`[0.0, -1.0]`を返す。

- Actionが0から3でない
- navigationとvehicleの両方からlaneを取得できない
- 速度、lane座標、heading、PID結果のいずれかが非有限
- ActionMapperまたはPIDが例外を出す

`action_info`へAction、target speed、steering、throttle/brake、fail-safe flag、理由を入れる。
例外をMetaDrive engine loopの外へ漏らさず、安全側へ閉じる。

## Control smoke lifecycle

既存`mad_driving.cli.smoke`はPhase 1/2回帰用として変更しない。新しい
`mad_driving.cli.control_smoke`を追加する。

### Reset

1. custom Policy付きMetaDrive environmentを作る。
2. seedを指定してresetする。
3. step 0のSceneSnapshotを作る。
4. AgentSuiteでclaimsとreviewを作る。

### Decision step

各stepで次を行う。

1. 現在snapshot、claims、reviewからCoordinatorがrequested actionを返す。
2. Shieldがrequested actionをfilterする。
3. executed actionを`env.step()`へ渡す。
4. MetaDriveがPolicyのsteeringとthrottle/brakeで1 decision step進む。
5. previous actionとshield interventionを含む次snapshotを作る。
6. 次decision用のclaimsとreviewを作る。
7. そのstepのDecisionTraceを作る。

DecisionTraceのclaimsとreviewは、そのstepの判断に使ったstep前の値を保持する。
`reward_components`はPhase 3では空dictとする。

### Result

`ControlSmokeResult`は次を持つ。

```python
steps_completed: int
terminated: bool
truncated: bool
final_snapshot: SceneSnapshot
final_claims: tuple[RiskClaim, ...]
final_review: CriticReview
final_trace: DecisionTrace
action_counts: tuple[int, int, int, int]
shield_intervention_count: int
```

CLIは`dataclasses.asdict()`を有限JSONへ変換する。100件のtrace全体は標準出力へ出さず、
最終traceと集計だけを出す。

## Failure handling

control smokeは安全側に閉じ、resourceを必ず解放する。

- AgentSuiteが例外を出した場合、そのdecisionではclaimsを空にし、fallback reviewを使う。
- fallback reviewは`conflict_score=1.0`、`unresolved_conflict=True`、
  `max_severity=1.0`、reason `agent_analysis_failed`とする。
- claimsが空なのでCoordinatorはPREPARE_STOP、Shieldは複数Agent欠損としてSTOPを要求する。
- Criticを含むAgentSuite全体の例外を同じ経路で処理し、部分的で未検証のclaimを使わない。
- Policy内部の失敗は最大brakeへ変換する。
- MetaDrive reset、step、snapshot生成の例外は呼び出し元へ返すが、`finally`で必ずcloseする。

Phase 3のAgentは同期・ローカル・決定論的であり、外部I/Oを行わない。そのためwall-clock timeoutや
thread/processによるpreemptionは追加しない。将来外部Agentを導入する場合は、実行境界でtimeoutを
設け、同じclaims欠損経路へ変換する。

## Determinism

同じ型付き入力と設定に対し、ActionMapper、Coordinator、Shield、PIDの初期状態からの出力は
exact equalityを満たす。PIDは明示状態を持つため、比較時は同じ入力系列とreset位置を使う。

MetaDrive vehicle IDがUUIDである問題はPhase 2と同じ外部挙動であり、本Phaseでは変更しない。
Agentと制御モジュール自体はsnapshot内のIDを生成し直さない。

## Testing

実装はTDDで進める。

### Unit tests

- Config: default、strictness、全範囲、閾値間関係
- ActionMapper: 4 Action、0 speed limit、25%・60%境界、invalid input
- Coordinator: claim speed制約、hard stop、conflict、severity、欠損、exact equality
- Shield: 全理由、固定順、3 mode、境界、invalid injected claim、missing Agent
- Shield property: `executed >= requested`、危険増加時にAction indexが減らない
- PID: P/I/D、上下限、conditional anti-windup、integral clamp、reset、invalid input
- Policy: steeringの向き、target speed、STOP brake、normalized output、reset、fail-safe
- Control smoke: lifecycle、Action集計、trace、analysis failure、step failure時close、finite JSON

### Integration tests

- custom Policyのaction spaceがMetaDriveでDiscrete(4)として認識される。
- 実MetaDriveでreset、4 Action入力、snapshot、Agent解析、closeが成功する。
- 100 decision stepsのrule-based control smokeがheadlessで終了する。
- 最終JSONが有限で、Action count合計がsteps completedと一致する。
- lane offset、steering、throttle/brakeが有限である。
- 強制STOPの短いintegrationで速度が開始時より低下する。

### Quality gate

- 全既存テストを含むpytestが成功する。
- branch coverage 80%以上を維持する。
- Ruff checkとformat checkが成功する。
- mypy strictが成功する。
- `git diff --check`が成功する。
- upstream Matplotlib/Pyparsing警告以外の新規警告を追加しない。

## Acceptance criteria

Phase 3は次をすべて満たしたとき完了する。

1. DrivingActionの安全順序と4目標速度がテストで固定されている。
2. RuleBasedCoordinatorがAgent制約から決定論的にActionを選ぶ。
3. SafetyShieldがenforce modeで危険側へActionを変更しない。
4. off、monitor、enforceの差がテストとresultで確認できる。
5. PIDがresetとanti-windupを実装し、出力を設定範囲に制限する。
6. custom MetaDrive Policyがlane keepingと速度制御を独立して行う。
7. Coordinatorはsteeringを出力しない。
8. AgentまたはPolicy失敗時にSTOPまたは最大brakeへ閉じる。
9. 実MetaDriveの100-step control smokeがheadlessで成功する。
10. Phase 1/2 smokeと145件の既存テストが回帰しない。
11. Phase 4以降のObservation、Reward、PPO、ScenarioManagerを実装していない。
12. 設計判断、MetaDrive API観測、検証結果を実装記録へ残す。

この設計では、Phase 3の終点が明確である。Agentの意見は車両へ届くが、学習器にはまだ届かない。
