# Phase 4 RL環境設計

## Status and authority

本設計は、`docs/multi_agent_driving_mvp_spec.md` のPhase 4「RL環境」を実装可能な
粒度へ固定する。最上位要件はMVP仕様書であり、本設計はそこにない機能を追加しない。

Phase 4では、Phase 3の決定論的Agent、Safety Shield、車線追従、速度PIDを固定したまま、
CoordinatorだけをStable-Baselines3 PPOで学習できるGymnasium環境へ接続する。

## Goals

1. Agent出力、Critic、履歴を仕様どおり24次元`float32` Observationへ変換する。
2. 仕様の10成分から有限なRewardを計算し、各成分を`info`と`DecisionTrace`へ出す。
3. `MultiAgentSpeedEnv(gymnasium.Env)`が`check_env()`を通る。
4. PPO smoke training、checkpoint保存・再読込、TensorBoard出力をheadless CPUで確認する。
5. 同一seedのresetとAgent出力を再現可能にする。

## Non-goals

Phase 4では次を実装しない。

- Lead Brake、Cut-in、Occluded Crossing ActorとScenarioManager
- curriculum、train/eval seed集合の分離
- baseline、ablation、比較実験
- JSONL全step永続化、CSV metrics、PNG plot、GIF
- 評価・描画・比較CLI
- 画像入力、LLM/VLM、学習操舵、複数学習対象車両

上記は仕様書のPhase 5またはPhase 6に残す。Phase 4はrun directory、checkpoint、
TensorBoard eventだけを生成する。

## Architecture

```text
PPO Coordinator action (requested)
  -> MultiAgentSpeedEnv
  -> SafetyShield (executed)
  -> LaneKeepingLongitudinalPolicy
  -> MetaDrive.step
  -> SceneSnapshotBuilder
  -> AgentSuite + Critic
  -> RewardCalculator
  -> ObservationBuilder
  -> obs, reward, terminated, truncated, info
```

責務境界は次のとおりである。

- `coordinator/observation.py`: simulator非依存の24次元変換。
- `envs/reward.py`: simulator非依存の遷移Reward計算。
- `envs/multi_agent_speed_env.py`: Gymnasium APIと全runtime lifecycle。
- `training/train.py`: SB3 PPO構築、再開、checkpoint、TensorBoard。
- `training/callbacks.py`: Reward成分のTensorBoard記録。
- `cli/train.py`: 引数、設定読込、エラー表示だけ。

SB3 importは`training/`と`cli/train.py`へ限定する。Observation、Reward、Gym wrapperは
SB3なしでunit testできる。

## Configuration

既存`AppConfig`へdefault factory付きで次を追加する。既存Phase 1-3設定は引き続き読める。
新しいmodelはstrict、frozen、unknown-key拒否、有限値・範囲検証を行う。

### ObservationConfig

```yaml
observation:
  max_speed_mps: 40.0
  max_abs_acceleration_mps2: 10.0
  max_abs_lane_offset_m: 3.5
  max_ttc_s: 10.0
  max_abs_stopping_margin_m: 50.0
```

すべて正の有限値とする。

### RewardConfig

```yaml
reward:
  progress_per_meter: 0.10
  arrival: 100.0
  collision_vehicle: 200.0
  collision_crossing_actor: 500.0
  near_miss_max: 50.0
  near_miss_ttc_s: 3.0
  offroad: 100.0
  hard_rule_violation: 100.0
  jerk_scale: 0.05
  unnecessary_brake_scale: 0.20
  unnecessary_brake_severity_threshold: 0.25
  unnecessary_brake_safe_ttc_s: 5.0
  unnecessary_brake_lookahead_steps: 3
  standstill_per_second: 0.50
  standstill_speed_mps: 0.10
  shield_intervention: 2.0
```

仕様書に値がある10 weightはその値を使う。仕様書に値がないnear-miss閾値、
unnecessary-brake判定閾値、lookahead、standstill速度だけを明示設定として補う。
weightは有限・非負、時間・速度閾値は正、lookaheadは正の整数とする。

### PPOConfig

```yaml
training:
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
  smoke_timesteps: 5000
  total_timesteps: 500000
  num_envs: 1
  checkpoint_interval_steps: 10000
  eval_interval_steps: 10000
  eval_episodes: 5
  run_root: runs
```

PPO既定値は仕様書をそのまま使う。`n_steps * num_envs`は`batch_size`以上かつ
`batch_size`で割り切れることを起動時に検証する。複数環境では各MetaDrive engineを
別processへ隔離する。subprocess vector envの構築に失敗した場合は、部分構築された
workerの停止を確認して明示的に失敗する。同数の`DummyVecEnv`へのフォールバックは、
MetaDrive 0.4.3のprocess単位のengine singleton制約を破るため行わない。

## Observation contract

`ObservationBuilder.build(snapshot, claims, review)`はshape `(24,)`、dtype `float32`、
全要素有限、範囲`[-1, 1]`のNumPy arrayを返す。

### Normalization rules

- 非負量: `clip(value / configured_max, 0, 1)`
- 符号付き量: `clip(value / configured_abs_max, -1, 1)`
- probability、severity、confidence、flag、route progress: `[0, 1]`のままclip
- TTC: `clip(ttc / max_ttc_s, 0, 1)`。`None`は危険なしを表す`1.0`
- previous action: `action / 3`
- supported ratio: Criticのunique supported Agent数を3で割ってclip

固定indexは次のとおりである。

| Index | Feature |
|---:|---|
| 0 | ego speed / max speed |
| 1 | previous action target speed / speed limit |
| 2 | acceleration / max absolute acceleration |
| 3 | lane offset / max absolute lane offset |
| 4 | route progress |
| 5 | speed limit / max speed |
| 6 | Nominal min TTC / max TTC |
| 7 | Nominal collision probability |
| 8 | Nominal confidence |
| 9 | Nominal recommended speed / max speed |
| 10 | Hazard worst TTC / max TTC |
| 11 | Hazard stopping margin / max absolute margin |
| 12 | Hazard severity |
| 13 | Hazard confidence |
| 14 | Hazard recommended speed / max speed |
| 15 | Rule recommended speed / max speed |
| 16 | Rule hard stop flag |
| 17 | predicted hard violation flag |
| 18 | Critic conflict score |
| 19 | Critic unresolved conflict flag |
| 20 | Critic supported Agent ratio |
| 21 | Critic max severity |
| 22 | previous action / 3 |
| 23 | previous Shield intervention flag |

Feature 1は`target_speed_mps(snapshot.previous_action, current speed, speed limit)`から導出する。
Feature 17は`stop_required`、`intersection_entry_prohibited`、`collision_occurred`、
`off_road`の論理和とする。

claimは`agent_id`で検索する。Agent欠損時は有限な安全側値にする。

- Nominal欠損: TTC `0`、probability `1`、confidence `0`、recommended speed `0`
- Hazard欠損: TTC `0`、margin `-1`、severity `1`、confidence `0`、recommended speed `0`
- Rule欠損: recommended speed `0`、hard stop `1`、predicted violation `1`

重複agent ID、無効Snapshot/Claim/Reviewは`ValueError`にする。通常runtimeでは
Agent失敗を空claimsへ変換するため、欠損値経路でPPOへ有限Observationを返せる。

## Reward contract

`RewardCalculator`はepisodeごとにresetする小さな状態機械である。`calculate(context)`は
`RewardResult(total: float, components: dict[str, float])`を返す。componentsは次の固定キーを
持つ符号付き寄与で、`total == sum(components.values())`とする。

```text
progress_reward
arrival_reward
collision_penalty
near_miss_penalty
offroad_penalty
rule_violation_penalty
jerk_penalty
unnecessary_brake_penalty
standstill_penalty
shield_intervention_penalty
```

- progress: 前Snapshotのheadingへ位置差を射影し、正の前進距離だけをweight倍する。
- arrival: MetaDrive infoの`arrive_dest=True`で一度だけ加点する。
- collision: `crash_human=True`ならcrossing actor weight、それ以外の衝突はvehicle weight。
- near miss: post-step claimsの最小TTCに対し
  `-near_miss_max * max(0, 1 - ttc / threshold)^2`とする。連続で境界値は0。
- offroad: post-step Snapshotがoff-roadなら固定減点。
- rule violation: Rule hard stopまたはSnapshot制約があり、executed actionがSTOP未満なら減点。
- jerk: `abs(next_acceleration - previous_acceleration) / dt`をscale倍して減点。
- unnecessary brake: SLOW以上、Hazard severityが閾値未満、Rule制約なし、TTCが安全域、
  かつ同条件がlookahead steps連続したstepから`-scale * action_index`を与える。
  将来状態はObservationへ入れない。
- standstill: post-step speedが閾値以下なら`-weight * dt`。
- shield intervention: 実介入時だけ固定減点。

`dt`は連続Snapshotの`sim_time_s`差を使い、正でなければ設定済みdecision intervalを使う。

## Gymnasium environment

```python
class MultiAgentSpeedEnv(gym.Env[np.ndarray, int]):
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(-1.0, 1.0, (24,), np.float32)

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict[str, object]]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...
    def close(self) -> None: ...
```

`reset`は`super().reset(seed=seed)`を先に呼び、未指定seedでは`AppConfig.seed`を使う。
simulator、Agent、Reward、履歴を初期化し、初期Observationを返す。

`step`はPPO actionをrequested actionとしてShieldへ渡す。MetaDriveへはexecuted actionだけを
渡す。post-step Snapshot/claims/reviewからRewardと次Observationを作り、次を`info`へ入れる。

```text
requested_action
executed_action
shield_intervened
shield_reasons
target_speed_mps
reward_components
decision_trace
```

MetaDriveの`terminated`と`truncated`を保持し、collision、off-road、arrivalはterminated、
horizonはtruncatedとする。MetaDrive例外は`terminated=False`、`truncated=True`、
`info["simulator_error"]`として安全にepisodeを閉じる。次reset時はsimulatorを再生成する。
`close()`は例外安全かつ冪等とする。

## PPO training and artifacts

`run_training`はrun directoryを作り、resolved configをYAML保存し、PPOを新規構築または
checkpointから再開する。Agent、Critic、Shield、PIDはoptimizerへ渡さず、PPO policyだけを
学習する。

```text
runs/<run_id>/
├─ config_resolved.yaml
├─ checkpoints/
│  ├─ checkpoint_<steps>_steps.zip
│  ├─ best_model.zip
│  └─ final_model.zip
└─ tensorboard/
   └─ PPO_*/events.out.tfevents.*
```

SB3の`CheckpointCallback`と`EvalCallback`を使い、追加callbackで
`info["reward_components"]`をTensorBoardへ記録する。評価envは学習envと別instanceにする。
`EvalCallback`はbest checkpoint選択にだけ使い、Phase 6の評価artifactは保存しない。
全envは成功・例外の両方でcloseし、subprocess workerは期限付きjoin、terminate、killの順で
停止を確認する。cleanup失敗は成功として返さず、既存の学習例外がある場合はそれを保持して
cleanup失敗をnoteへ記録する。

CLIは次を提供する。

```powershell
python -m mad_driving.cli.train --config configs/train.yaml --smoke
python -m mad_driving.cli.train --config configs/train.yaml
python -m mad_driving.cli.train --config configs/train.yaml --resume-from <checkpoint.zip>
```

`--smoke`は5,000 timesteps、通常は500,000 timestepsを選ぶ。エラーはtracebackなしで
stderrへ明示し、非0終了する。

## Verification

- 24 indexの値、dtype、shape、範囲、欠損、安全側fallbackをunit testする。
- Rewardの全10成分、連続near-miss、unnecessary-brake lookahead、resetをunit testする。
- fake simulatorでreset/step/seed/termination/truncation/例外/closeをunit testする。
- 実MetaDriveで`gymnasium.utils.env_checker.check_env()`と100 stepsを通す。
- 小さいPPO integrationでcheckpoint保存・読込とTensorBoard event生成を確認する。
- canonical 5,000 timestep headless CPU smoke trainingを実行する。
- pytest coverage 80%以上、Ruff、format、mypy strictを通す。
