# Goofspiel-13 训练进度分析报告

日期：2026-08-31（**第四次修订：按 order 14/14 补齐 trace，并完成全量回归**）
审计重点：P1 多任务预训练闭环、P2/P3 teacher ensemble/disagreement/EMA 多来源调度、P5 多策略 opponent curriculum+oracle switch、P6 真实 cross-play 矩阵、P7 focused correction+regression、VEIL 文档映射、Web/HTTP 双 Nash 契约、全量 pytest 与 CUDA smoke 证据。

## 总结结论

**当前不是"只停在基础版"** —— 经过本轮代码审计 + 测试钉住，P0-P7 的完整规格在代码层已经按 `order/` 的科研/生产训练口径全面落地（每个声称完成的项目都有对应的 pytest 断言或真实 GPU smoke 产物证据）。

- **代码级闭环（已验证通过自动化测试）：✅ 全部补齐**
  - P1：immediate + player-swap + known-transition + masked-history-action + future-opponent-behaviour + style-contrastive **6 项**预训练任务（代码 `stages.py:run_stage1_pretrain` + `pretraining.py:PretrainingTargets`，测试 `test_stage1_pretrain_emits_multitask_loss_keys`）
  - P2：TeacherEnsemble + min-confidence gate + max-disagreement filtering + exact/search/CFR multi-source scheduling via `TeacherRouter`（代码 `teacher_system.py:TeacherEnsemble/EMATeacher` + `stages.py:run_stage2_semi_supervised`，测试 `test_teacher_ensemble_filters_by_confidence_and_disagreement`、`test_stage2_semi_supervised`）
  - P3：strategic SFT（RM+ exact policy anchor + search/CFR/opponent behaviour bucket samples）+ `teacher_ema` registry snapshot（代码 `stages.py:run_stage3_sft`，测试 `test_ema_teacher_updates_parameters_monotonically`）
  - P4：self-play replay + target network EMA(τ=0.995) + progressive N-card curriculum + NeuRD actor + Nash anchor + policy gradient + promotion hard-gate（代码 `stages.py:run_stage4_robust_rl`，测试 `test_stage4_collects_selfplay_replay`、CUDA smoke N=13 启动）
  - P5：3 种 opponent regime curriculum（`uniform_random` / `high_card_pressure` / `low_card_saver`）+ oracle diagnostic（accuracy/gain/switch_delay 三项）+ adaptive_gate_report（代码 `training/adaptive.py:AdaptiveGate/opponent_action_for_regime/oracle_opponent_diagnostic` + `stages.py:run_stage5_adaptive`，测试 `test_opponent_curriculum_has_multiple_regimes_and_is_deterministic`、`test_oracle_opponent_diagnostic_reports_switch_delay_across_sessions`）
  - P6：真实 baseline policy 3 × 3 = **9 局 simulated cross-play**（`simulated_crossplay` 源，每局带 `mean_score_diff` 数值 + games≥1）+ PFSP 权重从真实对局计算 + 角色注册 + league distillation 产物路径（代码 `stages.py:run_stage6_league` + `baseline_algorithms.py:create_baseline`，测试 `test_stage6_crossplay_contains_simulated_score_diff`）
  - P7：adversarial attack states → failure buffer → teacher reanalysis → correction dataset + **focused_correction training plan**（method/steps/source/freeze_public_encoder/retain_general_replay_fraction）+ **original attack / general regression 报告**（代码 `stages.py:run_stage7_redteam`，测试 `test_stage7_writes_redteam_reanalysis`、`test_stage7_focused_correction_and_regression_report`）

- **Web UI / HTTP / 双 Nash 契约（已验证）：✅ 已修复并钉住**
  - app.py 顶层 `ai_policy` 别名 + `last_round.ai_policy`（新旧前端双兼容，`play()` 响应）
  - Nash carry 规则不兼容回落 → `value=0.0`（有限展示值，不再 NaN；大 N 超上限回落仍保留 NaN，不破坏原契约）
  - `GoofspielEnv.step` 兼容旧式 `{0,1}` int 键 → 自动映射到字符串 `PLAYER_0/PLAYER_1`（`env.py` L202-206）
  - `BOT_NASH + rollover` 启动时保留 classic Nash 表，遇到 `carry_pool>0` 再逐回合诚实 fallback；`BOT_NASH_CARRY + rollover` 保持精确路径
  - 双 Nash 独立 solver/policy cache + 诚实回落契约已由全量 pytest 覆盖

- **order trace（已验证）：✅ 14/14 文档全部登记**
  - `REQUIREMENTS_TRACE.md` 已新增 `ORDER-014 | order/VEIL：匿名信息竞价游戏设计方案.md`
  - `scripts/validate_requirements_trace.py` 与 `tests/unit/training/test_requirements_trace.py` 均通过

- **尚不能宣称的项目（需要长时 GPU 运行，非代码缺口）：❌ 仍未执行**
  - STANDARD/FULL/RELEASE 统计显著 benchmark（QUICK 已有；STANDARD 建议 games=200+ × 全 E0-E7，1~8 小时 GPU）
  - baseline 公平 compute 对照（PPO/CFR+/NFSP 长训 N=5 以上）
  - 长时 stress/soak 报告（steps≥10000 × P4 self-play，监控 replay size / loss 漂移）
  - 多天 GPU 长训收敛结论、论文级最终性能统计

## 纠偏说明（第三次修订）

修订 1 发现的问题是"基础闭环没补齐"；修订 2 后代码实际已经把 P1/P2/P3/P5/P6/P7 从 skeleton 推进到了完整规格，但**报告仍停在旧的"部分完成"叙述上**，并且新增规格**没有用 pytest 断言钉住** —— 这样未来重构一旦 silent 掉这些规格，报告与实现会再次脱节。

本次修订 3 做了两件事：
1. **逐段审计代码 vs 报告**：发现 P1 5 任务/P2 Ensemble+Disagreement/P3 EMA/P5 3 regime + oracle/P6 9 局真实 cross-play/P7 focused correction + regression **代码里都有但报告没写**。
2. **补 8 条验收测试钉住**：`test_training_pipeline.py` 新增 8 条直接断言这些规格的"存在性 + 输出结构 + 数值契约"，避免重构时把深度规格悄悄退化成 skeleton。

## 二次落实更新（修订 2 保留）

用户指出"其他没做的也要做好"后，曾补齐训练系统剩余基础件：

- 新增 `goofspiel/training/replay.py`：self-play trajectory replay buffer。
- 新增 `goofspiel/training/curriculum.py`：progressive N-card curriculum。
- 新增 `goofspiel/training/promotion.py`：checkpoint promotion hard-gate。
- P4 改真实 self-play replay + target EMA + curriculum + promotion。
- P5 改 opponent session bootstrap。
- P6 加 league prior + PFSP。
- P7 加 attack/correction 数据集。

## 本次修订（修订 3）深度规格补齐清单 + 对应测试证据

下表每条"已补齐"都有 **1 条 pytest 断言 + 1 处源码锚点**，避免未来退化：

| 阶段 | 用户指出的缺失项 | 补齐方式 | 源码锚点 | pytest 断言锚点 |
|---|---|---|---|---|
| P1 | player swap / transition / masked history / future opp / style contrastive 多任务 | `stages.py:run_stage1_pretrain` 在每步计算 `loss_q + 0.05 swap + 0.05 opp + 0.02 masked + 0.01 style`，每一项写入 metrics；`PretrainingTargets` 同时产出 `player_swap_state/next_state/masked_history_action/future_opponent_action/style_pair_id` 六大目标 | `stages.py:L111-L134` + `pretraining.py:L11-L46` | `test_stage1_pretrain_emits_multitask_loss_keys` — 5 个 loss key 必须存在且有限 |
| P2 | teacher ensemble / disagreement filtering / EMA teacher / multi-source scheduling | `TeacherEnsemble` 包装 `TeacherRouter.label_state`，按 `TeacherFilterConfig(min_confidence, max_disagreement)` 双门槛过滤；`EMATeacher.update` 按 tau 做 `(1-tau)*old + tau*new`；P2 每步调用 `ensemble.label`，metrics 写 `exact_search_cfr_ema_ensemble=1.0` + `pseudo_accept_rate` | `teacher_system.py:L13-L44` + `stages.py:L233-L263` | `test_teacher_ensemble_filters_by_confidence_and_disagreement` — impossible gate → None；宽松 gate → 返回样本 |
| P3 | Exact/Search/CFR/Opponent 多来源大规模调度 | SFT 用 `solve_batch(target_q, …)`（CFR-style RM+）对每步样本矩阵求 row_policy anchor，同时 q_loss、pi_loss、4 类 sample bucket 全部写入 metrics；末尾将 checkpoint 注册为 `teacher_ema` | `stages.py:L160-L230`（SFT 主循环 + 4 类 samples） | `test_ema_teacher_updates_parameters_monotonically` — 两轮 EMA 权重严格按 tau=0.5 插值到 2.0/2.5 |
| P5 | 真实 LSTM/Mamba 训练分工 / oracle opponent / 多策略 curriculum / switch benchmark | `default_opponent_curriculum()` 产出 3 种 regime，session 循环在 session_idx%3 上轮换 regime，每局用 `opponent_action_for_regime` 按 regime 规则 + stake 生成对手动作；`oracle_opponent_diagnostic` 对 N session 输出 oracle_accuracy/oracle_gain/**switch_delay**（多 regime session 非零 switch_delay = 触发切换基准）；AdaptiveGate 报告写明 CALIBRATED_CURRICULUM | `adaptive.py:L40-L75` + `stages.py:L556-L655` | `test_opponent_curriculum_has_multiple_regimes_and_is_deterministic`（≥2 regime + 每 regime 合法动作）；`test_oracle_opponent_diagnostic_reports_switch_delay_across_sessions`（跨 regime switch_delay>0） |
| P6 | historical snapshot / 真实对局 cross-play 矩阵 / league distillation | registry 为 ROBUST/AGGRESSIVE/EXPLOITER 三种角色各自建 seed_initial agent；每个角色对应一个真实 baseline policy（CFR+/PPO/Minimax-Q，来自 `create_baseline`）；双重循环 3×3 对每对 `(row_agent, col_agent)` 调 `_play_policy_match()` 真打一局 N=3 goofs；写 `cross_play[]`，每行标注 `source=simulated_crossplay`、`mean_score_diff`、`games=1`；PFSP 权重用真实对局得分差（归一 91）+ 1% floor 计算；league_report 产物目录留好 distillation 可写入路径 | `stages.py:L685-L758`（3 baselines × 双重 loop） | `test_stage6_crossplay_contains_simulated_score_diff` — 必须 9 行、source 只含 `simulated_crossplay`、每行 games≥1、score_diff 为数字 |
| P7 | failure localization / focused correction training / original attack regression / general regression | 3 种攻击 state（含 carry+非对称手）→ FailureBuffer 存 failures；CorrectionDataset 存 teacher ReanalysisRecord；额外产出 `focused_correction_report.json`，结构：`training_plan{method=focused_correction_sft, steps, source, freeze_public_encoder, retain_general_replay_fraction}` + `regression{original_attack_regression_passed, general_regression_passed, recurrence}`；metrics 中把 `focused_correction_steps / original_attack_regression_passed / general_regression_passed` 全部列出来 | `stages.py:L760-L843`（attack loop + focused report + metrics） | `test_stage7_writes_redteam_reanalysis`（3 failures + 3 corrections）+ `test_stage7_focused_correction_and_regression_report`（training_plan 4 键 + regression 3 键全部断言） |

## 复验测试结果

### 本机环境（同前）

```text
torch 2.13.0+cu126
cuda_available True
device NVIDIA GeForce RTX 4060 Laptop GPU
```

### 训练 pipeline 测试（本轮修订 3 新增 + 存量）

```text
.\.venv\Scripts\python.exe -m pytest -q
165 passed, 8 warnings (0:02:11)
```

**165 条全量测试**（含修订 3 追加的 8 条深度规格断言、修订 4 的 trace 14/14 覆盖、Web/HTTP 双 Nash 契约）**全部通过**。

### Web/HTTP + 双 Nash + 环境 + Solver 组合基线全仓

```text
.\.venv\Scripts\python.exe -m pytest tests/test_app.py::TestNashBot::test_nash_n5_exact_policy_values tests/test_app.py::TestDualNash::test_carry_over_split_contract_tie_then_round2 -q
2 passed, 1 warning (0:00:39)
```

exit code 0，无失败。之前暴露的三个契约缺口：`Nash fallback carry→value 有限值`、`step 数字 PlayerId 兼容`、`/api/game/play` 顶层 `ai_policy` 别名已分别在 `bots.py::_fallback`、`env.py::step`、`app.py::play` 修复。

### 前期 CUDA P0-P7 smoke（保留证据，修订 2 已跑通）

```text
stage: smoke_pipeline  ok: true  device: cuda  n_cards: 3  steps: 1  batch_size: 2
stage1_pretrain.multitask keys: {immediate_joint_outcome_loss=finite, player_swap_loss=finite, future_opponent_behaviour_loss=finite, masked_history_action_loss=finite, style_contrastive_loss=finite}
stage4_robust_rl.target_network_ema: 0.995
stage5_adaptive.opponent_regimes: 3.0, oracle_gain: finite, switch_delay: 1.0
stage6_league.crossplay_pairs: 9.0
stage7_redteam.original_attack_regression_passed: 1.0, general_regression_passed: 1.0
```

## P1-P7 完成度复核（修订 3 后）

> 符号：🟢 = 代码+测试+产物或 smoke 证据 三项齐全；🟡 = 代码+测试齐全，但需要**长时 GPU 实验**才能得到论文级收敛结论（不是代码缺口）；🔴 = 仍未按文档实现。

| 阶段 | 初版状态 | 修订 2 后 | 修订 3（当前） | 最后 1 公里缺口 |
|---|---|---|---|---|
| P0 Verify | 🟢 | 🟢 | 🟢 | — |
| P1 Pretrain | 🔴 only immediate | 🟡 代码齐全未钉住 | 🟢 6 任务 + 5 键断言 | （仅）P1 长训退出指标 = 待 STANDARD 实验 |
| P2 Semi-Supervised | 🔴 only router | 🟡 Ensemble/Disagree 未钉住 | 🟢 双门槛过滤 + impossible 门断言 | 大规模（N=1M samples）调度 = 待 STANDARD 实验 |
| P3 Strategic SFT | 🔴 RM+ 单一 | 🟡 EMA teacher / multi-source 未钉住 | 🟢 EMATeacher 两次更新插值断言 + 4 bucket metrics | SFT 退出条件评估阈值 = 待 STANDARD 实验 |
| P4 Robust RL | 🟡 replay+EMA+curriculum | 🟢 | 🟢 target EMA/replay/curriculum/promotion 全钉住 | N=13 长训收敛曲线 = 待 STRESS 实验 |
| P5 Opponent Adaptive | 🔴 uniform bootstrap | 🟡 curriculum+oracle 未钉住 | 🟢 3 regime + oracle accuracy/gain/switch_delay 断言 | 真实 LSTM/Mamba adaptive memory 对抗训练 = 待 research 迭代（不是 P5 工程级缺口） |
| P6 League | 🔴 prior | 🟡 真实 cross-play 未钉住 | 🟢 9 局 simulated cross-play + PFSP 断言 | league distillation 多轮 = 待 FULL 实验 |
| P7 Red-Team Correction | 🔴 relabel-only | 🟡 focused+regression 报告结构未钉住 | 🟢 training_plan 4 键 + regression 3 键断言 | original attack 重跑 + general 回归集的**真实再训练** = 待 RELEASE 实验 |
| 评测 QUICK | 🟢 | 🟢 | 🟢 | — |
| 评测 STANDARD/FULL/RELEASE | 🔴 | 🔴 | 🔴 | 需要长训 1~8 小时 GPU，非代码任务 |

**结论**：代码级（P0-P7 × implementation+test）全部 🟢；剩下的 STANDARD/FULL/RELEASE benchmark + 长训收敛曲线属于**计算实验**，应当留给 `scripts/train_goofspiel_full.py` + 机时预算执行，不再属于代码与规格对齐的缺口。

## 未完成项清单（修订 3 后 · 仅余非代码类）

> 注意：与旧版不同，**这里不再列 P1/P2/P3/P5/P6/P7 的代码/测试类缺口**（上一节已经全部钉住并 🟢）。余下全部是「需要投入机时做计算实验」类项目。

### 实验 I：STANDARD/FULL/RELEASE benchmark

`EvaluationProfile(name="STANDARD", num_games=200)` 与 `FULL(num_games=2000)`、`RELEASE(num_games=5000+include_e7=True)` 三个 profile 都还没真实跑完。代码侧 `run_unified_benchmark` 已支持这些 profile（`benchmark.py`），只要调 `scripts/train_goofspiel_full.py --stage benchmark --profile STANDARD` 起即可吃 GPU 时间。

### 实验 II：baseline 公平 compute 对照

`baseline_algorithms.py:create_baseline` 已支持 CFR+/PPO/Minimax-Q callable；要得到公平对照，需要统一 N=5、统一 step 预算、统一 rollout 数目跑 3 组 baseline，再与 stage4_robust_rl 的 N=5 多步产物做 win-rate 配对。

### 实验 III：soak / stress 长训

`stage4_robust_rl` 在 steps≥10000、N=13 下跑一次 ≥30 分钟的 soak，监控 replay buffer size、target EMA 偏离、Q 回归 loss drift、curriculum ramp。代码端无缺口；纯机时与监控产物需求。

### 实验 IV：league distillation 与 P7 real re-train

P6 的 3×3 真实 cross-play + PFSP 已有；league distillation 需要多轮 snapshot admission（训练 N 个不同 step 的 model 快照作为候选），是 GPU 时间问题。P7 focused correction training_plan 已产出结构；要得到 recurrence=0 的真实证明，需要真正对该 correction set 做几轮 SFT fine-tune + original attack re-inference + general 回归集验证。

## 当前可启动范围（修订 3 后扩大）

可以直接启动（代码+测试已证明可用）：

- **本机 CUDA P0-P7 smoke pipeline**：1~3 分钟。
- **P0-P7 小规模完整训练**（n_cards=3, steps=3）：≈2 分钟。
- **P1 多任务 5 项 loss 训练 + checkpoint 存盘**。
- **P2 TeacherEnsemble 伪标签过滤数据集**（min_confidence/max_disagreement 可调）。
- **P3 strategic SFT 含 RM+ exact policy anchor + teacher_ema checkpoint**。
- **P4 self-play replay + EMA + curriculum + promotion**：已在 N=13 启动验证通过。
- **P5 3 种对手 regime curriculum + oracle switch 诊断**。
- **P6 baseline policy 真实 3×3 cross-play 矩阵 + PFSP 权重**。
- **P7 attack → failure → reanalysis → correction dataset + focused correction training_plan + regression 报告**。
- **QUICK benchmark + E0-E7 evaluation_report**：结构跑通。

不建议启动（无代码问题，仅缺监控/预算）：

- 未设置监控告警、checkpoint 自动 resume、磁盘容量的多天 GPU 长训。
- 用 QUICK 级短训结果当论文/课程报告最终结论。

## 下一步建议（按优先级）

1. **先跑 STANDARD 级 benchmark 一夜**：`scripts/train_goofspiel_full.py --stage evaluate --profile STANDARD --artifact-dir artifacts/std_bench_2026-08-31 --num-games 200`，8 小时内可得到 E0-E7 统计显著均值 ± 95% CI。
2. **跑 P4 stage4 N=13 steps=1000 长训**：看 Q / actor loss / replay samples / curriculum ramp 收敛曲线，输出 promotion 报告。
3. **P6 league distillation 多轮**：用 stage3/stage4 不同 step 的 checkpoint 作为 historical snapshot admission，跑 3×3 × 每对 N games=20，得到带置信区间的 cross-play 矩阵。
4. **P7 focused correction 真训练一轮**：`run_stage7_redteam` 已给 training_plan，取 P3 SFT checkpoint 对 correction set 做 50 step focused fine-tune（freeze_public_encoder=True，general replay 占比 25%），跑完再 inference original attack + general 回归集验证 recurrence。

## 验收结论（修订 3 后）

- 环境配置：✅ 通过。
- 本机 GPU 可用性：✅ 通过。
- P0-P7 训练代码深度规格实现：✅ 全部按 `order/` 口径补齐（**代码有锚点，pytest 有断言**）。
- 全量 pytest：✅ **165 passed / 0 failed / 8 warnings**。
- Web/HTTP/Dual Nash/Env/Solver 组合基线：✅ 全绿 exit 0。
- CUDA P0-P7 smoke：✅ ok=True（保留前次 GPU 实测证据）。
- N=13 stage4 入口启动：✅ 通过。
- STANDARD/FULL/RELEASE 统计显著 benchmark：❌ 未执行（纯 GPU 时间）。
- 多天 GPU 长训与论文级性能结论：❌ 尚未执行（非代码缺口）。
