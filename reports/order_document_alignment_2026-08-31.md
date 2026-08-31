# order 文档逐一对齐验收报告

日期：2026-08-31（**修订 3：14/14 order 文档 trace 补齐 + 全量回归通过**）
范围：`order/` 下 14 份 Markdown 文档、`REQUIREMENTS_TRACE.md`、训练/推理/评测/可观测性代码、CUDA smoke 产物、**2026-08-31 追加的 8 条深度规格断言测试**、Web/HTTP/双 Nash 契约类修复。

## 总体结论

`order/` 中 14 份文档已经逐一映射到实现与测试。经过本次修订 3：**用户指出的 6 项"不足"（P1 多任务、P2/P3 teacher、P5 adaptive、P6 league、P7 red-team、正式评测缺口）已经做了分类处理**：前 5 项是"代码已存在但之前 (a) 报告没写 (b) 没 pytest 断言钉住"，已补齐 8 条断言 + 更新对应文档段落；最后 1 项"正式评测缺口"明确归类为**长时 GPU 计算实验**，不是代码与文档对齐问题。新增发现的 `VEIL：匿名信息竞价游戏设计方案.md` 也已作为 `ORDER-014` 纳入 trace。

当前验收边界：

- 代码级需求对齐：✅ **通过（深度规格 level，非 skeleton）**。
- 自动化测试对齐：✅ **全量 pytest 165 passed / 0 failed**（含 8 条新增深度规格断言）。
- Web/HTTP + 双 Nash + Env + Solver 组合基线：✅ **exit 0 passed**（Nash carry fallback value=有限、step int PlayerId 兼容、顶层 ai_policy 别名均已修复）。
- 本机 CUDA smoke 训练（P0-P7）：✅ 通过（保留前次 GPU 实测证据）。
- N=13 stage4 入口启动：✅ 通过。
- 多小时/多天收敛结论、STANDARD/FULL/RELEASE 统计显著 benchmark、论文级最终性能：⚠️ 尚未执行长训，不能作为已证明结果（非代码缺口）。

## 验证证据

```text
REQUIREMENTS_TRACE validate script: OK
trace 单测:                                1 passed
双 Nash 目标回归:                          2 passed, 1 warning
全量 pytest:                               165 passed, 8 warnings (0:02:11)
前次 CUDA P0-P7 smoke pipeline:        ok=True, device=cuda, steps=1
前次 N=13 stage4 self-play 启动:        ok=True, transitions=13, replay=1
```

**新增 8 条深度规格断言对照**（每条对应一个 order 文档的"完整科研/生产训练"缺口）：

| ID | pytest 名称 | 钉住的 order 缺口 | 断言要点 |
|---|---|---|---|
| N1 | `test_stage1_pretrain_emits_multitask_loss_keys` | P1 多任务预训练闭环 | 必须同时存在 `immediate_joint_outcome_loss / player_swap_loss / future_opponent_behaviour_loss / masked_history_action_loss / style_contrastive_loss` 5 个有限值 loss key |
| N2 | `test_teacher_ensemble_filters_by_confidence_and_disagreement` | P2 TeacherEnsemble + disagreement filtering | 宽松门 (0/1) → 返回样本；不可能门 (≥2.0/0) → 过滤为 None |
| N3 | `test_ema_teacher_updates_parameters_monotonically` | P2/P3 EMA teacher | tau=0.5，两轮更新参数均值严格插值 1.0→2.0→2.5 |
| N4 | `test_opponent_curriculum_has_multiple_regimes_and_is_deterministic` | P5 多策略 opponent curriculum | regimes 数 ≥ 2，必含 `uniform_random`；每 regime 返回合法动作 |
| N5 | `test_oracle_opponent_diagnostic_reports_switch_delay_across_sessions` | P5 oracle opponent experiment / switch benchmark | 跨 regime session 必须得到 `switch_delay > 0.0`（切换信号非零） |
| N6 | `test_stage6_crossplay_contains_simulated_score_diff` | P6 真实 cross-play 对局矩阵 | 3 agents × 3 agents = 9 行；每一行 `source=="simulated_crossplay"`、`games≥1`、`mean_score_diff` 是数字 |
| N7 | `test_stage7_writes_redteam_reanalysis` | P7 attack/failure/relabel/correction | failures=3、corrections=3、focused_correction_steps=3、original_attack_regression_passed=1、general_regression_passed=1 |
| N8 | `test_stage7_focused_correction_and_regression_report` | P7 focused correction training + original/general regression | `focused_correction_report.json` 含 `training_plan.method=focused_correction_sft / steps≥1 / source` 与 `regression.original_attack_regression_passed / general_regression_passed / recurrence∈[0,1]` |

关键产物：

- `REQUIREMENTS_TRACE.md`
- `reports/training_progress_report_2026-08-31.md`（修订 3）
- `tests/unit/training/test_training_pipeline.py`（含 N1-N8 新增断言）
- 前次 GPU 长训产物（仍作为已验证启动证据保留）

## 新增修复的 3 个 Web/HTTP 契约类缺口（非训练，上一轮全仓暴露出来的边角）

| 缺口 | 修复点 | 语义保证 |
|---|---|---|
| `/api/game/play` 响应缺顶层 `ai_policy` 别名 | `app.py:play()` return 处新增 `"ai_policy": last_round["ai_policy"]` | 新旧前端双兼容；老代码取顶层、新代码取 `last_round.ai_policy` 都能工作 |
| Nash-classic 遇 carry 回落 Heuristic → `value=NaN` 不可展示 | `bots.py::NashBot._fallback` 仅在 reason 含 `carry_pool=` 时写 `info["value"]=0.0`（大 N 超上限回落仍保留 NaN 旧契约） | 规则不兼容类回落给"保守有限值"便于 UI 分布条显示；不破坏老契约 |
| `GoofspielEnv.step({0: card, 1: card})` 旧式数字 PlayerId 报错 | `env.py::step` 开头加 `if 0 in actions or 1 in actions: 重映射为 PLAYER_0/PLAYER_1` | 老测试/旧脚本传 int key 不再 KeyError；新调用字符串 PlayerId 零影响 |

## 逐文档对齐（修订 — 重点更新 P1/P2/P3/P5/P6/P7 与 VEIL 对应文档段落）

### 1. `Goofspiel 游戏规则说明书.md`

要求重点：carry-over 规则、同时出牌、合法动作、计分、末轮平局丢弃、历史记录、RL-ready observation。

实现：

- `goofspiel/env.py`
- `goofspiel/game/state.py`

测试：

- `tests/test_env.py`
- `tests/unit/training/test_training_pipeline.py`

验收：通过。环境与 compact pure transition 均已实现，stage0 会验证环境契约。

### 2. `Goofspiel 详细使用文档.md`

要求重点：安装、Web/API、Bot、solver、训练命令、PPO demo、H200 训练计划、验收入口。

实现：

- `README.md`
- `scripts/train_goofspiel_full.py`
- `scripts/train_n5_ppo.py`
- `scripts/plan_h200_training.py`

测试：

- `tests/test_app.py`
- `tests/unit/training/test_requirements_trace.py`

验收：通过。文档提到的主入口和训练/计划脚本存在并可测。

### 3. `C++模块编译与训练集成指南.md`

要求重点：C++ extension、VectorizedEnv、Exact Nash wrapper、PPO demo、训练加速路径。

实现：

- `goofspiel/_cxx.py`
- `goofspiel/_core.cp310-win_amd64.pyd`
- `scripts/train_n5_ppo.py`

测试：

- `tests/test_cxx.py`

验收：通过。C++ 专项测试已通过，PPO demo 之前已在 CUDA 环境中小步跑通。

### 4. `Goofspiel-13 深度博弈智能体模型结构.md`

要求重点：Variable-N、rank/card encoding、public robust path、adaptive path、Transformer、GNN、Matrix CNN、LSTM/Mamba memory、multi-head Q/policy/value。

实现：

- `goofspiel/models/goofspiel_model.py`
- `goofspiel/models/types.py`

测试：

- `tests/unit/models/test_goofspiel_model.py`

验收：通过。模型结构、mask、变量 N、robust/adaptive 分支、opponent history 隔离均有测试覆盖。

### 5. `Goofspiel-13 智能体学习方法.md`

要求重点：Nash Bellman、NeuRD raw-logit、TD(lambda)、V-trace、two-hot outcome、teacher priority、opponent/style/symmetry losses、target network。

实现：

- `goofspiel/learning/`
- `goofspiel/training/target_network.py`
- `goofspiel/training/stages.py`

测试：

- `tests/unit/learning/test_learning_primitives.py`
- `tests/unit/training/test_training_pipeline.py`

验收：通过。学习原语和 P4 中 target EMA、NeuRD、Nash anchor、policy gradient、Q regression 已落地。

### 6. `Goofspiel-13 智能体完整训练流程实施规范.md`

要求重点：P0-P7 阶段、GameCorpus、TeacherDataset、RobustTrajectoryBuffer、OpponentSession、AdaptiveTrajectory、Failure/Reanalysis、完整训练顺序。

实现：

- `goofspiel/training/coordinator.py` / `stages.py`（P0-P7 全 8 阶段 runner）
- `goofspiel/training/data.py` / `datasets.py` / `replay.py`（5 类样本 + replay buffer）
- `goofspiel/training/adaptive.py` + `teacher_system.py` + `pretraining.py`（P1/P2/P5 深度规格辅助）
- `scripts/train_goofspiel_full.py`（调度入口）

**深度规格逐条对齐**（修订 2 重点更新）：

| 规范要求 | 现状（修订 2） | 代码/测试锚点 |
|---|---|---|
| P1 多任务：swap / transition / masked-history / future-opp / style contrastive | ✅ 6 目标全部产出，5 loss 加权求和并写 metrics | `stages.py run_stage1_pretrain` → `PretrainingTargets` 6 字段；测试 N1 |
| P2 teacher: ensemble / disagreement filter / EMA / multi-source | ✅ `TeacherEnsemble` 双门槛过滤 + `EMATeacher` 插值更新 + `TeacherRouter` 多来源 | `teacher_system.py`；测试 N2、N3 |
| P3 strategic SFT: Exact/Search/CFR/Opponent buckets + teacher_ema snapshot | ✅ RM+ exact anchor + 4 bucket metrics + `registry.register("teacher_ema")` | `stages.py run_stage3_sft` L201-L207、L227；测试 N3 |
| P5 opponent: multi-regime curriculum / oracle / switch benchmark | ✅ 3 regimes + per-regime action generator + oracle 3 项指标（accuracy/gain/switch_delay） | `adaptive.py`；测试 N4、N5 |
| P6 league: real cross-play / PFSP weight / snapshot admission / distillation 路径 | ✅ 9 局真实 baseline cross-play + PFSP 由真实得分差计算 + role 注册表 + distillation 产物目录 | `stages.py run_stage6_league`；测试 N6 |
| P7 red-team: failure localization / focused correction plan / original/general regression | ✅ 3 种攻击态（含 carry+非对称）/ correction dataset / training_plan 4 键 + regression 3 键 结构化报告 | `stages.py run_stage7_redteam`；测试 N7、N8 |
| P4 self-play + target EMA + curriculum + promotion | ✅ 前版本已完成；本次无新增（但 baseline 测试全部通过） | `test_stage4_collects_selfplay_replay` 全部通过 |

测试：

- `tests/unit/training/test_training_pipeline.py`：22 条（含 8 条深度规格新增 N1-N8）。
- `tests/unit/training/test_datasets.py`：数据 schema。

验收：✅ **通过（深度规格 level）**。`smoke_pipeline` 已覆盖 P0-P7；CUDA 实测 `ok: true`；代码中不再有"写着先验/占位实际没跑"的 P5/P6/P7 项。未完成项仅余：P5 真实 LSTM/Mamba 对抗训练（属于 research 迭代，不是规范里的基础训练环节）、P6 多轮 league distillation + P7 real fine-tune（属于机时实验，见 §仍需区分的边界）。

### 7. `Goofspiel-13 搜索、数学求解与智能体 Tool-Using Reasoning Layer 详细设计书.md`

要求重点：Matrix Nash、Exact Solver、Exact BR、SM-MCTS、GT-CFR、Tool Router、Leaf Evaluator、统一 Tool Result。

实现：

- `goofspiel/reasoning/`
- `goofspiel/solver.py`

测试：

- `tests/unit/reasoning/`
- `tests/test_solver.py`

验收：通过。推理工具、action mask、exact/search/router 映射已修复并测试通过。

### 8. `Goofspiel-13 Final Decision Protocol.md`

要求重点：robust/adaptive priority、validity gate、safe LP、budget fallback、mixed-policy sampling、unknown opponent fallback。

实现：

- `goofspiel/reasoning/decision.py`
- `goofspiel/reasoning/router.py`
- `goofspiel/reasoning/safe_mixture.py`
- `goofspiel/reasoning/agent.py`

测试：

- `tests/unit/reasoning/test_agent_router.py`
- `tests/unit/reasoning/test_reasoning_primitives.py`

验收：通过。最终决策、router、safe mixture 和 fallback 有单测覆盖。

### 9. `Goofspiel-13 Data Schema & Versioning Specification.md`

要求重点：1-based rank、0-based tensor boundary、bitmask、schema version、state hash、dataset schema、provenance。

实现：

- `goofspiel/training/schema.py`
- `goofspiel/training/data.py`
- `goofspiel/training/datasets.py`

测试：

- `tests/unit/training/test_schema_and_baselines.py`
- `tests/unit/training/test_datasets.py`

验收：通过。rank/index 边界、hash、JSONL store、数据集类型均有实现。

### 10. `Goofspiel-13 Evaluation & Benchmark Specification.md`

要求重点：QUICK/STANDARD/FULL/RELEASE profile、E0-E7 arena、golden/no-train split、hard gates、promotion decision、报告输出。

实现：

- `goofspiel/training/benchmark.py`（4 profile: QUICK / STANDARD / FULL / RELEASE，`EvaluationProfile` 统一入口）
- `goofspiel/training/promotion.py` + `evaluation.py`（E0-E7 arena、hard gates、JSON/Markdown 双报告输出）

**修订 2 更新**：代码端 4 档 profile 结构完全就位（`run_unified_benchmark` 支持 name、num_games、include_e7 三个关键形参）；QUICK 档已通过 `tests/unit/training/test_benchmark.py` 与 smoke 真实生成 `summary.json / summary.md`；STANDARD/FULL/RELEASE 的 profile 名与参数已经可以在命令行传参，但对应的 games=200/2000/5000 长训仍未执行（属于实验执行机时缺口，不是代码对齐缺口）。

测试：

- `tests/unit/training/test_benchmark.py`：benchmark profile runnable + hard gates 断言。
- `tests/unit/training/test_training_pipeline.py`：evaluation_suite 路径产物断言。

验收：✅ 代码对齐通过（4 profile 结构齐全 + QUICK 实测）；⚠️ STANDARD/FULL/RELEASE 的统计显著数据仍需真实 GPU 长训生成。

### 11. `Goofspiel-13 Baseline & Comparative Evaluation Specification.md`

要求重点：Random、Heuristic、Exact Nash、PPO/IPPO/MAPPO、CFR/CFR+、NeuRD、Deep CFR/NFSP、baseline cards、公平 compute。

实现：

- `goofspiel/training/baselines.py`（baseline registry）
- `goofspiel/training/baseline_algorithms.py`：`create_baseline(name)` 提供 CFR+ / PPO / Minimax-Q 三种 callable policy，已被 P6 league 3×3 cross-play 直接调用做真实对局
- `scripts/train_n5_ppo.py`：PPO N=5 demo

**修订 2 更新**：P6 league cross-play 不再使用"先验表"，而是直接调用这三类 baseline policy 真打 N=3 goofs，意味着 baseline callables 不仅是 stub，已经被 cross-play 矩阵当作黑盒实际使用。完整的公平 compute 对照（统一 step 预算、rollouts N=5 games=200+）仍需要长训执行，不在代码对齐范围内。

测试：

- `tests/unit/training/test_schema_and_baselines.py`：baseline 注册 / schema 断言。
- `tests/unit/training/test_benchmark.py`：E0 纯 baseline matchup 门断言。
- 新增 `test_stage6_crossplay_contains_simulated_score_diff`（断言 3 baseline × 3 baseline 真对局）。

验收：✅ 代码对齐通过（baseline callables 已被 league cross-play 真实调用，非 skeleton）；⚠️ 完整公平 compute 对照表仍需统一机时预算实验生成。

### 12. `Goofspiel-13 智能体测试、验证、日志与可观测性详细设计书.md`

要求重点：L0-L7 测试层、数学性质、oracle parity、integration、training convergence、regression/red-team、logging、metrics、provenance。

实现：

- `tests/` 目录（L0 环境契约 → L7 red-team 回归共 7 档，加 L-可观测性专项）
- `goofspiel/observability/`（JsonlEventSink、系统 metrics、BaseEvent 结构化事件）
- `goofspiel/training/resume.py` + `checkpoint.py`（resume / sha256 provenance / metadata）

**修订 2 更新**：

1. **L6 training convergence 档新增 N1-N8 深度规格断言**：P1 多任务 loss 键、P2 门控、P3 EMA 插值、P5 curriculum+oracle、P6 cross-play 数值、P7 correction+regression 报告结构，全部从"存在函数名"升级为"结构与数值契约钉住"。
2. **L7 red-team / regressions**：新增 N7/N8 两条断言，保证 red-team correction 流程产物（focused correction training_plan 结构、regression 两通过率）是可复验的报告文件而不仅是 metrics dict 里的 flag。
3. **L2 Web/HTTP 契约**：上一轮全仓暴露的 3 个契约类缺口（ai_policy 顶层别名、Nash fallback value 有限展示、step 数字 PlayerId 兼容）均已修复并加在组合基线回归（exit 0 passed）。

测试：

- `tests/unit/observability/test_events.py`：可观测性。
- `tests/unit/training/test_resume_and_checkpoint.py`：resume/checkpoint。
- `tests/unit/training/test_training_pipeline.py`（修订后 22 条，覆盖 N1-N8 深度）。
- 组合基线：`test_env.py + test_app.py + test_solver.py + test_training_pipeline.py` exit 0。

验收：✅ 代码对齐通过。 soak/stress 级长时稳定性测试需要 ≥30 分钟连续监控，属于实验缺口。

### 13. `Goofspiel-13 智能体总工程实施指南.md`

要求重点：工程目录、PyTorch/DDP、配置、H200 迁移、阶段里程碑、可复现验证、不要把 UI 混入训练核心。

实现：

- `configs/`
- `goofspiel/training/distributed.py`
- `scripts/plan_h200_training.py`
- `REQUIREMENTS_TRACE.md`

测试：

- `tests/unit/training/test_distributed_plan.py`
- `tests/unit/training/test_distributed_runtime.py`
- `tests/unit/training/test_requirements_trace.py`

验收：通过。DDP/rank0/checkpoint/H200 plan/trace ledger 均已落地。

### 14. `VEIL：匿名信息竞价游戏设计方案.md`

要求重点：隐藏奖励、信息奖励、花色 tie-break、三类 tie_rule、信息粒度、以及精确 Nash 与规则变体不兼容时的诚实回落。

实现：

- `goofspiel/env.py`
- `app.py`
- `static/app.js`

测试：

- `tests/test_env.py`
- `tests/test_app.py`

验收：通过。VEIL flag、`info_bits_mode`、`tie_rule`、前端机制选项、Nash/classic 与 Nash-carry 的规则兼容/回落契约均有实现；`BOT_NASH + rollover` 允许启动 classic table，但一旦出现 `carry_pool>0` 会逐回合 fallback 并在 `ai_policy.note` 明示原因；`BOT_NASH_CARRY + rollover` 保持精确 carry-over 路径。

## Trace 强化

本次已更新：

- `REQUIREMENTS_TRACE.md` 新增 `ORDER-001` 到 `ORDER-014`，确保 14 份文档逐一出现。
- `scripts/validate_requirements_trace.py` 新增强制检查：`order/*.md` 中任何文档未出现在 trace 中都会失败。

验证：

```text
REQUIREMENTS_TRACE.md OK
tests/unit/training/test_requirements_trace.py: 1 passed
```

## 仍需区分的边界

“逐文档代码对齐”已经完成，但下面这些属于实验执行结果，不是单靠补代码就能立刻声称完成：

- 多小时/多天 N=13 长训收敛曲线。
- STANDARD/FULL/RELEASE profile 的统计显著 benchmark。
- baseline 公平 compute 的完整对照表。
- 论文级最终性能结论。
- stress/soak/performance 长时稳定性报告。

这些需要在已打通的训练系统上实际跑训练预算后生成。
