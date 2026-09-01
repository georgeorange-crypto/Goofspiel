# Goofspiel 训练全流程 — 文件级调用链与产物接线审计

> 本文档不是凭记忆写的，是逐行读代码追出来的。每一条"断线"结论旁边都标了证据来源
> （文件:行号）。日期 2026-09-01。
>
> **更新 2026-09-01：§4 的两条断线已修复。** 见 §6"修复记录"。
> **续（同日）：** 从"接线对"推进到"训练语义诚实"——两模式契约（严格全序列）、数据来源
> content-addressed、θ mutation 量化。见 §6.3–6.5。

## 0. 两个编排入口（这是你"开始训练"时真正跑的东西）

代码里有**两个**端到端入口，行为**不完全一致**——这点很重要：

| 入口 | 位置 | 用途 | θ 接线 | 下游接线（stage6/7/eval） |
|------|------|------|--------|--------------------------|
| `run_smoke_pipeline` | `stages.py:1737` | 最小端到端冒烟，写真实产物 | ✅ 显式串好 | ✅ **显式传入前段 checkpoint** |
| `TrainingCoordinator.run_full_sequence` | `coordinator.py:226` | 正式全量、多 GPU/torchrun | ✅ 自动接线 + 硬失败保护 | ⚠️ **不传前段 checkpoint**（见 §4） |

`run` 方法（`coordinator.py:194`）根据 `config.stage` 分派：
- `stage in ("all","full","full_sequence")` → 走 `run_full_sequence`（全量）
- 单个 stage 名 → 只跑一段，但会 `_discover_parent_on_disk` 自动从磁盘捞父 checkpoint（`coordinator.py:214`）

---

## 1. 冻结的阶段顺序（单一事实来源）

`STAGE_SEQUENCE`（`distributed.py:10`）：

```
stage0_verify → build_corpus → stage1_pretrain → stage2_semi_supervised
  → stage3_sft → stage4_robust_rl → stage5_adaptive → stage6_league
  → stage7_redteam → evaluate → smoke_pipeline
```

`validate_stage_sequence`（`distributed.py:62`）强制这个顺序不被打乱。
`run_full_sequence` 会 **跳过** `smoke_pipeline`（`coordinator.py:248`，否则重复跑一遍）。

---

## 2. θ 继承链 —— 这是你最担心的部分，结论：**是真的，且自动**

### 2.1 单一事实来源（`coordinator.py:37-52`）

```
stage1_pretrain ──θ──▶ stage3_sft ──θ──▶ stage4_robust_rl ──θ──▶ stage5_adaptive
```

```python
THETA_PRODUCERS = ("stage1_pretrain", "stage3_sft", "stage4_robust_rl", "stage5_adaptive")
THETA_PARENT = {
    "stage3_sft":       "stage1_pretrain",   # 注意：跳过 stage2（stage2 是数据阶段，无 θ）
    "stage4_robust_rl": "stage3_sft",
    "stage5_adaptive":  "stage4_robust_rl",
}
```

### 2.2 自动接线怎么做的（`coordinator.py:247-285`）

`run_full_sequence` 每跑完一段，把它产出的 checkpoint 路径存进 `produced[stage]`；
下一段若在 `THETA_PARENT` 里，就从 `produced[parent]` 取父 checkpoint 作为
`init_from_checkpoint`。**关键安全装置**（`coordinator.py:257-262`）：

```python
init_ckpt = produced.get(parent)
if init_ckpt is None:
    raise RuntimeError(
        f"θ auto-wiring broken: stage {stage!r} must inherit weights from {parent!r}, "
        f"but {parent!r} produced no checkpoint. Refusing to train {stage!r} from scratch."
    )
```

**这正是你担心的场景的护栏**：如果上一级没产出 checkpoint，下一级不会"偷偷从零训练"，
而是**直接崩**。这是设计上刻意的（防止"每个阶段自己孤立训练"这个正是要防的失败模式）。

### 2.3 θ 到底是不是真加载了（不是空操作）

`init_from_checkpoint`（`checkpoint.py:119-155`）做三件事，全是真的：
1. `load_checkpoint(path)` 读盘 + **SHA256 校验**（`checkpoint.py:100-104`，manifest 不匹配就抛错）
2. **架构签名校验** `model_config_hash`（`checkpoint.py:139-146`）——防止 θ 被静默 reshape
3. `model.load_state_dict(payload["model_state"], strict=strict)` —— 真正把权重灌进去

> ⚠️ **一个隐患**：架构校验是 `if want is not None`（`checkpoint.py:142`）。若某 checkpoint
> 存的时候没写 `model_config_hash`，校验会**静默跳过**。已核实**四个 θ 阶段全都 stamp 了**
> 这个字段（`stages.py:256, 429, 768, 1121`），所以当前链路安全。但这是个"未来若新增阶段忘了
> stamp 就会静默失效"的脆弱点。

### 2.4 init vs resume 两个语义严格分离（`checkpoint.py` + `stages.py:110`）

- `init_from_checkpoint`：**阶段过渡**——只拷 θ，重置优化器/step=0（新阶段是新的优化问题）
- `resume_checkpoint`：**崩溃恢复**——恢复 model + 每个优化器 + target 网络 + global_step
- 两者互斥，同时传会抛错（`stages.py:126-131`）。混淆这两者正是 Phase 3.1 要防的 bug。

---

## 3. 逐阶段：输入 → 做什么 → 输出

| # | 阶段 | 函数位置 | 消费（输入） | 产出（输出） | θ 父 |
|---|------|---------|-------------|-------------|------|
| 0 | stage0_verify | `stage0_verify.py` | 无 | `stage0_verify/` 报告（模块可导入性 + 算法自检） | — |
| — | build_corpus | `corpus.py:13` | 无 | `data/game_corpus.jsonl` ⚠️**无人消费，见 §4.1** | — |
| 1 | stage1_pretrain | `stages.py:156` | `sample_reachable_states` 现场生成（**非** corpus） | `checkpoints/stage1_pretrain.pt` + registry `latest` + coverage | 无（链头） |
| 2 | stage2_semi_supervised | `stages.py:442` | `sample_reachable_states` 现场生成 | `data/teacher_dataset.jsonl`（CFR/SEARCH/EXACT/PSEUDO 四源标注） | 无（数据阶段） |
| 3 | stage3_sft | `stages.py:291` | **读 `data/teacher_dataset.jsonl`**（`stages.py:337-339`；缺失则现场求解，源计数标 `fallback`） | `checkpoints/stage3_sft.pt` | stage1 |
| 4 | stage4_robust_rl | `stages.py:602` | θ←stage3；自博弈 rollout | `checkpoints/stage4_robust_rl.pt`（+target 网络在 extra） | stage3 |
| 5 | stage5_adaptive | `stages.py:967` | θ←stage4；对手条件化 | `stage5_adaptive.pt` | stage4 |
| 6 | stage6_league | `stages.py:1246` | **role_checkpoints（可选）** | `league/registry.json` + 交叉对弈表 | —（见 §4.2） |
| 7 | stage7_redteam | `stages.py:1409` | **init_from_checkpoint（可选）** | `redteam/stage7_corrected.pt` + 回归报告 | —（见 §4.2） |
| E | evaluate | `stages.py:1601` | **checkpoint（可选）** | benchmark 报告 | —（见 §4.2） |

### 关键数据接线（stage2 → stage3）验证过是真的

`run_stage3_sft` 默认 `teacher_dataset_path = out_dir.parent/"data"/"teacher_dataset.jsonl"`
（`stages.py:337-338`），正好是 stage2 的写出路径。loss 真的依赖文件内容（按 teacher_source
分别 KL），不是别名成一个数（`stages.py:342-365`）。**这条接线是真的。**

---

## 4. 发现的实在断线 / 隐患（证据在此）

### 4.1 🔴 `game_corpus.jsonl` 是死产物

- `generate_random_game_corpus` 在两个入口都被调用生成文件
  （`coordinator.py:117`、`stages.py:1758`）。
- **但全 training 目录里没有任何 `load`/读取它的地方**——grep `game_corpus|load.*corpus|corpus_path`
  只命中"写"的两处，零个"读"。
- stage1/stage2 都用 `sample_reachable_states` **现场生成**训练状态（`stages.py:186, 461`），
  根本不碰这个 corpus。
- **结论**：`build_corpus` 阶段产出的语料没喂给任何下游。要么是历史遗留死枝，要么 stage1
  本应从它读取却没接上。**这是一条真实断线**（但不影响 θ 链，因为 stage1 自给自足）。

### 4.2 🟡 正式 `run_full_sequence` 下游接线比 smoke 松

`_dispatch_stage`（`coordinator.py:172-180`）里：

```python
if stage == "stage6_league":
    metrics = run_stage6_league(out_dir=self.artifact_dir)          # ← 不传 role_checkpoints
if stage == "stage7_redteam":
    metrics = run_stage7_redteam(out_dir=self.artifact_dir)         # ← 不传 init_from_checkpoint
if stage == "evaluate":
    payload = run_evaluation_suite(out_dir=self.artifact_dir, num_games=16)  # ← 不传 checkpoint
```

对比 smoke（`stages.py:1784-1800`）**显式**传了 `role_checkpoints={robust:stage4,...}`、
`init_from_checkpoint=stage4.checkpoint`、`checkpoint=stage4.checkpoint`。

后果（都不是崩溃，是"降级/自给"，但确实是产物没串上）：
- **stage6**：收不到真实 stage3/4/5 快照 → 自己 mint 三个 role-seeded 快照
  （`stages.py:1264-1265`）。联赛打的是**新铸的种子模型，不是本次训练链的产物**。
- **stage7**：`init_from_checkpoint` 为空 → 自己用 `run_stage1_pretrain` 铸一个种子
  （`stages.py:1480-1484`）。红队修正**不是修在 stage4 鲁棒骨干上**，而是修在一个临时种子上。
- **evaluate**：无 checkpoint → E2/E6 退回 heuristic 参考，G2=None（未跑）。**正式全量跑完，
  benchmark 却没评到本次训练出的模型。**
- **且 `run_full_sequence` 完全不调用 `run_axis_promotion_selection`**（只有 smoke 调，
  `stages.py:1807`）→ 正式全量**不产 best_* 别名**，Phase 5 的成果在正式路径上没接。

> 这几条不违反 θ 链的正确性（stage1→3→4→5 仍然串得死死的），但确实符合你的担心：
> **"上一级的训练产物没有正确给到下一级作为开始"** —— 在 stage6/7/eval 这三段上，正式路径
> 是成立的。smoke 路径则没有这个问题。

---

## 5. 一句话总结

- **核心 θ 学习链（1→3→4→5）**：真加载、真校验、全自动、断链即硬崩。**放心。**
- **数据链 stage2→stage3**：真接上了。**放心。**
- **build_corpus**：~~产物没人读，死枝。~~ **已修（§6.1）** — P1 现在训练在 corpus 状态上。
- **正式 run_full_sequence 的 stage6/7/eval + 轴选择**：~~没接本次训练产物~~ **已修（§6.2）**。

---

## 6. 修复记录（2026-09-01）

### 6.1 build_corpus 接入为 P1 训练数据

- 新增 `_load_corpus_states(path)`（`stages.py`，紧邻 `_load_teacher_dataset`）：从
  `game_corpus.jsonl` 重建 `GameState`，跳过终局态和无合法动作的状态。
- `run_stage1_pretrain` 新增 `corpus_path` 参数：有 corpus 时按 batch 轮取 corpus 状态训练；
  无则回退到 `sample_reachable_states`（阶段单独跑时仍可用）。metrics 增加 `corpus_states` /
  `trained_on_corpus` 两个可审计字段。
- 两个入口都接上：smoke（`stages.py`）和 coordinator 的 `stage1_pretrain` 分派
  （`coordinator.py`）都传 `corpus_path = artifact_dir/data/game_corpus.jsonl`。
- **测试**（`test_downstream_wiring.py::test_stage1_trains_on_corpus_states_not_synthetic`）：
  独立重算 corpus 状态数，要求与 stage1 上报的 `corpus_states` 完全一致——不是读字段，是重算。

### 6.2 run_full_sequence 下游接线

- 新增 `TrainingCoordinator._resolve_checkpoint(stage, produced)`：优先取内存 `produced` 映射
  （全序列路径），回退到磁盘约定路径（单阶段分进程路径），都没有才返回 `None`（诚实降级，不假装）。
- `_dispatch_stage` 新增 `produced` 参数，并改写三段：
  - **stage6**：解析 P4/P3/P5 → `role_checkpoints={ROLE_ROBUST, ROLE_AGGRESSIVE, ROLE_EXPLOITER}`
  - **stage7**：解析 P4 → `init_from_checkpoint`（修正修在本次 P4 骨干上）
  - **evaluate**：解析 P4 → `checkpoint`（E2 变真实模型对弈，G2 变计算判定）
- `run_full_sequence` 末尾补上 `run_axis_promotion_selection`（P3/P4/P5/P7 四候选），
  并把 `axis_selection` 写进 `full_sequence_summary.json`。之前只有 smoke 会做轴选择。
- 单阶段 `run()` 路径 `produced=None`，走磁盘发现——与既有 `_discover_parent_on_disk` 行为一致。
- **测试**（`test_downstream_wiring.py::test_full_sequence_wires_downstream_to_real_checkpoints`）：
  跑完整序列后，(a) 重载 stage7 corrected 的 metadata，断言 `init_checkpoint_id` 解析到本次 P4；
  (b) 对 P3/P4/P5 与联赛引用的 checkpoint **重算 SHA256** 求交集，证明联赛打的是本次产物；
  (c) 读 benchmark `summary.json` 断言 E2 `source==trained_model_vs_random` 且 G2 非 None；
  (d) 断言 best_* 别名已注册且文件存在。全部重执行事实，不读状态位。

> 注意：`_resolve_checkpoint` 找不到父 checkpoint 时返回 `None`。**这条降级路径现在只对
> 单阶段 `run()`（Standalone 模式）开放**——见 §6.3，全序列已改为严格模式。降级路径绝不
> 伪造"消费了 checkpoint"。

### 6.3 Standalone vs Full-sequence 两模式契约（优先级①，2026-09-01）

上一版 §6.2 让**所有**路径都走"诚实降级"。但在全序列里，下游阶段（league/redteam/evaluate）
按契约必须消费本次的真实产物；缺产物却静默铸一个种子、再把结果标成"本次 league/eval"，
正是本次整改要消灭的那类不诚实。于是同一段 dispatch 代码按模式分叉:

- `_dispatch_stage(..., strict: bool)`。单阶段 `run()` 传 `strict=False`（Standalone，缺产物
  诚实降级、如实汇报 `evaluated_checkpoint=None`）；`run_full_sequence` 传 `strict=True`。
- 新增 `_require_checkpoint(stage, produced, consumer=)`：strict 下解析不到必需产物即
  `RuntimeError`，且**点名**缺的角色（"robust backbone (P4)"）与生产阶段，便于运维定位要重建哪个产物。
- strict 覆盖三段:stage6 要 P4/P3/P5 三个角色、stage7 要 P4、evaluate 要 P4。θ 链（1→3→4→5）
  本就在 `run_full_sequence` 里硬失败;这把同样的拒绝延伸到下游非 θ 阶段。
- **测试**（`test_strict_lineage.py`）:(a) 破坏**终端** θ 阶段 P5 的 checkpoint（P5 无下游 θ 继承，
  故 θ 链不受影响），全序列在 stage6 因缺 P5 exploiter 角色而 `RuntimeError`——隔离了下游契约
  与 θ 硬失败两条路径;(b) 直接 strict dispatch 空 produced，断言报错点名 consumer+角色;
  (c) 反面:同样缺 P4 在 Standalone `--stage evaluate` 下**不**报错,如实降级。

### 6.4 数据来源 content-addressed（优先级④，2026-09-01）

`teacher_dataset_ids` 只记路径字符串——同名文件改了内容,lineage 看不出来。新增:

- `CheckpointMetadata.datasets: list[dict]`,每项 `{path, sha256, num_samples, role}`;
  `checkpoint.dataset_provenance(path, num_samples, role)` 算**文件字节** SHA256。
- 接入 stage1（`game_corpus` 角色）与 stage3（`teacher_dataset` 角色）的存档 metadata。
- **测试**（`test_dataset_provenance.py`）:存的 sha256 == 独立重算的文件 sha256、num_samples ==
  独立重数;并**改数据同路径重跑**,断言新 sha256 与旧不同——这正是路径字符串给不了的性质。

### 6.5 θ mutation 量化（优先级②,2026-09-01）

`test_checkpoint_chaining.py` 只断言"θ 变了"（`any(not torch.equal)`）——单个 ULP 抖动即可通过,
分不清"真训练了"和"冻结的父本挪了一个权重"。新增 `test_theta_mutation.py`:从两个 checkpoint 的
**加载字节**自算 `parameter_l2_delta`/`changed_parameter_ratio`/`cosine_similarity`,断言:
移动实在（l2>1e-4、changed_ratio>10%）、方向仍是父本的孩子（cos>0.90）;以**全新随机模型**为对照锚
（l2 远大于、cos 远小于训练后的孩子）,让阈值有意义而非拍脑袋。并加单调性:训得越久 θ 离同一 init 越远。

> 优先级③（optimizer reset/resume 契约）经审计已被 `test_checkpoint_chaining.py::
> test_init_copies_theta_only_but_resume_restores_optimizer` 充分覆盖(init:θ拷贝+优化器空;
> resume:θ+优化器+target+global_step 全恢复)。重写会造出重言式测试,故**不重复**。

### 6.6 能力保持检测 + 血缘树一致性（优先级⑤⑥,2026-09-01）

**⑤ capability retention**（`model_eval.capability_retention`,`test_capability_retention.py`）:
父/子 checkpoint 在**同一批能力**上过诚实测温计（每个 N、每个对手的胜率），子比父下跌超过
`win_rate_tolerance` 即判定该能力"崩塌"(catastrophic forgetting)。`retained` 是所有能力的 AND。
三层再执行测试:自比自 delta 恒为 0(不无中生有回归)、受控胜率下只有真崩塌被标记(阈值逻辑)、
真父子上 `regressions` 集合 == 对同一 deltas 独立手算 `>tol` 规则(诚实执行自己的契约)。

**⑥ lineage tree**（`lineage.build_lineage_tree`,`test_lineage_tree.py`）:真链一致且有序;
把父文件原地重跑成**合法但不同**的字节后,子的 `parent_checkpoint_sha256` 不再匹配 →
树报 `parent_content_changed`(裸 parent-id 字符串看不见的漂移);父缺席 → `dangling_parent`。
`run_full_sequence` 摘要盖 `lineage_consistent` 绿章。

> **⑤ 挖出的真 bug（已修）**:`HeuristicBot._choose` 用**全局** `np.random.choice` 采样邻牌,
> 无视 `create_bot(seed=...)` 播下的 `self._rng`,导致**同 seed 同 checkpoint 的对局不可复现**
> ——整条测温计对启发式对手都带隐性噪声。改为 per-instance `self._np_rng =
> np.random.default_rng(self._rng.randint(...))`(与 NashBot 一致)。修后自比自 delta 精确为 0。
> 这正是"测试必须再执行事实、不读回存量字段"的价值:自比自 `==0.0` 断言不是过严,是它抓到了真缺陷。

---

## 7. 可观测性 / 日志（2026-09-01）

在此之前，`training/*` 全模块**零日志**——没有 `logging`、没有 `print`、没有计时、没有进度。
跑一次正式全序列，控制台什么都不显示，唯一的可观测手段是事后翻产物落盘。本节记录补上的
**双通道日志**。

### 7.1 双通道架构 —— `goofspiel/observability/run_logger.py::TrainingLogger`

一个对象，两条**同步**通道，每个语义方法**同时**写两边（控制台行不会与机器事件流漂移）:

| 通道 | 载体 | 位置 | 用途 |
|------|------|------|------|
| ① 人类可读 | `logging.Logger("goofspiel.training")` + `StreamHandler`(控制台) + `FileHandler` | `artifact_dir/run.log` | 运维现场肉眼追踪、`tail -f` |
| ② 结构化 | `JsonlEventSink` | `artifact_dir/events/run.jsonl` | 机器解析、指标提取、回归断言 |

formatter 统一为 `时间 | 级别 | 阶段 | 消息`。控制台 handler 只收 `INFO+`，文件 handler 收
`DEBUG+`(含 `SYSTEM` 指标行)。

**四条设计不变量**:
- **rank0-only**:构造时传 `is_rank0`（coordinator 用 `current_runtime().is_rank0` 从 torchrun
  环境变量轻量判定，无需 `torch.distributed` init）。非 rank0 的每个方法都是 no-op，多卡不刷屏、
  不抢写文件。
- **宿主安全**:命名 logger + 显式 handler + `propagate=False`，**绝不**碰 root logger 或
  `basicConfig`——不污染宿主进程的日志配置。
- **幂等**:重复为同一 `artifact_dir` 构造会先清掉旧 handler 再装，重入不会每行打印两遍。
- **处处可选**:所有 stage 的 `logger` 参数默认 `None`。阶段单独跑(Standalone)时零日志依赖、
  零行为变化——日志纯属编排层附加物。

### 7.2 事件类型（`event_type`）

| event_type | 何时发 | 关键 payload |
|------------|--------|-------------|
| `RUN_START` | 编排开始 | `config`(整份 resolved config) |
| `RUN_END` | 编排结束 | `summary`、`elapsed_s` |
| `STAGE_START` | 每阶段起 | `stage`、`init_from`、`inherited_from` |
| `STAGE_END` | 每阶段止 | `stage`、`ok`、`metrics`、`checkpoint`、`elapsed_s`（`ok=False` 时 severity=ERROR） |
| `THETA_WIRED` | θ 产物接线时 | `child`、`parent`、`init_checkpoint`、`produced_checkpoint` |
| `STEP_METRICS` | 每 ~10% 步(见下) | `stage`、`step`、`total`、`metrics`(当步 loss 分量) |
| `CHECKPOINT_SAVED` | rank0 存档后 | `stage`、`path`、`global_step` |
| `LINEAGE_VERDICT` | 全序列末尾 | `consistent`、`inconsistencies`、`order`（不一致时 severity=ERROR） |
| `SYSTEM_METRICS` | run 起止 | `collect_system_metrics()`(cpu/mem/gpu) |
| `WARNING` / `EXCEPTION` | `warn()` / `error(exc=)` | `error(exc=)` 走 `emit_exception_event` 拼完整 traceback |

**每步日志的采样**（`stages._should_log_step`）:约每 10% 进度打一行、末步必打。少步数(冒烟/测试)
逐步全打，长跑保持 `run.log` 可读又能画出 loss 轨迹——把**过去算完即丢**的 per-step 标量真正记下来
(stage1 的 `loss/loss_q/swap/opp`；stage3 的 `loss/q_loss/pi_loss`；stage4 的
`q/actor/pg/entropy/curriculum_n`；stage5 的 `nll/acc/ece/adaptive_grad_norm`)。

### 7.3 接入点

- **coordinator**:`run()` 与 `run_full_sequence()` 在 `write_resolved_config()` 后
  `_build_logger()`，发 `run_start`；循环里每阶段 `stage_start`/计时/`stage_end`，θ 产物
  append 时 `theta_wired`；θ 接线 `RuntimeError` 与 strict `_require_checkpoint` 失败**先
  `logger.error` 再 raise**（失败在控制台+run.log+JSONL 三处留痕）；末尾 `lineage_verdict` +
  `run_end`。`full_sequence_summary.json` 增补 `run_log`/`event_log`/`event_count` 三个字段
  （对齐 smoke 的 `event_log` 惯例，产物从 summary 即可发现）。
- **四个 θ 阶段**(stage1/3/4/5):加可选 `logger` 参，训练循环内按采样发 `step_metrics`，rank0
  存档后发 `checkpoint_saved`。
- **保留** stage4 私有的 `JsonlEventSink(stage4_robust_rl.jsonl)`(细粒度自博弈事件)与 smoke 的
  sink——它们是阶段私有细粒度流，与 run 级 logger **并存**，不删。

### 7.4 怎么看

```bash
# 人类可读，实时跟：
tail -f <artifact_dir>/run.log

# 结构化，提某类事件（例：所有 STEP_METRICS 的 loss 轨迹）：
python -c "import json,sys; [print(e['payload']) for l in open(sys.argv[1],encoding='utf-8') \
  if (e:=json.loads(l))['event_type']=='STEP_METRICS']" <artifact_dir>/events/run.jsonl
```

### 7.5 测试

`test_full_sequence_e2e.py::test_full_sequence_produces_every_wired_artifact`（`integration` 标记）
跑完整全序列后，除逐阶段再执行校验每个产物(加载 checkpoint、重算父 SHA 匹配子记录、stage2 数据
非空、league/redteam/eval 产物合法、summary 从盘回读且 `lineage_consistent`)外，**把"日志确实写出"
本身当作再执行事实校验**:`run.log` 存在且含每个阶段名;`events/run.jsonl` 每行合法 JSON、事件弧为
`RUN_START → 各 STAGE_START/STAGE_END → LINEAGE_VERDICT → RUN_END`，每 θ 阶段的 `STEP_METRICS`
带非空 metrics，三个继承阶段都有 `THETA_WIRED`。不读状态位——重新解析日志文件本身。
