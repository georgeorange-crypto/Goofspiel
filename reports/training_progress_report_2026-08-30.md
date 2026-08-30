# Goofspiel-13 训练部分进度报告

日期：2026-08-30  
审计范围：`order/` 下 13 份规格文档、`goofspiel/training`、`goofspiel/models`、`goofspiel/learning`、`goofspiel/reasoning`、`scripts/train_*.py`、`configs/*`、`tests/*`。

## 结论

当前项目还不适合直接开启“完整 Goofspiel-13 生产训练流程”。

当前代码已经具备训练入口、模型骨架、学习原语、数据 schema、stage-gated coordinator、轻量验证和若干非 torch 阶段 runner；但离 `order/` 中定义的完整训练系统还有明显差距。主要阻塞是：

1. 本机全局 Python 的 PyTorch 已损坏，`import torch` 因 `c10_cuda.dll` 依赖缺失失败，导致神经模型、PPO demo、stage1/stage3/stage4 训练不能运行。
2. C++ 扩展可以 import，但 C++ 专项测试未通过，尤其是 C++ Nash policy 的数学不变量失败，不能作为完整训练的可信 fast backend。
3. 训练阶段多数仍是“可调用骨架/冒烟 runner”，不是完整 P1-P7 生产训练闭环。
4. 完整 benchmark、baseline、公平评测、日志、MLflow、golden dataset、promotion gate、checkpoint resume validation 等科研训练必需设施尚未完整落地。

因此，目前只能启动“小规模功能验证/冒烟训练”，不能启动正式完整训练。

## 已具备的训练基础

- `scripts/train_goofspiel_full.py` 存在统一训练入口，支持 `stage0_verify`、`build_corpus`、`stage1_pretrain`、`stage2_semi_supervised`、`stage3_sft`、`stage4_robust_rl`、`stage5_adaptive`、`stage6_league`、`stage7_redteam`、`evaluate`。
- `goofspiel/training/coordinator.py` 已实现 stage 路由，并会写 `resolved_config.json`。
- `goofspiel/training/stages.py` 已实现各阶段 runner：
  - `stage0_verify`：环境、solver、schema、teacher priority 检查。
  - `build_corpus`：生成随机 game corpus JSONL。
  - `stage1_pretrain`：用 immediate joint outcome 做小规模 Q 预训练。
  - `stage2_semi_supervised`：通过 teacher router 生成 teacher dataset。
  - `stage3_sft`：使用 RM+ matrix teacher 做战略 SFT 骨架。
  - `stage4_robust_rl`：NeuRD + Nash anchor 的轻量 robust RL runner。
  - `stage5_adaptive`：写 adaptive gate 报告，但当前明确 blocked until calibration。
  - `stage6_league`：建立 bootstrap league registry。
  - `stage7_redteam`：写 failure/correction schema smoke。
- `goofspiel/models/goofspiel_model.py` 已有主模型结构骨架，包含 public backbone、Transformer、GNN、Matrix CNN、LSTM、Mamba-style memory、robust/adaptive heads。
- `goofspiel/learning` 已有 NeuRD、RM+ matrix solver、lambda return、joint V-trace、teacher priority、opponent/style/symmetry loss 等学习原语。
- `goofspiel/training/data.py` 和 `schema.py` 已有 JSONL store、hash、rank/index 边界转换、若干样本 dataclass。
- `goofspiel/reasoning` 已有 matrix tool、exact tool、SM-MCTS/GT-CFR 风格入口、final decision、safe mixture、router/agent 等模块。
- `configs/training/pipeline.yaml` 已声明全部训练阶段 enabled。

## 当前实测结果

在系统 Python 3.10 环境下：

- `python -m pytest -q`：`106 passed / 3 skipped / 9 failed`。
- 失败集中在 `tests/test_cxx.py`：
  - 8 条失败：测试期待 `ComplexityReport.C_N`，但 Python dataclass 当前字段为 `chance_states`。
  - 1 条失败：`TestCppNashSolver.test_n3_root_value_zero_and_policy_nash_invariant` 中 C++ Nash policy 不满足 `M y <= V` 不变量。
- `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`：失败，错误为加载 `c10_cuda.dll` 或其依赖失败。
- `python scripts/train_goofspiel_full.py --stage stage0_verify ...`：通过，返回 `ok: true`。
- `python scripts/train_goofspiel_full.py --stage build_corpus --num-corpus-games 3 ...`：通过，能写 corpus 样本。
- `python scripts/train_goofspiel_full.py --stage stage1_pretrain ...`：失败，原因是 torch 无法 import。
- `python scripts/train_n5_ppo.py ...`：失败，原因是 torch 无法 import。
- 单独跑非 C++/unit 组合测试时曾在 `tests/test_app.py` 双 Nash carry 场景触发 Windows access violation，说明当前 C++/Web/Nash 组合仍有稳定性风险。

## 与完整训练规格的差距

### P0：环境、Solver 与数据系统校准

部分完成。`stage0_verify` 可运行，环境和 Python solver 基础通过。  
未完成：GPU matrix solver verification、C++/Python parity 全绿、完整 calibration 和正式 report。

### P1：Game Representation Pre-training

部分完成。已有 immediate joint outcome 的最小预训练 loop。  
未完成：player swap、known-transition prediction、masked history action、future opponent behaviour、style contrastive、多任务 loss、退出条件评估。

### P2：Teacher-Based Semi-Supervised Learning

部分完成。已有 teacher router 和 teacher dataset 生成。  
未完成：teacher ensemble、disagreement filtering、EMA teacher、exact anchoring 全流程、pseudo-label 质量报告。

### P3：Strategic SFT

部分完成。已有 RM+ teacher 的轻量 SFT。  
未完成：Exact/Search/CFR/Opponent 多来源 SFT 数据、teacher priority 全落地、SFT 退出条件。

### P4：Game-Theoretic RL Post-training

部分完成。已有 NeuRD + Nash anchor 的轻量 runner。  
未完成：真正 self-play、replay/buffer、target network、progressive curriculum、remaining-horizon curriculum、N=13 扩展、在线 evaluator。

### P5：Opponent-Adaptive Post-training

未完成为主。当前 `stage5_adaptive` 明确写出 `BLOCKED_UNTIL_CALIBRATION`。  
缺 opponent curriculum、session construction、LSTM/Mamba 分工训练、oracle opponent experiment、adaptive safety gate。

### P6：League

仅 bootstrap registry。  
缺 robust/aggressive/exploiter lineage、PFSP sampling、historical snapshot admission、cross-play、distillation。

### P7：Red-Team Continual Correction

仅 schema smoke。  
缺 attack、failure detection/localization、strong relabel、focused correction、original attack regression、general regression。

## 是否可以现在开训练

可以开：

- `stage0_verify`
- `build_corpus`
- 修复 torch 后的小步 `stage1_pretrain`
- 修复 torch 后的小步 `stage3_sft`
- 修复 torch 后的小步 `stage4_robust_rl`
- 修复 torch 后的小规模 PPO demo

不建议开：

- 完整 N=13 生产训练
- 长时间 GPU 训练
- 使用 C++ Nash 作为可信 teacher 或 fast backend 的训练
- 用当前结果做论文/报告主结论
- 启动完整 league / adaptive / red-team 闭环

## 下一步执行计划

1. 建立项目隔离 `.venv`，安装 CUDA 可用的 PyTorch，不再依赖已损坏的全局 Python。
2. 在 `.venv` 中安装项目依赖并验证：
   - `import torch`
   - `torch.cuda.is_available()`
   - CUDA tensor 运算
   - `python -m pytest tests/unit/models tests/unit/learning tests/unit/reasoning -q`
3. 修复 C++/Python complexity report 字段兼容问题。
4. 定位并修复 C++ Nash policy 不变量失败。
5. 把训练 runner 从“分散小阶段”补成一个可顺序执行的完整 smoke pipeline：
   - stage0 verify
   - build corpus
   - stage1 pretrain
   - stage2 teacher dataset
   - stage3 SFT
   - stage4 robust RL
   - evaluate
   - 写统一 training report
6. 对仍不能在本机完整实现的科研级部分，保留明确 blocked 状态和可追踪 TODO，不假装完成。

## 复验更新：本机环境与 smoke 训练已打通

更新时间：2026-08-30

已完成环境配置：

- 新建项目隔离环境：`.venv`
- 安装项目依赖：`pip install -r requirements-train.txt`、`pip install -e .`
- 安装 CUDA 版 PyTorch：`torch 2.13.0+cu126`
- 本机 CUDA 环境变量：`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- GPU 验证：`torch.cuda.is_available() == True`
- GPU 型号：`NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA tensor 运算验证通过

已完成代码修复：

- 修复 reasoning/search/exact_br/router 中 `immediate_q_matrix()` 返回值未解包的问题。
- 修复 compact action matrix 到真实牌面 rank 的映射，避免在非连续合法牌状态下错位。
- 修复 `exact_tool` teacher 调用参数与策略映射。
- 修复 NeuRD loss 的梯度测试，使 raw logits 在 regret 方向上产生有效梯度。
- 补齐 `ComplexityReport.C_N/L_N/E_N` 兼容属性。
- 新增 `smoke_pipeline` 训练 stage，把 stage0、corpus、P1、P2、P3、P4、evaluate 串成单次可复验流程。

最新实测结果：

- 训练相关单元测试：`32 passed, 3 warnings`
- C++ 专项测试：`18 passed`
- 全仓测试：`143 passed, 4 warnings`
- CUDA smoke pipeline：通过，返回 `ok: true`

本次 smoke pipeline 产物：

- 统一摘要：`artifacts/venv_smoke/training_smoke_summary.json`
- 事件日志：`artifacts/venv_smoke/events/training_smoke.jsonl`
- P1 checkpoint：`artifacts/venv_smoke/checkpoints/stage1_pretrain.pt`
- P3 checkpoint：`artifacts/venv_smoke/checkpoints/stage3_sft.pt`
- P4 checkpoint：`artifacts/venv_smoke/checkpoints/stage4_robust_rl.pt`
- quick benchmark：`artifacts/venv_smoke/evaluation/reports/quick/summary.json`、`artifacts/venv_smoke/evaluation/reports/quick/summary.md`

## 最新结论

现在可以开启“本机 CUDA 上的小规模完整 smoke 训练流程”。可用命令：

```powershell
.\.venv\Scripts\python.exe scripts\train_goofspiel_full.py --stage smoke_pipeline --steps 1 --batch-size 2 --n-cards 3 --num-corpus-games 2 --device cuda --artifact-dir artifacts\venv_smoke
```

也可以分别启动这些阶段：

- `stage0_verify`
- `build_corpus`
- `stage1_pretrain`
- `stage2_semi_supervised`
- `stage3_sft`
- `stage4_robust_rl`
- `evaluate`
- `smoke_pipeline`

但仍不建议直接开启“完整 N=13 生产级长训”。原因已经从“环境和测试失败”变成“算法闭环尚未达到 order 规格的生产完成度”：

- P1 仍不是完整 representation pre-training 多任务目标。
- P2 仍不是完整 teacher ensemble / EMA / disagreement filtering。
- P4 仍不是真正 self-play + replay + target network + curriculum 的长训系统。
- P5 仍明确 blocked until calibration。
- P6/P7 仍是 registry/schema smoke，不是完整 league 和 red-team continual correction。

因此，验收口径应拆开：

- 本机训练环境：通过。
- 训练代码基础可运行性：通过。
- 端到端 smoke pipeline：通过。
- 全仓测试：通过。
- 完整 Goofspiel-13 生产训练：未完成，不应宣称已经具备。
