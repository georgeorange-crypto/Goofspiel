<!--
  Author: 陈子聪 (Chen Zicong)
  Date: 2026-08-30
  Purpose: Project README — rules / install / run / env API example.
-->

# Goofspiel

> Author: 陈子聪 (Chen Zicong) · 2026-08-30

标准 13 张 Goofspiel (Game of Pure Strategy, "纯策略扑克") 的**完整工程实现**。
交付范围包括：规则环境、精确 Nash 求解器、C++ 向量环境、FastAPI 对战 Web UI、PPO baseline，以及 `order/` 规范要求的 Goofspiel-13 完整训练系统代码。正式长训由使用者在目标服务器上启动；仓库内提供 dry-run、单测和小规模验证入口。

```
┌───────────────────────────────────────────────────────────────────┐
│  Product / UI Layer   ·  FastAPI + HTML/CSS/JS 人机对战页面        │
│                        ·  N=1..13 + AI=Random/Heuristic/2×Nash    │
│                        ·  AI 决策分布 + 你每张牌的反事实 win/tie/lose │
├───────────────────────────────────────────────────────────────────┤
│  Agent Reasoning      ·  RandomBot / HeuristicBot / **2 NashBot**  │
│                        ·  Nash-classic (N≤7): 经典「平局弃奖」奖牌型 │
│                        ·  **Nash-carry  (N≤4): 平局滚入 carry-over**│
│                        ·  Nash 超 N / 规则不兼容 → 诚实回落 Heuristic│
├───────────────────────────────────────────────────────────────────┤
│  Training Backend     ·  P0-P7 完整训练流水线 + teacher/search/replay │
│                        ·  self-play RL + adaptive + league + red-team│
│                        ·  typed checkpoints + resume 校验 + reports  │
├───────────────────────────────────────────────────────────────────┤
│  C++ Accelerated Core ·  cxxgoof/ CMake + pybind11                  │
│                        ·  VectorizedEnv (SoA, bit-pack state)       │
│                        ·  GoofspielExactSolverCpp (LP callback      │
│                           scipy HiGHS 默认 / HiGHS native 可选)     │
│                        ·  观测 Gymnasium API — 不编 C++ 也有 fallback│
├───────────────────────────────────────────────────────────────────┤
│  Game / Math Core     ·  GoofspielEnv (Python reference)            │
│                        ·  Carry-over 平局滚入规则 + 奖池守恒不变量    │
│                        ·  OEIS A000172 Level-A+B 复杂度预检 (5 级风险)│
│                        ·  **两套独立精确 Solver（缓存 / policy 全隔离）**│
└───────────────────────────────────────────────────────────────────┘
```

核心原则：**Reference (Python 慢但可审计) 永远保留；Fast (C++/训练) 永远定期 cross-check。**

> **双精确 Nash 铁律**：本项目共存两套永不交叉污染缓存/策略的独立 Solver。绝不"拿错规则的分布冒充另一个精确解"。下面对照表是**契约**（所有测试强制执行）。

| `bot_type` | 奖牌型（训练/求解的规则） | State 维 | 默认 N 上限 | `carry_pool > 0` 时行为 | 回落触发条件 |
|---|---|---|---|---|---|
| `"nash"`       | 经典「平局 → 丢弃奖金」 | (A,B,R) 3-tuple | NASH_MAX_N = 7   | **诚实回落 Heuristic**（solver 无 carry 维度，绝不伪造） | (1) N > 7 或 (2) **任何回合出现 carry** |
| `"nash_carry"` | **新增 · 平局 → 滚入 carry-over（唯末轮平局才永久丢弃）** | (A,B,R,carry) 4-tuple | NASH_CARRY_MAX_N = 4 | **继续精确查表**（x*/y*/V 已在 carry 维度上求解，状态包含底池 stake） | 仅当 **N > 4** |
| `"random"` / `"heuristic"` | 随机 / 启发式（无需求解） | — | 13 | Heuristic 是 carry 适配的（见 §6.3 五条规则 inflate 系数） | 永不回落 |

数学独立性证据（已在 `tests/test_solver.py::TestCarrySolver` 强制断言）：对子状态 A={2,3}, B={1,3}, R={2,3}
```
V(nash_classic)                       = −1.40000
V(nash_carry, carry=0)                = −1.78571   ≠ classic ✓
V(nash_carry, carry=2)                = −3.15152   ≠ carry=0 且 ≠ classic ✓
F(A,B,R,c) = −F(B,A,R,c) 对两种模式、任意 c 都成立
```

---

## 1. 游戏规则（Carry-Over 变体）

双方各持有 `A, 2, 3, ..., 10, J, Q, K` 共 `N` 张牌，内部数值映射：

```
A=1, 2=2, ..., 10=10, J=11, Q=12, K=13
```

公共奖品牌是洗好的另一套 `1..N`。每轮"待分配总额"定义为：

\[
\text{prize\_at\_stake} = \text{round\_prize} + \text{carry\_in}
\]

其中 `carry_in` = 之前所有平局滚入下一轮的累计奖池。结算三分支：

1. **胜负**：出牌**大**的一方一次性拿全额 `prize_at_stake`；carry 清零。
2. **平局 & 非末轮**：双方本轮都不得分；**整包 `prize_at_stake` 滚入下一轮**（`carry_out = prize_at_stake`）。
3. **平局 & 末轮**：无下一轮可滚，**整包 `prize_at_stake` 永久丢弃** —— 这是整局**唯一**会真正丢奖金的场景。

出过的牌永久移除。`N` 轮后总分高者胜；分相同 `draw`。

### 关键不变量（所有测试强制验证）

\[
\mathrm{score_{human}} + \mathrm{score_{bot}} + [\text{末轮平局才会 > 0 的 discard}]
\equiv \frac{N(N+1)}{2}
\]

**随机性契约**：默认 RNG = `secrets.SystemRandom()`（密码学级别）。可复现/测试/训练用固定 seed 必须**显式注入** `GoofspielEnv(rng=random.Random(seed))`，绝不允许环境内部偷偷用全局 `random.random()`。

---

## 2. 安装（两条路线）

### 2a. 简单版：纯 Python（Web + Nash Exact 都能跑；C++ 环境自动 fallback 到 Python 串行）

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2b. 训练版：启用 C++ 加速（VectorizedEnv 吞吐 ~260× + Nash 精确解递归快 1.3~5×）

```bash
# 1. 建虚拟环境（上面一样）
# 2. 先装编译工具 + Python 依赖
pip install --upgrade cmake pybind11 ninja numpy scipy torch cmake-build-extension

# 3. pip install -e .  会自动跑 cxxgoof/CMakeLists.txt → 产出 goofspiel/_core.pyd
pip install -e .
```

> **Windows 注意**：Ninja + MSVC 环境要打开 **"x64 Native Tools Command Prompt for VS 2022"**，否则 `cmake -G Ninja` 找不到 `cl.exe`。更详细三种编译方式（pip / 手工 cmake / HiGHS native）、HiGHS 原生加速、FAQ 坑点 → 见 [order/C++模块编译与训练集成指南.md](order/C++模块编译与训练集成指南.md)。

### 冒烟验证（成功标志）

```bash
python -c "from goofspiel import _core; c=_core.estimate_complexity(5); print('C(5)=',c['C_N']); from goofspiel._cxx import cpp_solve_with_policy as s; print('N=3 exact value=',s(3).value)"
# 预期输出:
#   C(5)= 2252
#   N=3 exact value= 0.0
```

---

## 3. 启动 Web UI

```bash
python app.py
```

**特性：自动端口扫描** — 从 8000 开始，碰到 `WinError 10013/10048`（系统服务/之前你开的 Python）就**自动往后最多扫 50 个端口**，最后控制台打印最终 URL：

```text
Goofspiel: checking http://127.0.0.1:8000 ...
  -> port 8000 busy, trying :8001 ...
Goofspiel: port 8000 not available; using :8001.
Goofspiel: starting server on http://127.0.0.1:8001
           (Press CTRL+C to stop)
INFO:     Uvicorn running on http://127.0.0.1:8001
```

手动覆盖：

```bash
python app.py --port 9000                           # 换起始端口
python app.py --host 0.0.0.0 --port 8000            # 让局域网朋友访问
python app.py --port 8000 --max-port-tries 200      # 加大扫描范围
```

### 3.1 开局你能选什么（Game Setup 面板）

打开页面后**先看到开局表单**（不是默认开一局 13+Random）：

```
┌─ 开局设置 · Game Setup ───────────────────────────────────────┐
│ 牌数 N (1~13)    [  13  ] ← 任意 1..13                         │
│ AI 对手          [ Nash · 精确 · 经典平局弃奖 (N≤7) ▼ ]         │
│                 (Random / Heuristic / **Nash-carry N≤4** 共 4) │
│  ⚠️ 选 Nash 但本局出现平局 carry → Nash-classic 诚实回落；       │
│     想要 carry 局面仍精确 → 选 Nash-carry (平局滚入奖牌型 N≤4)    │
│                                                                 │
│ [ 开始游戏 · Start ]                                            │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 对战中新增两个可解释性面板

上一轮结束后除了 Round History 文字，还会多出两块条形面板：

| 面板 | 含义 | 颜色图例 |
|---|---|---|
| 🤖 **AI 决策分布条 `ai_policy`** | AI 在这一轮每张牌上的出手概率 % + V 值（期望净胜你分） | 单色蓝紫渐变，总和 100 |
| 🧍 **你每张牌的反事实 `human_policy`** | 如果你上一轮**改出另一张牌**，会得到 Win / Tie / Lose 中的哪一种（AI 仍保持它实际的出那张不变） | **三色**：绿=Win（赢多少） / 黄=Tie（0） / 红=Lose（输多少）；你实际打的那张画虚线框标出 |

反事实面板的目的：立刻让你理解 —— "这把我出 3 输了，但其实改出 K 就能赢 `prize_at_stake` + carry = 23 分"。

---

## 4. 运行全部测试

```bash
pytest -v
# pytest 默认跳过 tests/test_cxx.py（没编 C++ 扩展时 auto-importorskip）

# 编完 C++ 以后再跑的专项：
pytest tests/test_cxx.py -v        # C++ env 单步对齐 Python + 吞吐 + Nash N=3 不变式
pytest tests/test_solver.py -v     # Python solver 数学正确性（OEIS + 零和对称 + Nash 不等式）
pytest tests/test_env.py -v        # Env 契约 23 条（carry 规则 8 条）
pytest tests/test_app.py -v        # FastAPI HTTP 端点 E2E（反事实字段对齐）
```

**当前状态**：`tests/test_*.py` 非 C++ 用例 **89 passed / 0 failed**（1 条 slow Nash N=5 value 测默认 deselected）。已覆盖：双 Nash 独立数学不变量（7 条）+ 双 Nash HTTP 路由契约回落对照（4 条）+ 双 Nash carry 分支诚实回落 vs 精确（关键契约）+ carry 规则 8 条不变量 + env/app/solver 基础。

---

## 5. 训练 PPO（最小 demo，证明 C++ 环境能直接喂模型）

```bash
# 编了 C++： 4096 env × 256 step / update，默认 100k 步
python scripts/train_n5_ppo.py --num-cards 5 --total-timesteps 100000 --seed 1

# 没编 C++ 也能跑（fallback 到 256 个 Python env，慢一些但能证明 pipeline 通）
python scripts/train_n5_ppo.py --num-cards 5 --num-envs 256 --total-timesteps 20000
```

每 1 update 打印一行：

```
[upd   1/10] steps=  1048576  SPS= 31241  |  avg_return(±200)=+1.847  |  pg_loss=-0.0112 v_loss=+3.218 entropy=+1.442
...
[upd  10/10] steps= 10485760  SPS= 34891  |  avg_return=+6.213        |  pg_loss=-0.0081 v_loss=+2.714 entropy=+0.873
[goof] checkpoint saved to checkpoints/ppo_n5_seed1.pt
```

> 注意：PPO 只是 baseline/demo，不是 `order/` 里冻结的 Goofspiel-13 主训练路线。

## 6. Goofspiel-13 完整训练工程入口

本仓库已经把 `order/` 里的主训练系统拆成可迁移到服务器的代码入口；本机只做静态检查、单测、dry-run、小规模验证，不启动正式长训。核心模块：

| 需求 | 代码入口 |
|---|---|
| 统一 1-based rank / 0-based tensor 边界、schema version、state hash | `goofspiel/training/schema.py`、`goofspiel/training/data.py` |
| 变量 N 主模型，Robust/Adaptive 分支隔离，joint-action Q `[B,N,N]` | `goofspiel/models/` |
| Nash Bellman、RM+、NeuRD、TD(lambda)、V-trace、two-hot、teacher priority | `goofspiel/learning/` |
| Matrix Nash、Exact、Exact BR、SM-MCTS、GT-CFR、LeafEvaluator、ToolRouter、Final Decision | `goofspiel/reasoning/` |
| P0-P7 stage-gated 训练流水线、分池数据、checkpoint registry、league、red-team、reanalysis | `goofspiel/training/` |
| 结构化事件、JSONL sink、metric aggregator、system/GPU metrics | `goofspiel/observability/` |
| 七 Arena benchmark、主表/search/adaptive/opponent/generalization 表、baseline registry、promotion report | `goofspiel/training/benchmark.py`、`goofspiel/training/baselines.py` |
| 八卡 H200 服务器 torchrun 命令生成 | `goofspiel/training/distributed.py`、`scripts/plan_h200_training.py`、`configs/training/h200_8gpu.yaml` |

服务器安装建议：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install -r requirements-train.txt
python -m pip install -e .
```

训练前先生成八卡 H200 执行计划，不会启动训练：

```bash
python scripts/plan_h200_training.py --gpus-per-node 8 --steps 100000 --batch-size 512
```

正式训练由你在服务器上按阶段执行，例如：

```bash
torchrun --nnodes 1 --nproc_per_node 8 scripts/train_goofspiel_full.py \
  --artifact-dir artifacts/runs/h200_full \
  --stage stage4_robust_rl \
  --steps 100000 \
  --batch-size 512 \
  --n-cards 13 \
  --device cuda
```

`torchrun` 下训练代码会读取 `RANK/WORLD_SIZE/LOCAL_RANK`，使用 DDP 包装神经模型，并且只有 rank0 写 checkpoint，避免八个进程同时覆盖同一文件。

本机轻量验收命令：

```bash
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
python -m compileall goofspiel scripts tests -q
python -m pytest -q
python scripts/train_goofspiel_full.py --dry-run
python scripts/plan_h200_training.py --gpus-per-node 8 --steps 10 --batch-size 4
python scripts/validate_requirements_trace.py
```

本机 CUDA 仅用于轻量测试；生产 H200 服务器不要硬编码上述 Windows 路径，按服务器驱动/CUDA/PyTorch 版本匹配安装。

完整训练流水线阶段：

```text
stage0_verify            规则、schema、teacher priority、复杂度预检
build_corpus             环境自生成 GameCorpus
stage1_pretrain          player-swap / transition / joint outcome / masked action / opponent / style
stage2_semi_supervised   Exact/Search/CFR/EMA teacher ensemble + confidence/disagreement filter
stage3_sft               Exact/Search/CFR SFT + opponent behaviour + pseudo SFT anchors
stage4_robust_rl         self-play trajectories -> replay -> target network -> Nash/NeuRD/teacher anchors -> promotion
stage5_adaptive          opponent session、校准 gate、adaptive/oracle diagnostics
stage6_league            robust/aggressive/exploiter lineage、PFSP、cross-play、student distillation interface
stage7_redteam           attack、failure persistence、strong relabel、correction、regression
evaluate                 E0-E7 benchmark + summary/table artifacts
```

快速自检：

```bash
# 先确认 torch 能 import；如果这里报 c10_cuda.dll，先修 PyTorch 环境再训练
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 全量工程回归（没编 C++ 会自动 skip C++ 专项；torch 坏时 torch 测试会 skip）
python -m pytest -q

# torch 环境修好后，单独跑新增模型/学习测试
python -m pytest tests/unit/models tests/unit/learning tests/unit/reasoning -q
```

最小模型 forward 冒烟：

```python
from goofspiel.game import GameState
from goofspiel.models import GoofspielModel, public_state_from_game

model = GoofspielModel(max_cards=13)
batch = public_state_from_game([GameState.initial(5, current_prize=2)], max_cards=13)
out = model(batch)
print(out.q_robust.shape)          # torch.Size([1, 13, 13])
print(out.robust_policy_logits.shape)  # torch.Size([1, 13])
```

注意：当前机器的全局 Python 曾在 `import torch` 时因 `c10_cuda.dll` 缺依赖失败；这属于本机 PyTorch/CUDA wheel 环境问题，不是本项目代码语法问题。训练前请先换成可导入的 CPU 或 CUDA 版 PyTorch。

---

## 6. Python 环境 API 示例

核心环境 `GoofspielEnv` 与 AI / 训练解耦，接口精简、自解释：

```python
import random
from goofspiel import GoofspielEnv, RandomBot, PLAYER_0, PLAYER_1

env = GoofspielEnv(num_cards=13)                 # 默认使用 secrets.SystemRandom()
# env = GoofspielEnv(num_cards=13, rng=random.Random(42))   # 测试/可复现场景

obs = env.reset()
print(obs)
# {
#   "round": 1,
#   "current_prize": 12,
#   "scores": {"player_0": 0, "player_1": 0},
#   "remaining_cards": {
#       "player_0": [1,2,3,4,5,6,7,8,9,10,11,12,13],
#       "player_1": [1,2,3,4,5,6,7,8,9,10,11,12,13]
#   },
#   "remaining_prizes": [...],
#   "done": False,
#   "result": None
# }

# --- 单步必须一次提交双方动作；动作不合法抛 ValueError ---
bot = RandomBot()
while not obs["done"]:
    a0 = env.legal_actions(PLAYER_0)[0]           # 玩家：固定打最小的合法牌
    a1 = bot.choose_action(env, PLAYER_1)         # Bot：随机出牌
    obs, rewards, done, info = env.step({
        PLAYER_0: a0,
        PLAYER_1: a1,
    })
    print(rewards)     # { player_0: p 或 0, player_1: p 或 0 }

# 游戏结束后查询结果：
print(env.scores)          # {"player_0": ..., "player_1": ...}
print(env.result())        # "player_0" / "player_1" / "draw"
print(len(env.history))    # 13 (每一轮一条)
```

### 接口要点

| Item | 说明 |
| --- | --- |
| `env.reset()` | 重开一局，返回首条 observation |
| `env.step({p0: a, p1: b})` | **同时**接收双方动作；缺任一玩家都会抛异常 |
| `env.legal_actions(player)` | 返回该玩家剩余牌列表（可打的合法动作） |
| `env.result()` | `done=True` 后返回 `"player_0"` / `"player_1"` / `"draw"`，未结束返回 `None` |
| `env.history` | 每轮字典：`{round, prize, actions, winner, rewards}` |
| `observation` | 纯 `dict`，字段见示例，**不含对方未揭晓动作** |

Reward 默认只按真实得分（非零和）：赢奖品 `p` 的一方 reward = `p`，另一方 0；平局双方 0。需要零和收益可以在外部写一个简单的 wrapper。
