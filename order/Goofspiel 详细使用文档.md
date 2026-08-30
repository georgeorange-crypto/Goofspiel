# Goofspiel 项目 · 详细使用文档

> **Author**：陈子聪 (Chen Zicong)  
> **Date**：2026-08-30  
> **Purpose**：覆盖项目全貌的面向使用者与二次开发者的操作手册  
> - 普通玩家 → 读「快速开始」+「网页玩法指南」即可  
> - Python / RL 开发者 → 读「Python API 详解」+「Bot 家族详解」+「扩展指南」  
> - Web / HTTP 客户端开发者 → 读「FastAPI HTTP API 完全手册」

---

## 1. 项目简介

Goofspiel（Game of Pure Strategy，纯策略扑克游戏）的完整 Python 实现。交付内容分三层：

| 层级 | 职责 | 主要文件 |
|---|---|---|
| **游戏核心层** | 规则正确的 RL-ready 环境 | `goofspiel/env.py` |
| **AI / Bot 层** | Random · Heuristic · 精确 Nash 三种对手，统一接口 | `goofspiel/bots.py`、`goofspiel/solver.py` |
| **应用 / 网页层** | FastAPI + 原生 HTML/CSS/JS 的人机对战页面；后端保存所有游戏真值 | `app.py`、`templates/index.html`、`static/style.css`、`static/app.js` |

设计优先级严格为：

```
规则正确  >  环境接口（对 AI 友好）  >  网页可玩性  >  测试覆盖  >  美观
```

关键约束 **已经强制**：
- 双方动作必须**同时**提交到 `env.step()`，从技术上杜绝"偷看对手动作后再决定"的违规。
- `observation` 不包含任何对方未揭晓动作；动作只有在 `step()` 结算之后才进入 `history`，对双方同步可见。
- 默认 RNG 是 `secrets.SystemRandom()`（密码学级别），测试 / 复现通过构造参数显式注入 `random.Random(seed)`。
- **平局 Carry-Over 变体（本项目采用的规则）**：平局 **不丢弃奖金**，而是滚入下一轮作为"累计奖池 carry_pool"。结算三分支：
  1. 胜负 → 胜者拿 `prize_at_stake = round_prize + carry_in`，carry 清零；
  2. 平局 & 非末轮 → 双方 0 分，`carry_out = prize_at_stake`（滚入下一轮）；
  3. 平局 & 末轮 → 整包 `prize_at_stake` 永久丢弃（唯一会真正丢奖的场景，因无下一轮可滚）。
  奖池守恒不变量：人类总分 + Bot 总分 +（仅末轮平局时有值）丢弃量 ≡ `N(N+1)/2`。
- 由于精确 Nash solver 是基于经典"平局弃奖"Goofspiel 推导的闭式解，**与 carry-over 规则在策略空间根本不兼容**；因此只要 `carry_pool > 0`，NashBot 会**诚实回退 HeuristicBot**，并在 `note / meta.fallback_reason` 明示原因（不会偷偷输出错误的"精确分布"蒙骗用户）。

---

## 2. 目录结构

```text
Poker/
├── goofspiel/                     # 游戏核心包（Python Reference + C++ Accel 鸭子兼容）
│   ├── __init__.py                # 对外统一导出：Env / Bots / Solver / 常量
│   ├── env.py                     # GoofspielEnv (Python Reference)：reset/step/legal_actions/result/history
│   ├── bots.py                    # BaseBot + RandomBot / HeuristicBot / NashBot + create_bot() 工厂
│   ├── solver.py                  # Python 精确 Nash：OEIS A000172 Level-A+B 预检 + 两阶段递归
│   └── _cxx.py                    # ⭐ C++ 鸭子兼容层；import goofspiel._core 成功则接 C++ 加速，否则 fallback
│
├── cxxgoof/                       # ⭐ C++ 加速模块 (CXX20 + pybind11 + Eigen + HiGHS 可选)
│   ├── CMakeLists.txt             #   产物 goofspiel/_core.{pyd,so} 直接落 goofspiel 包旁
│   ├── include/goofspiel/         #   goof_env.h / goof_estimate.h / goof_nash.h（header-only 内核）
│   └── src/                       #   goof_env.cc (VectorEnv SoA) / goof_nash.cc / bindings.cc (pybind)
│
├── templates/
│   └── index.html                 # 页面结构：Game Setup(N+AI 下拉) + 对战 + ai_policy + human_policy 两块
├── static/
│   ├── style.css                  # 样式（含反事实条三色 win-h/tie-h/lose-h）
│   └── app.js                     # Vanilla JS：状态机 + 渲染 AI 条 + 渲染 反事实三色条 + 虚线框
│
├── scripts/
│   └── train_n5_ppo.py            # ⭐ CleanRL-style PPO 最小自博弈 demo：证明 C++ VectorEnv → torch 管道通
│
├── tests/
│   ├── __init__.py
│   ├── test_env.py                # 23 条环境契约（carry-over 8 条）
│   ├── test_app.py                # FastAPI 端点 E2E：反事实 winner/delta 对齐 env.step
│   ├── test_solver.py             # Python Nash：OEIS / 零和对称 / xᵀM≥V 与 My≤V 不变式
│   └── test_cxx.py                # ⭐ C++ 扩展专项（扩展未编时 pytest auto-skip）
│
├── order/                         # 完整工程规范文档（13+ 份）
│   └── Goofspiel 详细使用文档.md   # 本文档
│
├── app.py                         # FastAPI 应用 + 端口自动扫描启动入口
├── pyproject.toml                 # ⭐ pip install -e . + cmake-build-extension 自动编译 C++ 扩展
├── requirements.txt               # 最小 Python 依赖 (FastAPI/Scipy/Pytest 等，不含 torch/cmake)
└── README.md                      # 项目概览 / 两种安装方式 / Web 面板说明 / PPO demo 启动
```

---

## 3. 快速开始

### 3.1 环境要求

| 项 | 要求 |
|---|---|
| Python 版本 | **≥ 3.10**（训练脚本 train_n5_ppo.py 使用新 typing；非训练使用 3.9 也可） |
| 操作系统 | Windows / macOS / Linux（FastAPI、solver、CMake 构建、PPO 脚本均跨平台） |
| 内存 · 普通玩法（Random/Heuristic/N≤3） | < 200 MB |
| 内存 · 精确 Nash N=7（Python solver 冷启动） | ≥ 4 GB 可用内存（solver 会用 psutil 预检 + GREEN/YELLOW/ORANGE/RED/BLACK 五级风险） |
| 内存 · 精确 Nash N=7（C++ solver，默认 scipy 回调） | ≥ 2 GB（递归 cache 比 Python dict 紧凑 ~3×） |
| 内存 · 精确 Nash N=7（C++ solver + 原生 HiGHS 不开 scipy GIL） | ≥ 1.5 GB（预计，见编译指南 §HiGHS 章节） |
| 训练 · N=5 PPO + 4096 VectorEnv | ≥ 2 GB（torch 2 层 MLP，状态 tensor 内存 < 100 MB） |
| C++ 构建工具（可选，仅训练加速） | MSVC 2022 或 GCC≥11/Clang≥15，CMake≥3.20，Ninja（或 MSBuild），pybind11≥2.10 |
| 磁盘依赖 | 无数据库，无本地缓存写入；训练 checkpoint 写入 `checkpoints/`（每局几 MB） |

> 精确 Nash 求解依赖 `scipy.optimize.linprog`（单纯形 / 内点法）和 `numpy`；文档第 7 章有细节。

### 3.2 安装

```bash
# 1. 建虚拟环境（强烈推荐）
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# 2. 安装依赖（含 FastAPI/Nash solver/scipy/numpy/psutil/pytest）
pip install -r requirements.txt
```

### 3.3 启动 Web 服务

```bash
python app.py
```

启动器内置**自动端口扫描**，从 8000 开始尝试；如果碰到：
- **WinError 10013**（8000 被系统 Apache/IIS 等高权限进程占着，你 kill 不掉）
- **WinError 10048**（端口上已有你之前开的 Python 服务）

程序会自动往后最多扫 50 个端口，找到第一个空闲就绑定，并在终端打印最终访问地址：

```text
Goofspiel: checking http://127.0.0.1:8000 ...
  -> port 8000 busy, trying :8001 ...
Goofspiel: port 8000 not available; using :8001.
Goofspiel: starting server on http://127.0.0.1:8001
           (Press CTRL+C to stop)
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

跟着提示里的 URL 在浏览器打开即可。常用的**显式覆盖写法**：

```bash
# 换起始端口（扫 9000 起）
python app.py --port 9000

# 允许局域网朋友访问（防火墙放行端口后）
python app.py --host 0.0.0.0 --port 8000

# 加大扫描范围
python app.py --port 8000 --max-port-tries 200

# 用环境变量（PowerShell 示例）
$env:HOST="0.0.0.0"; $env:PORT="9000"; $env:MAX_PORT_TRIES="100"; python app.py
```

### 3.4 运行全部测试

```bash
pytest -v
```

三个测试文件：
- `test_env.py` —— 17 个**环境契约**测试（手牌、奖品牌、合法性、平局、胜负、轮数、可复现性等）。
- `test_app.py` —— FastAPI `config/new/state/play` 端点的 E2E（含非法 bot_type、非法 action、自动回落、完整 13 轮跑通）。
- `test_solver.py` —— 精确 Nash solver：解是否符合零和、对称性 `F(A,B,R) = -F(B,A,R)`、策略概率归一化、复杂度估算函数正确性。

---

## 4. 网页玩法指南（Human vs AI）

### 4.1 进入首页后的页面

打开后先看到「开局设置 · Game Setup」面板：

```
┌─ 开局设置 ─────────────────────────────────────┐
│ 牌数 N (1~13)：  [  13  ]  ← 可改 1..13        │
│ AI 对手：        [ Random · 纯随机 ▾ ]         │
│                                                  │
│ [开始游戏]                                        │
└──────────────────────────────────────────────────┘
```

#### 字段说明

| 字段 | 取值 | 解释 |
|---|---|---|
| **牌数 N** | `1..13`（整数） | `1` = 只有 A；`13` = 完整一副 `A..K`（完整规则） |
| **AI 对手** | 3 选 1（下拉框由 `/api/game/config` 动态填充） | 见 4.2 节详解 |

下拉框选项**文字标签**含义：

```
Random · 纯随机 (baseline)                 → 所有合法牌均匀抽
Heuristic · 启发式 (出价≈奖金比例)          → 轻量规则 AI，对普通人类非常强
Nash · 数学精确混合策略 (仅 N ≤ 7)          → 零和精确纳什均衡；N>7 自动回落 Heuristic
```

> ⚠️ 首次选 `Nash` 且 N=5~7 时，服务端会在**开局阶段**一次性完成冷启动策略预计算（为了不在你首张牌点下去才卡顿）。加载提示 `正在生成精确策略 (Nash N>5 可能耗时数秒~数十秒) …` 会出现，正常等待即可。结果会类级缓存，**同 N 的后续局** 0ms 返回。

### 4.2 三种 AI 的人类体验

| AI | 对普通人类强度 | 典型使用场景 | 前端会额外看到 |
|---|---|---|---|
| **Random** | 很弱（胜率 ~50%，看脸） | 熟悉规则、练手 | AI 决策分布：每张牌概率都是 `100/牌数%` |
| **Heuristic** | 较强（普通人类 70% 概率会输） | 认真玩的朋友 | 分布是「目标排位 ±1」加权，并有人类可读 note（例如 `Target rank 5/9 (prize percentile 6/12)`、`Tail go-for-broke · 梭哈大牌抢最大奖`） |
| **Nash（N≤7 精确）** | 理论最强（无人类能长期击败它） | 研究策略 / 训练 RL 基线 / 感受理论最优的风格 | 分布是严格 x* 混合策略，`当前状态价值 V` 会显示该局面下 AI 期望净胜分（正=AI 占优） |

#### 自动回落（Nash · N>7）

你选了 Nash，但把 N 调到 8~13 → 服务器不会报错，而是**内部自动改用 HeuristicBot**，并在副标题给出黄色提示，例如：

```
实际 AI：Heuristic（要求 N=8；Nash 精确上限 N=7，已自动回落）
```

这是为了保证界面使用流畅、不抛 422 给用户造成困惑。同时 API 响应 `meta.fallback_reason` 会带完整原因字符串。

### 4.3 对战界面区块总览

点击「开始游戏」→ 进入对战界面：

```
┌ Header: 标题 / 副标题（AI 型号 + Nash N>7 回落黄色提示） / 重新开局 ───┐
├ 状态栏（6 格）：Round / Prize / Carry Pool / Total at Stake / 你的分 / AI 分 ─┤
├ 🤖 AI 决策分布条 `ai_policy`（蓝紫渐变，每张牌出手概率 %，V 值显示行）─┤
├ 🧍 你每张牌的反事实 `human_policy`（三色 Win/Tie/Lose 条形 + 实际出牌虚框）┐
│   · 图例： ██ Win (+Δ)  ██ Tie (0)  ██ Lose (−Δ)   [虚线] = 你实际出的那张 │
├ 上一轮横幅（后端返回 result_text，含 carry-aware 中文措辞） ─────────────┤
├ 终局横幅（You win! 42-36 / Bot wins / Draw） ──────────────────────────┤
├ AI 剩余手牌（只读 · 蓝色卡片） ────────────────────────────────────────┤
├ 你的剩余手牌（可点 · 红色卡片，已用卡片自动移除 & 禁点） ────────────────┤
├ 已用牌（你出过 / AI 出过 2 列） ────────────────────────────────────────┤
├ Round History（OL，每回合 R1..RN，胜/负/平配色 · 每回合 result_text） ─┤
```

> **`human_policy` 反事实条怎么用？**  
> 结算上一轮之后，它会展示："如果我上一轮**没出我实际打的那张 h_real**，而改出 h' ∈ 我当时剩下的合法牌，且 AI 仍出它实际出的 b_real —— 结果会是 Win / Tie / Lose 中的哪一种？"  
> 绿条长度 = 你赢多少（正数，Δ = prize_at_stake），黄条 = Tie（0），红条 = 你输多少（负），你实际打的那张周围会有虚线黑色边框标记出来，一目了然判断"有没有打对"。  
> 反事实的 winner/delta 与真实 GoofspielEnv step() 结果字节级对齐（`tests/test_app.py::test_human_counterfactual_agrees_with_env` 强制验证，100 个随机动作全过）。

### 4.4 完整一局的流程（Human 视角）

1. 进入第 1 轮，状态栏显示 **Round 1/N**，**Current Prize** 显示当前奖金牌（例如 `Q = 12`）。
2. 在 **Your Remaining Cards** 点击任意一张（例如 `7`）。
   - 你点下的瞬间：服务端
     - 校验 7 确实在你剩余牌里（非法 → 400）
     - 让 AI **基于当前未结算状态** 独立选牌（同时原则）
     - 一次 `env.step({player_0:7, player_1: bot_choice})` 原子结算
   - 返回后页面：横幅显示结果，AI 决策分布条会展示 AI 在这一步对每张合法牌的出手概率百分比。
3. 用过的牌自动从上方「剩余手牌」消失，追加到「已用牌」和「Round History」。
4. 自动进入 Round 2，重复 1..3。
5. Round N 结束后，屏幕中央出现绿色/红色/黄色终局横幅：
   - `You win! 73 – 18`（绿）
   - `Bot wins. 48 – 43`（红）
   - `Draw. 35 – 35`（黄）
6. 点「重新开局 / New Game」回到开局设置面板，可以重新调 N 和 AI 类型。

---

## 5. Python API 详解（GoofspielEnv 环境）

这是给**程序化玩家 / RL 训练脚本**用的核心接口，完全独立于 FastAPI/Web 层。

### 5.1 导入

```python
from goofspiel import (
    GoofspielEnv,      # 环境主类
    RandomBot,         # 3 种 bot
    HeuristicBot,
    NashBot,
    create_bot,        # 字符串 -> bot 工厂
    PLAYER_0,          # "player_0"  常量，避免硬编码
    PLAYER_1,          # "player_1"
)
```

### 5.2 构造函数

```python
GoofspielEnv(
    num_cards: int = 13,        # 手牌 / 奖品牌张数（N）
    rng       = None,           # 随机源
)
```

| 参数 | 说明 |
|---|---|
| `num_cards` | 默认 13（完整 Goofspiel）。合法值 ≥ 1。传入 0/负数抛 `ValueError: num_cards must be >= 1`。 |
| `rng` | `None` → 默认用 **`secrets.SystemRandom()`**（不可复现，生产用）。<br />测试 / 复现用 **`random.Random(seed)`**。 |

> 🔑 可复现性原则：相同 `seed` 构造的 RNG → 相同 prize deck 洗牌结果；若 bot 也用同样 seeded RNG → 整局 history/scores/result 字节级一致。`tests/test_env.py::TestReproducibility::test_fixed_seed_reproduces_prize_order_and_outcome` 强制验证此契约。

### 5.3 `obs = env.reset()`

重置一局，**返回首条 observation**（首回合的 prize 已翻开）。

### 5.4 `obs, rewards, done, info = env.step(actions)`

推进**一个回合**，必须**同时**包含双方动作：

```python
obs, rewards, done, info = env.step({
    PLAYER_0: 5,   # 人类 / 玩家 0
    PLAYER_1: 9,   # Bot / 玩家 1
})
```

| 返回项 | 类型 | 说明 |
|---|---|---|
| `obs` | `dict` | 当前 observation（与 `reset()`/`get_observation()` 结构相同，见下表） |
| `rewards` | `{player_0: int, player_1: int}` | **真值得分 reward**（carry-aware）：胜负 → 胜者 = `prize_at_stake = round_prize + carry_in`，负方 = 0；平局（含末轮平局丢弃）→ 双方 = 0。<br />**不会**出现"平局负奖励"或"丢弃被当作奖励"的脏语义。后续要做零和训练时，在环境外加一行 wrapper 即可（不破坏真值得分的审计性）。 |
| `done` | `bool` | `True` = 所有奖品已翻完 & 结算。 |
| `info` | `{"winner": str\|None, "carry_in": int, "carry_out": int, "prize_at_stake": int, "discarded": bool}` | 额外调试信息：winner=`player_0/player_1` 平局 = `None`；carry 字段还原本回合结算三分支（胜 / 平滚 / 平末丢弃）；`discarded=True` **当且仅当**末轮平局。 |

#### step() 触发的异常语义

| 场景 | 异常 |
|---|---|
| `game.done == True` 时调用 | `RuntimeError("Cannot call step() on a finished game. Call reset() first.")` |
| 缺某一方动作（`{"player_0": 1}` 或 `{}`） | `ValueError("step() requires actions for both player_0 and player_1")` |
| 某玩家 action 不在其剩余牌中 | `ValueError(f"Illegal action {a} for {pid}. Remaining cards: [...]")` |
| `player` 传入了错误的 player id（非 0/1 常量） | `ValueError("Unknown player ... Must be one of ('player_0', 'player_1')")` |

### 5.5 Observation 字段表（`dict`，RL-ready）

| 字段 | 类型 | 说明 |
|---|---|---|
| `round` | `int` | 当前已揭晓的回合编号（1..N；game over 时 = N） |
| `current_prize` | `int \| None` | 当前奖金面值。游戏结束时 = `None`。**注意：前端显示时要再用 `card_display_name()` 映射成 A/J/Q/K。** |
| `carry_pool` | `int` | **之前所有平局累计滚入**的奖池金额（≥ 0）。如果前序未平过 = 0。 |
| `total_prize_at_stake` | `int \| None` | 当前这一轮若分出大小，胜者能**一次性拿到**的总额 = `current_prize + carry_pool`。这是决策时最重要的"有效奖金额"，Bot 启发式排位 & AI 反事实条都用它。 |
| `scores` | `{"player_0": int, "player_1": int}` | 双方**真值得分累计**。总和 ≤ N(N+1)/2（仅无平局且非末轮平局时才等于 91 for N=13；若末轮平局则丢分 = N(N+1)/2 - 总分）。 |
| `remaining_cards` | `{"player_0": [int…], "player_1": [int…]}` | 双方剩余牌，**升序排序**。回合开始时 `len` = N - round + 1。 |
| `remaining_prizes` | `[int…]` | 奖品堆中**尚未揭晓**的牌（**不含** `current_prize`），升序排序。仅用于 AI / 展示，不要作为玩家"偷看奖品牌顺序"的依据。 |
| `done` | `bool` | 是否结束。 |
| `result` | `None \| "player_0" \| "player_1" \| "draw"` | 未结束 = `None`；结束后根据分数判断。 |

> 🚩 信息隐藏契约：**`observation` 从不包含任何一方"下一步即将打但还没结算"的动作**。`history` 只在 `step()` 后写入已揭晓回合。这保证了环境可以直接接任何不完全信息 AI 算法，无需额外 wrapper。

### 5.6 其它公共 API

```python
env.legal_actions(player: str) -> List[int]   # 返回升序的合法动作（剩余牌）
env.result()                  -> Optional[str]  # 未结束 None; 否则 PLAYER_0/PLAYER_1/"draw"
env.get_observation()         -> dict           # 取当前 observation（和 reset/step 返回结构一致）
env.history                   -> List[dict]     # 每回合一条，见下 5.7
```

### 5.7 `env.history` 每条字段

```python
{
  "round":   int,                        # 1..N
  "prize":   int,                        # 兼容字段：= round_prize（本轮奖金面值，不含 carry）
  "round_prize":      int,               # 本轮奖金面值（= prize，语义更清晰）
  "carry_in":         int,               # 本回合开始时已有的累计滚入奖池
  "prize_at_stake":   int,               # 本回合实际待分配总额 = round_prize + carry_in
  "carry_out":        int,               # 本回合结算后，要带到下一轮的累计奖池；胜负=0，平非末=prize_at_stake，平末=0
  "discarded":        bool,              # True ⟺ 末轮平局且整包 prize_at_stake 被永久丢弃
  "actions": {"player_0": int, "player_1": int},  # 双方真实出的牌
  "winner":  "player_0" | "player_1" | None,      # None = 平局
  "rewards": {"player_0": int, "player_1": int}   # 同 step() 的 rewards（胜负=prize_at_stake；平=0）
}
```

不变量（每条 history entry 都满足）：
- `prize_at_stake == round_prize + carry_in`
- 胜：`winner != None, carry_out == 0, discarded == False, rewards[winner] == prize_at_stake`
- 平非末：`winner is None, carry_out == prize_at_stake, discarded == False, rewards == {0, 0}`
- 平末：`winner is None, carry_out == 0, discarded == True, rewards == {0, 0}`

### 5.8 完整最小示例（Python）

```python
import random
from goofspiel import GoofspielEnv, RandomBot, PLAYER_0, PLAYER_1

env = GoofspielEnv(num_cards=13, rng=random.Random(42))  # 可复现
bot = RandomBot(rng=random.Random(7))

obs = env.reset()
while not obs["done"]:
    a0 = min(env.legal_actions(PLAYER_0))                # "Always play smallest" 策略
    a1 = bot.choose_action(env, PLAYER_1)
    obs, r, d, i = env.step({PLAYER_0: a0, PLAYER_1: a1})
    print(f"R{env.history[-1]['round']:2d}  prize={env.history[-1]['prize']:2d}  "
          f"you={a0:2d} bot={a1:2d}  →  winner={i['winner']}  scores={env.scores}")

print(f"\nFinal: scores={env.scores}, result={env.result()}")
print(f"Total rounds played = {len(env.history)}")
```

---

## 6. Bot 家族详解

所有 bot 都继承 `BaseBot`，因此接口统一，互替换零成本。

```python
from goofspiel.bots import BaseBot
```

### 6.1 BaseBot 双接口

```python
# 老接口 —— 只返回动作
action: int = bot.choose_action(env: GoofspielEnv, player: str)

# 新接口 —— 返回动作 + 决策透明化信息（Web 前端用来画分布条）
action, info = bot.choose_action_with_policy(env, player)
# info 键（通用）：
#   "distribution" : List[[card_value:int, pct_0_to_100:float]]   # AI 对每张合法牌的出手概率
#   "value"        : float              # 该局面 AI 期望净胜分；Random/Heuristic = NaN；Nash = 精确 V
#   "bot_type"     : str                # "random" / "heuristic" / "nash"
#   "note"         : str                # 人类可读说明（前端展示）
```

> 写自定义 AI 时，只需覆盖 `_choose(env, player) -> Tuple[int, Dict]`；`BaseBot.choose_action` 会自动用它兼容老环境接口。

### 6.2 RandomBot

**均匀随机** —— 所有合法动作等概率。

```python
from goofspiel import RandomBot
import random

# 默认：secrets.SystemRandom() → 真随机
r0 = RandomBot()

# 复现模式（例如测试）：
r1 = RandomBot(rng=random.Random(12345))
```

`distribution` 每张牌都是 `100 / len(legal)`；`value = NaN`；`note = "Uniform random · 1/len(legal)"`。

### 6.3 HeuristicBot

**人类可读规则 AI**（Numpy 做加权采样）。对普通人类水平约 70% 胜率。

五条规则（按优先级，**全部基于有效奖金额 `prize_at_stake = round_prize + carry_in`**，而非单轮 prize）：

1. **Tail go-for-broke**：末段（剩余奖品 ≤ max(2, N//4)）、我落后、且（当前奖是最大剩余奖 **或** carry 导致 `prize_at_stake ≥ 最大剩余奖`） → 直接梭哈最大牌，`note` 会写 `Tail go-for-broke · 梭哈抢 (含carry) total=N`。
2. 否则：先用 `inflation = prize_at_stake / round_prize  (若 round_prize=0 则 1)` 计算"含 carry 的膨胀系数"，再在 percentile 排位上追加 `boost_from_carry = min(0.22, (inflation-1.0)*0.14)` 的向上偏移 —— carry 越大，整体目标牌等级越高。
3. **小奖阻尼**：若 `prize_at_stake < 0.75 * mean_eff_ref_prize`，原始排位 × 0.55（避免小奖耗大牌）。
4. **大奖增强**：若 `prize_at_stake > 1.25 * mean_eff_ref_prize`，排位略超匹配 × 1.12 + 0.04。
5. 把排位夹到 `[0, len(legal)-1]`，以 0.70 / 0.22 / 0.08 权重在目标排位 ±1 邻居中抽样。

`distribution` 就是该权重向量 × 100；`value = NaN`（启发式没有理论值）。**当 env.carry_pool > 0 时，note 前缀会打印 `[Carry +N]`，提示决策是基于膨胀后的奖金额。**

### 6.4 NashBot —— 数学精确解（零和 Nash 均衡）

> **2026-08-30 升级：从单 Nash → 双 Nash 独立模型。**
> 由于项目同时支持两套奖牌型（经典「平局弃奖」 vs Carry-Over「平局滚入」），它们的 MDP 子结构 V 值**数学上不同**（已证明：对同一 (A,B,R)，V_classic ≠ V_carry_c=0 ≠ V_carry_c=2），因此我们**不用一套 solver + 几个参数凑合**，而是交付两套**状态/cache/policy/复杂度预检全部隔离**的独立精确 Nash Solver + NashBot 两种 `exact_mode`。永不"拿模型 A 的分布冒充模型 B 的精确解"。

| 模式 (`exact_mode`) | `bot_type` | 对应 Solver 类 | state 缓存 key | policy lookup key | 默认 N 上限 | `carry_pool > 0` 行为 | 回落 #1（规则不兼容） | 回落 #2（N 超阈值） |
|---|---|---|---|---|---|---|---|---|
| `EXACT_MODE_CLASSIC`（经典） | `"nash"` | `GoofspielExactSolver` | (A,B,R) 3-tuple | (A,B,R,prize) 4-tuple | `NASH_MAX_N = 7` | **诚实回落 Heuristic**（solver 根本没有 carry 维度，**绝不伪造分布**） | ✅ **触发**：任何回合 carry>0 | N > 7 |
| `EXACT_MODE_CARRY`（平局滚入）**新增** | `"nash_carry"` **新增** | `GoofspielCarrySolver` **新增** | (A,B,R,carry) **4-tuple** | (A,B,R,carry,prize) **5-tuple** | `NASH_CARRY_MAX_N = 4` | **继续精确查表**（求解已完整覆盖 carry 维度 + stake=p+c 的三分支） | ❌ **不触发**（carry 是规则内置） | N > 4 |

离线分别用 `solve_with_policy(N)` / `solve_with_policy_carry(N)` 预计算所有可达状态的 x*/y*/V；在线查表 + 按 x* 采样。

```python
from goofspiel import NashBot, EXACT_MODE_CLASSIC, EXACT_MODE_CARRY

# —— 单模式构造 ——
bot_a = NashBot(exact_mode=EXACT_MODE_CLASSIC)   # 经典奖牌型（默认 N≤7）
bot_b = NashBot(exact_mode=EXACT_MODE_CARRY)     # Carry-over 奖牌型（默认 N≤4）
# 自定义更保守的上限
bot_c = NashBot(exact_mode=EXACT_MODE_CARRY, max_nash_n=3)
```

#### 关键机制（两种模式共用 + 各自差异）

| 机制 | 说明 |
|---|---|
| **类级双缓存（严格隔离）** | `_policy_cache_classic: Dict[int, SolveResult]` 经典专用；<br />`_policy_cache_carry:   Dict[int, SolveResult]` carry 专用。**永不交叉读。** 同一 N 的两种模式缓存完全独立，第一局冷启动代价，第二局起 0ms。 |
| **`max_nash_n`（各自默认上限）** | CLASSIC 上限 7；CARRY 上限 4（因 carry 状态 = 经典 × ~(N(N+1)/2+1) branch，N=7 → ~3M GREEN/ORANGE 边界）。超上限**不抛异常**，内部用 `HeuristicBot`，`info.note` 前缀 `[Nash-classic fallback…]` / `[Nash-carry-over fallback…]`。 |
| **回落契约 #1（规则不兼容）CLASSIC 专用** | 由于 `GoofspielExactSolver` 的精确解是基于**经典 Goofspiel「平局弃奖」**规则离线推导的 MDP，与 carry-over 变体在 V 值上不兼容。**只要 `env.carry_pool > 0`（前面出现过平局滚入），CLASSIC NashBot 立刻切换 HeuristicBot**，并在 `note / meta.fallback_reason` 写清：<br />`[Nash-classic fallback to Heuristic · carry_pool=2 存在平局累计奖池；当前 Nash 精确 solver 为经典「平局弃奖」奖牌型推导，未适配 carry-over 规则。请切换到 bot_type='nash_carry'。]`<br />**绝不输出伪造的"精确分布"**（这是强契约，在 `tests/test_app.py::TestDualNash::test_carry_over_split_contract_tie_then_round2` 强制验证）。 |
| **回落契约 #1（CARRY = 无）**  | CARRY 模式 solver 本来就是为 carry-over 奖牌型写的三分支 terminal（胜负清零 / 平非末累加到 stake / 平末清零丢弃），carry>0 是合法子状态，**照常精确**，note 前缀显示：<br />`Nash-carry-over(tie→rollover; carry_in=2) · bot=row, opponent=col · bot expected score-diff vs human = -0.000` |
| **视角无关**： | 由于 `F(A,B,R,c) = -F(B,A,R,c)` 反对称对任意 c 都成立 → canonical sign 翻转机制使 Bot 无论当 player_0/player_1 都正确，carry 是公共底池不影响身份。 |
| **数值稳定性** | 策略在使用前 `np.clip(x, 0, None)`；若 `sum <= 0` 退化成均匀分布再归一化，防止 LP 下溢崩溃。 |
| **求解算法** | `goofspiel/solver.py`：两阶段递归（Phase1 eager 解所有子状态存 canonical cache；Phase2 纯读构建 M 矩阵，避免符号翻转泄漏→交叉污染）→ 每子状态 `scipy.optimize.linprog(method='highs')` 求矩阵游戏 V/x*/y*。 |

`info.value` 是精确 `V`（从 bot=row 视角的 AI 期望净胜你分数），前端显示在 **状态价值 V** 行。若回落 heuristic，`value` 退化成 heuristic 期望分差。

### 6.5 `create_bot(bot_type, *, seed=None)` 工厂

```python
from goofspiel import (
    create_bot,
    BOT_RANDOM, BOT_HEURISTIC, BOT_NASH, BOT_NASH_CARRY,   # 4 种 bot_type
)

a = create_bot(BOT_RANDOM)                      # secrets.SystemRandom
b = create_bot(BOT_HEURISTIC, seed=42)          # seeded 可复现
c = create_bot(BOT_NASH)                        # Nash-classic (N≤7 默认)
d = create_bot(BOT_NASH_CARRY)                  # Nash-carry-over (N≤4 默认)
e = create_bot("unknown")                       # ValueError: Unknown bot_type 'unknown'
```

**`BOT_TYPES = {"random", "heuristic", "nash", "nash_carry"}`（4 项，2026-08-30 双 Nash 扩展）。** 后端 `app.py` 用工厂实例化用户选的 AI；前端表单下拉直接从 `GET /api/game/config` 拉 `bots` 数组生成。

---

## 7. FastAPI HTTP API 完全手册

**重要**：服务器为了保持"极简无数据库"原则，使用**单内存 session**（全局 `_env, _bot`）。并发多浏览器会互相覆盖游戏状态。要做多用户 / 房间，可参考 §11.3。

所有端点都位于 `/api/game/*`。返回体统一 JSON：

```json
{
  "state":       { /* UI 完整状态，§7.5 字段表 */ },
  "last_round":  { /* 最近一轮详情；没有时 null */ },
  "meta":        { /* num_cards / actual_bot / fallback_reason */ }
}
```

### 7.1 `GET /api/game/config`

返回开局设置面板要填的所有合法参数（前端不要硬编码，避免与后端脱节）。

**响应示例（2026-08-30：bots 扩至 4 项 + 新增 `nash_rule_model`）**：

```json
{
  "num_cards": { "min": 1, "max": 13, "default": 13 },
  "bots": [
    { "id": "random",     "label": "Random · 纯随机 (baseline)",                                             "max_n_for_exact_nash": 13, "nash_rule_model": null    },
    { "id": "heuristic",  "label": "Heuristic · 启发式 (出价≈奖金比例 + carry 适配)",                         "max_n_for_exact_nash": 13, "nash_rule_model": null    },
    { "id": "nash",       "label": "Nash · 精确纳什 · 经典平局弃奖牌型 (仅 N ≤ 7, carry>0 会诚实回落)",        "max_n_for_exact_nash": 7,  "nash_rule_model": "classic" },
    { "id": "nash_carry", "label": "Nash · 精确纳什 · Carry-Over 平局滚入奖牌型 (仅 N ≤ 4 默认)",              "max_n_for_exact_nash": 4,  "nash_rule_model": "carry"   }
  ],
  "card_display": { "1": "A", "11": "J", "12": "Q", "13": "K" }
}
```

| 字段 | 前端用途 |
|---|---|
| `max_n_for_exact_nash` | 选中对应 bot 时，给 N 输入框标红/禁用超上限值。Nash-classic: 7；Nash-carry: 4；random/heuristic: 13（无上限）。 |
| `nash_rule_model`（**新增**） | 三态枚举：`"classic"` / `"carry"` / `null`。<br />null = 该 bot 无"精确 Nash 奖牌模型"。<br />前端可用此在副标题显示：「精确 Nash · 经典(平局弃奖)」 vs 「精确 Nash · Carry-Over(平局滚入)」。 |
| `label` | 中文标签直接可渲染，已经带规则差异 & N 上限提示。 |

### 7.2 `POST /api/game/new`

创建新游戏。**请求体是可选的**（缺 body → 13 + Random，兼容最老的前端 & curl 快速测试）。

```json
{ "num_cards": 5, "bot_type": "nash" }           // Nash-classic 默认 N≤7
{ "num_cards": 4, "bot_type": "nash_carry" }     // Nash-carry 默认 N≤4
```

| 字段 | 默认 | 校验失败 |
|---|---|---|
| `num_cards` | 13 | 不在 `[1, 13]` → **422** `detail: "num_cards must be in [1, 13], got n"` |
| `bot_type` | `"random"` | 不在 `BOT_TYPES={random, heuristic, nash, nash_carry}` → **422** `detail: "Unknown bot_type 'xxx'. Must be one of [...]"` |

**响应**：`state + last_round(null) + meta`。`meta` 字段：

| meta 字段 | 说明 |
|---|---|
| `num_cards` | 实际使用的 N（校验通过后） |
| `requested_bot` | 用户在下拉选的 bot_type（即使回落也保留原值，方便前端副标题说明） |
| `actual_bot` | 真正实例化的 bot。回落情形：<br />• `nash` 且 N>7 → `heuristic`<br />• `nash_carry` 且 N>4 → `heuristic`<br />• 其它 = `requested_bot` |
| `actual_bot_label` | 前端副标题直接显示的中文字串（对应 actual_bot） |
| `fallback_reason` | **严格语义红线（2026-08-30 重构）**：非 `null` ⇔ `actual_bot != requested_bot`。<br />绝不把"Nash 冷启动耗时"塞进这个字段（用 `nash_warmup_sec` 替代）。<br />典型内容例：<br />`所选精确 Nash 为「Nash-carry…」；N=5 超过 NASH_CARRY_MAX_N=4 (平局滚入奖牌型)，自动回落为 Heuristic 启发式。` |
| `nash_warmup_sec`（**新增**） | 精确 Nash 开局冷启动耗时（秒，float）。非精确 bot = `null`。<br />慢 ≠ 回落：前端可显示"加载 5.6s，策略已缓存"的信息条，但不应显示黄色"回落警告"。 |
| `nash_rule_model`（**新增**） | 三态：若 `actual_bot ∈ {nash, nash_carry}` 且 **真的跑了精确未回落** → `"classic"` / `"carry"`；一旦回落（actual_bot = heuristic）→ `null`。<br />前端可据此决定"AI 徽章"显示「精确-classic」/「精确-carry」/「Heuristic 回落中」。 |

### 7.3 `GET /api/game/state`

无需参数。如果未开局 → **400** `detail: "No active game. Call POST /api/game/new first."`

### 7.4 `POST /api/game/play`

人类点击某张牌触发。请求体：

```json
{ "action": 7 }       // 牌的数值（不是显示名！ 7 就是 7；A=1, J=11, Q=12, K=13）
```

后端处理流程（与 Goofspiel 规则严格对齐）：

1. 校验 `action` 在 `env.legal_actions(PLAYER_0)` 里 → 不在 → **400** `Illegal action ...`
2. **Bot 基于 step 前状态** 独立调用 `choose_action_with_policy(env, PLAYER_1)` 生成 `bot_action + policy_info`（同时原则）
3. 双重防线：若 bot_action 非法 → **500** `Bot 'type' produced illegal action ...` （说明 bot 有 bug，避免写脏数据）
4. 一次 `env.step({p0:action, p1:bot_action})` 原子结算
5. 返回 `state + last_round + meta`，其中 `last_round.ai_policy` 就是前端要画的 **AI 决策分布条**数据、`last_round.human_policy` 就是前端要画的**你的反事实三色条**数据：

```json
"ai_policy": {
  "bot_type":     "nash",                 // ∈ {"random","heuristic","nash","nash_carry"}（2026-08-30 新增第 4 项）
  "value":        0.4123,                 // 精确 Nash = V；Random/Heuristic (或回落) = NaN
  "note":         "Nash-classic(tie-discard) · bot=row, opponent=col · bot expected score-diff vs human = +0.412",
  "distribution": [[1, 10.0], [3, 50.0], [5, 40.0]]
},
"human_policy": {
  "legend":       "Length = delta=human_score_diff_if_we_play_h'. Colors: GREEN=win, YELLOW=tie, RED=lose. Border=--- is the card human *actually played*.",
  "total_prize_at_stake": 23,
  "bars": [
    { "card": 1, "card_display": "A",  "outcome": "lose", "delta": -23, "was_played": false },
    { "card": 5, "card_display": "5",  "outcome": "tie",  "delta":   0, "was_played": true  },
    { "card":13, "card_display": "K",  "outcome": "win",  "delta": +23, "was_played": false }
  ]
}
```

> - `ai_policy.bot_type` = 当前回合实际的决策引擎 ID。关键差异：
>   • **Nash-classic 选 N=3，R1 平局 carry>0 → R2 这个字段 = `"nash"`（但 note 以 `[Nash-classic fallback…]` 开头，实际用了 Heuristic 分布）—— 诚实回落契约**。
>   • **Nash-carry 选 N=3，R1 平局 carry>0 → R2 这个字段 = `"nash_carry"`，note 以 `Nash-carry-over(tie→rollover; carry_in=2)…` 开头，value = 真精确 V。**（carry>0 不回落契约）
>   • 回落时 bot_type 保持用户选择（用 note 透明化"我们用了 heuristic 分布"），不突然换 bot 标签让前端困惑。
> - `ai_policy.distribution` 每条是 `[card_value, 出手概率%]`，和 = 100 ± 浮点误差。
> - `ai_policy.note` 前缀规则（快速做 UI 徽章）：
>   | 前缀 | 含义 |
>   |---|---|
>   | `Uniform random …` | RandomBot |
>   | `[carry=N …] Target rank …` 或 `Tail go-for-broke …` | HeuristicBot（含 carry 适配） |
>   | `Nash-classic(tie-discard) …` | 经典 Nash，真精确 |
>   | `Nash-carry-over(tie→rollover; carry_in=N) …` | Carry Nash，真精确（carry>0 也精确 ✓） |
>   | `[Nash-classic fallback …]` | 原选 nash 但规则不兼容（carry>0）或 N>7 → 诚实回落 heuristic |
>   | `[Nash-carry-over fallback …]` | 原选 nash_carry 但 N>4 → 回落 heuristic |
> - `human_policy.bars[*].outcome ∈ {"win","tie","lose"}` 语义：**如果人类改出 card，AI 仍打它实际打的那张 b_real**，真实环境 step() 后会发生的结果；`delta` = 人类单步分差（负 = 输 `prize_at_stake`，正 = 赢 `prize_at_stake`，平 = 0）；`was_played=true` 就是人类实际出的那张，前端应该用 `border: 2px dashed #111` 画出来便于比较。

### 7.5 state 字段表（HTTP 返回用，更偏 UI）

| 字段 | 类型 | 说明 |
|---|---|---|
| `round` | `int` | 当前回合号（已揭晓） |
| `num_cards` | `int` | 总局数 N |
| `current_prize` | `int \| null` | 奖金面值（数值） |
| `current_prize_display` | `str \| null` | 奖金牌显示名（A/J/Q/K） |
| `carry_pool` | `int` | 平局累计滚入奖池（≥ 0） |
| `carry_pool_display` | `str` | 前端直接显示：carry_pool=0 → `0`；否则 `"{value} (+{value} rollover)"` 中文等价格式，详见实际 UI |
| `total_prize_at_stake` | `int \| null` | 当前回合胜方一次拿的总额 = current_prize + carry_pool |
| `total_prize_at_stake_display` | `str \| null` | 前端显示串（例如 `"Q+J=23"` 或 `"23"`），由后端 helper `_stake_display_text` 生成 |
| `scores` | `{human, bot}` | 累计分 |
| `remaining_cards.human` | `[{value, display}…]` | 你的剩余牌（value=数值，display=可显示） |
| `remaining_cards.bot` | `[{value, display}…]` | AI 剩余牌 |
| `remaining_prizes_display` | `[str…]` | 还没揭晓的奖品牌（排序列表） |
| `used_cards.human` / `bot` | `[{value, display}…]` | 已用牌（历史顺序） |
| `history[]` | 每回合对象 | 每回合 R1..RN，除胜负平外还带 `carry_in/out/prize_at_stake/discarded` 与后端生成 `result_text`（carry-aware 中文文案） |
| `done` / `result` | bool / str | 是否结束 & 结果（与 §5 一致） |

### 7.6 HTTP 错误码总表

| 码 | 典型场景 |
|---|---|
| **400** | 未开局就 `/state` 或 `/play`；游戏已结束仍 `/play`；人类出牌非法 |
| **422** | Pydantic schema 校验失败：`action` <1 或 >13（即使是合法牌值，也会先过 schema）；`/new` 的 N 越界 / bot_type 未知 |
| **500** | Bot 返回了非法动作（开发/自定义 bot 时才会出现） |
| **404** | 没这条路由 |
| **405** | 方法错（例如 POST `/state`） |

### 7.7 完整 E2E 调用脚本（requests）

```python
import requests

BASE = "http://127.0.0.1:8001"  # 根据你启动时终端打印的 URL 改

# 1. 查配置（可选）
cfg = requests.get(f"{BASE}/api/game/config").json()
print(f"Valid N ∈ [1, {cfg['num_cards']['max']}], bots = {[b['id'] for b in cfg['bots']]}")

# 2. 新游戏：N=7 + Nash
r = requests.post(f"{BASE}/api/game/new",
                  json={"num_cards": 7, "bot_type": "nash"}).json()
print("meta:", r["meta"])

rounds = 0
while not r["state"]["done"]:
    # 策略：人类永远打最小的合法牌
    my_cards = sorted(c["value"] for c in r["state"]["remaining_cards"]["human"])
    action = my_cards[0]
    r = requests.post(f"{BASE}/api/game/play", json={"action": action}).json()
    rounds += 1
    lr = r["last_round"]
    ai_type   = lr["ai_policy"]["bot_type"]
    ai_note   = lr["ai_policy"]["note"]
    ai_value  = lr["ai_policy"]["value"]
    ai_dist   = lr["ai_policy"]["distribution"]
    print(f"R{lr['round']} prize={lr['prize_display']}: "
          f"me={lr['human_action_display']} bot={lr['bot_action_display']} "
          f"-> {lr['winner']}  |  AI[{ai_type}] V={ai_value:+.3f}  dist={ai_dist}  note=({ai_note[:50]}..)")

final = r["state"]
print(f"\nDone after {rounds} rounds: result={final['result']}  scores={final['scores']}")
print("History entries:", len(final["history"]))
```

---

## 8. 测试体系

### 8.1 测试分类

| 文件 | 测试数 | 覆盖内容 | 目标 |
|---|---|---|---|
| `tests/test_env.py` | 23 | 手牌/奖品堆初始化、合法性校验、平局→carry 累计、连平累计、平后胜拿全额 stake、末轮平局单/连带 carry 丢弃、history 不变量、observation 字段 match、奖池守恒（总分+discard=N(N+1)/2）、单局 13 轮、任意 N 轮数、fixed seed 复现、result 正确区分 draw/win | 守住**规则契约**，任何 PR 必须全绿 |
| `tests/test_app.py` | 46（含 counterfactual 专项） | `/config` 选项合法、`/new` 对 N/bot_type 校验、Nash N>7 回落、carry-aware **win delta == prize_at_stake**（非单轮 prize）、完整 13 轮 HTTP E2E、非法动作返回 400 / 422 in、结束后 play 400、**反事实 outcome 与 env.winner 语义 + delta 字节一致** 100 动作全过 | 守住**前后端契约**；win delta 断言挂 = carry 结算链路漏接 |
| `tests/test_solver.py` | ~20 | Nash solver：V(A,B,R) = -V(B,A,R) 零和对称性、策略分布归一化（∑x ≈ 1）、复杂度 estimate() 对 N=1..8 精确等于 OEIS A000172 C(N)、N=13 preflight RED/BLACK 中断保护 + force 绕过、N=3 per-state **xᵀM ≥ V, My ≤ V** Nash 不变式全通过 | 守住**数学正确性**；solver **不会在 carry_pool>0 时被真实调用**（NashBot 已提前回退，这是有意设计，非 bypass） |
| `tests/test_cxx.py` | 4（C++ 扩展未编时 auto-skip） | OEIS C(N) 与 Python estimate 一致；C++ VectorizedEnv × 单局 vs Python GoofspielEnv 100 种子，每轮 score/bot_action/winner 完全一致；C++ 4096 envs × 256 step 总耗时 < 30s（吞吐基线）；C++ solve_exact_nash(N=3).policy_map 对每个 state 再建 M 矩阵验证 xᵀM≥V 与 My≤V 不变式 + 根值 0 | 守住**C++/Python 跨后端对等契约**（Reference ↔ Fast Backend cross-check 原则） |

### 8.2 运行

```bash
pytest                           # 全部（C++ 扩展未编时 test_cxx.py auto importorskip 安全跳过）
pytest tests/test_env.py -v      # 只跑环境契约
pytest tests/test_app.py -v      # 只跑 HTTP 端点
pytest tests/test_solver.py -v   # 只跑数学正确性
pytest tests/test_cxx.py -v      # 只跑 C++ 加速后端（需先编译 goofspiel._core.{pyd,so}）
```

**当前基线状态**（2026-08-30 版本）：纯 Python 三项文件共 **69 passed / 0 failed / 1 deselected**（1 条 slow Nash N=5 value 测试被标记为 slow，可单独 `pytest -m slow` 运行）。

---

## 9. 常见问题（FAQ）

### Q1：`[Errno 10013 / 10048]` 端口绑不上？
就是端口被别的进程占了。**解决**：不用手动查，现在 `python app.py` 会自动扫往后 50 个端口，等它打印 `starting server on http://x:port` 就行。若想手动指定：`python app.py --port 9000`。

### Q2：我选了 Nash + N=13，为什么实际上是 Heuristic？
本项目**同时存在两个独立的精确 Nash 奖牌型**，各自阈值不同，超过上限都会诚实回落 Heuristic：

| 下拉选项（`bot_type`） | 规则模型 | 精确解上限常量 | 默认阈值 | `meta.fallback_reason` 超阈值时包含字符串 | UI N 输入框上限（`bots[].max_n_for_exact_nash` 下发） |
|---|---|---|---|---|---|
| 第 3 项：Nash · 精确纳什 · 经典平局弃奖牌型（`nash`） | 经典 Goofspiel「平局丢弃奖」 | `NASH_MAX_N` | 7 | `NASH_MAX_N=7` | 7 |
| 第 4 项：Nash · 精确纳什 · Carry-Over 平局滚入奖牌型（`nash_carry`） | 平局→下一轮 carry（唯末轮丢弃） | `NASH_CARRY_MAX_N` | 4 | `NASH_CARRY_MAX_N=4 (平局滚入奖牌型)` | 4 |

> 为什么 nash_carry 上限更严？因为它的状态多了一维 `carry`，可达状态随 N 成 ~O(N²) 倍乘爆炸：N=4 ≈ 3,806 状态(GREEN 可解)，N=7 ≈ 3,043,840 状态(ORANGE)。具体分级请查 §6.4.2 的 Preflight Complexity 表。

回落之后，服务器 `meta.fallback_reason` 会写明对应的上限常量名与实际 N 值；Web UI 副标黄色警告条也会同步打印，**不会默默偷换策略**。

### Q3：Nash 预计算很慢怎么办？
N 越大越慢（大致组合爆炸），但结果会**类级缓存**（`NashBot._policy_cache`）：同 N 的第二局开始 0ms。如果你要反复打 N=7 的 Nash，可以接受一次冷启动成本。想更快就用 N≤5 或者选 Heuristic。

### Q4：平局时奖金去哪了？为什么 UI 有个「累计奖池 Carry Pool」栏位？
本项目采用 **平局滚入下一轮（carry-over）** 变体，而不是经典 Goofspiel 的"平局丢弃"。规则三分支：
- 胜负 → 胜者一次拿 `round_prize + carry_in` 全额，carry 清零；
- 平局 & 非末轮 → 双方 0 分，`carry_pool = round_prize + carry_in` 滚入下一轮；
- 平局 & 末轮 → 没有下一轮可滚，**这是整局唯一会真正丢奖金的情况**，UI 横幅会写「Tie (final round). … discarded · no rollover」。
末轮平局才会导致人类总分 + Bot 总分 < N(N+1)/2；其它所有场景，奖池只是在玩家之间重新分配或暂存到 carry，**不会凭空蒸发**。

### Q5：Nash AI 在 R1 显示「精确纳什分布」，R2 平局后突然变成 Heuristic？是 bug 吗？
**这取决于你开局选的是哪个 Nash 奖牌型**，两条路径的行为完全不同（这正是本项目「双 Nash 铁律」的核心透明性承诺）：

---

#### 情况 1：你选的是 `bot_type="nash"`（第 3 项 · 经典平局弃奖牌型）
→ **不是 bug，而是刻意的诚实回退契约**。
经典 `GoofspielExactSolver` 是离线按「平局丢弃奖」规则推导的，子状态里**根本没有 carry_pool 这一维**。一旦 `carry_pool > 0`，所谓"精确分布"就不再是该奖牌型下的均衡（甚至会系统性失真）。因此本项目**明确禁止**它在 carry 状态下输出伪装的精确分布；它会立即回落到已适配 carry 的 HeuristicBot，并在 AI 面板 note 前缀写：

```
[Nash-classic fallback to Heuristic · carry_pool=N 存在平局累计奖池；当前 Nash 精确 solver 为经典「平局弃奖」奖牌型推导，未适配 carry-over 规则。请切换到 bot_type='nash_carry'。]
```

同时 `meta.fallback_reason` 也会写入完整原因，前端黄色警告条同步显示。

> 旧建议（对 classic 仍适用）：想强制连续看纳什分布、不想回落？那就别让任何一回合平局 —— 这正好符合经典奖牌型的适用边界。

---

#### 情况 2：你选的是 `bot_type="nash_carry"`（第 4 项 · Carry-Over 平局滚入奖牌型 · 默认仅 N ≤ 4）
→ **不会回落，R2/R3…平局之后仍然精确 Nash ✓**。
这是 MSG7「实现两个版本的nash」新增的第二套独立求解器 `GoofspielCarrySolver`，**离线就是按 carry-over 奖牌型三分支全量求解**（胜负拿 stake；平非末滚入；平末丢弃）。状态天然带 `carry` 维度，`carry>0` 是它合法的 reachable child，因此：

- 即便 R1 打平，R2 开始 `carry_pool = 2`，AI 面板的 `ai_policy.bot_type` 仍然是 `nash_carry`；
- `ai_policy.note` 前缀会显示 `Nash-carry-over(tie→rollover; carry_in=N) · bot=row, opponent=col · bot expected score-diff vs human = V.VVV`（绝不含"fallback"或"回落"字样）；
- `meta.fallback_reason = None`，`meta.nash_rule_model = "carry"`。

**代价**：因为状态多了一维 carry，默认精确上限收紧到 `NASH_CARRY_MAX_N = 4`；N > 4 会回落 Heuristic（和 Q2 的阈值回落一致）。

---

**解决办法总结**：如果你不想在平局 carry>0 时突然掉到 Heuristic，**开局就把 AI 下拉切到第 4 项**
「`Nash · 精确纳什 · Carry-Over 平局滚入奖牌型 (仅 N ≤ 4 默认)`」（即 `bot_type=nash_carry`）即可。

### Q6：怎么让局域网朋友也访问？
```bash
python app.py --host 0.0.0.0 --port 9000
```
然后把 `http://<你电脑局域网IP>:9000` 给朋友。**记得在你电脑防火墙放行 9000 入站**。

### Q7：想要 100% 可复现的一局（调试 AI 用）？
- 环境：`GoofspielEnv(..., rng=random.Random(seed))`
- Bot：`RandomBot(rng=random.Random(seed2))`、`HeuristicBot(rng=...)`、`NashBot(rng=...)`
- Nash 的**策略本身**是 deterministic（由 solver 保证），随机仅在按 x* 抽样时影响最终出牌，所以只要 seed 固定就能复现。

### Q8：为什么 reward 是"胜者 +p, 负方 0"，不是零和？
默认保留**真值得分**，便于审计（总分 ≤ 91，若末轮平局还要 + discard 才等于 91）、网页横幅展示。要零和训练的话，在外部加 **1 行 wrapper**：
```python
def zero_sum_wrap(rewards, stake):
    if rewards["player_0"]:   return {"player_0":  stake, "player_1": -stake}
    if rewards["player_1"]:   return {"player_0": -stake, "player_1":  stake}
    return {"player_0": 0, "player_1": 0}
```
> 注意 carry-over 下应该传 `stake = prize_at_stake`（= round_prize + carry_in）而不是单轮 prize，否则零和收益会低估"平后胜"那轮的差值。

这样既能保留真值得分，又不影响 RL 算法收敛。核心环境不内嵌，是为了**遵循单一职责**。

### Q9：C++ 扩展怎么编译？最快最简单一条命令？
打开 **x64 Native Tools Command Prompt for VS 2022**（Windows，避免 cl.exe 找不到）后执行：

```powershell
pip install --upgrade cmake pybind11 ninja cmake-build-extension numpy scipy torch
pip install -e .
```

`pip install -e .` 会用 `pyproject.toml` 里的 cmake-build-extension 配置自动跑 `cxxgoof/CMakeLists.txt`，把产物 `goofspiel/_core.cp3XX-win_amd64.pyd` 直接放到 `goofspiel/` 目录（import 就能用）。

如果你想精细控制构建或接入原生 HiGHS（~5× 更快的 Nash 精确解），三种手动构建方式 + 详细 FAQ 坑点（`cl.exe` 找不到、pyd 路径不对、HiGHS link 失败）都在 [order/C++模块编译与训练集成指南.md](C++模块编译与训练集成指南.md) 里。

编完后冒烟：
```powershell
python -c "from goofspiel import _core; c=_core.estimate_complexity(5); print('C(5)=',c['C_N']); from goofspiel._cxx import cpp_solve_with_policy as s; print('N=3 exact value=', s(3).value)"
```

### Q10：我想跑 PPO 自博弈训练，怎么启动？预期输出什么？
```powershell
# 编了 C++（推荐）：4096 并行环境 × 100k 步 = 约 10 次 PPO update，5~15 分钟内跑完
python scripts/train_n5_ppo.py --num-cards 5 --total-timesteps 100000 --seed 1

# 没编 C++：退化成 Python 串行 × 256 env，慢一些但证明 pipeline 通
python scripts/train_n5_ppo.py --num-cards 5 --num-envs 256 --total-timesteps 20000
```

预期 stdout 每 1 update 一行：
```
[upd   1/10] steps=  1048576  SPS= 31241  |  avg_return(±200)=+1.847  |  pg_loss=-0.0112 v_loss=+3.218 entropy=+1.442
...
[upd  10/10] steps= 10485760  SPS= 34891  |  avg_return=+6.213        |  pg_loss=-0.0081 v_loss=+2.714 entropy=+0.873
[goof] checkpoint saved to checkpoints/ppo_n5_seed1.pt
```

指标解释：
- `SPS`：每秒处理的 env-step 数（C++ VectorEnv 下应该 ≥ 15000，普通机器 ≥ 30000）。
- `avg_return`：近 200 局平均 `分差(你 − 对手)`。范围约 [−91, +91]；训练过程中从 0 附近稳步向正方向爬升 = 在学习。
- `pg_loss / v_loss / entropy`：标准 PPO 三项，用来诊断模型是否塌缩（entropy→0 或 v_loss 爆炸）。
- 最后会把权重存到 `checkpoints/ppo_n5_seed1.pt`，里面是 dict：`{model_state_dict, optim_state_dict, args, global_step, last_200_avg_return}`，后续接 `bots.py::TorchPolicyBot` 即可前端对战使用。

全规范 N=13 训练路线 (`N=5→9→13 + GNN骨干 + FSP策略池 + Nash-MCTS`) 参考 `order/` 下 13 份规范设计文档。

### Q11：Goofspiel-13 完整训练工程代码怎么验收？

本次工程不是启动训练，而是把 `order/` 里的完整训练、推理、评测与可观测性需求落成可迁移代码。PPO 仍保留为 baseline/demo；Goofspiel-13 主路线使用下面这些入口。

| 模块 | 路径 | 说明 |
|---|---|---|
| 纯状态转移 | `goofspiel/game/state.py` | immutable `GameState` + 无 RNG 的 `transition()`，Search/Teacher/Learning 共用同一规则语义 |
| 数据契约 | `goofspiel/training/schema.py`、`goofspiel/training/data.py` | `MAX_N=13`、rank/index 边界函数、schema version、`state_hash`、分池样本 dataclass |
| 神经模型 | `goofspiel/models/` | 变量 N、Transformer/GNN/CNN/LSTM/Mamba 风格结构、Robust/Adaptive 分支隔离、joint-action Q |
| 学习原语 | `goofspiel/learning/` | RM+、Matrix solution、NeuRD raw-logit loss、TD(lambda)、joint V-trace、two-hot、teacher priority、opponent/style/symmetry loss |
| 推理工具 | `goofspiel/reasoning/` | Matrix Nash、Exact、Exact BR、SM-MCTS、GT-CFR、LeafEvaluator、ToolRouter、Final Decision Protocol |
| 训练流水线 | `goofspiel/training/coordinator.py`、`goofspiel/training/stages.py` | P0-P7 stage-gated pipeline，含 corpus/teacher/robust/adaptive/league/red-team/evaluate |
| 训练任务 | `goofspiel/training/pretraining.py`、`goofspiel/training/teacher_system.py` | P1 player-swap/transition/joint-outcome/masked-action/opponent/style；P2/P3 teacher ensemble/filter/SFT anchors |
| 数据池 | `goofspiel/training/datasets.py`、`goofspiel/training/replay.py`、`goofspiel/training/redteam.py` | GameCorpus、ExactDataset、TeacherDataset、Robust/Adaptive/Opponent/Failure/Reanalysis 分池持久化 |
| Checkpoint | `goofspiel/training/checkpoint.py`、`goofspiel/training/checkpoint_registry.py`、`goofspiel/training/resume.py` | `latest`、`best_robust`、`best_raw`、`best_search`、`best_adaptive`、`best_generalization`、`best_opponent_model`、`teacher_ema` 类型与 checksum/resume 校验 |
| 分布式服务器 | `goofspiel/training/distributed.py`、`scripts/plan_h200_training.py`、`configs/training/h200_8gpu.yaml` | 八卡 H200 角色分配、stage 顺序校验、`torchrun` 命令生成、DDP rank/local_rank 支持 |
| 评测基线 | `goofspiel/training/benchmark.py`、`goofspiel/training/baselines.py` | E0-E7 Arena、baseline registry、promotion gates、`summary.json/summary.md`、主表、search/adaptive/opponent/generalization 表 |
| 可观测性 | `goofspiel/observability/` | 结构化 `BaseEvent`、JSONL sink、metric aggregator、system/GPU metric probe |
| 配置 | `configs/model/full.yaml`、`configs/learning/default.yaml`、`configs/training/pipeline.yaml` | 固化模型、学习、训练阶段默认配置 |
| 测试 | `tests/unit/models`、`tests/unit/learning`、`tests/unit/reasoning`、`tests/unit/training`、`tests/unit/observability` | shape、leakage、loss、router、schema、benchmark、distributed、logging 等 |

服务器生产环境建议：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install -r requirements-train.txt
python -m pip install -e .
```

生成八卡 H200 训练命令计划，但不启动训练：

```bash
python scripts/plan_h200_training.py --gpus-per-node 8 --steps 100000 --batch-size 512
```

正式训练由你在服务器上按阶段执行。例如 P4 主 RL 阶段：

```bash
torchrun --nnodes 1 --nproc_per_node 8 scripts/train_goofspiel_full.py \
  --artifact-dir artifacts/runs/h200_full \
  --stage stage4_robust_rl \
  --steps 100000 \
  --batch-size 512 \
  --n-cards 13 \
  --device cuda
```

`torchrun` 下训练代码会读取 `RANK/WORLD_SIZE/LOCAL_RANK`，用 DDP 包装神经模型，只有 rank0 保存 checkpoint。这样不会出现 8 个进程同时覆盖同一个 checkpoint 的问题。

本机只做轻量正确性验证：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
python -m compileall goofspiel scripts tests/unit -q
python -m compileall goofspiel scripts tests -q
python -m pytest -q
python scripts/train_goofspiel_full.py --dry-run
python scripts/plan_h200_training.py --gpus-per-node 8 --steps 10 --batch-size 4
python scripts/validate_requirements_trace.py
```

这里的 CUDA v13.0 路径只代表当前 Windows 本机测试环境。迁移到八卡 H200 服务器时，不要把这个路径写进代码；服务器按实际驱动/CUDA/PyTorch wheel 或集群镜像配置。

当前本机轻量验证结果：`python -m pytest -q` 为 **135 passed / 12 skipped**；skipped 来自本机系统 Python 的 torch CUDA DLL 依赖异常或需要 torch/C++ 的专项。生产服务器装好 CUDA 版 PyTorch 后，torch 相关模型/学习/搜索测试会自动执行。

---

## 10. 扩展指南（给二次开发者）

### 10.1 接入自定义 Bot

```python
from goofspiel.bots import BaseBot
import numpy as np

class MyBot(BaseBot):
    def _choose(self, env, player):
        legal = sorted(env.legal_actions(player))
        # 自定义策略：这里返回 (action, info_dict)
        p = np.ones(len(legal)) / len(legal)
        chosen = int(np.random.choice(legal, p=p))
        dist = [[c, float(p[i]*100)] for i, c in enumerate(legal)]
        return chosen, {
            "distribution": dist,
            "value": float("nan"),
            "bot_type": "my_bot",
            "note": "My first bot :)",
        }
```

然后把它注册到 `create_bot` 工厂（改 `bots.py` 的 `BOT_TYPES/BOT_DESCRIPTIONS`）和 `app.py` 的 `/api/game/new`，前端下拉框就会自动出现（因为下拉是 `/api/game/config` 动态生成的）。

### 10.2 Zero-Sum RL 环境包装（30 行）

```python
class ZeroSumGoofspielWrapper:
    def __init__(self, env: GoofspielEnv):
        self.env = env
    def reset(self): return self.env.reset()
    def step(self, actions):
        obs, r, done, info = self.env.step(actions)
        prize = self.env.history[-1]["prize"]
        if   r["player_0"]: zs = {"player_0": prize, "player_1": -prize}
        elif r["player_1"]: zs = {"player_0": -prize, "player_1": prize}
        else:               zs = {"player_0": 0, "player_1": 0}
        return obs, zs, done, info
```

### 10.3 多用户 / 多房间

目前 `app.py` 用全局 `_env / _bot`，只适合单 demo。升级步骤：
1. 加会话层：`_sessions: Dict[str, SessionTuple]`（session_id → `(env, bot, bot_type, N, fallback_reason)`），session_id 可以用 uuid4 生成给前端存 `localStorage` 或 URL param。
2. `/api/game/new` 返回 `session_id`，后续 `/state` `/play` 要求 header `X-Session-Id`。
3. 加 LRU/TTL 清理过期会话，避免内存泄露。
4. 需要持久化就把 session 扔 Redis / 数据库，读写同一 JSON。

### 10.4 新增 API / 修改接口字段
记住两条铁律：
- **后端永远是真值来源**：前端不能自己算分数/胜负/轮数，只能渲染 server 返回的 `state`。
- **动作必须同时**：不要为了"性能"先写人类动作再算 AI，要先在 step 前状态拿到两边独立动作，再一次 env.step。Goofspiel 的规则本质就是不完全信息博弈同时揭示，任何打破同时性都会产生数据污染/作弊空间。

---

> 文档版本：v1.2  
> 最后更新：2026-08-31（补 Goofspiel-13 完整训练工程入口、H200 torchrun 计划、DDP rank0 checkpoint、checkpoint registry、benchmark 表、trace 验收）  
> 作者：陈子聪 (Chen Zicong)
