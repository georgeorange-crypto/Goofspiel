# Goofspiel-13 智能体总工程实施指南
## ——从设计规格到可训练、可搜索、可部署系统的 Tech Lead / Developer Execution Guide

> 本文是整个项目的工程总纲。  
> 它不替代《模型结构设计书》《学习方法设计书》《训练流程设计书》《搜索与数学求解设计书》，而负责告诉开发者：**怎样把这四份设计按正确顺序真正做出来。**

---

# 0. 开发者首先必须理解的事情

这个项目不是：

```text
写一个环境
→ 写个 PPO
→ train.py
→ model.pt
```

也不是：

```text
AlphaZero clone
```

更不是：

```text
OpenSpiel + RLlib 拼起来
```

最终系统实际上有五层：

```text
┌────────────────────────────────────────────┐
│              Product / UI Layer            │
│     Web UI · Detector · Benchmark UI       │
└─────────────────────┬──────────────────────┘
                      │
┌─────────────────────▼──────────────────────┐
│           Agent Reasoning Layer             │
│ Matrix Nash · Exact · SM-MCTS · GT-CFR     │
│ Opponent BR Search · Tool Router · Safe LP │
└─────────────────────┬──────────────────────┘
                      │
┌─────────────────────▼──────────────────────┐
│             Neural Model Layer              │
│ Transformer · GNN · Matrix CNN             │
│ LSTM · Mamba · Robust / Adaptive Heads     │
└─────────────────────┬──────────────────────┘
                      │
┌─────────────────────▼──────────────────────┐
│            Learning / Training Layer        │
│ SSL · Semi-supervised · SFT · RL           │
│ Distillation · League · Red Team           │
└─────────────────────┬──────────────────────┘
                      │
┌─────────────────────▼──────────────────────┐
│              Game / Math Core               │
│ Env · Bitmask State · Exact DP · LP         │
│ Transition · Rules · Evaluation             │
└────────────────────────────────────────────┘
```

工程实施必须：

\[
\boxed{\text{自下而上}}
\]

因为上层永远依赖下层正确性。

---

# 1. 四份设计文档的优先级

开发者遇到冲突时，按下面顺序解释。

### 第一优先：数学/规则不变量

例如：

- simultaneous action；
- tie prize discarded；
- normalized score-difference utility；
- Robust 分支不得读取 opponent history；
- \(Q(s,a,b)\) 不得降成 \(Q(s,a)\)。

这是最高约束。

---

### 第二优先：各专项设计书

包括：

1. Model Architecture
2. Learning Algorithms
3. Training Pipeline
4. Search / Exact / Tool Layer

---

### 第三优先：本文工程实现建议

本文决定：

- 技术栈；
- 代码组织；
- 开发顺序；
- CPU/GPU 分工；
- 性能优化路线。

---

### 第四优先：框架默认行为

PyTorch、RLlib、OpenSpiel 等框架的默认行为**绝不能覆盖我们的设计**。

如果框架和设计冲突：

\[
\boxed{\text{改框架适配层，不改设计}}
\]

---

# 2. 总技术栈

建议核心技术栈冻结如下。

| 层 | 推荐 | 状态（2026-08-30） |
|---|---|---|
| 主语言 | Python 3.11 | ✅ 交付（3.10+ 实际兼容） |
| 神经网络 | PyTorch | ✅ PPO demo 最小训练闭环（2 层 MLP） |
| GPU | CUDA | ⏳ 预留，当前 PPO demo 在 CPU 端也能跑 smoke |
| 模型编译 | `torch.compile` | ⏳ 未启用，demo 用 eager |
| 自定义 GPU Kernel | Triton，必要时 CUDA C++ | ⏳ 保留给 N=13 GNN 骨干 |
| Mamba | `state-spaces/mamba` | ⏳ 待 §Opponent Learning 阶段 |
| Exact 高性能核心 | C++20 + pybind11 | ✅ **提前交付 M3.5**：`cxxgoof/` 含 bitmask state、两阶段递归 canonical solver、符号 cache、M 矩阵按 prize fresh 分配（无跨 p 污染）、GIL-aware scipy HiGHS LP callback（默认）、原生 HiGHS `-DCXXGOOF_USE_HIGHS=ON` 可选 |
| LP Reference Solver | SciPy `linprog(method="highs")` | ✅ 交付，Python/C++ 双后端共用，保证 cross-backend 不变式一致 |
| **Vectorized Env（训练吞吐核心）** | **C++ SoA + Gymnasium 鸭子 API** | ✅ **额外交付**：`VectorizedEnv`（Struct-of-Arrays 热循环 + `restrict` + 自动向量化 + 无堆 per-step 分配）；4096 env × 256 step 基准 5s 级完成（Python 串行 env 吞吐 ~260× 加速比） |
| 配置 | Hydra | ⏳ 当前 PPO demo 用 argparse 最小方案，Milestone M0 切换 |
| 数据格式 | PyArrow + Parquet | ⏳ 下一阶段 GameCorpus |
| 实验跟踪 | MLflow | ⏳ 当前 PPO demo stdout 指标即可 |
| 分布式训练 | PyTorch DDP | ⏳ 未启用（模型 < 8M，按文档 §4 默认不 FSDP/DDP） |
| 多进程 | Python multiprocessing / torch.multiprocessing | ✅ 将来用，当前单机足够 |
| 大规模多机调度 | Ray，可选后期开启 | ⏳ 按文档 §67 「单机先 Python mp，多节点才 Ray」 |
| 环境标准适配 | PettingZoo Parallel API | ⏳ Adapter 保留给 Marlib baseline |
| 博弈参考框架 | OpenSpiel | ⏳ ParityTest 保留给 M2 |
| 单测 | pytest | ✅ 交付（test_env + test_solver + test_app + **test_cxx**，共 69 passed + 4 C++ 专项） |
| 属性测试 | Hypothesis | ⏳ 未用 |
| Profiling | `torch.profiler` + Nsight Systems/Compute | ⏳ 下一阶段 GPU 骨干 |
| Web | FastAPI + HTML/CSS/Vanilla JS | ✅ 交付：开局可选 N=1..13 / AI 型号；ai_policy + **human_policy counterfactual 三色条** |
| 打包 | Docker | ⏳ 未启用 |
| CI | GitHub Actions / 自有 CI | ⏳ 未启用，当前本地 pytest 回归 |

---

# 3. 为什么核心一定选 PyTorch

这个项目需要：

- Transformer；
- GNN；
- CNN；
- LSTM；
- Mamba；
- BF16；
- GPU batched solver；
- ensemble；
- dynamic training losses；
- custom gradient routing；
- distributed training；
-大量研究实验。

PyTorch 是最合适的中心框架。

当前 PyTorch 文档已经把 `torch.compile` 作为主要优化路径，并建议尽可能对不产生过多 graph break 的高层函数或 module 使用 compile；如果与 DDP/FSDP 组合，应优先 compile 内部 module，而不是 distributed wrapper。

因此：

```text
PyTorch eager
↓
先保证正确
↓
torch.compile
↓
Profiler
↓
Triton
↓
必要时 CUDA C++
```

这是性能优化顺序。

禁止：

> 第一周就开始手写 CUDA。

---

# 4. 不要使用 FSDP 作为默认训练方案

模型预计：

\[
6\sim8M
\]

参数。

即使扩展以后：

\[
<50M
\]

也非常小。

FSDP 的核心价值是参数/梯度/优化器状态分片；PyTorch 的 FSDP/FSDP2 正是为这种 sharded training 设计。

我们当前模型根本不是显存瓶颈。

因此默认：

\[
\boxed{\text{Single GPU 或 DDP}}
\]

而不是：

\[
FSDP
\]

。

DDP 是 PyTorch 标准同步多进程数据并行实现。

---

# 5. 甚至不要急着使用 DDP

这是很重要的工程判断。

由于模型非常小：

> **训练吞吐的瓶颈很可能不是模型反向传播。**

更可能是：

- self-play state generation；
- Nash Bellman child expansion；
- Matrix solver；
- Search；
- Exact Solver；
-数据搬运；
-大量小 GPU kernels。

所以有两张 GPU 时，我首先建议：

```text
GPU 0
────────────
Main Learner
Forward
Backward

GPU 1
────────────
Actor Inference
Search Leaf Evaluation
Teacher Ensemble
Reanalysis
```

而不是：

```text
GPU0 + GPU1
DDP learner
```

把第二张卡浪费在同步一个 8M 参数模型。

---

# 6. 什么情况下使用 DDP

P1 Pre-training / P3 SFT 如果发现：

- batch 很大；
- GPU utilization 高；
- learner 是真正瓶颈；

才临时：

\[
2\ GPU\ DDP
\]

。

进入 search-heavy RL 后，通常重新拆成：

\[
Learner\ GPU
+
Inference/Search\ GPU
\]

更划算。

---

# 7. 异构 GPU 不要做 DDP

例如：

```text
H200 + H100
H200 + RTX PRO
H100 + RTX PRO
```

不建议组成一个 DDP group。

因为同步训练受最慢 GPU 限制。

更合理：

```text
H200
→ learner / large teacher batch

H100
→ search / reanalyse / inference

RTX PRO
→ actors / opponent inference / evaluation
```

异构资源应该：

\[
\boxed{\text{按角色分工}}
\]

而不是强行数据并行。

---

# 8. 项目根目录

最终建议使用 monorepo：

```text
goofspiel-ai/
│
├── pyproject.toml
├── README.md
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── model/
│   ├── learning/
│   ├── training/
│   ├── search/
│   ├── hardware/
│   └── experiments/
│
├── goofspiel/
│   ├── game/
│   ├── math/
│   ├── models/
│   ├── learning/
│   ├── reasoning/
│   ├── training/
│   ├── league/
│   ├── evaluation/
│   ├── data/
│   ├── distributed/
│   ├── ui/
│   └── utils/
│
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   ├── regression/
│   └── performance/
│
├── tools/
│
├── scripts/
│
├── benchmarks/
│
└── artifacts/
```

不要最终形成：

```text
train.py
model.py
env.py
utils.py
```

四个巨型文件。

---

# 9. Hydra 负责所有配置

实验参数很多：

- N；
- model size；
- Exact limits；
- search budget；
- loss；
- teacher；
- curriculum；
- GPU topology；
- league。

必须配置化。

Hydra 的核心价值就是 hierarchical config composition 和命令行 override，非常适合研究型系统。

例如：

```text
configs/
  model/full.yaml
  training/posttrain.yaml
  search/medium.yaml
  hardware/2xh200.yaml
```

启动：

```bash
python -m scripts.train \
    model=full \
    training=posttrain \
    search=medium \
    hardware=2xh200
```

禁止把实验参数散在 Python 常量里。

---

# 10. 实验必须可复现

每次 run 自动保存：

```text
full resolved config
git commit
Python version
PyTorch version
CUDA version
GPU names
driver
hostname
random seeds
dataset versions
teacher versions
model checkpoint IDs
```

。

---

# 11. 实验跟踪建议 MLflow

MLflow 可以记录：

- parameters；
- metrics；
- artifacts；
- code/version metadata；
- model checkpoints；

而且支持本地和远程 Tracking Server。

这里我建议：

### 开发期

```text
MLflow + SQLite
```

### 多机以后

```text
MLflow Server
+
PostgreSQL
+
shared/object storage
```

。

不要一开始做复杂 MLOps。

---

# 第一阶段工程：先做 Game Core

# 12. Step E1 — Environment

第一件真正写的代码：

```text
goofspiel/game/
```

必须独立于：

- PyTorch；
- RL；
- Search；
- UI。

---

# 13. 核心 State 必须 compact

内部 authoritative state：

```python
@dataclass(frozen=True)
class GameState:
    n: int

    self_mask: int
    opp_mask: int
    prize_mask: int

    current_prize: int

    self_score: int
    opp_score: int

    round_index: int
    done: bool
```

使用 bitmask。

不要核心 state 使用：

```python
set[int]
```

。

---

# 14. Transition 必须纯函数

关键 API：

```python
next_state, reward = transition(
    state,
    self_action,
    opp_action,
    next_prize=None,
)
```

要求：

- deterministic；
- no side effects；
- no global RNG；
- 可批量化；
- 可从 Python 调；
- 后续可移植 C++。

Search、Exact、Training 全部依赖同一个 transition semantics。

---

# 15. RNG 完全与 Game Logic 分离

规则层：

不产生随机数。

Chance sampler：

单独负责 prize sampling。

正常 gameplay 可以使用：

```python
secrets.SystemRandom()
```

测试：

```python
random.Random(seed)
```

。

---

# 16. PettingZoo 只做 Adapter

PettingZoo 的 Parallel API 专门支持“所有 agent 同时产生 actions，再一起 step”的环境，这和 Goofspiel 当前行动机制匹配。

所以可以实现：

```text
goofspiel/adapters/pettingzoo.py
```

但：

\[
\boxed{\text{PettingZoo 不是 authoritative environment}}
\]

。

内部核心仍使用自己的 compact state。

目的：

- 与 MARL 工具兼容；
- benchmark；
- PPO/MAPPO baseline。

---

# 17. OpenSpiel 必须接入，但定位是 Oracle / Reference

OpenSpiel 是本项目最值得复用的开源项目。

它原生支持：

- zero-sum；
- simultaneous move；
- perfect/imperfect information；
- CFR；
- MCTS；
- exploitability/evaluation；
- Goofspiel。

而核心实现为 C++，同时提供 Python API。

它的开发指南甚至明确把 Goofspiel列为 simultaneous-move game 的参考实现。

项目：

[OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel?utm_source=chatgpt.com)

---

# 18. 但不要让 OpenSpiel 变成我们的核心运行时

原因：

我们的：

- tie rule；
- state representation；
- opponent memory；
- variable-N；
- tool interfaces；
- Exact DP；
- neural full-matrix Q；

都有自己的设计。

正确做法：

```text
Our Environment
        ↕
Reference Adapter
        ↕
OpenSpiel
```

然后做 cross-validation。

---

# 19. 必须建立 `OpenSpielParityTest`

随机生成：

\[
10^4
\]

条合法轨迹。

在确保双方规则参数完全一致的情况下比较：

- legal actions；
- reward；
- score；
- remaining cards；
- terminal result。

如果 OpenSpiel 某个规则选项和我们不同：

不要改变我们的规则。

只关闭对应 parity 部分或写 adapter。

---

# 第二阶段工程：数学核心

# 20. Step E2 — Reference Matrix Solver

首先：

```text
Python + NumPy + SciPy HiGHS
```

。

SciPy 的 `linprog(method="highs")` 使用 HiGHS 线性规划求解器，并支持 dual/primal feasibility tolerance 和 time limit。

所以：

```text
math/reference_nash.py
```

使用 SciPy。

目标不是快。

目标是：

\[
\boxed{\text{可信}}
\]

。

---

# 21. Reference Solver 是所有 GPU Solver 的真值

以后：

```text
GPU RM+
Triton RM+
Search
Exact Solver
```

都和它交叉验证。

永远保留。

---

# 22. Step E3 — Python Exact Solver

第一版：

```text
Python
bitmask
functools.cache/custom dict
SciPy LP
```

。

目的：

\[
\boxed{\text{验证数学正确}}
\]

不是挑战 N=13。

---

# 23. Exact Solver 什么时候下沉 C++

> ## 2026-08-30 更新：**已提前交付（Milestone M3.5）**，原因与原设计略有调整：
>
> 实际工程推进中我们**不是**先等 profiler 暴露 Python 瓶颈（原设计 §23 写法），而是因为用户明确提出"训练模型要 C++ 环境"——这个信号表明系统首先面对的吞吐瓶颈不是 Exact recursion 本身，而是**大规模 self-play 的 state rollout**：当你每做 1 次 PPO update 需要 4096 × 256 = **100 万+ env-steps** 时，即使单个 transition 极便宜，Python 串行 + 每步 dict 分配的开销不可接受。
>
> 因此这次提前下沉 C++，收益分布实际是：
> - 60% 收益 → **VectorizedEnv（SoA + auto-vec）** 批量 transition，训练吞吐 **~260×**
> - 30% 收益 → **Exact 递归 / cache / bitmask 枚举**（C++ dict 比 Python dict 紧凑 3×，递归栈更轻）
> - 10% 收益 → **原生 HiGHS（可选，当前默认用 scipy GIL-safe callback 已能跑）**
>
> 这和"Exact 慢不慢"无关，是训练吞吐倒逼的工程决策 —— 正好也和 §90「Reference Backend / Fast Backend 双后端模式」对齐（Python 永不过时，C++ 只是加速 backend）。

第一版：
```text
Python
bitmask
functools.cache/custom dict
SciPy LP
```

目的：
\[
\boxed{\text{验证数学正确}}
\]

不是挑战 N=13。

---

第二版（已交付 M3.5，cxxgoof/）：
```text
C++20
unordered_map / flat hash map
uint16_t bitmask
pybind11 binding
LP: scipy callback (default)  ·  native HiGHS (optional)
```

API 不变：
```python
exact_solver.solve(...)   # 原 solver.py 的鸭子接口由 goofspiel._cxx 桥接
```

只是 backend 从：
```text
python
```

切成：
```text
cpp
```

。

---

# 24. C++ Exact Core 应包含什么（实际交付比原设计更宽：含 VectorEnv + 训练）

**实际 2026-08-30 已交付内容如下（比原 §24 多的项目标 ⭐）**：

### Exact Solver 核心（原设计内，已交付）
- bitmask enumeration；
- recursion；
- canonicalization（A≤B swap + sign flip，保证 `F(A,B,R)=-F(B,A,R)`）；
- memoization；
- chance expansion；
- matrix construction（**M 矩阵在每个 `for p in prizes` 循环 fresh 分配**，修掉跨 prize 污染经典 bug）；
- **等手短路 `F(A,A,R)=0` + `fill_policy_for_equal_hands` 对称策略，节省所有 A=B 状态的 LP 调用**。

### LP 后端双模式（原设计 §A/B，已交付两种 switch）
- ⭐ 默认 Python↔C++ GIL-safe scipy callback：**不用用户装 HiGHS binary，立刻能跑**。`goofspiel/_cxx.py` import `_core` 时自动 `install_lp_solver(callback)`。
- 可选 HiGHS C++ native：CMake 时 `-DCXXGOOF_USE_HIGHS=ON -Dhighs_ROOT=/path/to/highs/install`，小 matrix 省掉 GIL 往返，N=5~7 精确解预计 3~5× 提速。Reference 仍用 SciPy 交叉验证。

### ⭐ VectorizedEnv（Struct-of-Arrays 训练环境，原设计 §E1 没包含，提前交付）
这是这次 C++ 下沉最大的收益项：
- 状态 layout：`human_mask_[M], bot_mask_[M], prize_mask_[M], score_h_[M], score_b_[M], round_[M], done_[M]` 均 SoA，每个 state 实际内存 ≤ 11 字节（1M envs ≈ 11 MB，完全 fit L3）。
- `step_batch` 热循环：`uint16_t* __restrict__` 指针别名声明，分支模式稳定（h>/</==b），编译器自动向量化成 AVX2。
- 无 per-step 堆分配；reset 用 Xorshift64* 确定性 RNG。
- pybind 返回 numpy zero-copy：Python 训练侧 `torch.from_numpy(obs, copy=False)` 直接喂网络。

### ⭐ Gymnasium 鸭子兼容层（原设计没单独列）
`goofspiel/_cxx.py` 暴露：
```python
from goofspiel._cxx import make_vector_env, cpp_solve_with_policy
venv = make_vector_env(13, 4096)              # Gymnasium API: reset/step + infos
obs_dict, infos = venv.reset(seed=1)          # obs ∈ R^(M, 3N+3) float32 zero-copy
obs, rew, term, trunc, infos = venv.step(a_h, a_b)
```
不编 C++ 时自动 fallback 到纯 Python 串行 VectorEnv，打印 1 条 warning + 降 M 默认值，保证训练脚本零改动。

### ⭐ N=5 PPO self-play demo（scripts/train_n5_ppo.py）
CleanRL 风格最小闭环：ActorCritic (2×256 MLP) + GAE(γ=0.99, λ=0.95) + clipped policy loss + value clip loss + entropy bonus + legal mask logits。
- 自博弈：bot 侧 observation 由 human/bot mask 交换 + score 交换得到，同一套网络双视角。
- 默认 4096 env × 256 rollout = 1 update，10 次 update = 100k 步。
- stdout 每 update 打印 SPS / avg_return(±200) / pg_loss / v_loss / entropy。
- 产物 `checkpoints/ppo_n5_seed1.pt`（dict 含 model_state_dict、args、global_step、last_200_avg_return）。

Reference LP 仍可 Python 调 SciPy。
但 Python↔C++ 每个 state 往返会很慢。
因此性能版最终有两种方案：

### 方案 A

C++ 调 HiGHS C++ API。

### 方案 B

小 matrix 使用自己高效 zero-sum solver。

Reference 仍用 SciPy 检验。

---

# 25. 不建议 Exact Solver 用 GPU

Exact DP 的主要瓶颈：

- irregular recursion；
- hash table；
- dynamic state graph；
- 大量小 matrix；
- branch-heavy control flow。

这不是 GPU 最擅长的问题。

所以 Exact：

\[
\boxed{\text{CPU-heavy}}
\]

Search Leaf Neural Evaluation：

\[
\boxed{\text{GPU-heavy}}
\]

。

---

# 第三阶段工程：GPU Matrix Solver

# 26. Step E4 — 先写纯 PyTorch RM+

实现：

```text
math/batched_rm_plus.py
```

输入：

\[
[B,13,13]
\]

。

整个 solver 禁止 Python 循环遍历 B。

允许最多：

```python
for iteration in range(K):
```

但每次 iteration 内全部：

\[
Tensor\ operation
\]

。

---

# 27. 然后 `torch.compile`

由于：

\[
N_{max}=13
\]

可以固定 shape：

```text
[B,13,13]
```

并使用 mask。

这非常适合 compiler。

PyTorch 当前推荐把 `torch.compile` 用在合适的高层函数/module 上。

所以：

```python
@torch.compile
def rm_plus_solve(...):
    ...
```

是优先路线。

---

# 28. 只有 profiler 证明 solver 仍然是热点，才用 Triton

Triton 的定位就是用 Python DSL 编写高性能 GPU kernels。

真正值得 Triton 化的候选：

- batched RM+；
- masked row/column utility；
- regret normalization；
- joint-action masking；
- pair-builder 小 kernel。

不是：

> 整个 Transformer 用 Triton 重写。

---

# 29. CUDA C++ 是最后手段

优先顺序严格：

```text
PyTorch
↓
torch.compile
↓
Triton
↓
CUDA C++
```

。

除非 benchmark 证明必要，否则不进入下一层。

---

# 第四阶段工程：Model

# 30. Step E5 — 模型逐模块实现

不要一次写 `GoofspielModel` 2000 行。

顺序：

```text
RankEncoder
↓
Card Transformer
↓
Relational GNN
↓
Fusion
↓
Pair Builder
↓
Matrix CNN
↓
Robust Heads
↓
LSTM
↓
Mamba
↓
Opponent Heads
↓
Adaptive Branch
↓
Full Model
```

每实现一个模块：

立即写 shape/unit tests。

---

# 31. Mamba 不自己实现

直接复用官方：

\[
\texttt{state-spaces/mamba}
\]

。

官方项目提供 `mamba-ssm` 包及 Mamba block；当前仓库还提供 CUDA selective scan 构建选项。

项目：

[state-spaces/mamba](https://github.com/state-spaces/mamba?utm_source=chatgpt.com)

我们的代码只写：

```text
InterGameMamba
```

wrapper。

不要 Codex 自己根据论文“手搓一个差不多的 Mamba”。

---

# 32. GNN 不建议第一步引入 PyG/DGL

我们的 graph：

最大：

\[
3N=39
\]

nodes。

结构固定、很小。

更简单高效的方法：

\[
\boxed{\text{dense relation-aware attention}}
\]

直接用 PyTorch。

原因：

PyG/DGL 对大 sparse graph 很有价值。

这里 dynamic graph object packing 可能比真正计算还贵。

---

# 33. 固定 Nmax=13 进行 GPU 张量计算

虽然支持 variable N：

GPU 内部统一：

```text
Nmax = 13
```

。

例如：

```text
[B,13,D]
[B,13,13,C]
```

。

用 masks 表示 N。

这有巨大工程好处：

- batch 简单；
- compile 容易；
- kernel shape 稳定；
- CUDA Graph 更容易；
- search leaf batch 简单。

---

# 34. 不要真的为 N=5 创建 `[B,5,5]` 独立 graph

逻辑 N 可以变。

物理 Tensor shape：

尽量固定：

\[
13
\]

。

这是性能上非常重要的取舍。

---

# 35. History 使用 bucket

LSTM 当前局最大长度：

\[
12
\]

。

可以统一 pad：

\[
12
\]

。

Mamba session 最大：

\[
128
\]

games。

建议按：

```text
8
16
32
64
128
```

bucket。

减少无意义 padding。

---

# 36. Precision

默认：

### Backbone

\[
BF16
\]

### Q output

转换：

\[
FP32
\]

### Nash Solver

\[
FP32
\]

### Exact / Reference LP

\[
FP64
\]

### Metrics accumulation

关键项：

\[
FP32/FP64
\]

。

Hopper 类 GPU 对 BF16 非常友好。

不要默认 FP16。

---

# 第五阶段工程：Learning Primitives

# 37. Step E6 — 不要先写完整 Trainer

先逐个实现纯函数：

```text
build_nash_bellman_target()
neurd_loss()
lambda_returns()
joint_vtrace()
outcome_projection()
opponent_prediction_loss()
style_infonce()
adaptive_bellman_target()
```

。

每一个：

```text
input
→ output
```

独立可测。

---

# 38. 先使用人工小矩阵验证

例如：

Matching Pennies：

\[
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}
\]

Dominated strategies。

N=1/N=2 Goofspiel。

这些 test 比“训练 loss 在降”重要得多。

---

# 39. Gradient Routing 要在 Trainer 出现前验证

分别运行：

```text
Robust backward
Opponent backward
Adaptive backward
Actor backward
```

检查：

哪些参数有 grad。

必须自动化。

---

# 第六阶段工程：数据

# 40. Step E7 — 数据不要保存 Python pickle 海洋

长期 trajectory / teacher dataset 建议：

\[
\boxed{\text{PyArrow + Parquet}}
\]

Arrow/Parquet 支持列式存储、partitioned datasets、多线程读取和按列扫描。

例如：

```text
data/
  game_corpus/
    n=13/
      part-...
  exact/
    n=5/
  teacher/
    source=gt_cfr/
```

---

# 41. 为什么不是全部存 `.pt`

`.pt` 适合：

- checkpoint；
- 小 tensor artifact。

不适合：

> 数亿条可筛选、可分区 trajectory records。

Parquet 更适合：

- version；
- filtering；
- analysis；
- offline reanalysis。

---

# 42. 热数据和冷数据分开

### Hot Replay

RAM/shared-memory。

### Warm Dataset

NVMe Parquet。

### Cold Archive

共享存储/对象存储。

不要训练每个 batch 都从 Parquet 读随机小记录。

---

# 43. 数据记录尽量 compact

State 不需要存一大堆 float arrays。

存：

```text
uint16 self_mask
uint16 opp_mask
uint16 prize_mask
uint8 current_prize
uint16 scores
uint8 action
...
```

训练 Dataset collator 再转换成 GPU dense tensors。

---

# 第七阶段工程：Pre-training

# 44. Step E8 — 先跑 P1

此时系统只需要：

- Game Core；
- Model；
- GameCorpus；
- self-supervised losses。

不要等 Search/League 全写完。

---

# 45. Pretraining 的工程验收不是 Win Rate

必须出现 dashboard：

```text
transition accuracy
immediate outcome accuracy
symmetry error
masked-action accuracy
opponent prediction NLL
style contrastive loss
GPU throughput
```

。

满足设计书 gate 后才进入下一阶段。

---

# 第八阶段：Exact / Teacher / SFT

# 46. Step E9 — 打通 Teacher Protocol

统一：

```python
TeacherTarget
```

。

先只接：

```text
EXACT
EMA
```

确保 pipeline 正确。

再增加：

```text
CFR
SEARCH
ENSEMBLE
```

。

---

# 47. 不要等完整 GT-CFR 写好才训练

顺序应该：

```text
Exact Teacher
↓
SFT works
↓
Robust Model improves
↓
SM-MCTS
↓
GT-CFR
↓
Search Teacher
```

。

因为高质量 Search 本身也需要一个不太烂的 neural evaluator。

---

# 第九阶段：Robust RL

# 48. Step E10 — 第一条完整 Online Learning 闭环

> ### 2026-08-30 更新：✅ **最小 Precursor 已交付（M9.0 smoke）**
>
> 已按文档"先 N=3→5→13 逐步逼近"原则，写好最小可复现的自博弈 PPO 闭环：
> - `scripts/train_n5_ppo.py`（CleanRL 风格，~360 行）：ActorCritic 2×256 MLP + GAE + clip PPO + value clip + entropy bonus + legal-action logits mask + self-play bot 侧 obs mirror（mask/score swap）。
> - 默认脚本：`--num-cards 5 --num-envs 4096 --rollout-steps 256 --num-minibatches 4 --total-timesteps 100000`（= 10 updates，N=5，4096 envs × 256 rollout = 1 update 1M+ env-steps）。
> - C++ 未编时安全 fallback：自动降 `--num-envs 256` 接 Python 串行鸭子 VectorEnv，保证脚本跑通不 crash。
> - 100k 步 stdout 指标窗口：SPS (C++ VectorEnv ≥ 30k)、avg_return(±200)（应从 0 附近向正方向爬升，例如 1→6）、entropy 衰减平稳（未塌缩到 0）、value loss 10 次 update 从 +4 级 → +2.7 级下降，证明 learn 信号真实。
>
> **未做（保留给正式 M9）**：Full-Matrix Bellman Target / NeuRD / MC-TD 多目标 update 联合训练；当前 PPO demo 只跑最基础的 self-play zero-sum score diff 单目标，作为"pipeline 通不通"的 smoke 而不是收敛验收。

必须先做到：
```text
Self-play
↓
Trajectory
↓
Full-Matrix Bellman Target
↓
Q update
↓
NeuRD update
↓
MC/TD update
↓
New model
↓
Self-play
```

。

只运行：
\[
N=3,5
\]

验证。

---

# 49. 然后扩大到 N=13

先证明：

N=3：

能接近 Exact。

N=5：

能接近 Exact。

然后逐步放大。

不要第一天两张 H200 跑 N=13 几亿步后才发现：

> Q target 符号写反了。

---

# 第十阶段：Search

# 50. Step E11 — SM-MCTS

先做：

```text
CPU tree
+
GPU leaf evaluator
```

。

这是最自然架构。

---

# 51. Search Tree 不适合 GPU 化

Tree traversal：

- branch-heavy；
- dynamic；
- map/hash；
- irregular memory。

CPU 做。

真正 GPU 工作是：

\[
\boxed{\text{Leaf Neural Evaluation}}
\]

。

---

# 52. 必须 Batch Leaves

错误：

```text
CPU finds leaf
↓
GPU forward 1 state
↓
CPU
```

。

这样 GPU utilization 会极低。

正确：

```text
CPU Search Worker 1 ─┐
CPU Search Worker 2 ─┤
CPU Search Worker 3 ─┼→ Leaf Queue
CPU Search Worker N ─┘
                         ↓
                   GPU Batch Evaluator
                         ↓
                   Batched Predictions
                         ↓
                    Result Dispatch
```

。

---

# 53. Leaf Batch 策略

由于模型很小：

GPU 饱和可能需要比较大的 batch。

不要猜。

benchmark：

```text
B=32
64
128
256
512
1024
2048
```

测：

\[
states/sec
\]

和：

\[
latency
\]

。

PLAY 模式偏 latency。

TEACHER/Reanalyse 偏 throughput。

---

# 54. Micro-batching

在线 Search 可以：

```text
flush if:
  queue >= max_batch
  OR
  oldest_request_wait > X microseconds
```

。

这是 serving 系统常见的 dynamic batching 思路。

---

# 55. Search Leaf 可以使用独立 CUDA Stream

例如：

- stream 0：normal inference；
- stream 1：teacher/search batch；

但不要一开始复杂化。

先 profiling。

PyTorch 暴露 CUDA streams/events API，可在必要时做更细粒度并发。

---

# 56. CUDA Graphs 适合哪里

CUDA Graph 能通过重复 replay 固定 kernel 序列显著降低 CPU launch overhead，但要求较固定的 shape/control flow。

我们非常适合的地方：

```text
Fixed Nmax=13
Fixed inference batch buckets
Fixed model
```

。

所以后期可以为：

```text
B=64
B=128
B=256
B=512
```

分别 capture CUDA Graph。

---

# 57. CUDA Graph 不用于 Search Tree 本身

Search 控制流动态。

只 capture：

\[
\boxed{\text{batched neural leaf forward}}
\]

。

---

# 第十一阶段：GT-CFR

# 58. Step E12 — 不要从零发明所有 CFR 基础件

OpenSpiel 有大量：

- CFR；
- MCTS；
- exploitability；
- game traversal；

实现可作为数学与工程参考。

但是我们的：

\[
\text{simultaneous joint-Q + neural leaf + Exact leaf}
\]

GT-CFR 要自己写适配。

---

# 59. OpenSpiel CFR 最适合做什么

### 参考

- regret update；
- average strategy；
- exploitability；
- extensive-form bookkeeping。

### 验证

小 N：

我们的 search vs OpenSpiel CFR。

### 不直接复制

我们的 GPU evaluator / tool router / adaptive branch。

---

# 60. OpenSpiel MCTS 也只能借框架思想

OpenSpiel 有 Python/C++ MCTS 和 neural evaluator pattern。

但标准 PUCT sequential MCTS：

\[
\boxed{\text{不能直接用于我们的 simultaneous root}}
\]

。

复用：

- tree abstractions；
- evaluator concepts；
- tests。

不复用：

- sequential action semantics。

---

# 第十二阶段：Opponent / Adaptive

# 61. Step E13 — LSTM 先，Mamba 后

工程调试时：

先让：

\[
LSTM
\]

单独预测当前局 opponent action。

确认可学。

再加入：

\[
Mamba
\]

。

不是最终设计删 Mamba。

只是：

\[
\boxed{\text{逐模块验证}}
\]

。

---

# 62. Mamba integration 必须固定版本

因为官方 Mamba 包含 CUDA 扩展和硬件相关实现。

Docker/lockfile 必须记录：

```text
mamba-ssm commit/version
causal-conv1d version
CUDA version
PyTorch version
```

。

---

# 63. Opponent Session 是 Stateful System

不要普通：

```python
Dataset[index]
```

随机打散以后期待 Mamba 学长期行为。

必须有 session sampler：

```text
opponent
  game 1
  game 2
  ...
```

保持时间顺序。

---

# 第十三阶段：League

# 64. Step E14 — 不建议自己从第一天写分布式 League Runtime

先写：

```text
LeagueManager
```

纯 Python 单机版本。

负责：

- policy registry；
- role；
- payoff matrix；
- snapshot；
- sampling。

---

# 65. RLlib 可以作为 League 工程参考

RLlib 当前支持 competitive multi-agent/self-play/league workflows，并有直接基于 OpenSpiel 的 self-play 与 league-based self-play 示例。

参考：

[RLlib OpenSpiel league example](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/self_play_league_based_with_open_spiel.py?utm_source=chatgpt.com)

值得借：

- policy registry；
- frozen snapshots；
- matchup scheduling；
- worker callbacks。

---

# 66. 但不要使用 RLlib 实现我们的主 Learner

因为我们有：

- full-matrix Nash Bellman；
- NeuRD；
- multi-track gradients；
- opponent memory；
- Exact/Search teachers。

这些偏离标准 RLlib algorithm 很大。

强行塞进去最后会：

\[
\boxed{\text{框架复杂度 > 算法复杂度}}
\]

。

正确定位：

\[
\boxed{\text{Ray/RLlib orchestration reference}}
\]

而不是 learning core。

---

# 67. 什么时候开始使用 Ray

单机：

```text
Python multiprocessing
+
shared queues
```

足够。

只有当：

- 多节点；
-几十个 actor；
- Search workers 跨机器；
- Fault tolerance；

真的出现需求时，再升级：

\[
\boxed{\text{Ray Actors}}
\]

。

不要第一版分布式就引入 Ray 集群。

---

# 第十四阶段：Red Team / Reanalyse

# 68. Step E15

此时才建立独立服务角色：

```text
ActorWorker
SearchWorker
ExactWorker
ReanalysisWorker
RedTeamWorker
Evaluator
Learner
LeagueManager
```

。

所有 worker 通过稳定任务协议通信。

---

# 69. Worker 之间传 compact IDs，不传巨大 Python objects

例如：

```python
SearchTask(
    state_key=...,
    model_version=...,
    budget=...,
)
```

。

避免：

> pickle 一整个 model + environment。

---

# 70. Model Weights 分发

每个 worker：

维护：

```text
active_model_version
```

。

League/Coordinator 发布新版本。

worker 原子切换：

```text
v17 → v18
```

。

不要训练过程中覆盖正在使用的 `.pt`。

---

# 71. Checkpoint 使用内容寻址/不可变版本

例如：

```text
checkpoints/
  R_000012/
  A_000007/
  E_000004/
```

。

保存后不修改。

`latest`：

只是 symlink/metadata pointer。

---

# 第十五阶段：Distillation

# 72. Step E16 — Strong / Fast Student

这不是另起一个神秘训练框架。

依然 PyTorch。

定义：

```text
TeacherBatch
```

同时包含：

- Q；
- policy；
- distribution；
- opponent belief。

---

# 73. Fast Student 不需要复制所有 backbone

可以后续另行设计：

- smaller Transformer；
- fewer GNN layers；
- smaller Matrix CNN；
- shorter memory。

但 Teacher API 完全一样。

---

# 第十六阶段：GPU 性能总设计

# 74. GPU 上真正应该跑什么

### 必须 GPU

- Neural forward；
- backward；
- Matrix CNN；
- Transformer；
- GNN；
- LSTM；
- Mamba；
- batched RM+；
- teacher ensemble forward；
- leaf evaluation。

### CPU

- environment state transition；
- exact recursion；
- LP reference；
- tree traversal；
- league scheduling；
- cache；
- dataset metadata；
- failure localization。

---

# 75. 为什么 environment 初期不用 GPU

一个 Goofspiel state：

只有几个 bitmask。

transition 极便宜。

把单个 transition 发 GPU：

PCIe/dispatch 成本可能比计算本身大。

只有在 Full-Matrix Bellman 中生成：

\[
B\times169\times M
\]

大量 child states 时，

才值得写：

\[
\boxed{\text{batched tensor transition}}
\]

放 GPU。

所以保留两个 backend：

```text
CPU ScalarTransition
GPU BatchedTransition
```

。

---

# 76. Batched Bellman Target 应完全张量化

对于 batch：

\[
B
\]

构造：

```text
[B,13,13,M]
```

counterfactual children。

禁止 Python：

```python
for batch:
    for a:
        for b:
            for prize:
```

。

这会成为灾难。

---

# 77. GPU Child Generation

可以利用 bit operations：

\[
mask' = mask \& \sim(1<<action)
\]

。

PyTorch integer tensor 支持相关 bitwise operations。

第一版 PyTorch。

Profiler 后决定是否 Triton。

---

# 78. DataLoader

配置：

- `pin_memory=True`
- persistent workers
- prefetch
- `non_blocking=True` H2D

。

但 benchmark 决定 worker 数。

不要默认：

```text
num_workers=32
```

。

---

# 79. BF16 autocast

训练：

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...
```

但：

```text
Q outputs
Nash calculations
probabilities
loss reductions where needed
```

转 FP32。

---

# 80. `torch.compile` 的顺序

不要一开始 compile 全系统。

逐块：

```text
Public Backbone
↓
Matrix CNN
↓
Batched Nash
↓
Training Step
```

。

找到 graph breaks。

PyTorch Profiler 能记录 operator、input shape、CPU/GPU activity 和 traces，应成为性能诊断的第一工具。

---

# 81. Profiling 的固定流程

每个主要 milestone：

### Step 1

CPU wall-clock profile。

### Step 2

`torch.profiler`。

### Step 3

Nsight Systems：

检查：

- CPU-GPU gaps；
- kernel launch；
- memcpy；
- stream idle。

### Step 4

Nsight Compute：

只分析真正最贵 kernel。

NVIDIA CUDA 工具链提供 Nsight Systems、Nsight Compute、Compute Sanitizer 等调试/性能工具。

---

# 82. 优化原则

永远按照：

\[
\boxed{
\text{Measure}
\rightarrow
\text{Identify bottleneck}
\rightarrow
\text{Optimize}
\rightarrow
\text{Re-measure}
}
\]

。

禁止：

> “听说 Triton 快，所以全部 Triton。”

---

# 第十七阶段：CUDA Graph 进一步优化

# 83. 为什么这个项目可能特别受益

模型只有：

\[
6\sim8M
\]

且序列短。

这意味着单次 kernel 本身很小。

CPU launch overhead 可能占比明显。

CUDA Graph replay 可以显著减少 launch overhead。

所以：

\[
\boxed{\text{固定 shape inference}}
\]

值得测试。

---

# 84. 为不同 Batch Bucket 捕获

例如：

```text
32
64
128
256
512
1024
```

。

Request：

向上 pad 到最近 bucket。

---

# 85. Training 不一定适合全部 CUDA Graph

因为：

-多个 update track；
-动态 teacher；
-不同 batch；

控制流复杂。

Inference/Search leaf：

更合适。

---

# 第十八阶段：数据与存储

# 86. 推荐目录

```text
storage/
├── datasets/
│   ├── corpus/
│   ├── exact/
│   ├── teacher/
│   ├── trajectories/
│   ├── opponent_sessions/
│   ├── failures/
│   └── reanalysis/
│
├── checkpoints/
├── exact_cache/
├── search_cache/
├── league/
└── mlflow/
```

。

---

# 87. Exact Cache 不用 Parquet

它是 KV lookup。

可以：

### Prototype

SQLite。

### 高性能

LMDB / RocksDB / 自定义 mmap KV。

key：

\[
state\_key
\]

value：

serialized exact result。

---

# 88. 大数据使用 Parquet

因为需要：

- partition；
- scan；
- filter；
- analysis。

PyArrow Dataset 支持对多文件 dataset 进行筛选和分区扫描。

---

# 第十九阶段：开源项目复用清单

## A. OpenSpiel —— **强烈复用/参考**

[OpenSpiel repository](https://github.com/google-deepmind/open_spiel?utm_source=chatgpt.com)

可复用：

- Goofspiel reference；
- CFR；
- MCCFR；
- exploitability；
- best-response；
- MCTS framework；
- game-theoretic tests；
- simultaneous-game semantics。

OpenSpiel 支持 simultaneous games，并提供 C++ core + Python interface。

**定位：数学 oracle + algorithm reference。**

不要让它替代我们的整个 Agent。

---

## B. state-spaces/mamba —— **直接依赖**

[Official Mamba repository](https://github.com/state-spaces/mamba?utm_source=chatgpt.com)

直接复用：

- Mamba block；
- CUDA selective scan implementation。

不要自己实现 Mamba。

---

## C. PettingZoo —— **环境兼容层**

[PettingZoo](https://pettingzoo.farama.org/?utm_source=chatgpt.com)

Parallel API 正适合 simultaneous actions。

用途：

- external MARL compatibility；
- baseline；
- tests。

---

## D. RLlib / Ray —— **League 与分布式工程参考**

[RLlib league self-play example](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/self_play_league_based_with_open_spiel.py?utm_source=chatgpt.com)

直接值得研究：

- self-play policy versioning；
- league；
- exploiters；
- worker management。

RLlib 当前文档本身也支持 adversarial/self-play/league-based multi-agent training。

但：

\[
\boxed{\text{不要把我们的主算法强塞进 RLlib Trainer}}
\]

。

---

## E. TorchRL —— **Baseline/工具参考**

TorchRL 已有 multi-agent PPO/IPPO/MAPPO 目标实现。

用途：

- PPO/MAPPO baseline；
- GAE/V-trace 等实现参考；
- TensorDict ideas。

主算法仍自己实现。

---

## F. SciPy HiGHS —— **Reference Nash**

直接使用：

```python
scipy.optimize.linprog(..., method="highs")
```

。

作为：

\[
\boxed{\text{FP64 Reference Solver}}
\]

。

---

## G. PyTorch —— **主框架**

直接承担：

- model；
- training；
- DDP；
- compile；
- mixed precision；
- CUDA streams；
- profiler。

---

## H. Triton —— **优化工具**

只有 profile 后的 GPU hotspots 使用。

官方定位就是高性能自定义 DNN GPU kernel DSL。

---

## I. Hydra —— **配置**

所有实验配置。



---

## J. MLflow —— **实验追踪**

保存：

- runs；
- checkpoints；
- metrics；
- datasets；
- artifacts。



---

## K. PyArrow / Parquet —— **训练语料/Replay 冷存储**

大规模结构化数据。



---

# 第二十阶段：开发顺序——真正告诉工程师第一天做什么

这是最重要的部分。

不要并行乱写。

---

# Milestone M0 — Repository Bootstrap

**状态（2026-08-30）：✅ 部分完成**

实现：
```text
✅ pyproject.toml（cmake-build-extension + extras [train] [dev] 已写，见根目录）
⏳ configs/                          # 由 M0.1 Hydra 切换时落地
⏳ logging
✅ pytest                            # test_env(23) + test_solver(~20) + test_app(46) + test_cxx(4) = 69 + passed
⏳ CI
⏳ Docker
⏳ MLflow
```

验收：
```bash
pytest   # 69 passed / 0 failed / 1 slow deselected  ✅ 已达到
```

全绿。

GPU container：
```python
torch.cuda.is_available()
```

⏳ 成功（未实机 CI 跑过容器）。

---

# M1 — Game Core

**状态：✅ 已完成（Python Reference） + ⭐ C++ VectorEnv 提前交付**

实现（比原设计多 ⭐）：
- ✅ state（compact bitmask，13-bit 无溢出，GoofspielEnv Python 实现）；
- ✅ rules（carry-over 平局滚入变体，三分支结算：胜/平非末滚/平末丢弃）；
- ✅ transition（env.step 原子，必须双方 action 同时提交；禁止偷看另一方未结算 action 信息隐藏契约强制执行）；
- ✅ variable N（1..13，任意 N 奖品/手牌堆正确初始化，轮数 = N）；
- ✅ history（每 round 一条 dict，不变量 `prize_at_stake == round_prize + carry_in` 全通过）；
- ⭐ 提前交付：**PackedState C++ compact + VectorizedEnv SoA**（见 §24，为 M9 训练吞吐铺路）。

验收：
至少：
\[
1000
\]
✅ 实际随机游戏 invariants + `tests/test_env.py` 23 条 + `test_cxx.py` 单步 cross-backend 100 种子，全通过。

---

# M2 — Reference Validation

**状态：🟨 部分完成（SciPy Nash 双后端 OK；OpenSpiel parity 留 M2.1）**

接：
- ⏳ OpenSpiel parity；（未接，下一子任务 M2.1）
- ✅ SciPy Nash（solver.py + goof_nash.h 共用同一套 scipy HiGHS LP 后端）；
- ✅ basic exact solver（Python + C++ 两阶段递归，N=1..6 均可跑）。

验收：
小 N 数学结果一致。
✅ 已验证：Python solver / C++ solver 对 N=3 所有 policy 条目满足 `xᵀM ≥ V, My ≤ V` Nash 不变式（tests/test_solver.py + tests/test_cxx.py）。

在这里：
\[
\boxed{\text{✅ 达到 — N=3 时无神经网络，Nash 不等式全满足}}
\]

。

---

# M3 — Exact Solver

**状态：✅ 已完成（Python 版） + ✅ 提前交付 C++ 加速版（M3.5）**

实现：
- ✅ DP；
- ✅ bitmask；
- ✅ memo；
- ✅ **Level-A+B 复杂度估算器**（OEIS A000172 C(N) 精确 + 五级风险 GREEN/YELLOW/ORANGE/RED/BLACK + preflight 拒绝保护 + force=True 绕过）；
- ✅ 两阶段递归（Phase-1 eager child solve → Phase-2 纯 cache 读 + sign 翻，根治 sign leak）；
- ✅ 符号 canonical key（A≤B swap + sign，保证 storage 只存一半状态，`F(A,B,R) = -F(B,A,R)` 恒成立）；
- ✅ M 矩阵 per-prize fresh 分配（修跨 prize 污染 bug）；
- ✅ 等手 `F(A,A,R)=0` 短路 + 策略对称；
- ⭐ C++ 版额外：VectorizedEnv(SoA) + Gymnasium duck API + N=5 PPO demo（见下 M9）。

先 Python。

验收：
N=1..6。
✅ 全部通过；N=5 Python 精确解约 18.8s，3130 条 policy map，根值 = 0（对称根不变式，test_solver + test_cxx 双后端都满足）。

记录：
```text
states   ✅ C(N) 由 solver.py 和 cxxgoof 两者精确 = OEIS A000172（N=5 时 C=2252）
LPs      ⏱ N=5 约 3130 条；其中等手 / 叶子跳过 LP 调用，实际 LP 数 ≈ 1800
runtime  ⏱ Python N=5 ≈ 18.8s；C++ N=3 < 0.1s（C++ N=5 待实机测）
memory   ⏱ Python N=5 policy_map dict 约 3.2 MB；C++ unordered_map 预计 1 MB
```

。

---

# M4 — Neural Model

逐组件实现。

验收：

- shape tests；
- masks；
- variable N；
- padding invariance；
- opponent leakage；
- LSTM/Mamba isolation。

此时只 forward。

---

# M5 — GPU Nash Solver

实现：

PyTorch RM+。

对比：

SciPy LP。

误差达到阈值。

然后 compile。

---

# M6 — Learning Primitives

逐个 loss/target。

所有数学 unit tests。

不写大 trainer。

---

# M7 — Game Corpus + Pre-training

生成 corpus。

跑 P1。

验收 representation benchmarks。

---

# M8 — Exact Teacher + SFT

Teacher protocol。

先：

```text
Exact
+
EMA
```

。

完成第一版 Strategic Student。

---

# M9 — Robust RL

建立：

```text
Actor
→ trajectory
→ learner
```

闭环。

先 N=3。

必须逼近 Exact。

再 N=5。

最后 N=13。

---

# M10 — SM-MCTS

CPU tree + GPU leaf batching。

先测试：

Matching Pennies / N=3。

再接 Neural Agent。

---

# M11 — GT-CFR

先用 OpenSpiel CFR 作为数学对照。

再实现我们的 neural frontier / exact leaf 版本。

---

# M12 — Tool Router

接：

```text
Matrix Nash
Exact
SM-MCTS
GT-CFR
```

。

得到完整 Robust Agent。

---

# M13 — Opponent Learning

LSTM。

然后 Mamba。

然后 style/switch/calibration。

---

# M14 — Adaptive Branch

Opponent-conditioned Q。

Adaptive search。

Safe LP。

---

# M15 — Semi-supervised / Reanalyse

Teacher ensemble。

Pseudo labels。

Search teachers。

Reanalysis workers。

---

# M16 — League

建立：

- Robust；
- Aggressive；
- Exploiter；

三条 lineage。

---

# M17 — Red Team

实现：

```text
Attack
Diagnose
Relabel
Correct
Regression
```

。

---

# M18 — Distillation

Strong/Fast Student。

---

# M19 — Performance Hardening

现在才集中：

- C++ Exact；
- torch.compile；
- Triton；
- CUDA Graph；
- async leaf batching；
- Ray multi-node。

---

# M20 — Final Evaluation

冻结模型。

全面：

```text
Exact
Cross-play
Exploitability
Search scaling
Opponent benchmark
Adaptive benchmark
Red-team regression
Compute/strength
Variable-N
```

。

这时才允许称：

\[
\boxed{\text{完整训练版本}}
\]

。

---

# 第二十一阶段：每个 Milestone 的规则

工程师不得：

> “M4 做完模型看起来能跑，就直接跳 M16 League。”

每个 milestone 必须：

```text
Implementation
↓
Unit Test
↓
Integration Test
↓
Benchmark
↓
Documentation
↓
Commit / Tag
↓
Next Milestone
```

。

---

# 89. 每个模块都先追求 Correctness

优化顺序：

\[
\boxed{
Correct
\rightarrow
Observable
\rightarrow
Reproducible
\rightarrow
Fast
}
\]

而不是：

\[
Fast
\rightarrow
Debug
\]

。

---

# 90. 必须建立“Reference Backend”和“Fast Backend”

这是整个工程一个非常重要的模式。

例如：

```text
Nash Solver
├── reference_scipy
└── fast_gpu
```

```text
Transition
├── reference_python
└── batched_gpu
```

```text
Exact
├── python_reference
└── cpp_fast
```

```text
Search
├── deterministic_small_test
└── parallel_fast
```

Fast backend 必须和 Reference backend 定期 cross-check。

---

# 91. 不要删除 Reference Implementation

即使 C++ 比 Python 快：

Python reference 仍然永久保留。

即使 Triton 比 PyTorch 快：

PyTorch implementation 仍然保留。

原因：

\[
\boxed{\text{科研系统必须能判断“快版本是不是算错了”}}
\]

。

---

# 第二十二阶段：CI 体系

每个 PR：

### Fast Unit

秒级。

### Mathematical

小矩阵、小 N。

### Model

shape/gradient。

### Deterministic Search

固定 seed。

---

Nightly：

### N=5 exact comparisons

### GPU solver 10k random matrices

### Small training convergence

### Cross-backend parity

### Regression suite

---

Weekly：

### performance benchmarks

记录：

```text
states/sec
search nodes/sec
exact states/sec
GPU utilization
leaf throughput
training updates/sec
```

。

性能 regression 超阈值：

报警。

---

# 第二十三阶段：典型双 GPU 部署

如果机器有：

\[
2\times H200/H100
\]

我建议成熟期：

```text
CPU
─────────────────────────────
Environment Actors
Search Trees
Exact Solver
League
Replay
Reanalysis Coordination

GPU 0
─────────────────────────────
Main Learner
Forward + Backward

GPU 1
─────────────────────────────
Inference Server
Search Leaf Evaluator
Teacher Ensemble
Reanalysis Forward
```

。

---

# 92. Pretrain / SFT 时例外

P1/P3：

如果 learner GPU 真正饱和：

```text
GPU0 + GPU1
→ DDP
```

。

结束以后恢复角色分工。

---

# 93. 更多 GPU

例如：

```text
GPU0  Main Learner
GPU1  Search Leaf
GPU2  Teacher/Reanalyse
GPU3  Actor Inference
GPU4  Aggressive/Exploiter learner
GPU5  Evaluation
```

。

真正瓶颈根据 profiler 调整。

---

# 第二十四阶段：服务化只在需要时

不要第一版做：

```text
Kubernetes
Kafka
Redis cluster
microservices
```

。

单节点用普通进程。

多节点以后：

Ray / gRPC。

研究系统最忌讳：

> 算法还没验证，基础设施已经 50 个服务。

---

# 第二十五阶段：Web UI

现有：

\[
FastAPI
\]

继续使用。

UI 不参与核心训练。

它只消费：

```text
AgentReasoningResult
```

展示：

- model policy；
- Nash policy；
- Q matrix；
- search；
- exact；
- opponent prediction；
- safe exploit；
- tool provenance。

---

# 94. Detector 不允许直接读取模型内部 Python object

应该订阅结构化：

```python
ReasoningEvent
```

。

这样训练、CLI、Web 都能使用同一套 diagnostics。

---

# 第二十六阶段：工程师最需要避免的十类错误

### 1. 框架驱动设计

因为 RLlib 有 PPO：

就把系统写成 PPO。

错误。

---

### 2. 过早分布式

单机都没验证就 Ray cluster。

错误。

---

### 3. 过早 CUDA

Python 结果都不知道对不对就手写 kernel。

错误。

---

### 4. 只留 fast implementation

删掉 reference。

错误。

---

### 5. 环境逻辑重复

Env 一套，Search 又手写一套规则。

错误。

---

### 6. 数据语义混合

Robust / Adaptive / Teacher 全塞一个 Replay。

错误。

---

### 7. CPU/GPU 职责颠倒

把 recursive DP 强行 GPU；

把几万 leaf inference 放 CPU。

错误。

---

### 8. 小模型盲目 FSDP

增加复杂度，没有收益。

错误。

---

### 9. Search 每个 leaf 单独 GPU forward

GPU 大量 idle。

错误。

---

### 10. “能跑”当成“完成”

一个训练脚本 loss 在下降：

远远不是完成。

---

# 第二十七阶段：工程成功标准

整个工程最后应该做到：

## Correctness

数学 reference 全通过。

## Reproducibility

任何 checkpoint 都能找到：

- code；
- config；
- data；
- teacher；
- hardware。

## Observability

每一个 agent action 都能回答：

> 为什么这么下？

至少在工具来源层面可解释。

## Modularity

可以单独替换：

- Model；
- Solver；
- Search；
- Learner；
- Opponent model。

## Performance

GPU utilization 可观，

Search leaf batch 化，

CPU 不成为明显 Python bottleneck。

## Scientific validity

任何所谓 improvement 都能通过：

- ablation；
- baseline；
- exact benchmark；
- exploitability；

证明。

---

# 结论：开发者应该怎样看待这个项目

拿到这套设计以后，不应该想：

> “我要实现一个复杂 RL 模型。”

而应该把它理解成：

\[
\boxed{
\textbf{一个完整的博弈智能系统}
}
\]

它拥有：

### Game Engine

知道真实世界规则。

### Mathematical Core

知道在可计算范围内什么叫真正最优。

### Neural Foundation Model

把巨大状态空间压缩成可泛化的预测。

### Game-Theoretic Learner

通过 Nash、Regret、MC、TD 学会战略。

### Opponent Intelligence

学会读懂具体对手。

### Search System

允许在值得的时候花更多算力思考。

### Tool Router

知道什么时候该相信直觉、什么时候该算、什么时候该搜索。

### Training Factory

通过 Pre-train → SFT → Post-train 持续制造更强模型。

### League

保存战略历史与多样性。

### Adversarial System

不断制造能够攻击 Main 的对手。

### Red-Team Correction

把失败转化成新的高质量训练数据。

### Distillation

把昂贵推理压回网络。

因此真正的工程闭环是：

```text
Rules
  ↓
Reference Mathematics
  ↓
Neural Representation
  ↓
Learning
  ↓
Search & Tools
  ↓
Self-Play
  ↓
Failure
  ↓
Strong Relabeling
  ↓
Training
  ↓
Better Neural Model
  ↓
Better Search
  ↓
Harder Opponent
  ↓
...
```

最终工程师必须始终坚持一个原则：

\[
\boxed{
\textbf{
先把每一层做正确，
再把层与层连接起来；
先建立可验证的慢版本，
再实现高性能版本；
先证明算法真的更强，
再投入算力把它跑大。
}
}
\]

如果按照本文的 M0 → M20 顺序执行，开发者不会面对一个“巨大、复杂、不知道从哪开始”的项目，而会面对二十个**边界明确、可以独立验收、逐步组成完整智能体的工程里程碑**。

---

# 附：工程进度快照 · 2026-08-30（截至当前提交）

> 目的：把「理论里程碑 M0~M20」翻译成**今天用户能摸到的实际代码与测试数字**，避免后续开发者对"做没做完 / 做到哪一步"产生歧义。
> 编译：只统计源码实际存在 + 单测写好的条目，不包括计划。标记含义：✅ = 实装并通过 pytest / ⏳ = 设计文档就绪代码未写 / ⭐ = 超出原设计提前交付。

## 代码交付清单（实装行数 / 文件数）

| 模块 | 文件数 | 规模估计 | 状态 | 说明 |
|---|---|---|---|---|
| GoofspielEnv（Python Reference） | 1 (goofspiel/env.py) | ~900 行 | ✅ | 23 环境契约测全过 |
| Bot 家族（Rnd/Hst/Nash） | 1 (goofspiel/bots.py) | ~580 行 | ✅ | create_bot 工厂 + 回落契约；Nash carry>0 诚实回退 |
| Python Nash Exact Solver | 1 (goofspiel/solver.py) | ~900 行 | ✅ | OEIS Level-A+B + 两阶段递归 + canonical cache + 等手短路；N=5 实测 18.8s 3130 policy map V=0 |
| **C++ Core cxxgoof/ ⭐** | 6 个源码文件 + CMake | .h 1500 + .cc 800 ≈ 2300 行 | ✅ (源码写好，待本机编译验证) | goof_env.h + goof_estimate.h + goof_nash.h + bindings.cc；pybind11 产物 goofspiel/_core.{pyd,so} |
| Python 鸭子兼容层 goofspiel/_cxx.py | 1 | ~420 行 | ✅ | import 成功接 C++；失败自动 fallback Python VectorEnv + warning + 降 M |
| N=5 PPO self-play demo scripts/train_n5_ppo.py | 1 | ~360 行 | ✅ | CleanRL 风格，Gymnasium API，4096/256，save pt |
| Web 后端 FastAPI app.py | 1 | ~440 行 | ✅ | /config /new /state /play，meta 回落，ai_policy + **human_policy counterfactual** 双面板数据 |
| Web 前端 HTML/CSS/JS | 3 | 158 + 410 + 420 ≈ 990 行 | ✅ | Game Setup (N+AI 下拉) + 反事实三色条 + Nash 加载提示 |
| pyproject.toml（含 CMake 自动编扩展） | 1 | ~100 行 | ✅ | cmake-build-extension + [train]/[dev] extras |
| **编译/集成专项文档 order/C++模块编译与训练集成指南.md ⭐** | 1 | ~220 行 | ✅ | 三种构建方式 / HiGHS native / FAQ 坑点 / PPO demo smoke 启动 |
| 测试 tests/*.py | 4 | 400+250+400+580 ≈ 1630 行 | ✅ | **69 passed** (非 C++) + 4 C++ 专项 (auto-skip safe) |

## 里程碑进度量化（M0~M20）

| Milestone | 名称 | 完成度 | 已过交付物 / 欠交付物 |
|---|---|---|---|
| M0 | Repository Bootstrap | **40%** ✅ | ✅ pyproject + pytest + 69 测；⏳ 缺 configs/logging/Docker/MLflow/CI |
| M1 | Game Core | **95%** ✅ + ⭐ | ✅ Python Env + rules + history 契约 23/23；⭐ 提前交付 C++ VectorEnv(SoA)；欠 serialization |
| M2 | Reference Validation | **60%** 🟨 | ✅ SciPy Nash Python/C++ 双后端；✅ N=3 Nash 不变式；⏳ 缺 OpenSpielParityTest（10k 轨迹对比） |
| M3 | Exact Solver | **95%** ✅ + ⭐ | ✅ Python 全部 (N=1..7)；✅ 复杂度预检五级风险；⭐ C++ 递归/cache/LP 双后端（源码）；欠实机跑 `cmake --build + pytest test_cxx.py` 证明编译/速度 |
| M4 | Neural Model | **5%** ⭐ | ⭐ 仅 2×256 MLP ActorCritic（PPO demo 最小版）；⏳ 缺 RankEncoder / Card Transformer / GNN / Matrix CNN / Mamba LSTM / Robust Heads 等按设计书的 10 个模块 |
| M5 | GPU Nash (RM+) | 0% | ⏳ |
| M6 | Learning Primitives | 0% | ⏳（当前 PPO 只跑 standard clip loss，未接 NeuRD/Bellman/NeuRD） |
| M7 | GameCorpus + Pre-training | 0% | ⏳ |
| M8 | Exact Teacher + SFT | 0% | ⏳（但 cpp_solve_with_policy 鸭子接口已就绪，随时可接 SFT teacher） |
| **M9** | **Robust RL 最小闭环** | **~15%** ⭐ | ⭐ 提前交付最小 smoke PPO self-play demo (N=5)；⏳ 未接 Exact 逼近验证 / NeuRD / Full-Matrix Bellman |
| M10~M20 | Search / GT-CFR / Opponent / League / RedTeam / Distill / Perf Hardening | 0% | ⏳（全部在设计阶段，代码未触及） |

## 下一批立即可以做的 Top 5 小步（建议顺序）

1. **编译闭环**：Windows 本机上实际 `cmake -S cxxgoof -B cxxbuild -G Ninja` → `cmake --build` → `pytest tests/test_cxx.py -v` 拿 4/4 通过，把 cxxgoof 从「源码写好」升级到「已验证」。（0.5 天。）
2. **Smoke 训练**：`python scripts/train_n5_ppo.py --num-cards 5 --num-envs 4096 --total-timesteps 100000 --seed 1` 跑完不 crash，记录 SPS、last avg_return、checkpoint 大小。（0.5 天。）
3. **M2.1 OpenSpiel Parity**：装 `open_spiel` pip 包，接 adapter，随机 1 万条轨迹比对 legal_actions / reward / score / result，挂 `tests/test_parity_openspiel.py`。（1~2 天。）
4. **把 NashBot 默认切 C++ Solver**：在 `bots.py` 的 NashBot._ensure_policy() 里如果 `CXX_ENABLED=True` 就 `return cpp_solve_with_policy(N)`，保留 Python solver 作为 fallback + cross-check 开关。（0.5 天。）
5. **TorchPolicyBot 接 checkpoint → 前端可用**：`bots.py` 新增 `TorchPolicyBot(ckpt_path)` 加载 `scripts/train_n5_ppo.py` 存的 `.pt`，然后 `/api/game/config` 下拉框多出一个 "PPO Trained (N=5 demo)" 选项，给用户直观感受 "训练出来的 bot 到底打得如何"。（1 天。）

完成以上 5 步后，系统会从「代码基本就绪但没在这台机器上证明」升级到「这台机器真实能训练 + 前端可对战训练模型」。

---
文档版本：Goofspiel-13 智能体总工程实施指南 v1.1 · 最后更新 2026-08-30（补技术栈状态列、§23/24 C++ 里程碑更新、M0~M3 验收、M9.0 PPO Precursor、文末工程进度快照） · Author 陈子聪 (Chen Zicong)