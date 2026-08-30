# Goofspiel-13 搜索、数学求解与智能体 Tool-Using Reasoning Layer 详细设计书
## Exact Nash Solver · Matrix Nash Solver · SM-MCTS · GT-CFR · Adaptive Best-Response Search · Tool Router

---

# 0. 文档定位

本文定义 Goofspiel 智能体在**真正思考和决策时**如何使用数学算法与搜索算法。

本文不重新定义：

- 神经网络结构；
- 强化学习 loss；
- Pre-train / SFT / Post-train；
- League / Red Team 训练生态。

本文只回答：

> 神经网络给出预测以后，智能体怎样进一步调用数学工具和搜索工具，把快速神经判断升级为更可靠的博弈决策？

核心体系：

\[
\boxed{
\text{Neural Prediction}
\rightarrow
\text{Mathematical Reasoning Tools}
\rightarrow
\text{Search}
\rightarrow
\text{Tool Router}
\rightarrow
\text{Final Decision}
}
\]

---

# 1. 总体原则

最终 Agent 不是：

```text
state
→ neural policy
→ argmax
```

而必须是：

```text
State
  ↓
Neural Model
  ↓
Fast Strategic Estimates
  ↓
Matrix Nash Solver
  ↓
Exact Feasibility Check
  ↓
Optional Exact / Search
  ↓
Optional Opponent-Adaptive Reasoning
  ↓
Safe Robust–Exploit Controller
  ↓
Final Mixed Policy
  ↓
Sample Action
```

因此：

\[
\boxed{
\text{Model}
\neq
\text{Search}
\neq
\text{Exact Solver}
\neq
\text{Decision Router}
}
\]

四者必须作为独立模块实现。

---

# 2. Tool Layer 的四个基本层级

必须实现四种主要 reasoning tool。

## Tool A — Matrix Nash Solver

输入：

\[
Q(s,a,b)
\]

输出：

\[
\pi_{\text{self}},
\pi_{\text{opp}},
V
\]

作用：

> 求解**神经网络预测出来的当前 joint-action matrix**的 Nash equilibrium。

它是低成本工具，正常每个决策都调用。

---

## Tool B — Exact Nash Solver

输入：

真实 Goofspiel state。

输出：

\[
Q^*(s,a,b),
\pi^*(s),
V^*(s)
\]

作用：

> 对当前剩余游戏进行完整数学递归求解。

它不依赖 neural value。

只要可计算，优先级最高。

---

## Tool C — Simultaneous-Move Search

主在线形式：

\[
\boxed{\text{SM-MCTS}}
\]

输入：

- state；
- neural model；
- time/node budget。

输出：

改进后的：

\[
Q^{search},\pi^{search},V^{search}
\]

作用：

> Exact 太贵时进行有限预算 lookahead。

---

## Tool D — Game-Theoretic Deep Search

主高质量形式：

\[
\boxed{\text{GT-CFR-like Search}}
\]

用途主要是：

- Teacher generation；
- Reanalyse；
- Red-Team correction；
- 高预算 evaluation；
- 重要局面的深搜索。

其运行成本高于 SM-MCTS。

---

# 3. Robust 与 Adaptive 工具严格分开

必须存在：

```text
ROBUST
ADAPTIVE
```

两个 reasoning mode。

## Robust

只允许使用：

\[
s
\]

public game state。

目标：

\[
\max_\pi\min_\sigma U(\pi,\sigma)
\]

。

Opponent history：

\[
\boxed{\text{禁止进入 Robust Solver/Search}}
\]

---

## Adaptive

允许额外读取：

\[
h_{\text{opp}}
\]

以及：

\[
q_\phi(b|s,h)
\]

目标：

\[
\max_\pi E_{b\sim q_\phi}[U]
\]

。

Adaptive Search 得到的是：

\[
\boxed{\text{Exploit Candidate}}
\]

而不是最终无条件执行策略。

---

# 4. Tool Mode

所有工具调用必须带：

```python
mode: ToolMode
```

枚举：

```python
class ToolMode(Enum):
    PLAY = "play"
    TEACHER = "teacher"
    REANALYSE = "reanalyse"
    REDTEAM_CORRECTION = "redteam_correction"
    EVALUATION = "evaluation"
```

不同 mode 可以使用不同预算和质量阈值。

---

# 第一部分：统一状态与接口

# 5. ReasoningState

所有工具读取同一 immutable state。

建议：

```python
@dataclass(frozen=True)
class ReasoningState:
    num_cards: int

    self_cards_mask: int
    opponent_cards_mask: int

    remaining_prizes_mask: int

    current_prize: int

    self_score: int
    opponent_score: int

    round_index: int

    done: bool
```

---

# 6. 工具不得偷偷读取非法信息

Tool Layer 只能看到：

- 双方剩余牌；
- 已发生历史；
- 当前公开 prize；
- 当前得分。

当前 simultaneous round：

> 对手还没有 reveal 的 action 不允许任何工具读取。

禁止：

```python
search(state, opponent_current_action)
```

。

---

# 7. Canonical State Key

Exact/cache 使用：

```python
@dataclass(frozen=True)
class CanonicalStateKey:
    n: int
    self_mask: int
    opp_mask: int
    prize_mask: int
    current_prize: int
```

对于 future-before-prize state：

```python
@dataclass(frozen=True)
class ChanceStateKey:
    n: int
    self_mask: int
    opp_mask: int
    prize_mask: int
```

注意：

Robust future value 不需要把历史得分放入 DP key。

因为 Q 学习与 Exact objective 是：

\[
\text{future score difference}
\]

历史分数已经 sunk，不影响未来 optimal strategy。

---

# 8. Player-Swap Canonicalization

利用：

\[
F(A,B,R)
=
-F(B,A,R)
\]

。

若：

```text
A_mask > B_mask
```

可以交换双方进入 canonical key。

cache value 返回时修正符号。

必须单元测试。

---

# 9. Unified Tool Result

所有工具输出：

```python
@dataclass
class GameToolResult:
    source: str
    mode: str

    policy_self: Tensor
    policy_opponent: Tensor | None

    q_matrix: Tensor | None
    value: float | None

    valid_self_mask: Tensor
    valid_opponent_mask: Tensor

    quality_score: float
    duality_gap: float | None

    exactness: str

    runtime_ms: float
    expanded_nodes: int
    simulations: int

    exact_leaf_hits: int
    neural_leaf_hits: int

    state_key: object

    model_version: str | None
    opponent_model_version: str | None

    diagnostics: dict
    valid: bool
```

---

# 10. `exactness`

必须使用固定枚举：

```text
NONE
APPROXIMATE
NUMERICAL_EXACT
EXACT_WRT_OPPONENT_MODEL
RATIONAL_EXACT
```

当前 FP64 + LP 的 Exact Solver 返回：

```text
NUMERICAL_EXACT
```

不得称：

```text
RATIONAL_EXACT
```

。

---

# 第二部分：Matrix Nash Solver

# 11. Matrix Nash Solver 的定位

神经网络输出：

\[
Q_R(s)\in\mathbb R^{N\times N}
\]

当前 state 的 matrix game：

\[
\max_x\min_yx^TQ_Ry
\]

Matrix Solver 计算：

\[
x^*,y^*,V_Q
\]

。

重要：

\[
\boxed{
\text{这只是模型 Q 的 Nash 解，不是整个游戏的 Exact Nash}
}
\]

其 provenance 必须记录为：

```text
MODEL_MATRIX_NASH
```

。

---

# 12. Matrix Solver 两套实现

必须同时实现：

## Reference

```text
FP64
CPU
Linear Programming
```

用途：

- validation；
- Exact Solver；
- Search quality validation。

---

## Batched

```text
GPU
FP32
Regret Matching+ / suitable batched no-regret solver
```

用途：

- neural inference；
- online reasoning；
- training。

---

# 13. Reference LP

Row player：

\[
\max_{x,v}v
\]

subject to：

\[
Q^Tx\ge v\mathbf1
\]

\[
\sum_ax_a=1
\]

\[
x_a\ge0
\]

Column dual同理。

输出必须同时返回：

- row strategy；
- column strategy；
- primal value；
- dual value；
- duality gap。

---

# 14. Batched Solver

接口：

```python
def solve_batch(
    q: Tensor,              # [B,N,N]
    self_mask: Tensor,
    opponent_mask: Tensor,
    iterations: int,
) -> MatrixGameSolution:
    ...
```

输出：

```python
@dataclass
class MatrixGameSolution:
    row_policy: Tensor
    column_policy: Tensor
    value: Tensor
    duality_gap: Tensor
    iterations: int
```

---

# 15. Solver Failure

若：

- NaN；
- policy probability 非法；
- probability sum 偏差；
- duality gap 超标；

则：

```text
valid=False
```

Router 必须 fallback。

不得：

> Solver 报错以后随便 softmax Q。

---

# 16. Actor–Q Disagreement

每一步计算：

\[
D_{\pi,Q}
=
JSD(
\pi_R^{actor},
\pi_R^{matrix}
)
\]

。

这既是：

- diagnostics；
- search trigger；
- model consistency signal。

必须记录。

---

# 第三部分：Exact Nash Solver

# 17. Exact Solver 的数学对象

定义：

\[
F(A,B,R)
\]

为：

> 当前 prize 尚未 reveal，双方剩余 bid cards 为 \(A,B\)，剩余 prize set 为 \(R\) 时的最优 expected future normalized score difference。

terminal：

\[
F(\varnothing,\varnothing,\varnothing)=0
\]

。

---

# 18. Current Prize Node

若当前 prize：

\[
p
\]

已经 reveal，

对：

\[
a\in A,b\in B
\]

构造：

\[
Q^*_{ab}
=
\frac{p}{S_N}\operatorname{sgn}(a-b)
+
F(A-\{a\},B-\{b\},R)
\]

其中 \(R\) 不包含当前 prize。

然后：

\[
V_p
=
Val(Q^*)
\]

。

---

# 19. Chance Node

若下一 prize 尚未 reveal：

\[
F(A,B,R)
=
\frac1{|R|}
\sum_{p\in R}
V_p(A,B,R)
\]

。

这里必须完整枚举 chance。

Exact Solver 不允许 chance sampling。

---

# 20. Memoization

必须使用：

```python
cache[(A_mask, B_mask, R_mask)]
```

。

禁止用 recursive Python object graph 作为缓存。

---

# 21. Exact Solver Preflight

任何：

```python
solve()
```

之前必须：

```python
estimate()
```

。

接口：

```python
@dataclass
class ExactComplexityEstimate:
    estimated_states: int
    estimated_matrix_games: int
    estimated_joint_cells: int

    estimated_runtime_ms: float
    estimated_memory_bytes: int

    risk_level: str

    feasible_under_budget: bool
```

---

# 22. Complexity 计算

数学基础：

\[
C_k=
{N\choose k}^3
\]

。

局部 state 则应根据当前：

\[
|A|=|B|=|R|=k
\]

和具体 bitmask 剩余规模估计实际 subtree。

工程 estimator 必须使用：

- 理论 state count；
- 当前机器 LP benchmark；
- measured cache bytes/state。

---

# 23. Risk Level

统一：

```text
GREEN
YELLOW
ORANGE
RED
BLACK
```

Router：

```text
GREEN   → normally allowed
YELLOW  → allowed if budget comfortable
ORANGE  → teacher/offline only by default
RED     → reject unless force
BLACK   → reject
```

---

# 24. `force=True`

只允许：

```text
TEACHER
EVALUATION
REANALYSE
```

显式使用。

PLAY 模式默认：

```text
force=False
```

。

---

# 25. Exact Result

```python
@dataclass
class ExactNashResult(GameToolResult):
    q_matrix: Tensor
    policy_self: Tensor
    policy_opponent: Tensor

    value: float

    states_solved: int
    matrix_games_solved: int
    cache_hits: int
    cache_misses: int
```

---

# 26. Multiple Equilibria

必须定义稳定 tie-breaking。

第一阶段：

求 game value：

\[
V^*
\]

第二阶段：

在：

\[
\mathcal E_\epsilon
\]

即满足 equilibrium/value constraints 的策略集合中，优先求 high-entropy equilibrium。

目的：

- 减少数值跳变；
- 产生稳定 Teacher；
- 提高 regression reproducibility。

---

# 27. Exact Cache

Exact cache 可以跨：

- neural checkpoint；
- training stage；
- session；

永久复用。

Key 必须包含：

```text
rules_version
objective_version
state_key
solver_version
```

---

# 28. Exact Cache 不依赖模型版本

禁止：

```text
exact_cache_key += model_hash
```

。

Exact Solver 与神经模型无关。

---

# 第四部分：Exact Best Response

# 29. 另一个数学工具

除了 Nash，还必须支持：

\[
\boxed{\text{Exact Best Response}}
\]

给定固定 opponent policy：

\[
q_\phi(b|s,h)
\]

求：

\[
BR(q_\phi)
\]

。

---

# 30. Bellman

定义：

\[
Q^{BR}(s,a,b)
=
r+
E[V^{BR}(s')]
\]

。

然后：

\[
V^{BR}(s)
=
\max_a
\sum_b
q_\phi(b|s,h)
Q^{BR}(s,a,b)
\]

。

---

# 31. History Transition

由于：

\[
q_\phi
\]

可能依赖 history，

counterfactual child：

\[
h'
\]

必须使用真实 opponent-memory transition：

```python
h_next = opponent_memory.step_counterfactual(
    current_history=h,
    prize=p,
    self_action=a,
    opponent_action=b,
    outcome=...
)
```

。

同局内：

LSTM 更新。

Mamba：

\[
\boxed{\text{不更新}}
\]

。

---

# 32. Exact BR 的 exactness

返回：

```text
EXACT_WRT_OPPONENT_MODEL
```

。

不能宣传：

> “这是 game-theoretic exact solution。”

因为其正确性依赖：

\[
q_\phi
\]

是否正确。

---

# 第五部分：Leaf Evaluator

# 33. 所有 Search 共用 LeafEvaluator

禁止 SM-MCTS 与 GT-CFR 各自偷偷实现一套 leaf value logic。

统一：

```python
class LeafEvaluator:
    def evaluate_robust(...)
    def evaluate_adaptive(...)
```

---

# 34. Robust Leaf 优先级

严格：

```text
1. Exact cache hit
2. Exact solve if feasible
3. Neural Q_R
   → Matrix Nash
4. Failure
```

Neural leaf：

\[
V=
Val(Q_R^\theta)
\]

。

不得默认使用：

\[
E[Z_R]
\]

替代 Nash value。

---

# 35. Adaptive Leaf

优先：

```text
1. Exact BR cache/solve if feasible
2. Adaptive Q_A + opponent model
3. Neural fallback
```

。

---

# 36. Leaf Provenance

每一个 leaf 必须记录：

```text
EXACT
EXACT_BR
NEURAL_Q
```

最终 SearchResult 汇总：

```text
exact_leaf_hits
neural_leaf_hits
```

。

---

# 第六部分：SM-MCTS

# 37. 为什么必须是 Simultaneous-Move

禁止普通：

```text
Player A action
↓
Player B observes
↓
Player B action
```

。

正确状态节点：

\[
\boxed{\text{同时产生 }(a,b)}
\]

。

---

# 38. Node 数据结构

```python
@dataclass
class SMNode:
    state_key: object

    self_actions: Tensor
    opponent_actions: Tensor

    q_joint: Tensor
    visit_joint: Tensor

    regret_self: Tensor
    regret_opponent: Tensor

    strategy_sum_self: Tensor
    strategy_sum_opponent: Tensor

    current_strategy_self: Tensor
    current_strategy_opponent: Tensor

    total_visits: int

    children: dict
```

---

# 39. Q 初始化

初次 expand：

\[
Q_{tree}(a,b)
\leftarrow
Q_R^\theta(s,a,b)
\]

并设置：

```yaml
search:
  network_prior_pseudocount: 1.0
```

解释：

神经 Q 相当于少量虚拟访问。

---

# 40. 不允许高 pseudocount 锁死 Search

默认：

\[
N_0=1
\]

。

禁止：

```text
N0=100
```

导致搜索几乎无法修正错误 network。

---

# 41. Node Strategy

使用 Regret Matching+。

累计：

\[
R^+_a
\]

。

策略：

若：

\[
\sum_aR_a^+>0
\]

则：

\[
\sigma(a)
=
\frac{R_a^+}{\sum_jR_j^+}
\]

否则：

使用：

\[
(1-\epsilon)\pi_{network}
+
\epsilon U
\]

。

默认：

```yaml
search:
  zero_regret_uniform_mix: 0.10
```

---

# 42. Joint Action Sampling

必须：

\[
a\sim\sigma_A
\]

\[
b\sim\sigma_B
\]

独立采样。

禁止：

```python
joint_action = argmax(q_joint)
```

。

---

# 43. Chance

joint action 后：

下一 prize：

\[
p'\sim Uniform(R)
\]

。

SM-MCTS simulation 中：

默认随机 chance sampling。

高预算模式可以 stratified / balanced chance sampling。

---

# 44. Chance Visit Balancing

为避免某 prize 长期采样不足：

ChanceNode 可以维护：

```text
visit_count_per_prize
```

优先采样：

\[
N(p)
\]

较少的 prize。

但长期边际概率仍必须对应 uniform chance。

---

# 45. Expansion

到未展开 state：

1. 调用 LeafEvaluator；
2. 初始化 node；
3. 返回 leaf value；
4. backup。

---

# 46. Backup Return

若当前 immediate：

\[
r_t
\]

child：

\[
G_{t+1}
\]

：

\[
G_t=r_t+G_{t+1}
\]

因为：

\[
\gamma=1
\]

。

---

# 47. Joint Q Update

对于访问的：

\[
(a,b)
\]

：

\[
N_{ab}\leftarrow N_{ab}+1
\]

\[
Q_{ab}
\leftarrow
Q_{ab}
+
\frac{G-Q_{ab}}{N_{ab}+N_0}
\]

具体实现需把 pseudocount 纳入初始统计一致处理。

---

# 48. Regret Update

当前：

\[
Q
\]

与：

\[
\sigma_A,\sigma_B
\]

。

row action utility：

\[
u_A(a)
=
\sum_b\sigma_B(b)Q(a,b)
\]

当前 value：

\[
v=\sigma_A^TQ\sigma_B
\]

：

\[
R_A(a)
\leftarrow
\max(
0,
R_A(a)+u_A(a)-v
)
\]

。

column 使用：

\[
-Q^T
\]

同一函数计算。

---

# 49. Strategy Averaging

每次：

\[
S_A
\leftarrow
S_A+\sigma_A
\]

。

最终：

\[
\bar\sigma_A
=
S_A/\sum S_A
\]

。

正式 Search policy 使用：

\[
\boxed{\bar\sigma_A}
\]

而不是最后一步 current strategy。

---

# 50. SM-MCTS Stopping

支持三种 budget。

```python
SearchBudget(
    max_simulations=...,
    max_nodes=...,
    wall_time_ms=...,
)
```

任意一个达到：

立即停止。

---

# 51. 默认在线搜索档位

配置：

```yaml
sm_mcts:
  budgets:
    tiny: 128
    small: 512
    medium: 2048
    large: 8192
```

以 simulations 为主。

wall time 是硬上限。

---

# 52. Search Diagnostics

必须计算：

### Policy shift

\[
JSD(
\pi_{matrix},
\pi_{search}
)
\]

### Strategy stability

最近两个窗口：

\[
JSD(
\pi^{T},
\pi^{T-\Delta}
)
\]

### Coverage

访问过的 legal joint cells 比例。

### Root gap

用当前 root Q 运行 matrix solver 得 approximate duality gap。

---

# 53. SM-MCTS Quality Gate

至少检查：

```text
finite probabilities
probability sums correct
legal mask respected
strategy stability
minimum simulation count
root gap
```

。

若不可靠：

```text
valid=False
```

Router fallback。

---

# 第七部分：GT-CFR Search

# 54. 定位

GT-CFR-like 模块是：

\[
\boxed{\text{高预算 game-theoretic search}}
\]

不是主要低延迟在线工具。

默认用于：

```text
TEACHER
REANALYSE
REDTEAM_CORRECTION
EVALUATION
```

PLAY 模式只有大时间预算时才使用。

---

# 55. Tree Representation

必须直接支持 simultaneous node：

```python
class SimultaneousCFRNode:
    regret_self
    regret_opp

    strategy_sum_self
    strategy_sum_opp

    joint_children
```

。

不要为了省事变成：

```text
A acts
↓
B sees A
↓
B acts
```

。

---

# 56. Regret Matching+

双方：

\[
\sigma_A=RM^+(R_A)
\]

\[
\sigma_B=RM^+(R_B)
\]

。

---

# 57. Tree 非完整展开

初始 tree 只有 root。

每轮：

```text
CFR passes
↓
estimate frontier importance
↓
expand selected frontier
↓
evaluate leaves
↓
more CFR passes
```

。

---

# 58. Frontier Priority

一个 frontier 可用：

\[
P_{frontier}
=
Reach(s)
\cdot
U(s)
\cdot
I(s)
\]

其中：

### Reach

当前平均策略下 reach probability。

### U

neural/search uncertainty。

### I

对 root value/policy 的潜在影响。

---

# 59. 不只扩展最高概率路径

必须保留 exploration。

例如：

\[
P'(s)
=
(1-\epsilon)
\frac{P(s)}{\sum P}
+
\epsilon U_{\text{frontier}}
\]

避免 search 永远看不到低 reach 但高 exploit 的分支。

---

# 60. CFR Iteration

在当前 partial tree：

进行：

\[
K
\]

轮 regret update。

frontier 未展开部分：

由 LeafEvaluator 提供 value。

---

# 61. Expansion Schedule

默认：

```yaml
gt_cfr:
  initial_cfr_iterations: 64
  iterations_per_expansion: 32
  nodes_per_expansion: 4
```

高预算时可调整。

---

# 62. GT-CFR 输出

```python
@dataclass
class GTCFRResult(GameToolResult):
    cumulative_regret_self: Tensor
    cumulative_regret_opponent: Tensor

    average_policy_self: Tensor
    average_policy_opponent: Tensor

    tree_depth_max: int
    frontier_count: int

    cfr_iterations: int
```

---

# 63. Teacher Qualification

不是所有 GT-CFR result 都可以当 teacher。

例如必须满足：

```yaml
teacher_quality:
  max_policy_instability: 0.02
  max_duality_gap: 0.01
  min_iterations: 256
```

具体阈值可实验调整。

若不满足：

只作为 diagnostic。

---

# 第八部分：Adaptive Best-Response Search

# 64. Adaptive Search 输入

```python
adaptive_search(
    state,
    opponent_context,
    q_opponent,
    budget,
)
```

。

必须保存：

```text
opponent_model_version
```

。

---

# 65. Opponent Action Distribution

每个 search node：

调用：

\[
q_\phi(b|s,h)
\]

得到对手概率。

不能只在 root 预测一次，然后整棵树固定。

因为：

\[
q_\phi
\]

可能依赖：

- 当前剩余资源；
- 新历史；
- 当前局进展。

---

# 66. Counterfactual LSTM

每条模拟路径维护自己的：

\[
h_L
\]

。

joint action 后：

```python
h_L_next = lstm_step(
    h_L,
    round_event=(p,a,b,result),
)
```

。

禁止不同 simulation 共用并原地修改同一个 mutable LSTM state。

必须 copy / tensor branch。

---

# 67. Mamba

当前单局 search：

\[
\boxed{\text{Mamba state frozen}}
\]

。

因为 hypothetical round 不代表真实 completed game。

---

# 68. Adaptive Node Value

给 Q：

\[
Q_A(s,h)
\]

和 opponent distribution：

\[
q(b)
\]

自己 action utility：

\[
u(a)=\sum_bq(b)Q_A(a,b)
\]

。

soft best response：

\[
\pi_{BR}(a)
=
softmax(u(a)/\tau)
\]

。

---

# 69. Adaptive Search 可采用 Expectation 而不是采样

因为最多：

\[
13
\]

个 opponent actions。

在较小 action count：

可以直接：

\[
\sum_bq(b)
\]

枚举对手动作。

相比采样方差更低。

默认：

```text
if legal_opponent_actions <= 13:
    enumerate opponent actions
```

实际上标准 Goofspiel始终满足。

所以 Adaptive Search 优先使用 exact opponent expectation。

---

# 70. 仍然需要 chance handling

Prize chance：

可以 sampling 或枚举。

高预算：

剩余 prize 少时直接枚举。

---

# 71. Adaptive Search Result

必须：

```text
source=ADAPTIVE_BR_SEARCH
exactness=APPROXIMATE
```

除非整个 subtree 使用 Exact BR。

---

# 72. Adaptive Search 不能直接选最终 action

其输出必须进入：

\[
SafeExploitController
\]

。

---

# 第九部分：Safe Robust–Exploit Controller

# 73. 输入

最高质量 Robust result：

\[
Q_R,\pi_R,V_R
\]

Adaptive：

\[
Q_A,\pi_A
\]

Opponent belief：

\[
q_\phi
\]

Opponent uncertainty：

\[
U_{\text{opp}}
\]

。

---

# 74. Candidate Mixture

先定义：

\[
\pi_\alpha
=
(1-\alpha)\pi_R+\alpha\pi_A
\]

其中：

\[
\alpha\in[0,1]
\]

。

---

# 75. Robust Worst-Case Value

使用当前最高质量 robust Q：

\[
V_{worst}(\pi_\alpha)
=
\min_b
\sum_a
\pi_\alpha(a)Q_R(a,b)
\]

。

---

# 76. Safety Constraint

要求：

\[
V_{worst}(\pi_\alpha)
\ge
V_R-\epsilon
\]

。

其中：

\[
\epsilon
\]

是允许为了 exploit 放弃的 robust value。

---

# 77. Uncertainty Tightening

Opponent uncertainty 越高：

允许的：

\[
\epsilon
\]

越小。

例如：

\[
\epsilon_{eff}
=
\epsilon_{max}
\cdot
c_{\text{opp}}
\]

其中：

\[
c_{\text{opp}}\in[0,1]
\]

来自 calibrated opponent confidence。

未知 opponent：

\[
c\approx0
\]

则：

\[
\epsilon_{eff}\approx0
\]

几乎纯 Robust。

---

# 78. 求最大安全 \(\alpha\)

从：

\[
\alpha=1
\]

开始检查。

不满足则 binary search：

```python
alpha ∈ [0,1]
```

找到最大满足 safety constraint 的：

\[
\alpha^*
\]

。

---

# 79. 不简单以混合策略为最终唯一形式

后续可以实现更强 constrained optimisation：

\[
\max_\pi
\pi^TQ_Aq_\phi
\]

subject to：

\[
\min_b\pi^TQ_R[:,b]
\ge
V_R-\epsilon
\]

\[
\sum_a\pi_a=1,\quad\pi_a\ge0
\]

这是一个小型线性规划。

最终推荐同时实现：

```text
SAFE_MIXTURE
SAFE_LP
```

。

默认最终方案优先：

\[
\boxed{\text{SAFE\_LP}}
\]

因为只有最多 13 个 action。

---

# 80. Safe LP

优化：

\[
\max_\pi c^T\pi
\]

其中：

\[
c_a
=
\sum_bq_\phi(b)Q_A(a,b)
\]

subject to 对每一个 opponent pure action：

\[
\sum_a\pi_aQ_R(a,b)
\ge
V_R-\epsilon
\]

。

这直接保证：

> 无论真实对手突然采取哪一个纯动作，都不低于 robust safety floor。

这是比简单 interpolation 更漂亮的正式决策算法。

---

# 第十部分：Tool Router

# 81. Tool Router 不使用 learned gate 起步

第一版必须：

\[
\boxed{\text{Deterministic + Auditable}}
\]

。

输入：

```python
RouterInput(
    state,
    model_output,
    opponent_context,
    budget,
    mode,
)
```

。

---

# 82. Router 固定流程

严格顺序：

```text
1. Neural forward
2. Matrix Nash
3. Exact complexity preflight
4. Exact if feasible
5. Otherwise decide search budget
6. Run Robust Search if required
7. Select highest-quality Robust result
8. If opponent model usable:
       Adaptive reasoning
9. Safe exploit optimisation
10. Return final policy
```

不得擅自交换关键顺序。

---

# 83. Baseline Robust Result

无论如何，先得到：

\[
R_0=
MatrixNash(Q_R^\theta)
\]

。

这是所有 fallback 的底线。

---

# 84. Exact Preflight 每一步必做

必须：

```python
estimate = exact_solver.estimate(state)
```

成本应极低。

若：

```text
feasible=True
```

且：

\[
estimated\_runtime
\]

满足预算：

调用 Exact。

---

# 85. Exact 成功则 Robust 终止

如果：

```text
valid=True
exactness=NUMERICAL_EXACT
```

：

当前 Robust result：

\[
R_{robust}=R_{exact}
\]

Search 不允许覆盖。

除非 mode 为：

```text
EVALUATION
```

为了比较算法。

---

# 86. Search Trigger

若 Exact 不可行：

计算：

\[
S_{search}
\]

。

组件：

### Q Ensemble disagreement

归一化：

\[
U_Q
\]

### Actor–Matrix disagreement

\[
D_{\pi,Q}
\]

### Policy entropy

\[
H(\pi_{matrix})
\]

### Strategic Importance

\[
I(s)
\]

### Failure Prior

\[
F(s)
\]

。

---

# 87. Search Trigger 归一化

所有组件必须映射：

\[
[0,1]
\]

然后：

\[
S=
w_UU_Q+
w_DD+
w_HH+
w_II+
w_FF
\]

默认初始：

```yaml
router:
  uncertainty_weight: 0.30
  disagreement_weight: 0.25
  entropy_weight: 0.10
  importance_weight: 0.20
  failure_weight: 0.15
```

以后可实验调整。

---

# 88. Strategic Importance

默认可以定义：

\[
I(s)=
0.4I_{prize}
+
0.3I_{early}
+
0.3I_{future}
\]

例如：

### Current prize importance

\[
p/S_N
\]

### Early strategic importance

前几轮资源选择具有长程影响。

### Future prize mass

剩余总 prize value。

所有指标配置化。

---

# 89. PLAY Search Budget

例如：

```yaml
router:
  play:
    direct_threshold: 0.25
    small_threshold: 0.50
    medium_threshold: 0.75
```

对应：

```text
S < .25
→ Matrix Nash only

.25–.50
→ SM-MCTS 128

.50–.75
→ SM-MCTS 512

>.75
→ SM-MCTS 2048
```

如果 wall-clock budget 不够：

自动降档。

---

# 90. Teacher Mode

TEACHER：

优先：

```text
Exact
GT-CFR
High-budget SM-MCTS
```

。

不会因为：

\[
S_{search}
\]

低就完全不 Search。

Teacher mode 目标是：

\[
\boxed{\text{label quality}}
\]

而不是 latency。

---

# 91. Red-Team Correction

优先：

```text
Exact
↓
High-budget GT-CFR
↓
High-budget SM-MCTS
↓
Teacher Ensemble
```

。

---

# 92. Search Result Selection

不是：

> 只要 Search 跑过，就采用 Search。

定义：

\[
R_{base}
\]

和：

\[
R_{search}
\]

。

Search 必须通过质量 gate。

若：

```text
valid=False
```

：

回退 Base。

---

# 93. Search Quality Score

建议：

\[
Q_{score}
=
1
-
w_g\hat g
-
w_s\hat d_{stability}
-
w_e\hat e
\]

其中：

- \(g\)：duality gap；
- \(d\)：策略不稳定；
- \(e\)：solver/search numerical error。

限制：

\[
[0,1]
\]

。

---

# 94. Search 不要求必须改变策略

如果：

\[
\pi_{search}\approx\pi_{base}
\]

：

这不代表 Search 没用。

它可能确认：

> 网络本来就判断正确。

记录：

```text
search_confirmation=True
```

。

---

# 95. Tool Router 输出

```python
@dataclass
class AgentReasoningResult:
    final_policy: Tensor

    robust_policy: Tensor
    adaptive_policy: Tensor | None

    safe_exploit_alpha: float | None

    robust_source: str
    adaptive_source: str | None

    tool_results: list[GameToolResult]

    total_runtime_ms: float

    diagnostics: dict
```

---

# 第十一部分：Budget System

# 96. DecisionBudget

```python
@dataclass
class DecisionBudget:
    wall_time_ms: int

    exact_time_fraction: float
    search_time_fraction: float

    max_search_nodes: int
    max_exact_memory_bytes: int

    allow_gt_cfr: bool
    allow_adaptive_search: bool
```

---

# 97. 时间预算是硬限制

工具必须支持：

```python
deadline = monotonic_time + budget
```

循环内部定期检查。

不能只在搜索外面检查。

---

# 98. Budget 分配

PLAY 初始建议：

```text
Exact estimate       negligible
Exact solve          max 30%
Robust search        max 60%
Adaptive reasoning   max 30%
Safety solve         negligible
```

这些预算可以动态共享：

Exact 没调用：

Search 可以使用释放时间。

---

# 99. Exact 不应吃掉全部时间

例如预算：

\[
30s
\]

Exact estimator：

\[
27s
\]

即使可行也不一定值得。

Router 可定义：

```yaml
exact:
  max_play_budget_fraction: 0.30
```

所以预计超过：

\[
9s
\]

时：

PLAY 不调用完整 Exact。

但 Search leaf 仍可 exact solve 小残局。

---

# 第十二部分：Cache

# 100. Exact Cache

永久缓存。

建议：

```text
SQLite / LMDB / RocksDB
```

或高性能自定义 KV。

内存层再有 LRU。

---

# 101. Exact Cache Value

至少：

```text
value
Q matrix
row strategy
column strategy
solver tolerance
timestamp
solver version
```

。

---

# 102. Search Cache

与模型有关。

key：

```text
state_key
model_hash
search_algorithm
search_config_hash
opponent_context_hash
```

。

---

# 103. Search Tree Reuse

同一真实游戏：

上一轮 Search tree：

实际发生：

\[
a^*,b^*,p^*
\]

下一轮：

将对应 child：

```text
root = old_root.child[a*,b*,p*]
```

。

---

# 104. Tree Reuse 条件

必须满足：

```text
same model version
same search config
same opponent model version if adaptive
```

。

否则丢弃或重新 leaf evaluation。

---

# 第十三部分：工具作为训练教师

# 105. Search/Exact 不是只在部署使用

Tool Layer 必须为训练系统提供：

```python
tool_result.to_teacher_target()
```

。

---

# 106. Exact Teacher

直接：

\[
Q^*
\]

\[
\pi^*
\]

\[
V^*
\]

。

最高优先级。

---

# 107. SM-MCTS Teacher

只有 quality gate 通过才可：

```text
source=SM_MCTS
```

。

一般权重低于 GT-CFR。

---

# 108. GT-CFR Teacher

高质量：

\[
\pi^{CFR}
\]

以及 root value/Q。

用于：

- SFT；
- semi-supervised；
- failure correction；
- reanalyse。

---

# 109. Adaptive Teacher

Exact BR / Adaptive Search：

只能监督：

\[
Q_A,\pi_A
\]

。

禁止监督：

\[
Q_R
\]

。

---

# 110. Tool Provenance

所有训练 target 保存：

```text
source
model_version
search_budget
solver_version
quality_score
duality_gap
```

。

以后可以分析：

> 哪类 Teacher 真正带来提升。

---

# 第十四部分：Learned Value-of-Computation Router

# 111. 不在第一阶段启用

第一阶段 Router：

完全 deterministic。

但必须从第一天记录训练数据：

```text
state features
uncertainty
actor-Q disagreement

tool selected
tool cost

base policy/value
tool policy/value

policy shift
estimated improvement
actual downstream result
```

。

---

# 112. 后期学习

可以训练：

\[
C_\eta(s,budget,tool)
\]

预测：

\[
\boxed{
\text{Expected Improvement per Unit Compute}
}
\]

例如：

\[
VOC=
\frac{
ExpectedQualityGain
}{
milliseconds
}
\]

。

---

# 113. Learned Router 不能覆盖硬约束

即使未来：

\[
C_\eta
\]

说“不用 Exact”，

若状态进入明确 Exact-override 规则：

仍然可以调用。

同样：

安全约束不能由 learned router 关闭。

最终：

\[
\boxed{
Hard Rules
+
Learned Compute Allocation
}
\]

。

---

# 第十五部分：系统代码结构

# 114. 推荐目录

```text
reasoning/
│
├── types.py
├── budget.py
├── state_key.py
│
├── matrix/
│   ├── reference_lp.py
│   ├── batched_rm_plus.py
│   ├── equilibrium.py
│   └── tests/
│
├── exact/
│   ├── complexity.py
│   ├── future_value.py
│   ├── exact_nash.py
│   ├── exact_best_response.py
│   ├── canonical.py
│   ├── cache.py
│   └── tests/
│
├── search/
│   ├── leaf_evaluator.py
│   │
│   ├── sm_mcts/
│   │   ├── node.py
│   │   ├── regret.py
│   │   ├── search.py
│   │   └── result.py
│   │
│   ├── gt_cfr/
│   │   ├── node.py
│   │   ├── tree.py
│   │   ├── regret.py
│   │   ├── frontier.py
│   │   └── search.py
│   │
│   └── adaptive/
│       ├── history_branch.py
│       └── br_search.py
│
├── routing/
│   ├── search_trigger.py
│   ├── quality_gate.py
│   ├── exact_router.py
│   ├── compute_value.py
│   └── tool_router.py
│
├── decision/
│   ├── robust_result.py
│   ├── safe_mixture.py
│   ├── safe_lp.py
│   └── final_policy.py
│
├── teacher/
│   └── tool_to_teacher.py
│
└── agent.py
```

---

# 第十六部分：`GameAgent`

# 115. 顶层接口

```python
class GameAgent:

    def think(
        self,
        state: ReasoningState,
        opponent_context: OpponentContext | None,
        budget: DecisionBudget,
        mode: ToolMode = ToolMode.PLAY,
    ) -> AgentReasoningResult:
        ...
```

---

# 116. `think()` 固定逻辑

伪代码：

```python
def think(...):

    # 1. Neural prediction
    model_out = model.forward(...)

    # 2. Always solve model Q matrix
    matrix_result = matrix_nash.solve(
        model_out.q_robust
    )

    robust_result = matrix_result

    # 3. Exact preflight
    estimate = exact_solver.estimate(state)

    # 4. Exact if appropriate
    if exact_router.should_solve(
        estimate,
        budget,
        mode
    ):
        exact_result = exact_solver.solve(...)

        if exact_result.valid:
            robust_result = exact_result

    # 5. Search only if robust isn't exact
    if robust_result.exactness not in (
        "NUMERICAL_EXACT",
        "RATIONAL_EXACT",
    ):

        trigger = search_trigger.compute(
            model_out,
            matrix_result,
            state,
        )

        search_budget = router.allocate_search_budget(
            trigger,
            budget,
            mode
        )

        if search_budget.enabled:

            search_result = robust_search.search(...)

            if quality_gate.accept(search_result):
                robust_result = choose_robust_result(
                    robust_result,
                    search_result
                )

    # 6. Adaptive reasoning
    adaptive_result = None

    if opponent_context is not None:
        if opponent_controller.usable(
            opponent_context
        ):
            adaptive_result = run_adaptive_reasoning(...)

    # 7. Safe final decision
    if adaptive_result is not None:
        final_policy = safe_exploit.solve(
            robust_result,
            adaptive_result,
            opponent_context,
        )
    else:
        final_policy = robust_result.policy_self

    return AgentReasoningResult(...)
```

---

# 117. `act()`

另一个轻接口：

```python
def act(...):
    result = think(...)
    return categorical_sample(result.final_policy)
```

。

默认：

\[
\boxed{\text{sampling}}
\]

而不是 argmax。

---

# 第十七部分：Debug / Detector 输出

# 118. 每步必须能够显示推理来源

例如：

```text
#7 Prize=Q

MODEL
  Q uncertainty=0.071
  Actor↔Nash JSD=0.124

MATRIX NASH
  value=+0.031
  policy=...

EXACT
  estimate=RED
  skipped

SEARCH
  SM-MCTS 2048
  nodes=1342
  exact leaves=81
  neural leaves=927
  root gap=0.013
  policy shift=0.086

OPPONENT
  confidence=0.82
  predicted top action=9 (0.41)

ADAPTIVE
  expected gain=+0.052

SAFE LP
  safety epsilon=0.015
  exploit weight equivalent=0.37

FINAL
  policy=...
  sampled action=10
```

这对科研调试极其重要。

---

# 第十八部分：必须实现的 Unit Tests

# 119. Matrix Matching Pennies

\[
Q=
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}
\]

要求：

\[
\pi=(0.5,0.5)
\]

\[
V=0
\]

。

---

# 120. Dominated Strategy

构造具有严格 dominated action 的 matrix。

Nash policy 对该 action：

\[
\approx0
\]

。

---

# 121. Exact N=1

必须直接等于 immediate reward。

---

# 122. Exact Symmetry

随机 state：

\[
F(A,B,R)
=
-F(B,A,R)
\]

数值 tolerance 内成立。

---

# 123. Exact Cache

同状态第二次求：

cache hit。

结果完全一致。

---

# 124. Exact 不读取模型

替换 neural model 参数。

Exact result：

完全不变。

---

# 125. SM-MCTS Simultaneous Privacy

人为 instrument：

确保 opponent node 永远无法读取当前 sampled self action 后再决定自己的 action。

这是硬测试。

---

# 126. SM-MCTS Matching Pennies

搜索足够 iterations：

average strategy：

\[
\rightarrow(0.5,0.5)
\]

。

---

# 127. Average vs Current Policy

测试结果对象同时返回两者。

正式 `policy_self` 指向：

average strategy。

---

# 128. Leaf Exact Override

构造 Exact-feasible child。

确保 LeafEvaluator：

调用 Exact，而不是 neural Q。

---

# 129. Search Fallback

让 Search 故意返回 NaN。

Router 必须：

fallback Matrix Nash。

不能崩溃或输出随机 action。

---

# 130. Exact Priority

Exact 与 SM-MCTS 同时存在：

Robust final source 必须：

```text
EXACT
```

。

---

# 131. Adaptive Isolation

改变 opponent history：

Adaptive Search 可以变化。

Robust Exact/SM-MCTS：

必须不变化。

---

# 132. LSTM Branching

两条 search simulation 从相同 history 分叉。

修改 branch A history：

branch B 不得被污染。

---

# 133. Mamba Freeze

单局 Adaptive Search：

Mamba memory bytes/tensor 必须保持完全相同。

---

# 134. Safe Exploit

构造：

Adaptive policy 更赚钱但严重 exploitable。

Safe LP 必须降低其使用程度。

---

# 135. Unknown Opponent

Opponent confidence：

\[
0
\]

Safe result 应趋近：

\[
\pi_R
\]

。

---

# 136. Unlimited Confidence

若：

- opponent model oracle；
- adaptive policy 同时满足 robust floor；

Safe LP 应允许完全选择更优 adaptive policy。

---

# 137. Budget Hard Stop

给：

\[
10ms
\]

预算。

Search 必须在允许 tolerance 内结束。

---

# 138. Cache Versioning

同 state：

model hash 不同。

Search cache：

不能直接复用。

Exact cache：

可以复用。

---

# 第十九部分：Integration Tests

# 139. Full Agent N=3

对 N=3：

允许 Exact。

最终 action policy 应来自：

```text
NUMERICAL_EXACT
```

。

---

# 140. Full Agent N=13 Early Game

Exact estimator：

拒绝。

进入：

\[
MatrixNash
\]

或：

\[
SM-MCTS
\]

。

绝不能尝试完整 Exact 导致程序卡死。

---

# 141. Endgame Handoff

N=13 游戏进入残局。

前一手：

SM-MCTS。

剩余复杂度跌入 Exact threshold。

下一手自动：

\[
Search
\rightarrow
Exact
\]

。

不需要用户手动切模式。

---

# 142. Search Exact Leaves

N=13 中盘 root 不可 Exact。

SM-MCTS 深入后：

部分 leaves Exact-feasible。

要求：

```text
exact_leaf_hits > 0
```

。

---

# 143. Teacher Mode

同一个 state：

PLAY：

512 sims。

TEACHER：

GT-CFR 高预算。

确保 Router 按 mode 使用不同策略。

---

# 第二十部分：性能优化要求

# 144. Matrix Solver

必须 batched GPU。

---

# 145. Exact DP

核心 bitmask recursion 允许：

- Python prototype；
- 后续 C++/Rust/Cython 加速。

但 API 不变。

---

# 146. Search Model Evaluation

叶节点必须 batch。

禁止每遇到一个 leaf：

```python
model(single_state)
```

立即同步 GPU。

应建立：

```text
Leaf Evaluation Queue
```

累积：

\[
B
\]

个 leaf 后 batch inference。

---

# 147. Async Search

Search worker：

CPU tree traversal。

GPU：

batch neural evaluation。

建议：

```text
CPU Search Threads
        ↓
Leaf Queue
        ↓
GPU Batch Evaluator
        ↓
Result Queue
```

。

---

# 148. Exact Leaf Workers

Exact leaf solve 可独立 CPU worker pool。

Search 遇到 Exact-feasible leaf：

提交：

```text
ExactTask
```

。

若预算不足：

允许暂时 neural fallback。

---

# 149. Determinism

EVALUATION / regression mode：

使用固定 RNG seed。

PLAY：

正常随机 mixed strategy sampling。

---

# 第二十一部分：日志与科研指标

# 150. 每个工具必须记录

### Matrix Nash

- duality gap；
- entropy；
- Actor-Q JSD。

### Exact

- states；
- LP count；
- cache hit；
- runtime。

### SM-MCTS

- simulations；
- nodes；
- depth；
- root gap；
- strategy stability；
- exact leaf ratio。

### GT-CFR

- iterations；
- expanded nodes；
- cumulative regret；
- frontier count。

### Adaptive

- opponent entropy；
- opponent epistemic uncertainty；
- exploit expected gain。

### Router

- why tool was selected；
- total compute；
- policy change；
- fallback reason。

---

# 151. Value of Search

记录：

\[
\Delta_\pi
=
JSD(
\pi_{base},
\pi_{search}
)
\]

和：

\[
\Delta_V
=
V_{search}-V_{base}
\]

。

以后用于研究：

> 搜索到底在哪些状态真正有价值？

---

# 152. Exact Leaf Ratio

：

\[
R_{exact}
=
\frac{
ExactLeafHits
}{
TotalLeafEvaluations
}
\]

这个指标很重要。

它反映：

> Neural Search 在多大程度上能够把复杂开局自动连接到数学可解残局。

---

# 153. Compute-Strength Curve

Evaluator 必须测：

```text
Network only
SM-MCTS 128
SM-MCTS 512
SM-MCTS 2048
SM-MCTS 8192
GT-CFR low
GT-CFR high
Exact when available
```

。

形成：

\[
\boxed{\text{Strength vs Compute}}
\]

曲线。

这是整个 Tool-Using Agent 的关键科研结果之一。

---

# 第二十二部分：明确禁止 Codex 自作主张

## 禁止 1

禁止普通 sequential AlphaZero MCTS 替代 simultaneous search。

---

## 禁止 2

禁止让第二个玩家在搜索节点看到第一个玩家本轮 action。

---

## 禁止 3

禁止将：

\[
Q_R.mean(-1)
\]

后 argmax 当 Nash。

---

## 禁止 4

禁止用 raw Actor 直接替代 Matrix Nash。

---

## 禁止 5

禁止 Exact Solver 未运行 estimator 直接开算。

---

## 禁止 Exact chance sampling

Exact 必须完整 chance expectation。

---

## 禁止 7

禁止 Search leaf 永远只用 neural value。

可 Exact leaf 必须允许 Exact override。

---

## 禁止 8

禁止 Search 成功返回就无条件覆盖 baseline。

必须 Quality Gate。

---

## 禁止 9

禁止 Adaptive history 进入 Robust Search。

---

## 禁止 10

禁止 Adaptive Search 结果直接作为 final policy。

必须 Safe Exploit。

---

## 禁止 11

禁止 Mamba 在 hypothetical 单局搜索中更新。

---

## 禁止 12

禁止不同搜索分支共享 mutable LSTM state。

---

## 禁止 13

禁止 Search cache 跨模型版本无条件复用。

---

## 禁止 14

禁止 Exact cache 与神经模型绑定。

---

## 禁止 15

禁止只有 MCTS，没有 CFR 搜索接口。

完整系统必须保留：

\[
SM\text{-}MCTS
+
GT\text{-}CFR
\]

两条搜索路径。

---

## 禁止 16

禁止只提供 Exact Nash，不提供 Exact Best Response 接口。

---

## 禁止 17

禁止工具结果不记录 provenance。

---

## 禁止 18

禁止最终训练 TeacherDataset 不知道 target 来自哪一种 solver/search。

---

## 禁止 19

禁止 Learned Router 一开始取代硬规则。

---

## 禁止 20

禁止把 Tool Router 与 neural model forward 写成一个不可拆解的大函数。

---

# 第二十三部分：最终智能体推理体系

最终完整决策过程：

```text
                         PUBLIC STATE
                              │
                              ↓
                    ┌──────────────────┐
                    │   Neural Model   │
                    └──────────────────┘
                       │      │      │
                      Q_R    π_R   uncertainty
                       │
                       ↓
                MATRIX NASH SOLVER
                       │
                       ↓
                Baseline Robust Policy
                       │
                       ↓
               EXACT COMPLEXITY CHECK
                       │
             ┌─────────┴──────────┐
             │                    │
         feasible             infeasible
             │                    │
             ↓                    ↓
       EXACT NASH          SEARCH TRIGGER
             │                    │
             │           ┌────────┴────────┐
             │           │                 │
             │       SM-MCTS           GT-CFR
             │           │                 │
             └───────────┴────────┬────────┘
                                  ↓
                         BEST ROBUST RESULT
                                  │
                      ┌───────────┴────────────┐
                      │                        │
                no opponent model       opponent model
                      │                        │
                      │                        ↓
                      │              LSTM + Mamba belief
                      │                        │
                      │                        ↓
                      │               Adaptive BR Search
                      │                        │
                      │                 Exploit Candidate
                      │                        │
                      └───────────┬────────────┘
                                  ↓
                         SAFE EXPLOIT LP
                                  │
                                  ↓
                         FINAL MIXED POLICY
                                  │
                                  ↓
                            SAMPLE ACTION
```

---

# 结论

这一层的核心不是：

> “给神经网络加一个 MCTS。”

而是构建一个真正意义上的：

\[
\boxed{
\textbf{Tool-Using Game Reasoning Agent}
}
\]

神经模型负责：

\[
\boxed{\text{快速预测}}
\]

Matrix Nash Solver 负责：

\[
\boxed{\text{把 joint-action prediction 转成合理 mixed strategy}}
\]

Exact Solver 负责：

\[
\boxed{\text{可计算时用数学替代猜测}}
\]

SM-MCTS 负责：

\[
\boxed{\text{在线有限预算深思}}
\]

GT-CFR 负责：

\[
\boxed{\text{高质量博弈论搜索与教师生成}}
\]

Adaptive Best-Response Search 负责：

\[
\boxed{\text{针对具体对手推理}}
\]

Safe Exploit Controller 负责：

\[
\boxed{\text{在利用对手和避免被反利用之间做约束优化}}
\]

Tool Router 负责：

\[
\boxed{\text{决定什么时候值得花多少计算}}
\]

于是这个智能体最终拥有三个不同层次的“思考能力”：

\[
\boxed{
\text{Neural Intuition}
}
\]

快速、廉价、泛化。

\[
\boxed{
\text{Search Reasoning}
}
\]

预算越多，可以重新检验自己的直觉。

\[
\boxed{
\text{Mathematical Solving}
}
\]

当问题规模已经足够小时，不再依赖经验判断，而直接求解。

更重要的是，这三者不是孤立的。

训练过程中：

\[
Exact/Search
\rightarrow
Teacher
\rightarrow
Neural Model
\]

部署过程中：

\[
Neural Model
\rightarrow
Search
\rightarrow
Exact Leaves
\]

红队纠错时：

\[
Failure
\rightarrow
Exact/Search
\rightarrow
Correction Target
\]

长期以后：

\[
Tool\ Usage\ Logs
\rightarrow
Value\text{-}of\text{-}Computation
\rightarrow
Better\ Router
\]

最终形成：

\[
\boxed{
\textbf{
预测 → 推理 → 数学验证 → 决策 → 学习 → 再推理
}
}
\]

的闭环，而不是一个单纯依赖神经网络前向传播的 Goofspiel Bot。