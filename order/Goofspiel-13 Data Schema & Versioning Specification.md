# Goofspiel-13 Data Schema & Versioning Specification
## 数据契约、坐标系统、单位、Mask、Tensor Shape、持久化、版本控制与数据血缘规范

---

# 0. 文档地位

本文定义整个 Goofspiel 项目的唯一数据语义。

适用于：

- Environment
- Exact Solver
- Matrix Solver
- Neural Model
- Learning Algorithms
- Pre-training / SFT / Post-training
- Search
- Tool Router
- Opponent Modeling
- League
- Red Team
- Reanalyse
- Logging
- Evaluation
- Checkpoint / Dataset

任何模块不得自己重新解释：

- card rank；
- action index；
- round；
- score；
- reward；
- prize mask；
- self/opponent perspective；
- Q matrix axis；
- normalized value；
- history；
- state identity。

出现冲突时：

\[
\boxed{\texttt{DATA\_SPEC.md}}
\]

是唯一权威。

---

# 1. 最核心原则：Raw Fact 与 Derived Feature 严格分离

数据分成两类：

## A. Authoritative Raw Data

表示真实游戏事实。

例如：

- N；
- 剩余牌 bitmask；
- 当前 prize；
- 实际 action；
- 原始 score；
- RoundEvent。

这些是：

\[
\boxed{\text{Source of Truth}}
\]

---

## B. Derived Data

由 Raw Data 计算得到。

例如：

- normalized rank；
- normalized score；
- one-hot；
- neural tensor；
- LSTM feature；
- Mamba summary；
- two-hot outcome target；
- joint legal mask。

Derived Data：

\[
\boxed{\text{永远不得成为规则意义上的 Source of Truth}}
\]

如果 raw 和 derived 冲突：

\[
\boxed{\text{raw wins}}
\]

然后：

- derived 必须重新计算；
- 产生 ERROR event。

---

# 2. 永久数据优先保存事实，不保存模型解释

例如 history：

必须保存：

```text
RoundEvent
```

而不是：

```text
LSTM hidden state
```

Opponent session：

必须保存真实 games/actions。

而不是只保存：

```text
Mamba embedding
```

Search：

必须保存 root state + algorithm result + provenance。

不能只保存一个：

```text
best_action = 7
```

原则：

\[
\boxed{
\text{原始事实必须可重新生成未来版本的特征}
}
\]

---

# 3. 全局常量

```python
MAX_N = 13

DOMAIN_MIN_RANK = 1
DOMAIN_MAX_RANK = 13

TENSOR_MIN_INDEX = 0
TENSOR_MAX_INDEX = 12

PAD_INDEX = 13
IGNORE_INDEX = -100
```

---

# 4. 最重要的坐标规则：牌面值永远 1-based

游戏领域中的 card rank：

\[
\boxed{1,\dots,N}
\]

例如：

```text
Ace = 1
...
King = 13
```

所有：

- Environment API；
- JSON；
- Parquet；
- GameRecord；
- Human UI；
- Exact state；
-日志；

中的 action/prize rank：

统一：

\[
\boxed{1\text{-based}}
\]

---

# 5. Tensor Index 永远 0-based

进入 Neural / GPU tensor 后：

\[
index=rank-1
\]

所以：

```text
rank 1  → index 0
rank 13 → index 12
```

反向：

\[
rank=index+1
\]

。

---

# 6. 1-based / 0-based 转换只能出现在 Encoding Boundary

必须集中提供：

```python
def rank_to_index(rank: int) -> int:
    assert 1 <= rank <= 13
    return rank - 1


def index_to_rank(index: int) -> int:
    assert 0 <= index < 13
    return index + 1
```

禁止在业务代码里到处出现：

```python
a - 1
i + 1
```

。

Codex 不得自行进行隐式转换。

---

# 7. PAD 与 Rank 不得混淆

Tensor 中：

```text
0..12 = real rank index
13    = PAD
```

因此：

\[
\boxed{PAD\_INDEX=13}
\]

。

禁止使用：

```text
0 = PAD
```

因为 0 已经表示 rank 1 的 tensor index。

---

# 8. IGNORE_INDEX

只有 loss label 可以使用：

```text
-100
```

表示：

> 这个 label 不参与 loss。

它不表示：

- 没有牌；
- PAD；
- current prize absent。

---

# 第一部分：Bitmask Contract

# 9. Bitmask 是核心规则状态的权威牌集合表示

对于 rank：

\[
r\in1..13
\]

其 bit：

\[
\boxed{bit=r-1}
\]

。

因此：

```text
rank 1  → bit 0
rank 2  → bit 1
...
rank 13 → bit 12
```

。

---

# 10. Card Set → Mask

\[
mask=
\sum_{r\in S}
2^{r-1}
\]

例如：

\[
\{1,3,13\}
\]

：

```text
bit 0  = 1
bit 2  = 1
bit 12 = 1
```

。

---

# 11. Mask 类型

因为：

\[
N\le13
\]

磁盘/CPU canonical：

```text
uint16
```

足够。

禁止：

```text
int32 array[13]
```

作为规则核心 Source of Truth。

---

# 12. Mask 中 N 以外的 bit 必须为 0

对于：

\[
N=7
\]

：

bit 7..15：

必须全部：

\[
0
\]

。

validator 必须检查。

---

# 13. GPU Mask

bitmask 进入模型时派生为：

```text
[B, 13] torch.bool
```

例如：

```python
available[b, i] = bool(mask & (1 << i))
```

。

GPU bool mask 是 derived representation。

权威状态仍是 uint16 bitmask。

---

# 第二部分：状态类型

# 14. 不使用一个含义模糊的 `state`

整个系统区分：

\[
\boxed{
DecisionState
}
\]

\[
\boxed{
ChanceState
}
\]

\[
\boxed{
TerminalState
}
\]

三种逻辑 phase。

---

# 15. Phase Enum

```python
class GamePhase(IntEnum):
    CHANCE = 0
    DECISION = 1
    TERMINAL = 2
```

---

# 16. 为什么必须区分 Phase

因为：

Decision：

当前 prize 已公开。

Chance：

下一 prize 尚未公开。

Terminal：

游戏结束。

如果三者混成一个 state：

Codex 极容易错误地：

> 在 Exact Solver 中偷看下一张 prize。

---

# 第三部分：DecisionState

# 17. DecisionState 是 Neural Model 的主要输入状态

定义：

```python
@dataclass(frozen=True)
class DecisionState:
    schema_version: str
    rules_version: str

    n: int

    self_cards_mask: int
    opponent_cards_mask: int

    future_prizes_mask: int

    current_prize: int

    self_score: int
    opponent_score: int
```

---

# 18. `self_cards_mask`

含义：

> 当前决策前，自己仍然没有使用的 bid cards。

包括：

当前这一轮仍可出的所有牌。

---

# 19. `opponent_cards_mask`

完全相同：

> 当前决策前，对手尚未使用的 bid cards。

由于历史 action 已 reveal，这也是 public information。

---

# 20. `future_prizes_mask`

这是最容易写错的字段之一。

严格定义：

> 当前已经公开的 `current_prize` **不在** `future_prizes_mask` 中。

它只表示：

\[
\boxed{\text{当前回合之后尚未 reveal 的 prizes}}
\]

。

因此 DecisionState：

若当前双方还剩：

\[
k
\]

张 bid cards，

则：

\[
popcount(future\_prizes\_mask)=k-1
\]

。

---

# 21. `current_prize`

Domain rank：

\[
1..N
\]

。

DECISION phase：

必须非空。

---

# 22. Score

```text
self_score
opponent_score
```

始终是：

\[
\boxed{\text{原始整数累计得分}}
\]

。

范围：

\[
0..S_N
\]

其中：

\[
S_N=\frac{N(N+1)}2
\]

。

---

# 23. Score 不在 state 中 normalized

禁止 authoritative state 保存：

```text
score = 0.384615
```

。

必须保存：

```text
score = 35
```

。

normalized score 是 derived feature。

---

# 24. Round 不存为权威字段

当前已经完成轮数：

\[
completed\_rounds
=
N-popcount(self\_cards\_mask)
\]

。

因此：

\[
\boxed{\texttt{completed\_rounds}}
\]

是 derived。

当前 human-readable round：

\[
round\_number=completed\_rounds+1
\]

。

---

# 25. 为什么不把 `round` 作为 Source of Truth

因为：

```text
mask says 7 cards remain
round says 5
```

会形成两个冲突事实。

因此：

\[
\boxed{\text{mask 是权威，round 从 mask 计算}}
\]

。

---

# 26. DecisionState 必须满足的不变量

令：

\[
k=popcount(self\_cards\_mask)
\]

则：

\[
popcount(opponent\_cards\_mask)=k
\]

。

同时：

\[
popcount(future\_prizes\_mask)=k-1
\]

。

current prize：

不能存在于：

\[
future\_prizes\_mask
\]

。

并且：

\[
1\le current\_prize\le N
\]

。

---

# 第四部分：ChanceState

# 27. ChanceState 用于 Exact/Search 内部

```python
@dataclass(frozen=True)
class ChanceState:
    schema_version: str
    rules_version: str

    n: int

    self_cards_mask: int
    opponent_cards_mask: int

    remaining_prizes_mask: int

    self_score: int
    opponent_score: int
```

---

# 28. ChanceState 没有 current prize

因为：

\[
\boxed{\text{下一 prize 尚未 reveal}}
\]

。

禁止：

```text
current_prize = 0
```

冒充“没有 prize”。

Python 对象：

直接不存在该字段或使用严格 typed union。

---

# 29. ChanceState 不变量

令：

\[
k=popcount(self\_cards\_mask)
\]

则：

\[
popcount(opponent\_cards\_mask)=k
\]

且：

\[
popcount(remaining\_prizes\_mask)=k
\]

。

---

# 30. Exact DP 的 \(F(A,B,R)\)

直接对应：

```text
A = self_cards_mask
B = opponent_cards_mask
R = remaining_prizes_mask
```

。

这样 Exact Solver 不需要进行模糊字段转换。

---

# 第五部分：TerminalState

# 31. TerminalState

```python
@dataclass(frozen=True)
class TerminalState:
    schema_version: str
    rules_version: str

    n: int

    self_score: int
    opponent_score: int

    discarded_prize_total: int
```

。

必须满足：

\[
self\_score+
opponent\_score+
discarded\_prize\_total
=
S_N
\]

。

---

# 32. Terminal 不保留“假的 current prize”

禁止：

```text
current_prize = -1
```

写入领域层。

只有 Tensor encoding 可以使用 PAD。

---

# 第六部分：Raw Game Record 与 Perspective

# 33. 原始 GameRecord 采用 Seat-Based Neutral Representation

原始比赛记录不使用：

```text
self
opponent
```

。

而使用：

```text
player_0
player_1
```

。

原因：

同一局可以从双方视角训练。

---

# 34. Player Slot

```python
class PlayerSlot(IntEnum):
    P0 = 0
    P1 = 1
```

。

不存在：

```text
BLACK
WHITE
```

。

Goofspiel 不应引入棋类颜色语义。

---

# 35. Raw GameRecord

```python
@dataclass(frozen=True)
class GameRecord:
    schema_version: str
    rules_version: str

    game_id: str
    session_id: str | None

    n: int

    player_0_agent_id: str
    player_1_agent_id: str

    player_0_policy_version: str
    player_1_policy_version: str

    events: tuple["RoundEvent", ...]

    final_score_p0: int
    final_score_p1: int
    discarded_prize_total: int

    rng_seed: int | None

    created_at_ns: int
```

---

# 36. Raw Record 不直接定义赢家 utility

它保存事实：

```text
score P0
score P1
```

。

P0 perspective：

\[
\Delta=S_0-S_1
\]

P1 perspective：

\[
-\Delta
\]

。

---

# 37. Learning Example 才转换为 Ego-Centric

提供：

```python
view_as(game_record, PlayerSlot.P0)
view_as(game_record, PlayerSlot.P1)
```

。

产生：

```text
self
opponent
```

语义。

禁止 dataset 中一部分 row 是 P0-based，一部分是 ego-based，却没有 perspective 字段。

---

# 第七部分：RoundEvent

# 38. History 永远保存事件，不保存 feature

定义：

```python
@dataclass(frozen=True)
class RoundEvent:
    round_index: int

    prize_rank: int

    action_p0: int
    action_p1: int

    winner: int
    # -1 = tie
    # 0  = P0
    # 1  = P1

    score_p0_after: int
    score_p1_after: int
```

---

# 39. `round_index`

Raw Event 中：

\[
\boxed{0\text{-based}}
\]

范围：

\[
0..N-1
\]

。

注意：

牌 rank 是 1-based；

序列 index 是 0-based。

这两个概念不能混。

---

# 40. `prize_rank`

领域牌值：

\[
1..N
\]

。

---

# 41. `action_p0/action_p1`

同样：

\[
1..N
\]

。

不是 tensor index。

---

# 42. Winner

固定：

```text
-1 = tie
 0 = P0
 1 = P1
```

。

不要：

```text
0 = tie
1 = win
2 = loss
```

因为这是 neutral record，不存在 self/opp。

---

# 43. Immediate Raw Reward 不作为必须持久化字段

因为可以由：

```text
prize
action_p0
action_p1
```

精确重建。

如果为了性能缓存：

可以存：

```text
reward_p0_raw
```

但必须标记：

\[
\boxed{\text{derived}}
\]

并验证：

\[
reward_{P0}
=
prize\times sign(action_{P0}-action_{P1})
\]

。

---

# 44. History Feature

LSTM 使用的：

- rank embeddings；
- sign；
- normalized score；
- normalized round；

全部：

\[
\boxed{\text{运行时从 RoundEvent 派生}}
\]

。

不进入 canonical history schema。

---

# 第八部分：不得泄漏未来信息

# 45. Full GameRecord 天然包含未来事件

这是允许的。

但是当创建第 t 轮训练样本时：

模型只能看到：

\[
events[0:t]
\]

和：

\[
current\ prize_t
\]

。

绝不能看到：

\[
events[t:]
\]

。

---

# 46. 必须实现 Prefix Builder

```python
build_decision_example(
    game_record,
    perspective,
    round_index,
)
```

。

内部只允许读取：

```text
events[:round_index]
```

构造 history。

当前 event：

只允许读取：

```text
prize_rank
```

作为 current prize。

不得读取：

```text
action_p0
action_p1
```

直到监督 label 阶段单独加入 target。

---

# 47. Future Leakage Test

修改：

\[
events[t+1:]
\]

中的所有 action/prize。

重新 encode 第 t 轮输入。

要求：

\[
\boxed{\text{Model Input Bitwise Identical}}
\]

。

这是永久测试。

---

# 第九部分：Score / Reward / Value 单位

# 48. Raw Score

永远：

```text
integer prize points
```

。

例如：

```text
self_score = 38
opponent_score = 27
```

。

---

# 49. Raw Immediate Reward

从 self perspective：

\[
r^{raw}
=
p\operatorname{sign}(a-b)
\]

单位：

\[
\boxed{\text{prize points}}
\]

。

范围：

\[
[-N,N]
\]

。

磁盘 dtype 可：

```text
int8
```

。

---

# 50. Normalized Reward

学习系统使用：

\[
r
=
\frac{r^{raw}}{S_N}
\]

dtype：

```text
float32
```

。

它是 derived。

---

# 51. Final Raw Score Difference

\[
D^{raw}
=
Score_{self}-Score_{opp}
\]

范围：

\[
[-S_N,S_N]
\]

。

磁盘：

```text
int16
```

。

---

# 52. Normalized Final Score Difference

\[
D
=
\frac{D^{raw}}{S_N}
\in[-1,1]
\]

。

用于：

- outcome distribution；
- MC；
- metrics。

---

# 53. Robust Q 的单位

所有：

```text
Q_R
V_R
Exact Q
Search Robust Q
```

默认统一为：

\[
\boxed{\text{normalized future incremental score difference}}
\]

。

不是：

- absolute final score；
- WDL probability；
- raw prize point。

---

# 54. “Future Incremental” 的精确定义

给 DecisionState：

历史已经有：

\[
Score_{self},Score_{opp}
\]

。

但：

\[
Q_R(s,a,b)
\]

只预测：

\[
\boxed{\text{从当前开始到 terminal 新产生的 score difference}}
\]

。

历史 score 不加入 Q。

因此 Exact cache 可以忽略历史 score。

---

# 55. Outcome Distribution 的单位不同

\[
Z_R(s)
\]

对应：

\[
\boxed{\text{最终总 score difference}}
\]

normalized 后：

\[
[-1,1]
\]

。

它包括历史已经获得的 score。

因此：

\[
Q_R
\]

与：

\[
Z_R
\]

语义不可混。

---

# 56. 字段名必须体现单位

禁止模糊：

```text
score_diff
value2
reward
```

持久化 schema 中优先：

```text
score_diff_raw
score_diff_norm

reward_raw
reward_norm

future_value_norm
```

。

如果字段没有 suffix：

其单位必须由 schema 唯一定义。

---

# 第十部分：State Hash

# 57. 不使用 Python `hash()`

Python hash：

不保证跨进程/版本稳定。

禁止作为永久 ID。

---

# 58. 两种 State Identity

必须区分：

\[
\boxed{\text{Public State Identity}}
\]

和：

\[
\boxed{\text{Strategic State Identity}}
\]

。

---

# 59. Public State Hash

包含：

- rules version；
- N；
- phase；
- masks；
- current prize；
- raw scores。

用于：

-日志；
- trajectory；
- UI；
- failure；
- state lookup。

---

# 60. Strategic State Hash

Robust Q / Exact future value 不依赖历史 score。

因此 Strategic Key：

对于 Decision：

```text
N
self_cards_mask
opponent_cards_mask
future_prizes_mask
current_prize
```

。

对于 Chance：

```text
N
self_cards_mask
opponent_cards_mask
remaining_prizes_mask
```

。

不含 score。

---

# 61. 为什么必须有两个 Hash

两个状态可能：

牌完全相同，

但历史比分不同。

那么：

\[
Q_R^{future}
\]

相同。

可是：

\[
Z_R^{final}
\]

不同。

因此不能用一个 hash 同时承担两个语义。

---

# 62. Hash 算法

使用 canonical binary serialization 后：

```text
SHA-256
```

。

保存：

```text
64-char lowercase hex
```

。

不要为了省几十字节过早截断。

---

# 63. Canonical Serialization

必须：

- fixed field order；
- little-endian；
- fixed integer width；
- enum integer；
- schema/rules version 纳入 hash context。

禁止：

直接：

```python
sha256(repr(obj))
```

。

---

# 第十一部分：Trajectory Schema

# 64. Trajectory 是一场真实行为过程，不是 target

建议：

```python
@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str

    game_id: str
    session_id: str | None

    n: int

    agent_p0_id: str
    agent_p1_id: str

    policy_version_p0: str
    policy_version_p1: str

    generation_mode: str

    rounds: tuple["TrajectoryRound", ...]

    final_score_p0: int
    final_score_p1: int

    created_at_ns: int

    schema_version: str
    rules_version: str
```

---

# 65. Generation Mode

固定 enum：

```text
RANDOM
HEURISTIC
ROBUST_SELF_PLAY
POPULATION_PLAY
ADAPTIVE_PLAY
SEARCH_PLAY
REDTEAM_PLAY
EVALUATION
```

。

---

# 66. TrajectoryRound

```python
@dataclass(frozen=True)
class TrajectoryRound:
    round_index: int

    public_state_hash: str

    prize_rank: int

    action_p0: int
    action_p1: int

    action_prob_p0: float
    action_prob_p1: float

    behavior_policy_p0: tuple[float, ...]  # length 13
    behavior_policy_p1: tuple[float, ...]  # length 13

    reward_p0_raw: int

    score_p0_after: int
    score_p1_after: int

    decision_source_p0: str
    decision_source_p1: str
```

---

# 67. Behavior Policy Shape

持久化：

固定：

\[
13
\]

长度。

N 以外：

\[
0
\]

。

已经用过的 action：

\[
0
\]

。

合法概率和：

\[
1
\]

。

---

# 68. 为什么保存完整 Behavior Policy

因为未来需要：

- off-policy correction；
- behavior analysis；
- replay；
- policy drift；
- regression。

只保存 chosen action probability 不够做完整科研诊断。

---

# 69. Action Probability

必须等于：

\[
behavior\_policy[action-1]
\]

。

serializer validator 自动检查。

---

# 70. Robust / Adaptive 数据不得只靠推测区分

每个 decision 必须记录：

```text
decision_source
```

例如：

```text
ROBUST_MATRIX_NASH
ROBUST_SM_MCTS
ROBUST_EXACT
ADAPTIVE_SAFE_LP
```

。

整局 generation mode 只能作为粗分类。

---

# 第十二部分：Opponent Session Schema

# 71. Session 是跨局有序结构

```python
@dataclass(frozen=True)
class OpponentSessionRecord:
    session_id: str

    focal_agent_id: str
    opponent_id: str

    game_ids: tuple[str, ...]

    strategy_regimes: tuple["StrategyRegime", ...] | None

    created_at_ns: int

    schema_version: str
```

---

# 72. `game_ids` 顺序就是时间顺序

禁止：

```text
sort lexicographically
```

重新排序。

原顺序必须固定。

---

# 73. Opponent Session 不存 Mamba Hidden 作为 Source of Truth

禁止 canonical dataset 只保存：

```text
mamba_state
```

。

Mamba state：

可以作为 ephemeral inference cache。

不属于历史事实。

---

# 74. Strategy Regime

仅当已知：

```python
@dataclass(frozen=True)
class StrategyRegime:
    regime_id: str

    start_game_index: int
    start_round_index: int | None

    end_game_index: int | None
    end_round_index: int | None

    source: str
```

。

`source`：

```text
SYNTHETIC_KNOWN
HUMAN_LABEL
ALGORITHM_KNOWN
UNKNOWN
```

。

---

# 75. 不能把模型自己预测的 style 当 ground truth

如果是：

```text
model inferred style
```

必须存到 Prediction/Analysis record。

不能写回：

```text
strategy_regime_id
```

ground truth。

---

# 第十三部分：Opponent History

# 76. Opponent History 本质是 Perspective View

给 focal player：

过去 RoundEvent 转成：

```text
prize
self_action
opponent_action
score
```

。

它是：

\[
GameRecord+Perspective
\]

的派生视图。

---

# 77. History 不单独复制成另一套 Source of Truth

否则：

原 GameRecord 修改/迁移以后，

History 可能不一致。

因此 canonical：

\[
\boxed{\text{GameRecord}}
\]

History：

按需构造。

---

# 第十四部分：ExactTarget Schema

# 78. Exact Target 是高价值不可变数据

```python
@dataclass(frozen=True)
class ExactTarget:
    target_id: str

    strategic_state_hash: str

    n: int

    objective_version: str

    q_matrix_norm: tuple[float, ...]       # length 169
    policy_self: tuple[float, ...]         # length 13
    policy_opponent: tuple[float, ...]     # length 13

    value_norm: float

    legal_self_mask: int
    legal_opponent_mask: int

    exactness: str

    primal_dual_gap: float

    solver_name: str
    solver_version: str

    created_at_ns: int

    schema_version: str
    rules_version: str
```

---

# 79. Q Matrix Axis

永久冻结：

\[
\boxed{
Q[i,j]
=
Q(
self\ action=i+1,
opponent\ action=j+1
)
}
\]

。

第一维：

SELF。

第二维：

OPPONENT。

---

# 80. Flatten Order

持久化 length 169：

统一：

```text
row-major / C order
```

即：

\[
flat[i\times13+j]=Q[i,j]
\]

。

禁止不同模块自己 flatten。

---

# 81. Illegal Q Cells

磁盘：

固定为：

\[
0.0
\]

。

是否有效只由：

```text
legal_self_mask
legal_opponent_mask
```

决定。

禁止用：

- NaN；
- -inf；
- 99999；

表达非法 cell。

---

# 82. Exact Target dtype

落盘：

\[
\boxed{float64}
\]

优先。

进入训练：

按需要转换：

\[
FP32
\]

。

---

# 83. Exact policy

长度：

\[
13
\]

。

非法/N 外 rank：

\[
0
\]

。

合法 probability：

和为 1。

---

# 第十五部分：TeacherTarget Schema

# 84. Teacher 与 Exact 必须分开

Teacher 可能来自：

- Exact；
- GT-CFR；
- SM-MCTS；
- EMA；
- Ensemble；
- Historical。

所以：

```python
@dataclass(frozen=True)
class TeacherTarget:
    target_id: str

    strategic_state_hash: str

    source: str

    q_matrix_norm: tuple[float, ...] | None
    policy_self: tuple[float, ...] | None
    policy_opponent: tuple[float, ...] | None
    value_norm: float | None

    quality_score: float
    duality_gap: float | None
    disagreement: float | None

    legal_self_mask: int
    legal_opponent_mask: int

    teacher_model_version: str | None
    search_config_hash: str | None
    solver_version: str | None

    created_at_ns: int

    schema_version: str
    rules_version: str
    objective_version: str
```

---

# 85. Teacher Source Enum

固定：

```text
EXACT
GT_CFR
SM_MCTS
EMA
ENSEMBLE
HISTORICAL_ROBUST
HISTORICAL_AGGRESSIVE
```

。

---

# 86. Target 字段可以缺失，但语义必须明确

例如 CFR：

可能只有：

```text
policy
value
```

。

所以 optional 字段允许 null。

禁止拿：

```text
None
```

静默填 0 然后让 learner 误以为是 supervision。

必须同时有：

```text
has_q_target
has_policy_target
has_value_target
```

或 schema null validity。

---

# 第十六部分：SearchResult Schema

# 87. SearchResult 是推理产物，不是真值

```python
@dataclass(frozen=True)
class SearchResult:
    search_id: str

    public_state_hash: str
    strategic_state_hash: str

    search_type: str
    reasoning_mode: str

    q_matrix_norm: tuple[float, ...] | None
    policy_self: tuple[float, ...]
    policy_opponent: tuple[float, ...] | None
    value_norm: float | None

    quality_score: float
    duality_gap: float | None
    strategy_instability_jsd: float | None

    simulations: int
    expanded_nodes: int
    exact_leaf_hits: int
    neural_leaf_hits: int

    runtime_ms: float

    budget_config_hash: str
    search_config_hash: str

    model_version: str
    opponent_model_version: str | None

    rng_seed: int | None

    valid: bool
    failure_code: str | None

    created_at_ns: int

    schema_version: str
```

---

# 88. Search Type

```text
SM_MCTS
GT_CFR
ADAPTIVE_BR_SEARCH
```

。

---

# 89. Reasoning Mode

```text
ROBUST
ADAPTIVE
```

。

这是必须字段。

防止 Adaptive Search Q 被错误拿去监督 Robust Q。

---

# 90. Search Result 永不就地升级成 Teacher

必须显式转换：

```python
TeacherTarget.from_search_result(...)
```

并通过 quality gate。

这样 provenance 不会丢失。

---

# 第十七部分：Reanalysis Schema

# 91. Reanalysis 不覆盖旧数据

原则：

\[
\boxed{\text{Append-only}}
\]

。

永远保留旧 target。

---

# 92. ReanalysisRecord

```python
@dataclass(frozen=True)
class ReanalysisRecord:
    reanalysis_id: str

    state_hash: str

    old_target_id: str | None
    new_target_id: str

    reason: str

    old_model_version: str | None
    new_model_version: str | None

    old_value_norm: float | None
    new_value_norm: float | None

    value_delta: float | None
    policy_jsd: float | None

    created_at_ns: int

    schema_version: str
```

---

# 93. Reanalysis Reason

```text
MODEL_UPGRADE
SEARCH_UPGRADE
FAILURE_DISCOVERED
HIGH_UNCERTAINTY
HIGH_VISITATION
PERIODIC_REFRESH
TEACHER_DISAGREEMENT
```

。

---

# 94. 最新 Target 怎么找到

不要更新原 row。

建立：

```text
state_hash
→ latest accepted target_id
```

索引。

历史链保留。

---

# 第十八部分：Failure Schema

# 95. Failure 永久化

```python
@dataclass(frozen=True)
class FailureRecord:
    failure_id: str

    failure_type: str
    severity: str
    status: str

    game_id: str | None
    session_id: str | None

    public_state_hash: str
    strategic_state_hash: str

    round_index: int

    main_model_version: str
    attacker_agent_id: str | None
    attacker_policy_version: str | None

    predicted_value_norm: float | None
    teacher_value_norm: float | None
    realized_return_norm: float | None

    policy_disagreement_jsd: float | None

    discovery_source: str

    correction_target_id: str | None
    regression_fixture_id: str | None

    created_at_ns: int

    schema_version: str
```

---

# 96. Failure Status

固定：

```text
ACTIVE
CORRECTION_PENDING
FIXED
REGRESSED
OBSOLETE_BY_RULE_CHANGE
```

。

禁止删除。

---

# 97. Failure Type

至少：

```text
UNEXPECTED_LOSS
Q_MISMATCH
HIGH_REGRET_ACTION
SEARCH_DISAGREEMENT
OPPONENT_MODEL_OVERCONFIDENCE
STRATEGY_SWITCH_FAILURE
SAFE_EXPLOIT_FAILURE
NUMERICAL_FAILURE
```

。

---

# 第十九部分：League Matchup Schema

# 98. League Matchup 是实验事实

```python
@dataclass(frozen=True)
class LeagueMatchupRecord:
    matchup_id: str

    agent_a_id: str
    agent_a_role: str
    agent_a_policy_version: str

    agent_b_id: str
    agent_b_role: str
    agent_b_policy_version: str

    n: int

    games_played: int

    score_diff_sum_raw_from_a: int

    wins_a: int
    draws: int
    wins_b: int

    mean_score_diff_norm_from_a: float

    seed_start: int | None
    seed_count: int | None

    evaluation_config_hash: str

    created_at_ns: int

    schema_version: str
```

---

# 99. League Payoff Perspective

统一：

\[
\boxed{\text{always from Agent A perspective}}
\]

。

所以：

\[
G_{AB}
=
-\;G_{BA}
\]

统计误差范围内成立。

---

# 100. Matchup 不只记录 WDL

必须保存：

\[
mean\ normalized\ score\ difference
\]

因为主 utility 是 score difference。

---

# 第二十部分：Model-Facing Tensor Contract

# 101. Persistent Schema 与 Tensor Schema 必须分开

磁盘：

领域语义。

GPU：

固定 shape、高效计算。

由唯一：

```python
TensorEncoder
```

转换。

---

# 102. `PublicTensorBatch`

对于模型 Decision input：

```python
@dataclass
class PublicTensorBatch:
    n: Tensor

    valid_rank_mask: Tensor

    self_available: Tensor
    opponent_available: Tensor
    future_prize_available: Tensor

    current_prize_index: Tensor

    self_score_raw: Tensor
    opponent_score_raw: Tensor

    completed_rounds: Tensor

    legal_self_mask: Tensor
    legal_opponent_mask: Tensor
    joint_legal_mask: Tensor
```

---

# 103. Shape / dtype

```text
n
[B] int64

valid_rank_mask
[B,13] bool

self_available
[B,13] bool

opponent_available
[B,13] bool

future_prize_available
[B,13] bool

current_prize_index
[B] int64

self_score_raw
[B] float32

opponent_score_raw
[B] float32

completed_rounds
[B] int64

legal_self_mask
[B,13] bool

legal_opponent_mask
[B,13] bool

joint_legal_mask
[B,13,13] bool
```

---

# 104. 为什么 GPU score 使用 float32

虽然 raw score 语义是整数，

进入模型以后直接转换：

```text
float32
```

便于归一化。

这不改变权威磁盘表示仍为整数。

---

# 105. `valid_rank_mask`

对于 N：

\[
valid[i]=(i<N)
\]

。

例如 N=7：

```text
TTTTTTTFFFFFF
```

。

---

# 106. `current_prize_index`

DecisionState 永远有效：

\[
0..12
\]

。

ChanceState 不得传入普通 Decision model。

因此正常 model batch：

不需要 PAD current prize。

---

# 107. 如果特殊模型 API 允许 Terminal/Chance padding

则：

\[
current\_prize\_index=PAD\_INDEX=13
\]

并必须有：

```text
current_prize_valid=False
```

。

禁止用：

\[
-1
\]

直接索引 embedding。

---

# 108. Legal Mask

Decision state：

\[
legal\_self\_mask=self\_available
\]

。

Opponent 同理。

---

# 109. Joint Legal Mask

\[
M_{ij}
=
M^{self}_i
\land
M^{opp}_j
\]

shape：

\[
[B,13,13]
\]

。

---

# 第二十一部分：Temporal Tensor Contract

# 110. Current-game History 最大长度

N=13，

当前决策前最多已有：

\[
12
\]

个 completed rounds。

因此：

```text
MAX_ROUND_HISTORY = 12
```

。

---

# 111. RoundHistoryBatch

```text
prize_index
[B,12] int64

self_action_index
[B,12] int64

opponent_action_index
[B,12] int64

outcome_sign
[B,12] float32

score_diff_norm_after
[B,12] float32

round_index_norm
[B,12] float32

sequence_mask
[B,12] bool
```

。

---

# 112. PAD

padding position：

action/prize index：

\[
13
\]

。

其他 numeric feature：

\[
0
\]

但必须由：

```text
sequence_mask=False
```

屏蔽。

---

# 113. Outcome Sign

self perspective：

\[
+1
\]

self wins round。

\[
0
\]

tie。

\[
-1
\]

loss。

---

# 114. Score Difference

\[
\frac{
Score_{self}-Score_{opp}
}{
S_N
}
\]

。

---

# 115. `round_index_norm`

对第 r 个已完成 round event：

\[
r/N
\]

其中 r：

领域内部 sequence index：

\[
0..N-1
\]

。

---

# 第二十二部分：Mamba Session Data

# 116. Canonical Session 不保存 Game Summary Tensor

Game summary：

是 feature。

必须绑定：

```text
feature_spec_version
```

。

---

# 117. Cached GameSummary

如果为了训练效率落盘，可以保存：

```python
@dataclass
class DerivedGameSummary:
    game_id: str

    perspective: int

    features: tuple[float, ...]

    feature_spec_version: str

    source_model_version: str | None
```

。

必须能够从 raw GameRecord 重建。

---

# 118. LSTM Final Hidden 不应长期缓存为通用 Summary

因为 LSTM 参数变了：

hidden 也变。

因此：

```text
LSTM hidden
```

若落盘，必须包含：

```text
model_version
```

且只作为 cache。

模型升级：

cache 自动 invalid。

---

# 第二十三部分：Training Batch Contract

# 119. 不设计一个万能 `Batch`

必须按语义分。

至少：

```text
RobustQBatch
OutcomeTrajectoryBatch
OpponentBatch
AdaptiveBatch
SFTBatch
```

。

---

# 120. `RobustQBatch`

包含：

```text
public_tensor_batch

target_q_norm
[B,13,13] float32

joint_valid_mask
[B,13,13] bool

target_source
[B] enum

teacher_quality
[B] float32
```

。

---

# 121. `OutcomeTrajectoryBatch`

包含 sequence：

```text
rewards_norm
[T,B] float32

actions_self
[T,B] int64

actions_opp
[T,B] int64

behavior_prob_self
[T,B] float32

behavior_prob_opp
[T,B] float32

done
[T,B] bool

final_score_diff_norm
[B] float32
```

。

---

# 122. `OpponentBatch`

包含：

- current public state；
- current-game history；
- cross-game session；
- actual opponent action；
- optional regime/switch label。

Actual action Tensor：

\[
0..12
\]

。

---

# 123. `AdaptiveBatch`

必须有：

```text
opponent_context_id
opponent_model_version
```

以及：

```text
mode = ADAPTIVE
```

。

禁止没有 provenance 的 adaptive sample。

---

# 第二十四部分：Distribution Target

# 124. 201 bins 不永久保存 two-hot target

Canonical trajectory 保存：

\[
final\_score\_diff\_raw
\]

。

Two-hot：

训练时根据：

```text
distribution_spec_version
```

生成。

---

# 125. 为什么

未来可能：

```text
201 bins → 401 bins
```

。

如果只保存旧 two-hot：

无法无损重建。

保存 raw score：

永远可以重新投影。

---

# 126. Distribution Specification

```text
distribution_spec_version = "score_diff_201_v1"
range = [-1,1]
bins = 201
```

。

模型 checkpoint 必须保存这一版本。

---

# 第二十五部分：ID 与 Version

# 127. 所有长期对象都有独立 ID

例如：

```text
run_id
dataset_id
game_id
trajectory_id
session_id
target_id
search_id
failure_id
reanalysis_id
matchup_id
checkpoint_id
```

。

禁止拿文件名冒充永久 ID。

---

# 128. Version 不只有一个

必须区分：

```text
schema_version
rules_version
feature_spec_version
objective_version
distribution_spec_version
model_io_version
solver_version
search_version
```

。

---

# 129. `schema_version`

描述：

> 字段结构与字段语义。

采用 Semantic Versioning：

```text
MAJOR.MINOR.PATCH
```

。

---

# 130. Schema MAJOR

当发生：

- 字段含义改变；
- rank coordinate 改变；
- mask semantics 改变；
- Q axis 改变；

必须 MAJOR++。

例如：

```text
1.x → 2.0
```

。

---

# 131. Schema MINOR

只允许：

> 添加 backward-compatible optional field。

例如：

增加：

```text
search_policy_stability
```

。

---

# 132. Schema PATCH

只允许：

- 文档；
- validator bugfix；
- 不改变落盘语义。

---

# 133. `rules_version`

规则改变独立升级。

例如：

```text
goofspiel_discard_tie_v1
```

。

如果以后改 tie rule：

必须新 rules version。

旧数据不得混训而不标记。

---

# 134. `objective_version`

当前：

```text
future_normalized_score_difference_v1
```

。

如果以后改成：

WDL utility，

必须新 objective_version。

---

# 135. `feature_spec_version`

例如：

```text
public_features_v1
opponent_round_features_v1
game_summary_v1
```

。

因为 derived feature 可以变化，而 raw schema 不变。

---

# 第二十六部分：Dataset Manifest

# 136. 每个 Dataset 必须有 Manifest

```python
@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str

    dataset_type: str

    schema_version: str
    rules_version: str

    feature_spec_versions: dict
    objective_version: str | None

    created_at_ns: int

    git_commit: str
    config_hash: str

    parent_dataset_ids: tuple[str, ...]

    shard_count: int
    row_count: int

    checksums: dict[str, str]

    generation_seed: int | None

    description: str
```

---

# 137. Dataset 不允许“悄悄更新”

一旦发布：

\[
\boxed{\text{immutable}}
\]

。

需要修改：

生成：

```text
new dataset_id
```

。

---

# 138. Parent Dataset

Reanalysis / filtering / augmentation：

必须记录 parent。

形成：

\[
\boxed{\text{Data Lineage DAG}}
\]

。

---

# 第二十七部分：Parquet Contract

# 139. 大规模 Dataset 使用 Parquet

禁止长期：

```text
pickle
```

保存训练语料。

---

# 140. 基础 dtype

建议：

```text
n                       uint8
masks                   uint16
rank/action              uint8
score                    uint16
score_diff_raw           int16
reward_raw               int8
probability              float32
exact q/value             float64
timestamp_ns              int64
ID/version                string
boolean                   bool
```

。

---

# 141. Fixed Vector

Policy：

固定：

\[
13
\]

元素。

Q：

固定：

\[
169
\]

元素。

推荐 Arrow：

```text
FixedSizeList
```

而不是任意长度 list。

这样 schema 能直接发现 shape 错误。

---

# 142. Matrix Layout

Q flattened：

\[
169
\]

row-major。

必须在 metadata 中写：

```text
q_layout = "self_row_opponent_column_c_order"
```

。

---

# 143. Partition

例如：

```text
dataset/
  schema=1.0.0/
  n=13/
  source=robust_selfplay/
```

。

但 partition 字段仍由 manifest 约束。

---

# 第二十八部分：Runtime Validation

# 144. Hot Path 不使用重型 validation

Environment/Search 每一步不能大量 Pydantic 校验。

建议：

### Domain boundary

严格 validator。

### Hot path

dataclass / tensor assertions 可配置。

### Debug

full validation。

### Release

重要 invariants + sampled validation。

---

# 145. 所有持久化数据写入前必须 full validate

包括：

- rank；
- masks；
- probabilities；
- score conservation；
- schema version。

---

# 第二十九部分：Migration

# 146. 禁止 in-place 修改旧 dataset

升级：

```text
v1
↓
migration
↓
v2 new dataset
```

。

旧数据保留。

---

# 147. Migration 代码

目录：

```text
data/migrations/
  v1_0_to_v2_0.py
```

。

每次 migration：

输出：

```text
source dataset id
target dataset id
row count
checksum
migration code version
```

。

---

# 148. Migration 必须可重复

对同一 immutable source：

相同 migration/config：

结果 checksum 应相同。

---

# 149. MAJOR Version 不允许自动隐式转换

Reader 遇到未知 major：

必须：

```text
raise UnsupportedSchemaVersion
```

。

禁止：

> “字段差不多，我猜一下。”

---

# 第三十部分：Checkpoint Data Contract

# 150. Checkpoint 必须保存其数据语义依赖

metadata：

```text
schema_version
rules_version
objective_version
feature_spec_versions
distribution_spec_version
model_io_version
```

。

---

# 151. 加载模型时检查 Compatibility

如果 checkpoint：

```text
rules_version != runtime_rules_version
```

：

默认拒绝。

---

# 152. Feature Version 不匹配

若存在正式 migration：

转换。

否则拒绝。

禁止静默使用。

---

# 第三十一部分：Provenance

# 153. 每个学习 target 必须能回答

> 谁生成的？

至少：

```text
source_type
source_model_version
source_solver_version
source_search_config
created_at
state_hash
```

。

---

# 154. Pseudo Label 特别要求

必须保存：

```text
teacher ensemble members
teacher disagreement
accept threshold
quality
```

。

后续才能判断：

> 伪标签是不是造成错误强化。

---

# 155. Failure Correction

必须能建立：

```text
Failure
→ Relabel Target
→ Correction Dataset
→ Training Run
→ Candidate Checkpoint
→ Regression Result
```

完整链。

---

# 第三十二部分：Authoritative vs Derived 总表

| 数据 | Authoritative? | 单位/表示 |
|---|---|---|
| N | ✅ | integer 1..13 |
| card rank | ✅ | domain 1..N |
| tensor card index | ❌ | rank-1 |
| card bitmask | ✅ | uint16 |
| dense bool mask | ❌ | bool[13] |
| current prize | ✅ | rank 1..N |
| future prize mask | ✅ | uint16 |
| round event | ✅ | raw event |
| completed_rounds | ❌ | derived from mask |
| raw score | ✅ | integer prize points |
| normalized score | ❌ | score / \(S_N\) |
| raw reward | 可重建事实 | signed prize points |
| normalized reward | ❌ | raw / \(S_N\) |
| LSTM input features | ❌ | derived |
| LSTM hidden | ❌ | model-dependent |
| Mamba hidden | ❌ | model-dependent |
| game summary features | ❌ | versioned derived |
| exact Q | ✅ target | normalized future diff |
| teacher Q | ✅ teacher result | normalized future diff |
| behavior action | ✅ | rank |
| behavior policy | ✅ trajectory metadata | prob[13] |
| two-hot target | ❌ | derived from raw final diff |
| style embedding | ❌ | model-dependent |
| failure record | ✅ observation | versioned |
| reanalysis | ✅ lineage event | append-only |

---

# 第三十三部分：最关键的 Invariants

任何写入/读取都必须能够检查：

### Rank

\[
1\le rank\le N
\]

。

### Masks

没有 N 外 bit。

### Bid Counts

\[
popcount(self)=popcount(opponent)
\]

。

### Decision Prize Count

\[
popcount(future\ prizes)
=
popcount(self)-1
\]

。

### Chance Prize Count

\[
popcount(remaining\ prizes)
=
popcount(self)
\]

。

### Policy

\[
p_i\ge0
\]

非法：

\[
p_i=0
\]

合法和：

\[
1
\]

。

### Score

\[
0\le score\le S_N
\]

。

### Terminal Conservation

\[
score_0+score_1+discarded=S_N
\]

。

### Q Axis

row=self，

column=opponent。

---

# 第三十四部分：Player Swap Contract

# 156. State Swap

```text
self_cards ↔ opponent_cards
self_score ↔ opponent_score
```

prize：

不变。

---

# 157. Q Swap

\[
Q^{swap}
=
-Q^T
\]

。

---

# 158. Policy Swap

row policy：

变成 opponent-view row policy。

不能直接简单复制 self policy。

---

# 159. Raw GameRecord 不需要 swap

只在：

```text
view_as(P0/P1)
```

阶段进行 perspective transform。

---

# 第三十五部分：数据防泄漏

# 160. Robust 数据禁止 opponent private context

Robust training example：

不能包含：

- opponent ID embedding；
- opponent session hidden；
- Mamba state；
- future action。

---

# 161. Adaptive 数据必须显式标记

否则 learner 无法判断：

历史数据能否进入 adaptive branch。

---

# 162. Search Robust Result

不得保存：

```text
opponent_context_hash
```

作为计算依赖。

如果存在：

视为 leakage bug。

---

# 第三十六部分：Codex 禁止事项

## 禁止 1

禁止 domain action 一会儿 0-based，一会儿 1-based。

---

## 禁止 2

禁止 bit 1 表示 rank 1。

正确是：

\[
bit0=rank1
\]

。

---

## 禁止 3

禁止 `current_prize` 同时存在于 `future_prizes_mask`。

---

## 禁止 4

禁止把 round 当独立权威状态和 masks 冲突。

---

## 禁止 5

禁止 authoritative score 保存 normalized float。

---

## 禁止 6

禁止把 Q 值解释成 absolute final score。

---

## 禁止 7

禁止把 \(Z_R\) 与 \(Q_R\) 的 value 语义混合。

---

## 禁止 8

禁止将 history 存成只有 LSTM/Mamba hidden。

---

## 禁止 9

禁止持久化数据只存派生 feature 而没有 raw source。

---

## 禁止 10

禁止 Q matrix axis 在不同模块交换。

始终：

```text
row = self
column = opponent
```

。

---

## 禁止 11

禁止非法 Q cell 使用 NaN 作为持久化约定。

使用：

```text
0 + mask
```

。

---

## 禁止 12

禁止 SearchResult 自动成为 TeacherTarget。

必须经过显式质量过滤与转换。

---

## 禁止 13

禁止 Reanalysis 覆盖旧 target。

必须 append-only。

---

## 禁止 14

禁止 Failure 被删除。

---

## 禁止 15

禁止 schema major mismatch 自动猜测解析。

---

## 禁止 16

禁止 Python `hash()` 作为永久 state ID。

---

## 禁止 17

禁止 checkpoint 不保存 rules/schema/feature version。

---

## 禁止 18

禁止用未来 trajectory suffix 构造当前输入。

---

## 禁止 19

禁止把模型预测的 opponent style 写成 ground-truth regime。

---

## 禁止 20

禁止 dataset 原地修改。

---

# 第三十七部分：必须实现的数据测试

至少必须有：

```text
test_rank_index_roundtrip
test_mask_rank_roundtrip
test_mask_has_no_bits_above_n

test_decision_state_prize_count
test_chance_state_prize_count

test_current_prize_excluded_from_future_mask

test_score_raw_normalization

test_q_flatten_row_major
test_q_axis_self_opponent

test_player_swap_q_neg_transpose

test_game_record_perspective_p0
test_game_record_perspective_p1

test_history_future_leakage

test_serialization_roundtrip

test_schema_major_rejection

test_dataset_manifest_checksum

test_reanalysis_append_only

test_behavior_probability_matches_policy

test_illegal_behavior_probability_zero

test_tensor_padding_invariance
```

。

---

# 第三十八部分：推荐代码结构

```text
goofspiel/data/
│
├── schema/
│   ├── versions.py
│   ├── enums.py
│   ├── state.py
│   ├── game.py
│   ├── trajectory.py
│   ├── session.py
│   ├── target.py
│   ├── search.py
│   ├── failure.py
│   ├── reanalysis.py
│   ├── league.py
│   └── manifest.py
│
├── encoding/
│   ├── rank.py
│   ├── bitmask.py
│   ├── public_tensor.py
│   ├── history_tensor.py
│   ├── perspective.py
│   └── distribution.py
│
├── validation/
│   ├── state.py
│   ├── trajectory.py
│   ├── probability.py
│   └── dataset.py
│
├── storage/
│   ├── parquet.py
│   ├── dataset.py
│   └── checksum.py
│
├── migrations/
│
└── lineage/
    ├── provenance.py
    └── target_index.py
```

---

# 第三十九部分：开发者必须遵循的转换路径

最终只允许下面这条数据路径：

```text
Raw Game Facts
      │
      ↓
Canonical Domain Schema
      │
      ├────────────→ Persistent Dataset
      │
      ↓
Perspective Transformation
      │
      ↓
Versioned Feature Encoder
      │
      ↓
Fixed Tensor Contract
      │
      ↓
Model / Solver / Learner
```

禁止：

```text
Environment Python object
→ 随便拼几个 float
→ Model
```

。

---

# 第四十部分：一个完整例子

假设：

\[
N=13
\]

当前已经完成 4 轮。

自己剩：

```text
{1,2,4,5,7,8,9,10,12}
```

对手剩：

```text
{1,3,4,5,6,8,9,11,13}
```

当前公开 prize：

\[
10
\]

未来 prizes：

```text
{1,2,3,5,6,7,8,11}
```

那么：

```python
DecisionState(
    n=13,

    self_cards_mask=...,
    opponent_cards_mask=...,

    future_prizes_mask=...,

    current_prize=10,

    self_score=17,
    opponent_score=11,
)
```

。

这里：

\[
popcount(self)=9
\]

所以：

\[
completed\_rounds=13-9=4
\]

。

而：

\[
popcount(future\ prizes)=8
\]

满足：

\[
9-1=8
\]

。

---

# 164. 进入 Tensor

rank 10：

\[
index=9
\]

所以：

```text
current_prize_index = 9
```

。

自己 rank 1：

对应 tensor index 0。

自己 rank 13 不存在：

index 12：

```text
False
```

。

---

# 165. Score Feature

raw：

```text
17
11
```

。

总 prize：

\[
S_{13}=91
\]

派生：

\[
self\_score\_norm=17/91
\]

\[
opp\_score\_norm=11/91
\]

\[
score\_diff\_norm=6/91
\]

。

这些 float：

不是 canonical state。

---

# 166. 如果模型预测

\[
Q_R[6,8]=0.073
\]

注意 index：

6 对应 rank：

\[
7
\]

8 对应 rank：

\[
9
\]

所以语义：

> 自己当前出 7，对手当前出 9，之后双方 minimax continuation 时，未来 normalized incremental score difference 期望为 0.073。

绝不能解释成：

> 出第 6 张牌。

这就是 Schema Contract 存在的意义。

---

# 第四十一部分：版本升级例子

假设未来我们决定在 RoundEvent 新增：

```text
wall_time_ms
```

只是 optional metadata。

则：

```text
schema 1.0.0 → 1.1.0
```

。

---

如果未来把：

```text
future_prizes_mask
```

改为：

> 包含 current prize

这是字段语义改变。

必须：

```text
1.x → 2.0.0
```

而不能：

```text
1.1
```

。

---

如果只是修文档 typo：

```text
1.0.0 → 1.0.1
```

。

---

# 第四十二部分：最终原则

整个 Data Contract 最重要的原则可以压缩成八条：

\[
\boxed{
\textbf{领域中的牌永远 1-based，Tensor index 永远 0-based。}
}
\]

\[
\boxed{
\textbf{Bit 0 永远对应 rank 1。}
}
\]

\[
\boxed{
\textbf{DecisionState 的 current prize 永远不属于 future prize mask。}
}
\]

\[
\boxed{
\textbf{Score 永远以原始整数点数作为权威数据，归一化只属于派生特征。}
}
\]

\[
\boxed{
\textbf{History 保存真实事件，不保存模型 hidden state 作为事实。}
}
\]

\[
\boxed{
\textbf{所有 Q matrix 永远 row=self、column=opponent。}
}
\]

\[
\boxed{
\textbf{持久化 raw facts，feature 可以重建；旧 target 不覆盖，reanalysis 永远 append-only。}
}
\]

以及最后：

\[
\boxed{
\textbf{
任何数据如果不知道它的 schema version、rules version、单位、perspective、source 和 provenance，
就禁止进入训练。
}
}
\]

这就是整个项目的数据安全边界。