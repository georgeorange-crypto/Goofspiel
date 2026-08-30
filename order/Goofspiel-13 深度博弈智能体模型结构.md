# Goofspiel-13 深度博弈智能体
# 最终模型结构详细设计书

---

# 1. 文档范围

本文只规定 **Neural Model Architecture**。

本文回答的问题是：

> 给定当前 Goofspiel 状态、当前对手历史以及跨局交互历史，神经模型接收哪些信息，通过哪些模块计算，最终输出哪些张量？

本文**不讨论**：

- PPO
- CFR / Deep CFR
- NFSP
- PSRO
- R-NaD
- loss 权重
- replay buffer 采样
- self-play 调度
- MCTS 具体算法
- Exact Solver 的训练使用方式
- curriculum
- optimizer

这些属于 Learning / Planning System，不属于 Model Architecture。

最终系统严格区分：

\[
\boxed{
\text{Model}
\neq
\text{Learning Algorithm}
\neq
\text{Planner}
}
\]

本模型只负责从信息中构造：

\[
\boxed{
\text{representation}
+
\text{prediction}
}
\]

然后将结果交给外部算法使用。

---

# 2. 模型最终需要表达哪些信息

标准 Goofspiel 的核心决策并不是：

\[
\text{state}\rightarrow\text{one action}
\]

而是：

> 我出某张牌、对手同时出某张牌，这个 joint action 对整个后续游戏意味着什么？

因此模型最核心的对象确定为：

\[
\boxed{
Q(s,a,b)
}
\]

其中：

- \(s\)：当前公开游戏状态
- \(a\)：自己当前出的牌
- \(b\)：对手当前出的牌

对于 N 张牌：

\[
Q(s)\in\mathbb R^{N\times N}
\]

标准游戏：

\[
N=13
\]

所以标准状态下：

\[
Q(s)\in\mathbb R^{13\times13}
\]

而不是传统 RL 中：

\[
Q(s)\in\mathbb R^{13}
\]

这是模型结构最重要的设计决定。

---

# 3. 模型同时处理两种完全不同的信息

必须严格区分：

## 3.1 Public Game State

包括：

- 自己剩余哪些牌
- 对手剩余哪些牌
- 哪些奖品已经出现
- 当前奖品
- 当前比分
- 当前轮次

这些信息对双方完全公开。

它描述的是：

\[
\boxed{\text{Game}}
\]

---

## 3.2 Opponent Behaviour History

包括：

- 这个对手以前面对什么 prize
- 他出了什么牌
- 当前局此前的出牌规律
- 前几局的整体行为
- 长期 interaction pattern

它描述的是：

\[
\boxed{\text{Opponent}}
\]

两者必须在架构层分开。

因此最终模型采用：

\[
\boxed{\text{Public Game Backbone}}
\]

和：

\[
\boxed{\text{Opponent Memory Backbone}}
\]

两个主系统。

---

# 4. 为什么必须分开

我们希望模型能够同时回答两个不同问题：

### 问题 A

> 如果双方理性博弈，这个局面本身怎么样？

这是：

\[
Q_{\text{robust}}(s,a,b)
\]

这个值不应该因为：

> “我现在对手叫 Alice”

而变化。

---

### 问题 B

> 如果我现在面对的是这个具体对手，未来可能发生什么？

这是：

\[
Q_{\text{adapt}}(s,h,a,b)
\]

其中：

\[
h
\]

表示 opponent history。

所以最终模型明确同时输出：

\[
\boxed{Q_{\text{robust}}}
\]

和：

\[
\boxed{Q_{\text{adapt}}}
\]

但两个分支共享大量 game representation。

这样能够避免一个严重问题：

如果 opponent history 直接进入唯一的 Q 网络，那么我们将无法区分：

> 游戏本身的价值

和

> 对某一个对手的针对性估值。

---

# 5. 模型总体结构

最终模型结构为：

```text
                    PUBLIC GAME STATE
                           │
               ┌───────────┴───────────┐
               │                       │
       Card Transformer          Relational GNN
               │                       │
               └───────────┬───────────┘
                           │
                 Per-Card Fusion
                           │
                           ↓
                 Joint Pair Builder
                           │
                           ↓
                    Matrix CNN
                           │
                  Joint Pair Features
                           │
          ┌────────────────┼────────────────┐
          │                │                │
      Robust Q       Robust Policy     Robust Value
       Matrix             Prior          Distribution
          │
          │
          │
          │                   OPPONENT HISTORY
          │                         │
          │                ┌────────┴─────────┐
          │                │                  │
          │              LSTM               Mamba
          │          Intra-game       Inter-game Memory
          │                │                  │
          │                └────────┬─────────┘
          │                         │
          │                  Opponent Fusion
          │                         │
          │                Opponent Prediction
          │                         │
          │                         ↓
          └────────────→ Adaptive Pair Modulation
                                    │
                                    ↓
                           Adaptive Q Matrix
                                    │
                           Adaptive Policy Prior
```

它包含五个核心神经结构：

\[
\boxed{\text{Transformer}}
\]

\[
\boxed{\text{GNN}}
\]

\[
\boxed{\text{CNN}}
\]

\[
\boxed{\text{LSTM}}
\]

\[
\boxed{\text{Mamba}}
\]

每一个模块承担不同的 inductive bias。

不存在“为了堆模型而使用”。

---

# 6. Variable-N 是架构一级要求

虽然主游戏是：

\[
\boxed{N=13}
\]

但是网络不允许写死为：

```python
Linear(..., 13)
```

然后无法处理其他 N。

最终模型必须满足：

\[
f_\theta(S_N)
\]

对于：

\[
N=3,4,\dots,13
\]

使用完全相同的一套参数。

因此：

- rank 表示不能依赖固定 13 个 embedding class；
- Q head 不能是固定 169 输出；
- policy head 不能依赖固定 13 输出节点；
- GNN 必须允许不同节点数量；
- Transformer 必须支持 variable sequence length；
- CNN 必须支持 variable \(N\times N\) feature map；
- mask 必须贯穿模型。

这样以后能够真正研究：

\[
N_{\text{train}}\neq N_{\text{test}}
\]

而不是每个 N 重新训练一个完全不同的模型。

---

# 7. Rank 的连续表示

牌值不能单纯使用：

```python
nn.Embedding(13, d)
```

因为这样：

> 7 和 8

在网络初始表示中与：

> 1 和 13

没有任何本质区别。

但 Goofspiel 中：

\[
|7-8|=1
\]

显然具有重要意义。

因此每一个 rank：

\[
i
\]

首先归一化：

\[
r_i=\frac{i}{N}
\]

然后构造 continuous rank basis：

\[
\rho(r_i)=
[
r_i,\,
r_i^2,\,
\sin(\pi r_i),\,
\cos(\pi r_i),\,
\sin(2\pi r_i),\,
\cos(2\pi r_i)
]
\]

维度：

\[
6
\]

然后：

```text
6
↓
Linear(6, 64)
↓
SiLU
↓
Linear(64, 64)
```

获得：

\[
e^{rank}_i\in\mathbb R^{64}
\]

这个 RankEncoder 在：

- Card Transformer
- GNN
- history
- joint matrix

之间共享。

这样保证所有模块对于：

> “牌值大小”

使用同一个连续语义空间。

---

# 8. Public Card Token

对于每一个 rank：

\[
i\in\{1,\dots,N\}
\]

构造一个 public card entity。

基础状态 feature：

\[
x_i=
[
\rho(r_i),
m_i,
o_i,
u_i,
c_i
]
\]

其中：

### 自己是否还拥有

\[
m_i=
1[i\in A]
\]

### 对手是否还拥有

\[
o_i=
1[i\in B]
\]

### 奖品是否尚未出现

\[
u_i=
1[i\in R\cup\{p_t\}]
\]

### 是否当前 prize

\[
c_i=
1[i=p_t]
\]

所以基础 card feature 维度：

\[
64+4=68
\]

---

# 9. Card Token Projection

通过：

```text
Linear(68, 192)
LayerNorm
SiLU
Linear(192, 192)
```

得到：

\[
h_i^{(0)}\in\mathbb R^{192}
\]

所有牌使用同一个 projection。

没有：

```text
Card1Encoder
Card2Encoder
...
Card13Encoder
```

这种破坏泛化能力的结构。

---

# 10. Global State Features

单独构造：

\[
g
\]

包含：

\[
g=
[
N/N_{\max},
t/N,
k/N,
p_t/N,
S_{self}/S_N,
S_{opp}/S_N,
D/S_N,
P_{remaining}/S_N
]
\]

其中：

\[
S_N=\frac{N(N+1)}2
\]

\[
D=S_{self}-S_{opp}
\]

\[
P_{remaining}=
\sum_{p\in R\cup\{p_t\}}p
\]

因此：

\[
g\in\mathbb R^8
\]

通过：

```text
Linear(8, 128)
SiLU
Linear(128, 192)
LayerNorm
```

得到：

\[
h_g\in\mathbb R^{192}
\]

---

# 11. STATE Token

Card Transformer 不仅输入 N 个 card token。

再增加一个：

\[
[STATE]
\]

token。

初始化：

\[
h_{STATE}^{(0)}=h_g
\]

所以 Transformer 输入：

\[
H^{(0)}
=
[
h_{STATE},
h_1,
\dots,
h_N
]
\]

shape：

\[
[B,N+1,192]
\]

最终：

\[
H_{STATE}^{T}
\]

作为 Transformer 的 global representation。

---

# 12. Card Transformer

配置固定为：

```text
d_model       = 192
n_heads       = 6
n_layers      = 4
ffn_dim       = 768
norm          = Pre-LayerNorm
activation    = GELU
dropout       = 0.05
```

每层：

\[
H'
=
H+
MHA(LN(H))
\]

然后：

\[
H^{next}
=
H'
+
FFN(LN(H'))
\]

---

# 13. Transformer 的职责

Transformer 负责捕捉：

\[
\boxed{\text{global resource configuration}}
\]

例如：

> 我还剩 3、5、K。

本身意义有限。

如果：

> 对手还剩 2、Q、K。

以及：

> 未来还剩 Prize 4、J、K。

那么 K 的战略意义才真正确定。

Transformer 的 self-attention 可以直接建立：

\[
Card_i
\leftrightarrow
Card_j
\]

的全局依赖。

---

# 14. Transformer 输出

得到：

\[
H^T_{card}
\in
\mathbb R^{B\times N\times192}
\]

以及：

\[
h^T_{global}
\in
\mathbb R^{B\times192}
\]

---

# 15. Relational GNN 为什么仍然需要

Transformer 理论上能够学习任意 pair relation。

但是：

> “能够学习”

和：

> “架构明确告诉它应该重视什么”

不是一回事。

Goofspiel 存在非常明确的结构：

\[
SelfCard_i
\]

\[
OpponentCard_j
\]

\[
Prize_k
\]

这三类 entity 之间存在不同含义的关系。

因此另外建立 heterogeneous relational graph。

---

# 16. GNN 节点

构造最多：

\[
3N
\]

个节点。

三类：

### Self nodes

\[
S_1,\dots,S_N
\]

### Opponent nodes

\[
O_1,\dots,O_N
\]

### Prize nodes

\[
P_1,\dots,P_N
\]

每个 node 都拥有：

- rank representation
- role type
- available flag
- current prize flag
- global state conditioning

---

# 17. Role Embedding

定义：

\[
e_{self}
\]

\[
e_{opp}
\]

\[
e_{prize}
\]

每个：

\[
32
\]

维。

节点 feature：

\[
[
e^{rank}_i,
e^{role},
available,
current,
h_g
]
\]

先经过：

```text
NodeProjector
→ 128 dims
```

得到：

\[
h_v^{G,0}\in\mathbb R^{128}
\]

---

# 18. GNN Edge Types

不是所有 edge 意义相同。

定义至少四类 relation。

---

## 18.1 SELF-OPPONENT

\[
S_i\leftrightarrow O_j
\]

表示：

> 如果双方分别使用这两张牌，它们之间是什么关系？

edge feature：

\[
e_{ij}^{SO}
=
[
\Delta,
|\Delta|,
sign(\Delta),
r_i,
r_j
]
\]

其中：

\[
\Delta=\frac{i-j}{N}
\]

---

## 18.2 SELF-PRIZE

\[
S_i\leftrightarrow P_k
\]

表示：

> 使用自己牌 i 去争奖品 k 的资源关系。

edge feature：

\[
[
(i-k)/N,\,
|i-k|/N,\,
i/N,\,
k/N
]
\]

---

## 18.3 OPPONENT-PRIZE

\[
O_j\leftrightarrow P_k
\]

同理。

---

## 18.4 RANK-NEIGHBOUR

同一 role 中：

\[
i\leftrightarrow i\pm1
\]

用来显式建立：

\[
\text{ordinal locality}
\]

使网络知道：

> 8 与 9

是邻居。

---

# 19. GNN 不使用 sparse Python graph

由于：

\[
N=13
\]

最多只有：

\[
39
\]

个 node。

没有必要大量使用动态 graph object。

工程实现采用：

\[
\boxed{\text{Dense Relational Attention}}
\]

所有 pair relation tensor 化。

---

# 20. Relation-aware Message Passing

每层：

\[
m_{ij}
=
\phi(
h_i,
h_j,
e_{ij},
type_{ij}
)
\]

attention：

\[
\alpha_{ij}
=
softmax_j(
q_i^Tk_{ij}/\sqrt d
)
\]

更新：

\[
h_i'
=
LN
\left(
h_i+
\sum_j\alpha_{ij}m_{ij}
\right)
\]

然后 FFN residual。

---

# 21. GNN 配置

```text
hidden_dim       = 128
layers           = 3
heads            = 4
edge_embed_dim   = 64
activation       = SiLU
normalization    = LayerNorm
```

输出：

\[
H_S^G
\in
\mathbb R^{B\times N\times128}
\]

\[
H_O^G
\in
\mathbb R^{B\times N\times128}
\]

\[
H_P^G
\in
\mathbb R^{B\times N\times128}
\]

---

# 22. GNN Global Pooling

不能直接 mean 所有节点。

采用 masked attention pooling：

\[
\alpha_i
=
softmax(
w^Th_i
)
\]

然后：

\[
h_G^{global}
=
\sum_i\alpha_i h_i
\]

得到：

\[
h_G^{global}\in\mathbb R^{128}
\]

---

# 23. Transformer 与 GNN 的融合

两者不是简单：

```python
torch.cat(...)
```

然后全部丢给一个大 MLP。

对于每一个自己的合法 card：

\[
i
\]

首先：

\[
\tilde h_i^T
=
W_T H_i^T
\in\mathbb R^{192}
\]

\[
\tilde h_i^G
=
W_G H_{S,i}^G
\in\mathbb R^{192}
\]

计算 feature-wise gate：

\[
g_i
=
\sigma
(
MLP[
\tilde h_i^T;
\tilde h_i^G;
h_g
]
)
\]

其中：

\[
g_i\in\mathbb R^{192}
\]

然后：

\[
h_i^{self}
=
g_i\odot\tilde h_i^T
+
(1-g_i)\odot\tilde h_i^G
\]

---

# 24. Opponent Action Embedding

类似地：

\[
h_j^{opp}
\]

由：

\[
H_j^T
\]

和：

\[
H_{O,j}^G
\]

融合。

因此：

\[
H^{self}
\in
\mathbb R^{B\times N\times192}
\]

\[
H^{opp}
\in
\mathbb R^{B\times N\times192}
\]

---

# 25. Global Fusion

Transformer global：

\[
h_T^{global}\in\mathbb R^{192}
\]

GNN global：

\[
h_G^{global}\in\mathbb R^{128}
\]

global state：

\[
h_g\in\mathbb R^{192}
\]

拼接：

\[
[
h_T^{global};
h_G^{global};
h_g
]
\]

总维度：

\[
512
\]

通过：

```text
Linear(512, 256)
SiLU
Linear(256, 192)
LayerNorm
```

得到：

\[
h^{public}
\in\mathbb R^{192}
\]

它表示完整公共局面。

---

# 26. Joint Action Pair Builder

这是模型进入 Goofspiel 核心结构的地方。

对于所有：

\[
(i,j)
\]

即：

> 我出 i，对手出 j

构造 joint pair representation。

---

# 27. Pair Base Features

对于每一个：

\[
(i,j)
\]

拼接：

\[
[
h_i^{self};
h_j^{opp};
h^{public};
r_i;
r_j;
(r_i-r_j);
|r_i-r_j|;
sign(i-j);
p_t/N;
r_{immediate}
]
\]

其中当前即时收益归一化：

\[
r_{immediate}
=
\frac{
p_t\cdot sign(i-j)
}{
S_N
}
\]

总输入约：

\[
192+192+192+7
\approx583
\]

维。

---

# 28. Pair Projection

每个 pair 共享：

```text
Linear(~583, 256)
SiLU
Linear(256, 96)
LayerNorm
```

得到：

\[
M^{(0)}
\in
\mathbb R^{B\times N\times N\times96}
\]

转为 CNN 格式：

\[
[B,96,N,N]
\]

这就是：

\[
\boxed{\text{Strategic Interaction Map}}
\]

---

# 29. Matrix CNN 的真正角色

CNN 不负责“看牌”。

CNN 负责：

\[
\boxed{
\text{看 joint-action matrix 上的局部战略几何结构}
}
\]

在一个 \(N\times N\) matrix 中：

- 主对角线对应相同 bid；
- 对角线一边是 self win；
- 另一边是 opponent win；
- 相邻 cell 对应小幅改变自己/对手 bid；
- 高牌区域和低牌区域存在不同资源意义。

这种局部连续结构非常适合二维卷积。

---

# 30. Matrix CNN 结构

输入：

\[
[B,96,N,N]
\]

首先：

```text
Conv2d(96, 96, kernel=3, padding=1)
SiLU
```

然后 4 个 Residual Matrix Blocks。

每个 block：

```text
LayerNorm2D
Conv2d(96,96,3,padding=1)
SiLU
Conv2d(96,96,3,padding=1)
Residual Add
```

最后：

```text
Conv2d(96,128,1)
```

得到：

\[
M^{public}
\in
\mathbb R^{B\times128\times N\times N}
\]

---

# 31. CNN 不允许污染 padding

Variable-N batch 中，例如：

一个样本：

\[
N=7
\]

另一个：

\[
N=13
\]

统一 pad 到：

\[
13\times13
\]

定义：

\[
mask_{pair}
=
mask_{self}\otimes mask_{opp}
\]

每个 residual block 后：

\[
M
\leftarrow
M\odot mask_{pair}
\]

确保 padding area 不通过卷积逐层产生虚假 feature。

---

# 32. Robust Q Head

从：

\[
M^{public}_{ij}
\]

得到：

\[
Q_{\text{robust}}(s,i,j)
\]

使用共享 head：

```text
128
↓
Linear(128,64)
↓
SiLU
↓
Linear(64,1)
```

对每一个 pair 独立共享。

因此不是：

```python
Linear(..., 169)
```

而是：

\[
f_Q(m_{ij})
\]

所以支持任意 N。

输出：

\[
\boxed{
Q_R
\in
\mathbb R^{B\times N\times N}
}
\]

---

# 33. Q 不把非法动作设成 -∞

网络输出本身仍保持 finite。

另外返回：

\[
joint\_mask
\]

planner / solver 根据 mask 忽略非法 cell。

原因是如果直接向 Q tensor 写：

\[
-\infty
\]

容易导致：

- mean
- variance
- CNN
- debug
- visualization

出现问题。

---

# 34. Robust Policy Prior Head

模型同时输出一个快速 prior：

\[
\pi_R^{prior}(a|s)
\]

它不是最终 Nash 策略。

它只是一个直接神经近似。

对于每个：

\[
h_i^{self}
\]

再加入 Matrix CNN 的 row pooling：

\[
r_i
=
MaskedAttentionPool_j(M_{ij}^{public})
\]

得到：

\[
r_i\in\mathbb R^{128}
\]

然后：

\[
[
h_i^{self};
r_i;
h^{public}
]
\]

经过：

```text
Linear(512,192)
SiLU
Linear(192,1)
```

产生：

\[
l_i^R
\]

应用 legal mask：

\[
\pi_R^{prior}=Softmax(l^R)
\]

输出：

\[
[B,N]
\]

---

# 35. 为什么保留 Policy Prior

即使最终 planner 可以从 Q matrix 计算策略，direct prior 仍然有意义：

- 极低延迟直接决策；
- 为搜索提供 proposal prior；
- 作为 neural amortized solver；
- 可以与 Q-derived policy 做一致性分析。

---

# 36. Public Distributional Value

单一 scalar：

\[
V(s)
\]

信息不足。

因此 public backbone 输出：

\[
Z_R(s)
\]

表示：

\[
\frac{Score_{self}-Score_{opp}}{S_N}
\]

的分布。

归一化以后：

\[
z\in[-1,1]
\]

这样 variable-N 使用同一个 value head。

---

# 37. Distribution Support

固定：

\[
K=201
\]

个 support：

\[
z_k
=
-1+\frac{2k}{200}
\]

其中：

\[
k=0,\dots,200
\]

global matrix pooling：

\[
h_M
=
AttentionPool(M^{public})
\]

然后：

\[
[
h^{public};
h_M
]
\]

经过：

```text
Linear(...,256)
SiLU
Linear(256,201)
```

输出：

\[
Z_R\in\mathbb R^{B\times201}
\]

---

# 38. 从分布可以派生

Expected normalized score：

\[
V=
\sum_kP(z_k)z_k
\]

以及：

\[
P(win)
\]

\[
P(draw)
\]

\[
P(loss)
\]

所以不再单独堆三个冗余 value heads。

---

# 39. Opponent Memory 总体设计

现在进入第二条主干：

\[
\boxed{\text{Opponent Memory Backbone}}
\]

这里明确保留：

\[
\boxed{\text{LSTM + Mamba}}
\]

并且承担不同时间尺度。

---

# 40. 为什么 LSTM 和 Mamba 不应该做完全相同的事情

如果只是：

```text
history
→ LSTM
→ embedding

history
→ Mamba
→ embedding

concat
```

实际上两个模块会高度竞争相同信号。

最终很容易出现：

> 一个模块承担全部任务，另一个被旁路。

因此我们给二者明确分工：

\[
\boxed{
LSTM
=
\text{intra-game short-term opponent dynamics}
}
\]

\[
\boxed{
Mamba
=
\text{inter-game long-term opponent memory}
}
\]

---

# 41. LSTM：局内记忆

一局最多 13 轮。

当前决策第 t 轮，只允许看到：

\[
1,\dots,t-1
\]

轮。

每一个历史 round 构造：

\[
x_\tau^{game}
\]

包含：

- prize rank
- self action
- opponent action
- relative action difference
- round outcome
- score difference after round
- round index

---

# 42. History Rank Encoding

prize/action rank 不直接使用 one-hot。

使用与主模型共享的：

\[
RankEncoder
\]

因此：

\[
e(p_\tau)
\]

\[
e(a_\tau)
\]

\[
e(b_\tau)
\]

各：

\[
64
\]

维。

---

# 43. LSTM Token

拼接：

\[
[
e(p_\tau);
e(a_\tau);
e(b_\tau);
sign(a-b);
d_\tau/S_N;
\tau/N
]
\]

约：

\[
195
\]

维。

通过：

```text
Linear(~195,128)
LayerNorm
SiLU
```

得到：

\[
z_\tau^{game}\in\mathbb R^{128}
\]

---

# 44. LSTM 配置

最终：

```text
input_size     = 128
hidden_size    = 192
num_layers     = 2
dropout        = 0.05
bidirectional  = False
```

必须：

\[
bidirectional=False
\]

因为在线游戏不能看未来。

输出当前：

\[
h_t^{LSTM}\in\mathbb R^{192}
\]

以及：

\[
c_t^{LSTM}
\]

---

# 45. LSTM 支持在线状态更新

终端/网页真人游戏时，不需要每轮把所有历史重新跑一次。

维护：

```python
OpponentLSTMState(
    h,
    c,
)
```

每完成一轮：

```python
h, c = lstm.step(round_token, h, c)
```

所以单轮增加成本固定。

---

# 46. Mamba：跨局长期记忆

Mamba 不重复处理这一局 13 个 action。

它处理：

\[
\boxed{\text{Game-level opponent history}}
\]

也就是：

> 和同一个对手过去打过的多局游戏。

---

# 47. 每局生成 Game Summary Token

一局结束时，将该局：

\[
h_{final}^{LSTM}
\]

与：

- 最终分差
- 胜负
- 对手平均 bid
- prize/bid correlation
- high-card usage rate
- game length
- N

组合。

注意：

这些额外统计全部是确定性 public statistics。

不是另一个模型。

---

# 48. Game Summary

构造：

\[
g_m
\]

通过：

```text
concat
↓
Linear(...,192)
SiLU
Linear(192,192)
LayerNorm
```

得到：

\[
g_m\in\mathbb R^{192}
\]

每一局对应一个 token。

---

# 49. Mamba 输入序列

对于同一个 opponent：

\[
G=
[
g_1,
g_2,
\dots,
g_M
]
\]

保留最近：

\[
M_{\max}=128
\]

局。

因此 Mamba 处理的是：

\[
[B,M,192]
\]

而不是无限长原始动作序列。

---

# 50. Mamba 配置

最终配置：

```text
d_model     = 192
n_layers    = 4
d_state     = 32
d_conv      = 4
expand      = 2
norm        = RMSNorm
residual    = True
```

输出最新 long-term representation：

\[
h^{Mamba}\in\mathbb R^{192}
\]

---

# 51. 为什么 Mamba 放在 game-level

这种设计有三个优势。

### 第一

LSTM 与 Mamba 不重复劳动。

### 第二

长时间和某个人玩：

\[
100
\]

局时，Mamba 能积累长期行为规律。

### 第三

单局结束才更新长期 memory，语义清楚：

\[
\text{short-term dynamics}
\rightarrow
\text{game summary}
\rightarrow
\text{long-term memory}
\]

形成真正层次化 temporal architecture。

---

# 52. 新对手如何处理

没有历史时：

\[
h^{Mamba}
\]

使用 learned：

\[
[UNKNOWN\_OPPONENT]
\]

embedding：

\[
e_{unknown}\in\mathbb R^{192}
\]

当前局尚无历史时：

\[
h^{LSTM}
\]

使用：

\[
0
\]

或 learned initial hidden。

这样模型可以从：

> 完全不了解

自然过渡到：

> 当前局了解

再过渡到：

> 长期了解。

---

# 53. LSTM 与 Mamba Fusion

不要简单 concat。

分别计算：

\[
u_L=W_Lh^{LSTM}
\]

\[
u_M=W_Mh^{Mamba}
\]

均：

\[
192
\]

再加入：

\[
h^{public}
\]

计算 feature-wise gate。

---

# 54. Feature-wise Hierarchical Gate

输入：

\[
[
u_L;
u_M;
h^{public}
]
\]

通过：

```text
Linear(576,256)
SiLU
Linear(256,384)
```

reshape：

\[
[B,2,192]
\]

对 source dimension 做 softmax：

\[
[\alpha_L,\alpha_M]
\]

每个都是：

\[
[B,192]
\]

最终：

\[
h^{opp}
=
\alpha_L\odot u_L
+
\alpha_M\odot u_M
\]

然后：

\[
LayerNorm
\]

得到：

\[
\boxed{
h^{opp}\in\mathbb R^{192}
}
\]

---

# 55. 为什么 Gate 是 feature-wise

不使用：

\[
\alpha\in\mathbb R
\]

一个 scalar 决定：

> LSTM 70%，Mamba 30%。

因为长期和短期信号可能在不同语义维度发挥作用。

所以：

\[
\alpha_L,\alpha_M\in\mathbb R^{192}
\]

允许：

> 某些 feature 依赖短期变化，

而另外一些：

> 依赖长期习惯。

---

# 56. Opponent Model 输出三个 Head

为了确保 LSTM 和 Mamba 各自的信息能够被观察，Opponent Model 不只输出最终融合结果。

输出：

\[
q_L(b|s,h_L)
\]

\[
q_M(b|s,h_M)
\]

\[
q_F(b|s,h_L,h_M)
\]

分别表示：

### LSTM short-term prediction

### Mamba long-term prediction

### fused prediction

---

# 57. Action-conditioned Opponent Prediction

不能只：

```text
h_opp → Linear → 13 logits
```

因为 opponent action probability 明显与每张可用牌具体 representation 有关。

对于 opponent card \(j\)：

\[
y_j^L
=
[
h_j^{opp};
h^{LSTM};
h^{public}
]
\]

通过共享 scorer：

\[
l_j^L=f_L(y_j^L)
\]

Mamba 同理。

Fusion：

\[
y_j^F
=
[
h_j^{opp};
h^{opp};
h^{public}
]
\]

输出：

\[
l_j^F
\]

应用 opponent legal mask 后：

\[
q_L,q_M,q_F
\in
\mathbb R^{B\times N}
\]

---

# 58. Opponent Prediction 不改变 Robust Backbone

这是严格的信息隔离。

以下 outputs：

\[
Q_R
\]

\[
\pi_R^{prior}
\]

\[
Z_R
\]

只能读取：

\[
\text{Public State}
\]

绝不允许读取：

\[
h^{LSTM}
\]

或：

\[
h^{Mamba}
\]

这样才能保证：

\[
\boxed{
\text{Robust representation remains opponent-independent}
}
\]

---

# 59. Adaptive Pair Modulation

Opponent history 只进入：

\[
\boxed{\text{Adaptive Branch}}
\]

我们已有：

\[
M^{public}_{ij}
\in\mathbb R^{128}
\]

使用：

\[
h^{opp}
\]

进行 FiLM modulation。

---

# 60. FiLM

由：

\[
h^{opp}
\]

生成：

\[
\gamma,\beta
\in
\mathbb R^{128}
\]

然后：

\[
M^{adapt}_{ij}
=
(1+\gamma)\odot M^{public}_{ij}
+
\beta
\]

由于：

\[
\gamma,\beta
\]

对整张 pair matrix 共享，因此 opponent representation 改变的是：

> 如何解释整个战略矩阵，

而不是为每个 cell 硬编码独立参数。

---

# 61. 更细粒度的 Cross Conditioning

仅 global FiLM 还不够。

进一步计算：

\[
c_{ij}
=
MLP(
[
M^{public}_{ij};
h^{opp};
h_i^{self};
h_j^{opp}
]
)
\]

得到：

\[
c_{ij}\in\mathbb R^{128}
\]

然后：

\[
\tilde M^{adapt}_{ij}
=
M^{adapt}_{ij}
+
c_{ij}
\]

---

# 62. Adaptive Matrix Refinement

经过两个 lightweight residual matrix blocks：

```text
128 channels
ResBlock
ResBlock
```

产生：

\[
M^{adaptive}
\]

注意 adaptive branch 比 public Matrix CNN 更浅。

原因是绝大部分 game structure 已由 public backbone 建模。

Opponent branch 应该学习：

\[
\boxed{\text{residual adaptation}}
\]

而不是重新学一次整个游戏。

---

# 63. Adaptive Q

输出一个：

\[
\Delta Q_{opp}
\]

：

\[
\Delta Q_{ij}
=
f_{\Delta Q}
(
M^{adaptive}_{ij}
)
\]

然后：

\[
\boxed{
Q_A
=
Q_R+\Delta Q_{opp}
}
\]

模型同时返回：

\[
Q_R
\]

和：

\[
Q_A
\]

绝不把二者混成一个 Q。

---

# 64. Q_A 的语义

\[
Q_R(s,a,b)
\]

表示：

> opponent-independent strategic value representation。

而：

\[
Q_A(s,h,a,b)
\]

表示：

> 在当前 opponent behavioural context 下，该 joint action 的 opponent-conditioned continuation estimate。

后续最终采用哪个、如何组合，是决策算法的问题。

不在 Model 里自动做。

---

# 65. Adaptive Policy Prior

同理建立：

\[
\pi_A^{prior}
\]

使用：

\[
h_i^{self}
\]

adaptive matrix row feature：

\[
r_i^A
\]

以及：

\[
h^{opp}
\]

生成：

\[
l_i^A
\]

所以模型输出：

\[
\boxed{
\pi_R^{prior}
}
\]

和：

\[
\boxed{
\pi_A^{prior}
}
\]

分别代表：

- robust direct policy prior
- opponent-conditioned direct prior

---

# 66. Adaptive Distributional Value

为了判断：

> 针对当前 opponent 时风险分布如何变化，

adaptive branch 同样输出：

\[
Z_A
\in
\mathbb R^{B\times201}
\]

与 robust：

\[
Z_R
\]

分开。

这样后续能够比较：

\[
Z_R
\]

和：

\[
Z_A
\]

差异。

---

# 67. Uncertainty Architecture

未来搜索预算、Exact Solver fallback 等机制需要知道：

> 模型有多确定。

因此最终架构不采用一个：

```text
Linear → uncertainty scalar
```

的“自我报告”。

采用：

\[
\boxed{\text{shared backbone + multi-head ensemble}}
\]

---

# 68. Robust Q Ensemble

最终 Q output head 使用：

\[
K=4
\]

个独立 head：

\[
Q_R^{(1)},\dots,Q_R^{(4)}
\]

它们共享：

- Transformer
- GNN
- Matrix CNN

只复制最后约：

\[
128\rightarrow64\rightarrow1
\]

的 lightweight Q head。

因此参数增加很小。

---

# 69. Adaptive Ensemble

\[
\Delta Q
\]

同样设置：

\[
K=4
\]

heads。

Opponent fused prediction 也可以保留：

\[
K=4
\]

small scorers。

---

# 70. 模型输出原始 ensemble，而不是自己压成 uncertainty

ModelOutput 返回：

```python
q_robust_heads
q_adaptive_heads
opponent_logits_heads
```

上层可自行计算：

\[
variance
\]

\[
entropy
\]

\[
disagreement
\]

这样 Model 不绑定某一种 uncertainty 定义。

---

# 71. Final ModelOutput

最终：

```python
@dataclass
class GoofspielModelOutput:
    # robust game representation
    q_robust: Tensor
    q_robust_heads: Tensor

    robust_policy_logits: Tensor
    robust_score_logits: Tensor

    # opponent memory
    opponent_short_logits: Tensor
    opponent_long_logits: Tensor
    opponent_fused_logits: Tensor
    opponent_fused_heads: Tensor

    lstm_state: Tensor
    mamba_state: Tensor
    opponent_embedding: Tensor

    # adaptive representation
    q_adaptive: Tensor
    q_adaptive_heads: Tensor

    adaptive_policy_logits: Tensor
    adaptive_score_logits: Tensor

    # masks
    self_action_mask: Tensor
    opponent_action_mask: Tensor
    joint_action_mask: Tensor

    # optional intermediate representations
    public_embedding: Tensor
    self_action_embeddings: Tensor
    opponent_action_embeddings: Tensor
```

---

# 72. 标准 N=13 时 shape

假设：

\[
B=256
\]

则：

### Card features

\[
[256,13,68]
\]

### Transformer output

\[
[256,13,192]
\]

### GNN nodes

\[
[256,39,128]
\]

### Self action representation

\[
[256,13,192]
\]

### Opponent action representation

\[
[256,13,192]
\]

### Pair map

\[
[256,128,13,13]
\]

### Robust Q

\[
[256,13,13]
\]

### Robust Q ensemble

\[
[256,4,13,13]
\]

### Policy logits

\[
[256,13]
\]

### Value distribution

\[
[256,201]
\]

### LSTM state

\[
[2,256,192]
\]

### Mamba representation

\[
[256,192]
\]

### Opponent distribution

\[
[256,13]
\]

---

# 73. Variable-N Mask

统一定义：

\[
rank\_mask
\in
\{0,1\}^{B\times N_{max}}
\]

例如：

N=7：

```text
1 1 1 1 1 1 1 0 0 0 0 0 0
```

---

# 74. Self Action Mask

不仅判断：

\[
i\le N
\]

还判断：

\[
i\in A
\]

得到：

\[
M_S
\in
\{0,1\}^{B\times N_{max}}
\]

---

# 75. Opponent Mask

\[
M_O
\]

同理。

---

# 76. Joint Mask

\[
M_J
=
M_S[:,:,None]
\land
M_O[:,None,:]
\]

shape：

\[
[B,N,N]
\]

---

# 77. Attention Mask

Transformer：

padding rank 不能作为：

- query
- key
- value

GNN：

padding node 不参与 message passing。

CNN：

padding cell 每层重新置零。

Policy：

非法 action：

\[
logit=-10^9
\]

然后 softmax。

---

# 78. Role Symmetry

Self 和 Opponent 的 card encoder 必须尽量共享。

例如：

RankEncoder：

完全共享。

Transformer：

不区分两套 card token，它看到的是：

\[
m_i,o_i
\]

两个状态位。

GNN：

Self/Opp node 使用同一个 base node projector，再加 role embedding。

Q head：

所有：

\[
(i,j)
\]

使用同一个 scorer。

这样减少：

> “Player 0 天生和 Player 1 不一样”

这种错误 inductive bias。

---

# 79. 模型不输入固定 player ID

禁止输入：

```text
I am player 0
I am player 1
```

所有 observation 都转换为：

```text
self
opponent
```

视角。

所以同一个 model 可以同时控制两边。

---

# 80. Score 也采用 self-relative

输入：

\[
Score_{self}
\]

\[
Score_{opp}
\]

而不是：

```text
score_player_0
score_player_1
```

保证角色交换一致。

---

# 81. 当前模型中五个网络分别负责什么

最终职责冻结如下。

| 模块 | 主要职责 |
|---|---|
| **Transformer** | 全局资源配置、远距离 card dependency |
| **GNN** | 显式 self/opponent/prize relational reasoning |
| **CNN** | joint-action strategic surface 的局部几何结构 |
| **LSTM** | 当前局内短期 opponent dynamics |
| **Mamba** | 跨局长期 opponent behavioural memory |

这五个不是五个彼此竞争的全功能模型。

它们进入的是不同位置。

---

# 82. 为什么 CNN 不和 Transformer/GNN 平行

最终数据流不是：

```text
Transformer ─┐
CNN ─────────┼→ Gate
GNN ─────────┘
```

这种粗糙 MoE。

正确结构是：

\[
Transformer+GNN
\]

先建立：

\[
\text{action-level representation}
\]

然后构造：

\[
(a,b)
\]

pair map。

CNN 再在 joint-action plane 上工作。

所以：

\[
\boxed{
Transformer/GNN
\rightarrow
Matrix CNN
}
\]

有明确因果顺序。

---

# 83. 为什么 LSTM/Mamba 不进入 Robust Backbone

如果把：

\[
h^{opp}
\]

直接喂进：

\[
Q_R
\]

那么：

> 同样的 game state

面对两个不同真人，

会产生两个不同的所谓：

\[
Q_{\text{robust}}
\]

这是概念错误。

所以 public path 有硬信息隔离：

```text
Opponent Memory
      X
      │
      └── cannot enter Robust Backbone
```

它只能进入 Adaptive Branch。

---

# 84. Mamba Memory 生命周期

网页对战时：

每个对手/session 存：

```python
OpponentMemory:
    completed_game_summaries
    mamba_cache
```

当前游戏：

```python
CurrentGameMemory:
    lstm_h
    lstm_c
```

---

# 85. 一轮结束

更新：

\[
LSTM
\]

Mamba 不动。

---

# 86. 一整局结束

使用：

\[
h_{LSTM}^{final}
\]

生成：

\[
game\ summary
\]

然后更新：

\[
Mamba
\]

最后 reset 当前局 LSTM state。

因此：

```text
Round
→ LSTM update

Game end
→ Game summary
→ Mamba update
→ LSTM reset
```

语义非常清楚。

---

# 87. 对手切换

如果网页用户选择：

```text
New opponent/session
```

则：

LSTM：

reset。

Mamba：

加载对应 opponent memory。

如果不存在：

加载：

\[
UNKNOWN
\]

---

# 88. 模型参数规模预算

这里不追求上亿参数。

Goofspiel 不需要。

预计：

### Card Transformer

约：

\[
1.8M
\]

### GNN

约：

\[
0.6-0.9M
\]

### Matrix CNN

约：

\[
0.8-1.0M
\]

### LSTM

约：

\[
0.55M
\]

### Mamba

约：

\[
0.8-1.2M
\]

### Fusion + heads + adaptive branch

约：

\[
1.5-2.5M
\]

所以整体预计：

\[
\boxed{
6M\sim8M
}
\]

具体实现后必须自动打印：

```python
total_params
trainable_params
params_by_module
```

而不是文档里假装给出“精确 7.32M”。

---

# 89. 为什么模型不需要更大

状态空间虽然巨大，但单状态结构非常小。

\[
N=13
\]

意味着：

- 13 rank
- 39 GNN nodes
- 169 joint action cells
- 当前局 history ≤12
- 长期 history 按 game-level 压缩

所以这是：

\[
\boxed{\text{high combinatorial complexity}}
\]

而不是：

\[
\boxed{\text{high perceptual complexity}}
\]

问题。

因此重点应该放在 inductive bias 和 game-theoretic structure，而不是暴力增加参数。

---

# 90. Numerical Precision

模型主干：

\[
BF16
\]

即可。

但是：

### 输出 Q

建议 cast：

\[
FP32
\]

### softmax

FP32。

### opponent probability

FP32。

### value distribution normalization

FP32。

原因是后续 Nash solver/Search 对细微 Q 差异更敏感。

---

# 91. Forward API

最终主模型：

```python
class GoofspielModel(nn.Module):

    def forward(
        self,
        public_state: PublicStateBatch,
        current_game_history: HistoryBatch | None,
        long_term_memory: OpponentMemoryBatch | None,
        return_intermediates: bool = False,
    ) -> GoofspielModelOutput:
        ...
```

---

# 92. PublicStateBatch

建议：

```python
@dataclass
class PublicStateBatch:
    n_cards: Tensor

    self_cards: Tensor
    opponent_cards: Tensor

    remaining_prizes: Tensor
    current_prize: Tensor

    self_score: Tensor
    opponent_score: Tensor

    round_idx: Tensor

    rank_mask: Tensor
    self_action_mask: Tensor
    opponent_action_mask: Tensor
```

底层 env 可以 bitmask。

进入 GPU 前 decode 成 dense tensor。

---

# 93. HistoryBatch

```python
@dataclass
class HistoryBatch:
    prize: Tensor
    self_action: Tensor
    opponent_action: Tensor
    score_diff: Tensor
    outcome: Tensor
    round_idx: Tensor
    valid_mask: Tensor
```

---

# 94. OpponentMemoryBatch

不是把所有过去游戏原始历史重新传入。

应该存：

```python
@dataclass
class OpponentMemoryBatch:
    game_summary_sequence: Tensor
    valid_mask: Tensor
```

或者 inference 时直接提供 Mamba cache/state。

---

# 95. 模型内部类结构

建议：

```text
models/
│
├── rank_encoder.py
│
├── public/
│   ├── card_transformer.py
│   ├── relational_gnn.py
│   ├── card_fusion.py
│   ├── pair_builder.py
│   └── matrix_cnn.py
│
├── opponent/
│   ├── round_encoder.py
│   ├── intra_game_lstm.py
│   ├── game_summary.py
│   ├── inter_game_mamba.py
│   ├── memory_fusion.py
│   └── opponent_head.py
│
├── adaptive/
│   ├── film.py
│   ├── adaptive_matrix.py
│   └── adaptive_heads.py
│
├── heads/
│   ├── joint_q.py
│   ├── policy_prior.py
│   ├── distributional_value.py
│   └── ensemble.py
│
├── types.py
└── goofspiel_model.py
```

---

# 96. Public Backbone Forward

伪代码：

```python
rank_emb = rank_encoder(n_cards)

card_tokens = build_card_features(
    rank_emb,
    self_cards,
    opponent_cards,
    remaining_prizes,
    current_prize,
)

global_emb = global_encoder(global_features)

transformer_out = card_transformer(
    card_tokens,
    global_emb,
    rank_mask,
)

graph_out = relational_gnn(
    rank_emb,
    state,
    node_mask,
)

self_action_emb, opponent_action_emb = card_fusion(
    transformer_out,
    graph_out,
    global_emb,
)

public_emb = global_fusion(
    transformer_out.state_token,
    graph_out.global_embedding,
    global_emb,
)

pair_map = pair_builder(
    self_action_emb,
    opponent_action_emb,
    public_emb,
)

matrix_features = matrix_cnn(
    pair_map,
    joint_mask,
)

q_robust = robust_q_head(matrix_features)

robust_policy = robust_policy_head(
    self_action_emb,
    matrix_features,
    public_emb,
)

robust_value = robust_value_head(
    matrix_features,
    public_emb,
)
```

---

# 97. Opponent Backbone Forward

```python
short_memory = intra_game_lstm(
    current_game_history
)

long_memory = inter_game_mamba(
    long_term_memory
)

opponent_embedding = memory_fusion(
    short_memory,
    long_memory,
    public_emb,
)

opp_short = short_opponent_head(
    opponent_action_emb,
    short_memory,
    public_emb,
)

opp_long = long_opponent_head(
    opponent_action_emb,
    long_memory,
    public_emb,
)

opp_fused = fused_opponent_head(
    opponent_action_emb,
    opponent_embedding,
    public_emb,
)
```

---

# 98. Adaptive Branch

```python
adaptive_map = film_modulation(
    matrix_features,
    opponent_embedding,
)

adaptive_map = cross_condition(
    adaptive_map,
    self_action_emb,
    opponent_action_emb,
    opponent_embedding,
)

adaptive_map = adaptive_matrix_cnn(
    adaptive_map,
    joint_mask,
)

delta_q = adaptive_q_head(adaptive_map)

q_adaptive = q_robust + delta_q

adaptive_policy = adaptive_policy_head(...)

adaptive_value = adaptive_value_head(...)
```

---

# 99. 最终模型中明确不存在的东西

为了保证架构职责干净，以下内容不放进 model：

### MCTS Tree

不存在。

### Exact Solver

不存在。

### Nash LP Solver

不存在。

### PPO Buffer

不存在。

### Replay Buffer

不存在。

### Elo

不存在。

### PSRO Population

不存在。

### Action Sampling

严格来说最终 action sampling 也不是 model 责任。

Model 返回概率/logits/Q。

Decision layer 决定最后 action。

---

# 100. Planner 使用 Model 的方式

虽然 Planner 不属于本文，但 Model 的接口必须为它准备好。

Planner 获得：

\[
Q_R
\]

\[
Q_A
\]

\[
\pi_R^{prior}
\]

\[
\pi_A^{prior}
\]

\[
q_{opp}
\]

\[
Z_R
\]

\[
Z_A
\]

以及 ensemble heads。

因此后续可以自由实现：

- direct neural play
- Nash solve
- best response
- safe exploitation
- MCTS
- Exact Solver override

而无需修改神经网络本身。

---

# 101. 可解释性接口

这套模型非常适合做可视化。

因此 `return_intermediates=True` 时至少允许返回：

### Transformer attention

### GNN attention

### Matrix CNN feature summary

### Q matrix

### opponent prediction

### LSTM/Mamba fusion gate

### robust/adaptive difference

尤其网页可以直接画：

\[
13\times13
\]

Q heatmap。

这对于研究非常重要。

---

# 102. 可视化的五个核心对象

以后 localhost detector 可以展示：

### 1. Robust Q Matrix

\[
13\times13
\]

### 2. Adaptive Q Matrix

\[
13\times13
\]

### 3. Opponent predicted action distribution

13 bars。

### 4. Robust/Adaptive Policy Prior

两组概率。

### 5. LSTM/Mamba gate

例如：

```text
Short-term: 0.63
Long-term:  0.37
```

实际 feature-wise gate 可以再求 mean 做展示。

---

# 103. Architecture Assertions

代码中必须写 architecture-level tests。

不是等训练后才发现错误。

---

# 104. Variable-N Test

同一个 model：

```python
for n in [3, 5, 7, 10, 13]:
    output = model(state_n)
```

全部必须成功。

---

# 105. Mask Test

对于已使用 action：

policy probability：

\[
0
\]

joint Q mask：

false。

---

# 106. Opponent Leakage Test

给定完全相同 public state：

改变 opponent history。

要求：

\[
Q_R^{(history1)}
=
Q_R^{(history2)}
\]

数值完全一致。

同样：

\[
\pi_R^{prior}
\]

和：

\[
Z_R
\]

必须一致。

而：

\[
Q_A
\]

允许不同。

这是一个非常重要的单元测试。

---

# 107. LSTM Isolation Test

改变当前局历史：

Mamba output 不应该变化。

LSTM output 应该变化。

---

# 108. Mamba Isolation Test

改变过去 completed games：

Mamba output 应该变化。

当前游戏 LSTM 在相同局内 history 下保持一致。

---

# 109. Online LSTM Equivalence

一次性：

```python
lstm(full_history)
```

和逐轮：

```python
lstm.step(...)
```

最终 hidden 必须在数值 tolerance 内一致。

---

# 110. Padding Invariance

同一个 N=7 状态：

如果单独 batch：

\[
N_{max}=7
\]

与和 N=13 样本一起 batch：

\[
N_{max}=13
\]

有效区域输出必须基本相同。

这可以抓出 CNN padding leakage。

---

# 111. Player View Consistency

将：

\[
self
\leftrightarrow opponent
\]

并交换 score。

输出 shape 和 mask 必须正确。

严格 antisymmetry 可以由后续训练/约束处理，但架构不得写死 player identity。

---

# 112. 参数统计 Test

CI 中打印：

```text
Rank Encoder:
Card Transformer:
Relational GNN:
Matrix CNN:
LSTM:
Mamba:
Adaptive Branch:
Heads:
Total:
```

防止某个模块意外膨胀几十倍。

---

# 113. 计算复杂度

Public Transformer：

\[
O(N^2d)
\]

N=13，几乎可以忽略。

GNN dense relation：

\[
O((3N)^2d)
\]

最大：

\[
39^2
\]

也非常小。

Matrix CNN：

\[
O(N^2d^2L)
\]

N=13。

LSTM：

\[
O(Td^2)
\]

其中：

\[
T\le12
\]

Mamba：

\[
O(Gd)
\]

对 game sequence 近似线性。

因此整个模型 forward 是一个很轻量的网络。

---

# 114. 真正大的计算不会来自模型本身

以后整个系统真正重的是：

- 大量 self-play；
- full joint-action expansion；
- search；
- reanalysis；
- population evaluation。

不是：

> 单次 model forward。

所以模型完全可以拥有 Transformer + GNN + CNN + LSTM + Mamba，而不会像大型语言模型那样产生夸张计算成本。

---

# 115. 模型设计的核心哲学

最终模型不是一个：

\[
\text{state}\rightarrow action
\]

黑箱。

而是明确拆出四种内部表示。

---

## 115.1 Resource Representation

由：

\[
Transformer
\]

负责。

回答：

> 当前全部资源是如何配置的？

---

## 115.2 Relational Representation

由：

\[
GNN
\]

负责。

回答：

> 我的牌、对手牌和奖品之间是什么关系？

---

## 115.3 Strategic Interaction Representation

由：

\[
Matrix CNN
\]

负责。

回答：

> 所有 simultaneous joint actions 组成的 strategic surface 是什么样？

---

## 115.4 Temporal Opponent Representation

由：

\[
LSTM+Mamba
\]

负责。

回答：

> 这个具体对手短期正在做什么，以及长期是什么风格？

---

# 116. 最终神经模型定义

最终可以把整个网络形式化写成：

公共表示：

\[
H_T
=
Transformer(S)
\]

\[
H_G
=
GNN(S)
\]

动作表示：

\[
A_{self},A_{opp}
=
Fuse(H_T,H_G)
\]

pair map：

\[
M_0
=
PairBuilder(
A_{self},
A_{opp},
S
)
\]

战略矩阵：

\[
M_R
=
CNN(M_0)
\]

robust outputs：

\[
Q_R
=
f_Q(M_R)
\]

\[
\pi_R^{prior}
=
f_\pi(M_R,A_{self})
\]

\[
Z_R
=
f_Z(M_R)
\]

局内 opponent memory：

\[
h_L
=
LSTM(H_{game})
\]

跨局 opponent memory：

\[
h_M
=
Mamba(H_{games})
\]

融合：

\[
h_O
=
Fuse_{temporal}
(
h_L,h_M,S
)
\]

对手动作预测：

\[
q_O
=
f_O(
A_{opp},h_O,S
)
\]

adaptive representation：

\[
M_A
=
CNN_A(
Condition(M_R,h_O)
)
\]

adaptive Q：

\[
Q_A
=
Q_R+
f_{\Delta Q}(M_A)
\]

adaptive prior：

\[
\pi_A^{prior}
=
f_{\pi,A}(M_A,h_O)
\]

adaptive distribution：

\[
Z_A
=
f_{Z,A}(M_A,h_O)
\]

---

# 117. 最终输出的数学集合

因此完整神经模型：

\[
f_\theta(S,H_g,H_{session})
\]

返回：

\[
\boxed{
\{
Q_R,
Q_A,
\pi_R^{prior},
\pi_A^{prior},
Z_R,
Z_A,
q_L,
q_M,
q_F,
U\text{-heads}
\}
}
\]

其中：

\[
Q_R,Q_A
\in
\mathbb R^{N\times N}
\]

\[
\pi_R,\pi_A,q_L,q_M,q_F
\in
\Delta^N
\]

\[
Z_R,Z_A
\in
\Delta^{201}
\]

---

# 118. 为什么这个架构值得保留完整形态

这套架构不是为了宣称：

> “我们用了 Transformer、GNN、CNN、LSTM、Mamba。”

它真正表达的是五类不同问题：

\[
\text{global resource reasoning}
\]

\[
\text{relational reasoning}
\]

\[
\text{joint-action reasoning}
\]

\[
\text{short-term adaptation}
\]

\[
\text{long-term adaptation}
\]

并最终把它们投射到：

\[
\boxed{
Q(s,a,b)
}
\]

这个与 Goofspiel simultaneous game structure 高度匹配的核心对象上。

---

# 119. 架构中最关键的三条硬约束

如果后续工程实现发生取舍，下面三条不能破坏。

### 第一

\[
\boxed{
Q_R
\text{ 不允许读取 opponent history}
}
\]

robust game representation 必须保持纯净。

---

### 第二

\[
\boxed{
\text{核心输出必须是 joint-action matrix}
}
\]

不能最后为了实现简单退化成：

\[
13\text{-dimensional action Q}
\]

否则就丢掉 simultaneous-game 的核心结构。

---

### 第三

\[
\boxed{
LSTM 与 Mamba 必须分别承担局内和跨局时间尺度
}
\]

不能最终实现成两个并行 encoder 然后谁有用谁没用完全不知道。

---

# 120. 最终一句话定义

这个模型最终不是：

> 一个神经网络学习“当前出哪张牌”。

而是：

\[
\boxed{
\textbf{
一个以 joint-action value matrix 为核心，
通过 Transformer 建模全局资源，
通过 GNN 建模显式关系，
通过 CNN 建模战略矩阵几何，
通过 LSTM 建模局内短期对手行为，
通过 Mamba 建模跨局长期对手记忆，
并同时保持 robust 与 opponent-adaptive 两套价值表示的
hierarchical neural game model。
}
}
\]

这就是后续所有学习算法、Nash 求解、MCTS、Exact Solver、Red Team、Opponent Exploitation 所依赖的统一神经模型接口。