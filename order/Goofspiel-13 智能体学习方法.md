# Goofspiel-13 智能体学习方法详细设计书
## ——Learning Objectives、强化学习算法、博弈学习、对手学习与梯度更新实现规格

---

# 0. 文档定位

本文规定 Goofspiel 智能体**学习什么，以及每一种能力具体怎样学习**。

本文不是模型结构说明书，也不是训练集群/League/Replay/Red-Team 调度说明书。

已经确定的模型结构保持不变：

- Card Transformer
- Relational GNN
- Matrix CNN
- LSTM
- Mamba
- Robust / Adaptive 双分支
- Joint-action matrix \(Q(s,a,b)\)
- Distributional outcome head
- Opponent action prediction heads
- Ensemble uncertainty heads

本文负责规定这些输出分别对应什么数学对象、target 如何产生、loss 如何计算、梯度允许流向哪些模块。

训练生态，例如：

- League
- Historical Opponents
- Population
- Red Team 调度
- Replay Buffer 比例
- Reanalyse Worker 数量
- Actor/Learner 分布式架构

全部在后续《训练流程与训练生态设计书》中讨论。

---

# 1. 总原则：不要把“学习”缩成一个 RL reward

整个智能体最终当然服务于：

\[
\boxed{\text{赢得 Goofspiel}}
\]

但是“赢”只是最高层 utility。

为了做到这一点，智能体必须学习不同层次的知识。

整个学习系统按照：

\[
\boxed{
\text{Prediction}
\rightarrow
\text{Optimization}
\rightarrow
\text{Decision}
}
\]

组织。

---

# 2. Prediction：需要学习什么

必须学习至少以下内容：

### 游戏本身

\[
Q_R(s,a,b)
\]

预测 joint action 的长期 robust strategic value。

\[
Z_R(s)
\]

预测最终 score difference 的概率分布。

### 对手

\[
q_L(b|s,h_{\text{game}})
\]

预测当前局中对手下一动作。

\[
q_M(b|s,h_{\text{session}})
\]

依据长期历史预测对手行为。

\[
q_F(b|s,h_{\text{game}},h_{\text{session}})
\]

融合短期和长期信息后预测对手下一动作。

同时学习：

\[
z_{\text{short}}
\]

短期行为表征，

\[
z_{\text{long}}
\]

长期 opponent style 表征，

以及：

\[
P_{\text{switch}}
\]

对手是否发生策略变化。

### 针对具体对手

\[
Q_A(s,h,a,b)
\]

学习 opponent-conditioned joint-action value。

\[
Z_A(s,h)
\]

预测针对当前 opponent 时最终结果分布。

### 不确定性

学习/估计：

\[
U_Q
\]

和：

\[
U_{\text{opp}}
\]

供后续 Decision Layer 使用。

---

# 3. Optimization：学习什么战略目标

系统同时存在两个严格分开的 optimisation objective。

## Robust Objective

目标是：

\[
\boxed{
\max_{\pi}
\min_{\sigma}
E[U(\pi,\sigma)]
}
\]

即接近 Nash / minimax strategy，降低 exploitability。

---

## Exploit Objective

当存在可靠 opponent model：

\[
q_{\phi}
\]

时，目标是：

\[
\boxed{
\max_{\pi}
E_{b\sim q_\phi}[U]
}
\]

即针对实际对手获得比 Nash 更高的收益。

---

这两个目标不能混为一个 loss。

Robust 学习不允许被 opponent history 污染。

Adaptive 学习不能反过来修改 Robust representation。

---

# 4. Decision 不属于本文

本文学习：

- Q
- Policy
- Regret signal
- Opponent belief
- Outcome distribution
- Uncertainty

但最终一手牌到底采用：

- Nash(Q)
- Direct Policy
- CFR/Search Policy
- Adaptive Policy
- Safe Exploit
- Exact Solver

由后续 Decision / Search 文档决定。

Model/Learner 本身不直接作最终决策。

---

# 5. 首先冻结 reward 与 utility

标准 N-card Goofspiel：

\[
S_N=\frac{N(N+1)}{2}
\]

标准：

\[
N=13,\qquad S_{13}=91
\]

当前 prize：

\[
p_t
\]

自己出：

\[
a_t
\]

对手出：

\[
b_t
\]

定义 zero-sum normalized immediate reward：

\[
r_t=
\frac{p_t}{S_N}
\operatorname{sgn}(a_t-b_t)
\]

因此：

\[
r_t\in[-1,1]
\]

并且：

\[
r^{opp}_t=-r_t
\]

---

# 6. 主 Robust Utility

Robust Q 的优化 utility 固定为：

\[
\boxed{
U_\Delta
=
\frac{
Score_{\text{self}}-Score_{\text{opp}}
}{
S_N
}
}
\]

也就是 normalized final score difference。

不要由 Codex 擅自改成：

```text
win = +1
draw = 0
loss = -1
```

也不要改成：

```text
winner reward = 1
```

Robust Q 学的是：

\[
\boxed{\text{Expected Score Difference}}
\]

原因是它：

- 严格 zero-sum；
- 每轮 reward additive；
- 与 Exact Solver 一致；
- 适合动态规划；
- 信息比 binary win/loss 更丰富。

---

# 7. Win/Draw/Loss 不丢失

真正游戏结果仍然重要。

因此：

\[
Z_R
\]

和：

\[
Z_A
\]

学习完整 final score-difference distribution。

由其直接计算：

\[
P(W)
\]

\[
P(D)
\]

\[
P(L)
\]

所以：

> score difference optimisation

与：

> winning probability prediction

同时存在，但不要混成一个 Q target。

---

# 8. 三类 Value 必须严格区分

这是整份文档最不能写错的地方。

## 8.1 Robust Nash Q

\[
\boxed{
Q_R(s,a,b)
}
\]

语义：

> 当前执行 joint action \((a,b)\)，之后双方均按 minimax-optimal continuation play 时，从现在到游戏结束所能获得的 expected future score difference。

注意：

它预测的是：

\[
\text{future incremental return}
\]

不包含已经获得的历史分差。

---

## 8.2 Robust Outcome Distribution

\[
\boxed{
Z_R(s)
}
\]

语义：

> 当前实际 Robust Actor 从这个状态继续游戏时，最终总 score difference 的经验概率分布。

它由真实 trajectory 学。

因此：

\[
Z_R
\]

不是：

\[
Q_R
\]

的 distributional 版本。

它们语义不同。

---

## 8.3 Adaptive Q

\[
\boxed{
Q_A(s,h,a,b)
}
\]

语义：

> 当前 joint action \((a,b)\) 以后，对手继续按照当前 opponent model 所表示的行为产生动作，而我继续采用 opponent-conditioned best-response-like policy 时的 expected future score difference。

因此：

\[
Q_R
\neq
Q_A
\]

绝不允许用同一个 target 混着训练。

---

# 9. 为什么 MC/TD(λ) 不能直接覆盖 \(Q_R\)

实际 trajectory：

\[
s_t,a_t,b_t,\dots,s_T
\]

由某个行为策略产生。

得到：

\[
G_t^{MC}
\]

只说明：

> 这一次真实 continuation 的结果。

但：

\[
Q_R
\]

要求的是：

> 当前动作以后双方 optimal minimax continuation 的值。

如果实际 trajectory 不是 exact Nash continuation：

\[
G_t^{MC}
\neq Q_R(s_t,a_t,b_t)
\]

一般成立。

因此：

\[
\boxed{
\text{禁止}
}
\]

直接写：

```python
loss_q_robust = mse(q_robust[a,b], monte_carlo_return)
```

这是语义错误。

MC/TD(\(\lambda\)) 要训练的是：

\[
Z_R
\]

及其 expected value consistency，而不是覆盖 Nash Q。

---

# 第一部分：Robust Joint-Action Q Learning

# 10. Robust Q 使用 Nash Bellman Learning

经典 Minimax-Q 将普通 Q-learning 扩展到 two-player zero-sum Markov games：Q 函数依赖双方动作，并使用零和 matrix-game value 代替单智能体的 \(\max Q\)。

我们不照搬经典 sampled Minimax-Q。

因为 Goofspiel 有一个额外优势：

\[
\boxed{\text{transition model 完全已知}}
\]

所以采用：

\[
\boxed{
\text{Full-Matrix Model-Based Nash Bellman Learning}
}
\]

---

# 11. Robust Q Bellman Equation

当前 state：

\[
s
\]

合法自己的动作：

\[
a\in A(s)
\]

合法对手动作：

\[
b\in B(s)
\]

即时 reward：

\[
r(s,a,b)
\]

执行以后，下一 prize：

\[
p'
\]

来自 remaining prizes。

child state：

\[
s'_{a,b,p'}
\]

定义：

\[
V_R(s')
=
\operatorname{Val}(Q_R(s'))
\]

其中：

\[
\operatorname{Val}(Q)
=
\max_{x\in\Delta(A)}
\min_{y\in\Delta(B)}
x^TQy
\]

于是：

\[
\boxed{
Y^R_{ab}
=
r(s,a,b)
+
E_{p'}
[
V_R(s'_{a,b,p'})
]
}
\]

由于 finite horizon：

\[
\boxed{\gamma=1}
\]

禁止默认使用：

```python
gamma = 0.99
```

---

# 12. Terminal State

如果当前 round 是最后一轮：

\[
Y^R_{ab}
=
r(s,a,b)
\]

没有 bootstrap。

---

# 13. Full-Matrix Backup

训练一个 state 时：

不要只更新实际执行过的：

\[
(a_t,b_t)
\]

必须遍历所有 legal：

\[
a\in A
\]

\[
b\in B
\]

构造完整：

\[
k\times k
\]

target matrix。

N=13 开局最大：

\[
13^2=169
\]

个 joint action。

所以一个 root state 最多产生：

\[
169
\]

个 Q labels。

---

# 14. Counterfactual Child Generation

实现：

```python
children = env_model.expand_joint_actions(state)
```

返回逻辑上：

```text
for every legal self_action a
    for every legal opp_action b
        generate post-action state
```

禁止通过真正 environment object：

```python
deepcopy(env)
step(...)
```

循环 169 次。

必须使用纯 tensor / compact state transition function：

```python
transition(state, a, b, next_prize)
```

---

# 15. Chance Expectation

如果行动后 remaining prizes 为：

\[
R'
\]

理论 target：

\[
E_{p'}V(s')
=
\frac{1}{|R'|}
\sum_{p'\in R'}V(s'_{p'})
\]

默认实现：

### 当

\[
|R'|\le5
\]

全部枚举。

### 当

\[
|R'|>5
\]

无放回均匀采样：

\[
M=4
\]

个 prize。

配置：

```yaml
nash_bellman:
  chance_exact_threshold: 5
  chance_samples: 4
```

训练高精度配置可以提高 `chance_samples`，但不得改变均匀概率。

---

# 16. Target Network

必须维护：

\[
f_\theta
\]

online network，

和：

\[
f_{\bar\theta}
\]

target network。

Bellman child：

\[
Q_R(s')
\]

必须来自：

\[
f_{\bar\theta}
\]

默认 EMA：

\[
\bar\theta
\leftarrow
(1-\tau)\bar\theta+\tau\theta
\]

其中：

```yaml
target_network:
  tau: 0.005
```

禁止 target 与 online 完全同网络实时计算后再反传。

---

# 17. Nash Matrix Solver

Target 计算需要：

\[
V(s')=
\max_x\min_yx^TQy
\]

需要两个 solver。

### Reference Solver

CPU + FP64 + LP。

用途：

- 数学校验；
- Exact Teacher；
- 小规模 evaluation；
- GPU solver accuracy test。

### Training Solver

GPU batched zero-sum solver。

输入：

```text
[B, N, N]
```

返回：

```text
row_policy [B,N]
col_policy [B,N]
value      [B]
duality_gap[B]
```

训练 solver 使用：

- regret matching / regret matching+
- optimistic mirror descent
- 或其他明确实现的 batched no-regret solver

不得用：

```python
softmax(q.mean(-1))
```

假装这是 Nash。

NeuRD 的理论动机来自 replicator/no-regret dynamics，并且论文明确在 Goofspiel 上做过实验。

---

# 18. Nash Solver 精度保护

训练 solver 默认：

```yaml
matrix_solver:
  iterations: 64
  high_precision_iterations: 256
  max_duality_gap: 0.01
```

如果：

\[
gap>0.01
\]

则：

1. 重跑 256 iterations；
2. 仍超过 threshold，则 sample 标记 `solver_unreliable=True`；
3. 该 state 的 policy-anchor loss 不计算；
4. Q Bellman target可以使用 value 的高精度结果；
5. validation 时可送 Reference LP。

---

# 19. Q Robust Loss

Q 已归一化至大致：

\[
[-1,1]
\]

使用 Huber：

\[
L_{Q_R}
=
\frac{
\sum_{ab}M_{ab}
Huber_{\delta}
(
Q_R(s,a,b)-Y^R_{ab}
)
}{
\sum_{ab}M_{ab}
}
\]

其中：

```yaml
q_loss:
  type: huber
  delta: 0.1
```

`M` 为 joint legal mask。

非法 cell 不参与 loss。

---

# 20. Ensemble Q

已有：

\[
K=4
\]

Q heads。

每个 head 使用独立 bootstrap mask：

\[
m_k\sim Bernoulli(0.8)
\]

即：

```yaml
ensemble:
  heads: 4
  bootstrap_probability: 0.8
```

每个 head 只在自己的 bootstrap samples 上计算：

\[
L^{(k)}_{Q_R}
\]

平均得到：

\[
L^{ensemble}_{Q_R}
\]

不要加入人工 diversity loss。

Epistemic disagreement 应自然来自：

- 初始化；
- bootstrap sampling；
- target noise；
- approximation difference。

---

# 第二部分：Robust Mixed Policy Learning

# 21. 为什么 Policy 不只做 Nash Solver Distillation

如果每一步：

\[
Q_R
\rightarrow
LP
\rightarrow
\pi^{Nash}
\]

然后只训练：

\[
\pi_\theta\approx\pi^{Nash}
\]

这本质主要是 supervised distillation。

我们希望 robust Actor 自身拥有：

\[
\boxed{\text{no-regret learning dynamics}}
\]

因此主 Actor update 使用：

\[
\boxed{\text{NeuRD-style regret update}}
\]

再使用 Nash target 作为 anchor。

NeuRD 通过绕开 softmax Jacobian，让 policy logits 直接按照 advantage/regret 方向更新；其单状态 all-actions 情况与 exponential weights/Hedge 对应，并与 softmax CFR 存在形式联系。

---

# 22. Actor 输入

当前 Robust Q：

\[
Q=stopgrad(Q_R)
\]

当前 row actor：

\[
\pi_A
\]

当前 opponent-view actor：

\[
\pi_B
\]

对手 policy 必须由同一个 shared model 在 swapped player view 上得到。

不要创建固定：

```python
player_1_policy_network
```

---

# 23. Row Player Action Value

对于自己的每一个 action：

\[
u_A(a)
=
\sum_b
\pi_B(b)Q(a,b)
\]

当前策略 value：

\[
v
=
\sum_a
\pi_A(a)u_A(a)
=
\pi_A^TQ\pi_B
\]

定义 instantaneous regret / advantage：

\[
\boxed{
g_A(a)
=
u_A(a)-v
}
\]

---

# 24. Column Player Regret

对手希望最小化自己的负效用。

为了避免符号写错，推荐实现时：

> 直接构造 swap state，以 opponent self-view 再运行同一套 row calculation。

不要手写一套不同逻辑。

测试中要求：

\[
g_B
\]

与直接用：

\[
-Q^T
\]

计算结果一致。

---

# 25. NeuRD Loss

Actor 输出 raw legal logits：

\[
z(a)
\]

使用：

\[
\boxed{
L_{\text{NeuRD}}
=
-
\frac1{|A|}
\sum_a
stopgrad(g_A(a))
z(a)
}
\]

对非法动作 mask。

不要写成：

```python
-regret * log_softmax(logits)
```

那会重新引入普通 policy-gradient softmax Jacobian。

这里必须直接作用于：

\[
\boxed{\text{pre-softmax logits}}
\]

这是实现 NeuRD 思想最关键的一点。

---

# 26. 不添加普通 Entropy Bonus

Robust Actor 的目标本身可能需要 mixed strategy。

Nash 中随机化不是“为了探索”，而是战略本身。

因此默认：

```yaml
robust_actor:
  entropy_bonus: 0.0
```

不要 Codex 自动加：

```python
loss -= 0.01 * entropy
```

探索策略属于训练数据生成，不属于这里的 equilibrium objective。

---

# 27. Nash Anchor

NeuRD Actor 同时接受：

\[
Q_{\bar\theta}(s)
\]

经过高质量 matrix solver 得到：

\[
\pi_Q^{Nash}
\]

使用：

\[
L_{anchor}
=
KL(
stopgrad(\pi_Q^{Nash})
\|
\pi_R
)
\]

默认：

```yaml
robust_actor:
  neurd_weight: 1.0
  nash_anchor_weight: 0.10
```

所以：

\[
\boxed{
L_{\pi_R}
=
L_{\text{NeuRD}}
+
0.1L_{anchor}
}
\]

---

# 28. Nash Anchor 的作用

NeuRD 提供：

\[
\text{learning dynamics}
\]

Nash(Q) 提供：

\[
\text{equilibrium reference}
\]

二者职责不同。

不要把 Nash Anchor 权重设为 1 然后让 NeuRD 形同虚设。

---

# 29. CFR 在学习算法中的位置

CFR 不能被简单写成：

```text
CFR ≈ NeuRD
```

真正 CFR 计算：

\[
\text{counterfactual reach-weighted regret}
\]

并累计：

\[
R_T(I,a)
\]

再通过 regret matching 得到策略。

CFR+ 使用 regret-matching+，将累计 regret-like value 保持非负，并使用改进的 averaging；其实际求解速度可明显优于 vanilla CFR。

Discounted CFR 又进一步对早期 regrets 和平均策略赋予衰减/不同权重，在多种游戏中比 CFR+ 更快。

---

# 30. 本项目对 CFR 的明确处理

Robust Actor 主梯度算法：

\[
\boxed{\text{NeuRD}}
\]

CFR/CFR+/DCFR：

\[
\boxed{\text{独立 Game-Theoretic Teacher / Search Solver}}
\]

而不是把两种 update 的梯度直接相加。

如果 CFR solver 对某个 state/subgame 返回：

```python
CFRTarget(
    policy,
    counterfactual_values,
    regrets,
    quality,
)
```

学习系统使用：

\[
L_{CFR-policy}
=
KL(
\pi^{CFR}
\|
\pi_R
)
\]

作为高质量 teacher anchor。

若没有 CFR target：

不计算。

训练流程以后决定什么时候运行 CFR。

---

# 31. CFR Teacher Priority

Policy anchor target 优先级：

\[
\boxed{
Exact\ Equilibrium
>
Certified\ CFR/Search
>
Reference\ Nash(Q)
>
Training\ Nash(Q)
}
\]

只选择最高优先级的一种 anchor。

不要：

```python
loss = exact + cfr + nash + search
```

同时让四个稍有差异的 teacher 拉扯同一个 actor。

---

# 32. Deep CFR 的借鉴方式

Deep CFR 使用神经网络近似 CFR 所需的 regret/advantage，从而避免大型游戏中的 tabular regret table。

我们不照搬 Deep CFR 作为主算法。

原因：

Goofspiel：

- dynamics 已知；
- public state 小；
- joint action 最大只有 169；
- 已经显式学习 \(Q(s,a,b)\)。

但是 Deep CFR 提供一个重要原则：

\[
\boxed{
\text{Regret/advantage 是值得被函数逼近和蒸馏的战略信号}
}
\]

因此需要在训练日志中显式保存：

\[
g_A(a)
\]

和：

\[
g_B(b)
\]

即使当前不建立永久 Regret Head。

---

# 33. 不增加永久 Regret Head

最终冻结：

\[
\boxed{
\text{Regret 是学习信号，不增加第三套永久 value head}
}
\]

原因：

\[
g_A
\]

可以由：

\[
Q_R,\pi_A,\pi_B
\]

精确导出。

再单独预测：

\[
R(s,a)
\]

会增加一个可能与 Q、Policy 不一致的函数逼近对象。

如果以后研究 Deep CFR，另建 baseline model。

Codex 不得擅自在主模型增加 `regret_head.py`。

---

# 第三部分：Monte Carlo 与 TD(λ)

# 34. MC 的职责

MC 不训练 Nash Q。

MC 主要训练：

\[
\boxed{
Z_R(s)
}
\]

和：

\[
\boxed{
Z_A(s,h)
}
\]

即真实 outcome distribution。

---

# 35. 最终 Outcome

游戏结束：

\[
D_T
=
\frac{
Score_{\text{self}}-Score_{\text{opp}}
}{
S_N
}
\]

范围：

\[
[-1,1]
\]

模型 distribution support：

\[
K=201
\]

：

\[
z_i=-1+\frac{2i}{200}
\]

---

# 36. Two-Hot Projection

真实：

\[
D_T
\]

一般不恰好落在某个 bin。

找到左右：

\[
z_l\le D_T\le z_r
\]

target：

\[
P_l=
\frac{z_r-D_T}{z_r-z_l}
\]

\[
P_r=
1-P_l
\]

其余为 0。

于是：

\[
L_{MC}
=
-\sum_iP_i^{target}
\log Z_R(s)_i
\]

Distributional RL 的核心思想就是显式学习 return distribution，而不只学习其 expectation。

---

# 37. MC Target 的 trajectory 要求

\(Z_R\) 只能使用由 Robust Actor 生成，或与目标 policy 足够接近的 trajectory。

每条 trajectory 必须保存：

```text
behavior_policy_version
behavior_action_prob_self
behavior_action_prob_opp
```

如果 policy lag 超过学习系统配置：

```yaml
outcome_mc:
  max_policy_version_lag: 4
```

该 trajectory 不用于 \(Z_R\) MC loss。

因为完整 distribution 的 off-policy importance correction 很容易产生巨大方差。

---

# 38. 从 Distribution 得到 Scalar Value

定义：

\[
V_Z(s)
=
\sum_i
z_iP_i(s)
\]

它表示：

> 当前 Robust Actor 实际 policy 下的 expected final normalized score difference。

注意：

\[
V_Z
\neq
V_R^{Nash}
\]

训练早期尤其如此。

---

# 39. TD(λ) 训练谁

TD(\(\lambda\)) 用来训练：

\[
\boxed{
V_Z
}
\]

的一致性。

而不是训练：

\[
Q_R^{Nash}
\]

。

这保留我们五子棋设计中的：

\[
MC+TD(\lambda)
\]

思想，同时不破坏 Q 的数学语义。

---

# 40. On-Policy λ Return

定义：

\[
G_t^\lambda
=
r_t+
(1-\lambda)V_Z(s_{t+1})
+
\lambda G_{t+1}^\lambda
\]

因为：

\[
\gamma=1
\]

terminal：

\[
G_T^\lambda=r_T
\]

默认：

```yaml
td_lambda:
  lambda: 0.90
```

于是：

\[
L_{TD\lambda}
=
Huber(
V_Z(s_t)-stopgrad(G_t^\lambda)
)
\]

---

# 41. λ 的端点必须测试

当：

\[
\lambda=0
\]

必须退化到 one-step TD：

\[
r_t+V(s_{t+1})
\]

当：

\[
\lambda=1
\]

必须退化到 MC return：

\[
\sum_{\tau=t}^{T}r_\tau
\]

Codex 必须写单元测试。

---

# 42. Off-Policy TD Correction

如果 trajectory behavior：

\[
\mu
\]

与当前 target robust policy：

\[
\pi
\]

不同，

禁止直接使用裸 TD(\(\lambda\))。

定义 joint importance ratio：

\[
w_t
=
\frac{
\pi_A(a_t|s_t)
\pi_B(b_t|s_t)
}{
\mu_A(a_t|s_t)
\mu_B(b_t|s_t)
}
\]

由于双方动作在给定 simultaneous state 下独立采样，这里 joint policy probability 是双方概率乘积。

---

# 43. Joint V-trace(λ)

采用 V-trace/trace clipping 思想。

IMPALA 的 V-trace 正是用于解决 actor 与 learner policy 不一致时的 off-policy correction。

设置：

\[
\rho_t=\min(1,w_t)
\]

\[
c_t=\lambda\min(1,w_t)
\]

TD error：

\[
\delta_t
=
\rho_t
[
r_t+V(s_{t+1})-V(s_t)
]
\]

corrected target：

\[
v_t
=
V(s_t)
+
\sum_{k=t}^{T-1}
\left(
\prod_{i=t}^{k-1}c_i
\right)
\delta_k
\]

使用：

\[
L_{Vtrace}
=
Huber(V_Z(s_t),stopgrad(v_t))
\]

---

# 44. 不宣称理论保证直接继承

上述 joint V-trace 是针对当前 simultaneous two-player implementation 的工程适配。

不要在 README/论文里写：

> “因为 V-trace 收敛，所以我们的 Markov-game neural implementation 有收敛证明。”

这是不成立的。

Retrace/V-trace 与 function approximation/off-policy bootstrapping 的稳定性本身就需要谨慎处理。

---

# 45. MC + TD(λ) 最终 Loss

对于符合 MC policy-lag 要求的数据：

\[
L_Z=L_{MC}
+
\beta_{TD}L_{TD\lambda}
\]

默认：

```yaml
outcome_value:
  mc_weight: 1.0
  td_lambda_weight: 0.25
```

off-policy：

用：

\[
L_{Vtrace}
\]

替换：

\[
L_{TD\lambda}
\]

---

# 46. 为什么 MC 权重大于 TD

Distribution head 的最直接真值：

\[
\boxed{\text{terminal outcome}}
\]

TD 是 consistency / credit propagation signal。

所以 MC 是主监督。

---

# 第四部分：Exact / Search / Teacher Learning

# 47. External Teacher Interface

所有更强算法输出必须统一成：

```python
@dataclass
class TeacherTarget:
    q_matrix: Tensor | None
    policy_self: Tensor | None
    policy_opp: Tensor | None
    value: Tensor | None
    source: Literal[
        "exact",
        "cfr",
        "search",
        "nash"
    ]
    quality: float
```

学习代码只消费此接口。

不在 loss 文件中直接调用 search tree。

---

# 48. Exact Q

如果 Exact Solver 提供：

\[
Q^*
\]

则：

\[
Y_Q=Q^*
\]

直接覆盖 Nash Bellman target。

优先级：

\[
\boxed{
Exact > Search > Bellman
}
\]

---

# 49. Q Teacher Loss

\[
L_{Q,teacher}
=
Huber(
Q_R,
Q^{teacher}
)
\]

对于存在 teacher 的 state：

禁止同时对同一个 Q cell 再计算 Bellman Q loss。

---

# 50. Search-Bootstrapped Learning

如果高质量 search 返回：

\[
Q^{search}
\]

或者：

\[
V^{search}
\]

则可作为 Bellman continuation teacher。

例如：

\[
Y_{ab}
=
r_{ab}
+
E[
V^{search}(s')
]
\]

这是我们五子棋中：

\[
\boxed{\text{Search-Bootstrapped TD}}
\]

思想在 Goofspiel 中的直接继承。

但 search 如何产生 target 在后续 Search 文档定义。

---

# 51. Policy Teacher

Policy target 优先级：

\[
Exact
>
Certified\ CFR/Search
>
Reference\ Nash(Q)
>
Training\ Nash(Q)
\]

Teacher policy 只参与：

\[
L_{anchor}
\]

NeuRD loss仍然保留。

---

# 52. Teacher 不训练 Opponent Model

Exact Solver / CFR / Search 产生的是：

\[
\text{game-theoretic strategy}
\]

不是实际 opponent behavior label。

禁止使用：

\[
\pi^{Nash}
\]

去监督：

\[
q_F(b|\text{opponent history})
\]

Opponent Model 只学习实际观察到的 opponent actions。

---

# 第五部分：Opponent Action Prediction

# 53. Opponent Prediction 是一级学习目标

不是辅助装饰。

模型必须正式学习：

\[
\boxed{
P(b_t|s_t,h_t)
}
\]

因为 opponent exploitation 依赖其准确性。

---

# 54. Short-Term LSTM Objective

LSTM 只看当前局：

\[
h_{1:t-1}
\]

输出：

\[
q_L(b_t)
\]

真实 opponent action：

\[
b_t
\]

loss：

\[
L_L
=
-\log q_L(b_t)
\]

非法 opponent card 必须在 softmax 前：

\[
logit=-\infty
\]

训练实现中使用数值安全：

```python
masked_logits = logits.masked_fill(~legal_mask, -1e9)
```

---

# 55. Long-Term Mamba Objective

Mamba 只看当前局以前已经完成的 games。

它不能读取：

\[
b_t
\]

或当前局未来动作。

输出：

\[
q_M(b_t)
\]

loss：

\[
L_M
=
-\log q_M(b_t)
\]

这测试：

> 仅凭过去多局行为，对这个 opponent 当前策略可以预测多少。

---

# 56. Fused Prediction

融合：

\[
h_L
\]

和：

\[
h_M
\]

输出：

\[
q_F
\]

loss：

\[
L_F
=
-\log q_F(b_t)
\]

总 next-action loss：

\[
\boxed{
L_{\text{opp-action}}
=
L_F
+
0.3L_L
+
0.3L_M
}
\]

默认：

```yaml
opponent_prediction:
  fused_weight: 1.0
  lstm_aux_weight: 0.3
  mamba_aux_weight: 0.3
```

辅助 heads 必须保留，否则 Fusion 可能完全旁路 LSTM 或 Mamba 中一个。

---

# 57. Gradient Isolation

Opponent loss：

\[
L_{\text{opp-action}}
\]

只允许更新：

- Round Encoder
- LSTM
- Game Summary Projector
- Mamba
- Opponent Fusion
- Opponent Heads

公共 backbone：

- RankEncoder
- Transformer
- GNN
- Matrix CNN

全部：

\[
stop\_gradient
\]

即使 opponent history token 使用共享 RankEncoder，其输出也 detach。

---

# 第六部分：Opponent Style Learning

# 58. 下一动作预测与 Style 学习不是一回事

\[
q(b_t)
\]

回答：

> 下一张牌是什么？

\[
z_{\text{style}}
\]

回答：

> 这个对手长期是什么打法？

所以必须给 Mamba representation 一个额外 learning objective。

---

# 59. Long-Term Style Embedding

取：

\[
h_M
\]

通过 training-only projection：

```text
Linear(192,128)
SiLU
Linear(128,128)
L2 Normalize
```

得到：

\[
z_M\in\mathbb R^{128}
\]

这个 projection head：

\[
\boxed{\text{只用于训练}}
\]

可以不进入最终 Decision API。

---

# 60. Style Contrastive Learning

采用 InfoNCE。

Positive：

> 同一个 opponent、同一个已知 strategy regime 的两个不同历史窗口。

Negative：

> 不同 opponent 或不同 strategy regime。

similarity：

\[
sim(z_i,z_j)
=
z_i^Tz_j
\]

loss：

\[
L_{\text{style}}
=
-\log
\frac{
e^{sim(z_i,z_i^+)/\tau}
}{
e^{sim(z_i,z_i^+)/\tau}
+
\sum_j e^{sim(z_i,z_j^-)/\tau}
}
\]

默认：

```yaml
opponent_style:
  projection_dim: 128
  temperature: 0.10
  weight: 0.10
```

---

# 61. Positive Pair 规则

如果数据带有：

```text
opponent_id
strategy_regime_id
```

Positive 必须满足：

```text
same opponent_id
same strategy_regime_id
different temporal window
```

不能只因为：

```text
same opponent_id
```

就做 positive。

因为一个 opponent 可能会改变策略。

---

# 第七部分：Strategy Switch Learning

# 62. Switch Detection

需要显式学习：

\[
P_{\text{switch},t}
\]

表示：

> 当前短期行为是否已经偏离长期 style。

输入：

\[
[
h_L;
h_M;
h_L-h_M;
h^{public}
]
\]

通过一个 lightweight binary head。

这是对上一份模型结构的**明确学习接口补充**：

```python
opponent_switch_logit
```

Codex 必须加入 ModelOutput。

---

# 63. Switch Label

Switch-learning batch 必须提供：

```text
switch_label ∈ {0,1}
```

语义：

- `0`：当前行为仍属于此前 strategy regime；
- `1`：真实 opponent policy 已发生 regime switch。

训练流程以后负责怎样产生/获得这些标签。

Learner 不自己猜标签。

---

# 64. Switch Loss

使用 BCE：

\[
L_{\text{switch}}
=
BCEWithLogits(
l_{\text{switch}},
y_{\text{switch}}
)
\]

由于 switch 样本通常少：

使用正样本权重：

```yaml
switch_detection:
  positive_weight: 3.0
  loss_weight: 0.10
```

最终 weight 应允许由真实正负比例自动调整。

---

# 65. Opponent 总 Loss

\[
\boxed{
L_O
=
L_F
+
0.3L_L
+
0.3L_M
+
0.1L_{\text{style}}
+
0.1L_{\text{switch}}
}
\]

不要把 Adaptive Q reward 混进这个 loss。

---

# 第八部分：Opponent Uncertainty 与 Calibration

# 66. 不直接预测“confidence scalar”

禁止：

```python
confidence = sigmoid(confidence_head)
```

然后没有监督。

Opponent uncertainty 来自：

\[
K=4
\]

prediction ensemble。

每个 fused opponent head 输出：

\[
q_F^{(k)}
\]

---

# 67. Ensemble Training

每个 opponent prediction head 使用：

\[
Bernoulli(0.8)
\]

bootstrap mask。

主 LSTM/Mamba backbone共享。

预测 uncertainty 使用：

\[
U_{\text{opp}}
=
JSD(
q_F^{(1)},...,q_F^{(K)}
)
\]

即 ensemble Jensen-Shannon disagreement。

---

# 68. Aleatoric 与 Epistemic 不混

单个：

\[
q_F
\]

entropy：

\[
H(q_F)
\]

表示：

> 对手本身行为可能随机。

Ensemble disagreement：

\[
U_{\text{opp}}
\]

表示：

> 模型之间不知道该相信什么。

两者必须分别返回。

不要写：

```python
uncertainty = entropy
```

然后统称 uncertainty。

---

# 69. Calibration Metrics

Opponent predictor 必须记录：

- NLL
- Brier Score
- ECE
- Top-1 Accuracy
- Top-k Accuracy

决策层主要依赖：

\[
\boxed{\text{calibrated probability}}
\]

而不是 accuracy。

---

# 70. Temperature Calibration

在独立 validation 数据上学习一个 scalar：

\[
T>0
\]

推理：

\[
q_F
=
softmax(logits/T)
\]

Temperature：

不由主 training loss 更新。

保存为 checkpoint metadata/buffer。

---

# 第九部分：Adaptive / Exploit Learning

# 71. Adaptive Q 的目标

Opponent model 给出：

\[
q_{\bar\phi}(b|s,h)
\]

我们希望学习：

\[
Q_A(s,h,a,b)
\]

表示：

> 针对这种 opponent behaviour，未来采用 adaptive best response 时的 value。

---

# 72. Adaptive Bellman Equation

当前：

\[
(a,b)
\]

执行后：

\[
s',h'
\]

在 child state 对自己的某个 action：

\[
a'
\]

计算：

\[
u(a')
=
\sum_{b'}
q_{\bar\phi}(b'|s',h')
Q_A(s',h',a',b')
\]

---

# 73. Soft Best Response Policy

训练时不要直接 hard argmax。

定义：

\[
\pi_{BR}(a')
=
\frac{
\exp(u(a')/\tau_A)
}{
\sum_c\exp(u(c)/\tau_A)
}
\]

默认：

```yaml
adaptive:
  br_temperature: 0.05
```

不要使用：

\[
\tau\log\sum e^{u/\tau}
\]

作为 Bellman value，因为它会加入额外 entropy utility。

我们要的是：

\[
\boxed{
V_A(s',h')
=
\sum_{a'}
\pi_{BR}(a')u(a')
}
\]

---

# 74. Adaptive Bellman Target

\[
\boxed{
Y^A_{ab}
=
r_{ab}
+
E_{p'}[
V_A(s'_{ab,p'},h'_{ab})
]
}
\]

其中：

\[
h'_{ab}
\]

必须根据 counterfactual：

- current prize
- self action a
- opponent action b
- round result

调用 LSTM transition 得到。

---

# 75. Opponent Target Network

Adaptive Bellman 使用：

\[
q_{\bar\phi}
\]

而不是快速变化的 online：

\[
q_\phi
\]

Opponent model 维护 EMA target：

```yaml
opponent_target:
  tau: 0.01
```

---

# 76. Adaptive Q Residual

模型结构：

\[
Q_A
=
stopgrad(Q_R)
+
\Delta Q_A
\]

Adaptive loss：

\[
L_{Q_A}
=
Huber(
Q_A,
Y_A
)
\]

梯度只更新：

- Adaptive FiLM
- Adaptive Matrix Blocks
- Adaptive Q Head
- Adaptive Policy Head
- Adaptive Distribution Head

不能更新：

- Q_R
- Transformer
- GNN
- public Matrix CNN
- LSTM
- Mamba
- opponent prediction heads

---

# 77. 为什么 Opponent Encoder 也 detach

如果 adaptive return gradient 能修改 LSTM/Mamba，

它可能让：

\[
z_{\text{opp}}
\]

不再表示：

> “这个人如何行动”

而变成：

> “什么 latent 能让我现在赢钱”。

这样 opponent representation 会失去可解释和可校准语义。

因此：

\[
h_{\text{opp}}
=
stopgrad(h_{\text{opp}})
\]

进入 Adaptive branch。

---

# 78. Adaptive Policy Learning

Teacher：

\[
\pi_{BR}
\]

Adaptive direct prior：

\[
\pi_A^{prior}
\]

使用：

\[
L_{\pi_A}
=
KL(
stopgrad(\pi_{BR})
\|
\pi_A^{prior}
)
\]

默认：

```yaml
adaptive:
  policy_weight: 0.25
```

---

# 79. Adaptive Distributional MC

实际采用 adaptive policy 与实际 opponent 完成一局以后：

\[
D_T
\]

训练：

\[
Z_A(s,h)
\]

使用和 \(Z_R\) 相同的 201-bin two-hot MC target。

\[
L_{Z_A}=CE
\]

它表示实际 exploit behaviour 的 outcome risk。

---

# 80. Adaptive Residual Regularisation

在 opponent information 不可靠时：

\[
Q_A
\]

应该靠近：

\[
Q_R
\]

。

定义 opponent epistemic uncertainty：

\[
U_{\text{opp}}
\]

以及 observed rounds：

\[
n
\]

confidence：

\[
c_{\text{opp}}
=
(1-e^{-n/4})
e^{-U_{\text{opp}}/\tau_U}
\]

配置：

```yaml
adaptive:
  uncertainty_temperature: 0.1
```

residual regularisation：

\[
L_\Delta
=
(1-c_{\text{opp}})
\|
Q_A-stopgrad(Q_R)
\|^2
\]

默认：

```yaml
adaptive:
  residual_weight: 0.10
```

这不会直接决定最终 Safe Exploit 比例。

它只约束 Adaptive value representation 在信息不足时不要胡乱偏离 robust knowledge。

---

# 81. Adaptive 总 Loss

\[
\boxed{
L_A
=
L_{Q_A}
+
0.25L_{\pi_A}
+
0.5L_{Z_A}
+
0.1L_\Delta
}
\]

---

# 第十部分：Self-Supervised / Structural Learning

# 82. 必须保留的 Self-Supervised Objective

五子棋里讨论过多种 self-supervised learning。

迁移到 Goofspiel 后，不需要为了“全部保留”而机械加入没有意义的任务。

最终保留两个真正有理论结构的目标：

1. Player-Swap Symmetry
2. Opponent Style Contrastive Learning

Masked board reconstruction 不迁移。

原因：

Goofspiel public state 本身极低维且完全已知，随机遮住一张牌让网络猜回来并不是核心能力。

Codex 不得擅自加入 generic masked-autoencoder loss。

---

# 83. Player-Swap Q Symmetry

构造 swapped state：

\[
s^\leftrightarrow
\]

交换：

- self cards
- opponent cards
- scores

Q 必须满足：

\[
\boxed{
Q_R(s,a,b)
=
-Q_R(s^\leftrightarrow,b,a)
}
\]

loss：

\[
L_{\text{sym-Q}}
=
\|
Q_R(s)
+
Q_R(s^\leftrightarrow)^T
\|_2^2
\]

---

# 84. Distribution Symmetry

如果：

\[
Z_R(s)
\]

的 bins：

\[
[-1,\dots,1]
\]

swap 后：

\[
Z_R(s^\leftrightarrow)
\]

应该等于原 distribution reverse：

\[
P^\leftrightarrow(z)
=
P(-z)
\]

使用：

\[
L_{\text{sym-Z}}
=
KL(
reverse(Z_R(s))
\|
Z_R(s^\leftrightarrow)
)
\]

---

# 85. Policy Symmetry

shared model 在 swap state 输出 opponent-view policy。

不要求：

\[
\pi_A(s)=\pi_A(swap)
\]

因为双方 remaining resources 可能不同。

因此不写简单 policy equality loss。

不要 Codex 自作主张加。

---

# 86. Structural Loss Weight

Robust structural：

\[
L_{struct}
=
0.1L_{\text{sym-Q}}
+
0.05L_{\text{sym-Z}}
\]

---

# 第十一部分：Gradient Routing

# 87. 为什么必须明确 Gradient Routing

多个学习目标共享一个大模型。

如果所有 loss：

```python
total_loss.backward()
```

没有控制，

会导致：

- opponent prediction 修改 robust representation；
- exploit loss 修改 Nash critic；
- actor loss破坏 critic representation；
- teacher 与 MC 语义互相污染。

因此梯度路由是算法的一部分。

---

# 88. Robust Q Update

更新：

- RankEncoder
- Card Transformer
- Relational GNN
- Public Fusion
- Pair Builder
- Public Matrix CNN
- Robust Q Heads

loss：

\[
L_{Q_R}
+
L_{teacher}
+
L_{\text{sym-Q}}
\]

---

# 89. Robust Distribution Update

更新：

- public backbone
- Robust Distribution Head

loss：

\[
L_{MC}
+
L_{TD\lambda/Vtrace}
+
L_{\text{sym-Z}}
\]

允许 distribution loss 改进 public representation。

---

# 90. Robust Actor Update

Policy Head 输入 public features 时：

\[
\boxed{detach}
\]

即：

```python
actor_features = public_features.detach()
```

NeuRD / Nash anchor：

只更新 Robust Policy Head。

理由：

joint-Q critic承担主要 strategic representation learning；

policy head 的职责是：

\[
\boxed{
\text{amortized mixed-strategy solver}
}
\]

避免 policy-gradient dynamics 反过来扰动 Q representation。

---

# 91. Opponent Update

Public features：

detach。

共享 RankEncoder 输出：

detach。

更新：

- history projection
- LSTM
- Game Summary
- Mamba
- memory fusion
- action prediction heads
- style projection
- switch head

---

# 92. Adaptive Update

Robust features：

detach。

Opponent features：

detach。

只更新 Adaptive branch。

---

# 93. 梯度检查

必须写 test：

```text
Robust Q backward:
  opponent params grad = None/0
  adaptive params grad = None/0

Opponent backward:
  public backbone grad = None/0
  adaptive grad = None/0

Adaptive backward:
  public backbone grad = None/0
  opponent backbone grad = None/0
```

任何 violation：

测试失败。

---

# 第十二部分：Loss 总表

# 94. Robust Q

\[
L_{RQ}
=
L_{NashBellman/Teacher}
+
0.1L_{\text{sym-Q}}
\]

---

# 95. Robust Actor

\[
L_{R\pi}
=
L_{NeuRD}
+
0.1L_{anchor}
\]

---

# 96. Robust Outcome

\[
L_{RZ}
=
L_{MC}
+
0.25L_{TD\lambda/Vtrace}
+
0.05L_{\text{sym-Z}}
\]

---

# 97. Opponent

\[
L_O
=
L_F
+
0.3L_L
+
0.3L_M
+
0.1L_{\text{style}}
+
0.1L_{\text{switch}}
\]

---

# 98. Adaptive

\[
L_A
=
L_{QA}
+
0.25L_{\pi A}
+
0.5L_{ZA}
+
0.1L_\Delta
\]

---

# 99. 不使用一个 Total Loss

禁止：

```python
total_loss = robust_q_loss + actor_loss + opponent_loss + adaptive_loss
total_loss.backward()
optimizer.step()
```

整个系统使用四种明确 update：

```text
update_robust_q_and_value()
update_robust_actor()
update_opponent_model()
update_adaptive_branch()
```

它们可以在训练流程中以不同频率调用。

具体频率后续训练流程文档定义。

---

# 第十三部分：Target Priority

# 100. Robust Q Target

优先级严格固定：

```text
1. Exact Q
2. Certified Search Q/Value
3. Nash Bellman Target
```

不存在：

```text
MC Q Target
```

---

# 101. Robust Policy Anchor

```text
1. Exact equilibrium policy
2. Certified CFR/Search policy
3. Reference Nash(Q)
4. Batched Training Nash(Q)
```

NeuRD 始终独立存在。

---

# 102. Robust Outcome

只有：

```text
1. Real terminal MC outcome
2. TD(lambda)/V-trace consistency
```

Search/Exact 不覆盖真实 outcome distribution。

---

# 103. Opponent

只有：

```text
Actual opponent action
Actual strategy/regime metadata
```

禁止 Nash/Search teacher 教 opponent predictor。

---

# 104. Adaptive Q

使用：

```text
Opponent-conditioned Bellman target
```

不能用 Robust Exact Q 直接覆盖。

只有当 opponent 明确定义成 exact Nash opponent 时，两者才应接近。

---

# 第十四部分：哪些 RL 方法明确不组合进主算法

# 105. PPO

PPO 不进入 Robust 主更新。

后续可以作为：

- baseline；
- Best Response 算法；
- Adaptive Actor baseline。

不得：

```python
loss_actor = neurd_loss + ppo_loss
```

---

# 106. R-NaD

R-NaD / DeepNash 属于另一套 equilibrium-learning dynamics。

它非常值得作为强 baseline；其目标也是避免 zero-sum self-play dynamics 长期围绕 Nash 循环。它不与 NeuRD loss 同时更新同一个 actor。

实现时应作为：

```text
RobustActorAlgorithm = "rnad"
```

的独立 backend，而不是：

```text
NeuRD + RNaD
```

混合。

---

# 107. Deep CFR

Deep CFR 作为独立 baseline / CFR teacher architecture。

不要把 Deep CFR regret-network loss直接塞进当前 Q Actor。

Deep CFR 的价值是展示 neural function approximation 可以近似 CFR regret learning。

---

# 108. CFR+ / DCFR

作为：

- solver；
- teacher；
- future search algorithm；

而不是当前 gradient actor 的第二套并行 update。

---

# 109. Student of Games 的借鉴

Student of Games 将 guided search、self-play learning 和 game-theoretic reasoning统一起来，并使用 GT-CFR 进行 growing-tree regret search。

我们借鉴的是：

\[
\boxed{
\text{更强 game-theoretic search 可以反过来产生 policy/value target}
}
\]

但 Search 如何实现以后单独设计。

---

# 第十五部分：默认 Algorithm Configuration

Codex 必须提供如下配置，而不是把常量散落在代码里。

```yaml
learning:

  utility:
    type: normalized_score_difference
    gamma: 1.0

  robust_q:
    algorithm: full_matrix_nash_bellman
    huber_delta: 0.1
    target_tau: 0.005

    chance_exact_threshold: 5
    chance_samples: 4

  matrix_solver:
    algorithm: regret_matching_plus
    iterations: 64
    high_precision_iterations: 256
    max_duality_gap: 0.01

  robust_actor:
    algorithm: neurd
    neurd_weight: 1.0
    nash_anchor_weight: 0.10
    entropy_bonus: 0.0
    detach_backbone: true

  ensemble:
    heads: 4
    bootstrap_probability: 0.8

  outcome:
    bins: 201
    range: [-1.0, 1.0]

    mc_weight: 1.0

    td_lambda:
      enabled: true
      lambda: 0.90
      weight: 0.25

    off_policy:
      algorithm: joint_vtrace
      rho_clip: 1.0
      c_clip: 1.0

  opponent:
    fused_weight: 1.0
    lstm_aux_weight: 0.3
    mamba_aux_weight: 0.3

    style:
      enabled: true
      projection_dim: 128
      temperature: 0.10
      weight: 0.10

    switch:
      enabled: true
      positive_weight: 3.0
      weight: 0.10

    target_tau: 0.01

  adaptive:
    algorithm: opponent_conditioned_soft_best_response

    br_temperature: 0.05
    q_weight: 1.0
    policy_weight: 0.25
    distribution_weight: 0.5
    residual_weight: 0.10
    uncertainty_temperature: 0.10

  symmetry:
    q_weight: 0.10
    distribution_weight: 0.05
```

---

# 第十六部分：代码结构

学习算法建议严格拆成：

```text
learning/
│
├── types.py
│
├── targets/
│   ├── nash_bellman.py
│   ├── exact_teacher.py
│   ├── search_teacher.py
│   ├── lambda_return.py
│   ├── vtrace.py
│   ├── outcome_projection.py
│   └── adaptive_bellman.py
│
├── game_theory/
│   ├── matrix_solver.py
│   ├── regret_matching.py
│   ├── regret_matching_plus.py
│   ├── neurd.py
│   └── cfr_target.py
│
├── losses/
│   ├── robust_q.py
│   ├── robust_actor.py
│   ├── outcome.py
│   ├── opponent.py
│   ├── style.py
│   ├── switch.py
│   ├── adaptive.py
│   └── symmetry.py
│
├── updates/
│   ├── robust_value_update.py
│   ├── robust_actor_update.py
│   ├── opponent_update.py
│   └── adaptive_update.py
│
└── diagnostics/
    ├── calibration.py
    ├── duality_gap.py
    ├── gradient_flow.py
    └── target_comparison.py
```

---

# 第十七部分：关键类型

```python
@dataclass
class RobustQTarget:
    target_q: Tensor          # [B,N,N]
    valid_mask: Tensor        # [B,N,N]
    source: list[str]
    solver_gap: Tensor | None
```

---

```python
@dataclass
class PolicyTarget:
    target_policy: Tensor     # [B,N]
    source: list[str]
    quality: Tensor
```

---

```python
@dataclass
class TrajectoryLearningBatch:
    states: PublicStateBatch

    self_actions: Tensor
    opponent_actions: Tensor

    rewards: Tensor

    behavior_prob_self: Tensor
    behavior_prob_opp: Tensor

    final_score_diff: Tensor

    done: Tensor

    policy_version: Tensor
```

---

```python
@dataclass
class OpponentLearningBatch:
    public_states: PublicStateBatch
    current_game_history: HistoryBatch
    long_term_history: OpponentMemoryBatch

    actual_action: Tensor

    opponent_id: Tensor

    strategy_regime_id: Tensor | None
    switch_label: Tensor | None
```

---

# 第十八部分：核心函数接口

# 110. Nash Bellman

```python
def build_nash_bellman_target(
    states,
    target_model,
    transition_model,
    matrix_solver,
    config,
) -> RobustQTarget:
    ...
```

必须返回整张 Q target。

---

# 111. NeuRD

```python
def neurd_loss(
    logits_self,
    logits_opp,
    q_matrix,
    self_mask,
    opp_mask,
) -> Tensor:
    ...
```

Q 必须 detach。

---

# 112. MC Distribution

```python
def project_score_difference_two_hot(
    final_score_diff_normalized,
    num_bins=201,
) -> Tensor:
    ...
```

---

# 113. λ Return

```python
def lambda_returns(
    rewards,
    values,
    done,
    lambda_=0.9,
    gamma=1.0,
) -> Tensor:
    ...
```

---

# 114. Joint V-trace

```python
def joint_vtrace_targets(
    rewards,
    values,
    target_prob_self,
    target_prob_opp,
    behavior_prob_self,
    behavior_prob_opp,
    done,
    lambda_,
    rho_clip=1.0,
    c_clip=1.0,
) -> Tensor:
    ...
```

---

# 115. Opponent Loss

```python
def opponent_prediction_loss(
    short_logits,
    long_logits,
    fused_logits,
    actual_action,
    legal_mask,
) -> dict[str, Tensor]:
    ...
```

---

# 116. Style Loss

```python
def opponent_style_infonce(
    embeddings,
    opponent_ids,
    regime_ids,
    temperature=0.1,
) -> Tensor:
    ...
```

没有合法 positive pair 的 sample：

跳过，不产生 NaN。

---

# 117. Adaptive Bellman

```python
def build_adaptive_bellman_target(
    states,
    histories,
    target_adaptive_model,
    target_opponent_model,
    transition_model,
    config,
):
    ...
```

---

# 第十九部分：必须做的 Unit Tests

# 118. Terminal Q

构造最后一轮。

确认：

\[
Q_R(a,b)
\]

target 恰好等于 immediate reward matrix。

---

# 119. Matching Pennies Nash

输入：

\[
Q=
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}
\]

solver 应：

\[
V\approx0
\]

\[
\pi_A\approx(0.5,0.5)
\]

\[
\pi_B\approx(0.5,0.5)
\]

---

# 120. NeuRD Direction

人工：

\[
g=[+0.5,-0.5]
\]

更新一步后：

第一个 action logit 必须增大，

第二个减小。

---

# 121. NeuRD 不经过 Softmax Jacobian

自动梯度测试：

`loss` 对 raw logits gradient 应直接等于：

\[
-g
\]

在 mask/normalisation 范围内。

---

# 122. λ Endpoint

\[
\lambda=0
\]

等于 TD(0)。

\[
\lambda=1
\]

等于 MC。

---

# 123. V-trace On-Policy

若：

\[
\pi=\mu
\]

importance ratio：

\[
1
\]

corrected target 应退化到对应 on-policy trace。

---

# 124. Q-MC Semantic Isolation

运行 MC loss backward。

确认：

```text
robust_q_head.grad == 0/None
```

必须成立。

---

# 125. Opponent Isolation

改变 opponent history。

要求：

\[
Q_R
\]

完全不变。

而：

\[
q_F
\]

允许变化。

---

# 126. Adaptive Isolation

Adaptive loss backward：

Robust Q/Public Backbone：

无 gradient。

Opponent LSTM/Mamba：

无 gradient。

---

# 127. Symmetry Q

构造 state 与 swap。

target：

\[
Q(s)+Q(swap)^T
\]

应接近 0。

---

# 128. Distribution Symmetry

swap 后 target bin 必须 reverse。

---

# 129. Illegal Action

所有：

- policy loss
- NeuRD
- Nash solver
- opponent CE
- Q loss

都必须正确忽略非法 actions。

---

# 130. Teacher Priority

一个 state 同时提供：

- Bellman target
- Search target
- Exact target

最终 RobustQTarget 必须选择：

\[
Exact
\]

而不是平均。

---

# 131. CFR Priority

Policy 同时提供：

- Nash(Q)
- Search CFR
- Exact

选择：

\[
Exact
\]

作为 anchor。

NeuRD 仍计算。

---

# 132. Opponent Style

same opponent + same regime：

是 positive。

same opponent + different regime：

不能作为 positive。

---

# 133. Switch

给人工 sequence：

前半 policy A，

后半 policy B，

在已标记切换点：

switch BCE target 必须为 1。

---

# 第二十部分：训练日志必须输出什么

学习方法本身必须能够独立诊断。

每次 update 返回结构化 metrics。

Robust Q：

```text
q_loss
q_mean
q_std
q_target_mean
bellman_residual
teacher_fraction
solver_gap_mean
solver_gap_max
ensemble_disagreement
```

Actor：

```text
neurd_loss
nash_anchor_loss
policy_entropy
positive_regret_mean
negative_regret_mean
policy_nash_kl
```

Outcome：

```text
mc_ce
td_lambda_loss
value_mean
mc_value_error
win_prob_mean
draw_prob_mean
loss_prob_mean
```

Opponent：

```text
opp_fused_nll
opp_short_nll
opp_long_nll
opp_accuracy
opp_brier
opp_entropy
opp_ensemble_jsd
style_infonce
switch_bce
```

Adaptive：

```text
adaptive_q_loss
adaptive_policy_loss
adaptive_distribution_loss
delta_q_norm
opponent_confidence
adaptive_robust_q_distance
```

---

# 第二十一部分：明确禁止 Codex 自作主张的事项

以下全部属于硬约束。

## 禁止 1

不要把整个算法改成 PPO。

---

## 禁止 2

不要把 Robust Q target 改成 MC return。

---

## 禁止 3

不要把：

\[
Q(s,a,b)
\]

改成：

\[
Q(s,a)
\]

。

---

## 禁止 4

不要通过：

```python
q.mean(dim=-1).argmax()
```

替代 Nash matrix solving。

---

## 禁止 5

不要把 NeuRD loss 写成普通 policy-gradient：

```python
-log_prob * advantage
```

。

NeuRD 必须作用 raw logits。

---

## 禁止 6

不要把 CFR、NeuRD、PPO、R-NaD loss 全部加起来。

它们是不同 equilibrium-learning 方法。

主 Robust Actor：

\[
\boxed{NeuRD}
\]

CFR：

Teacher / Solver。

PPO、R-NaD：

独立 baseline backend。

---

## 禁止 7

不要添加默认 entropy regularization。

---

## 禁止 8

Opponent prediction 不能使用 Nash policy 作为 label。

只能使用真实 opponent action。

---

## 禁止 9

Adaptive loss 不能更新 Robust Backbone。

---

## 禁止 10

Adaptive loss 不能更新 LSTM/Mamba。

---

## 禁止 11

Opponent loss 不能更新 public backbone。

---

## 禁止 12

不要建立 learned transition/world model。

Goofspiel transition 已知。

---

## 禁止 13

不要擅自删除：

- MC
- TD(\(\lambda\))
- NeuRD
- Nash Bellman
- Exact/Search Teacher 接口
- opponent next-action prediction
- opponent style learning
- strategy-switch learning
- uncertainty/calibration
- adaptive value learning
- symmetry learning

这些不是“以后优化”。

它们属于完整学习设计。

---

# 第二十二部分：最终学习方法的完整逻辑

整个系统不是一个算法，而是多个语义严格不同的学习过程。

## 游戏规律学习

通过：

\[
\boxed{
\text{Full-Matrix Nash Bellman}
}
\]

学习：

\[
Q_R(s,a,b)
\]

回答：

> 如果这一轮双方这样出牌，之后双方都采取最优稳健策略，这个选择长期值多少？

---

## 均衡策略学习

通过：

\[
\boxed{
\text{NeuRD}
}
\]

根据 Q 导出的 regret/advantage 更新 mixed policy。

再通过：

\[
\boxed{
Nash/CFR/Exact\ Anchor
}
\]

限制 actor 漂移。

---

## 真实结果学习

通过：

\[
\boxed{
Monte\ Carlo
+
TD(\lambda)
+
Off\text{-}Policy\ Trace
}
\]

学习：

\[
Z_R
\]

回答：

> 当前实际策略从这里继续打，最终结果分布是什么？

---

## 对手动作学习

通过：

\[
\boxed{
Supervised\ Next\text{-}Action\ Prediction
}
\]

学习：

\[
q_L,\ q_M,\ q_F
\]

回答：

> 这个人下一轮大概出什么？

---

## 对手风格学习

通过：

\[
\boxed{
Long\text{-}Term\ Predictive
+
Contrastive\ Learning
}
\]

学习：

\[
z_{\text{style}}
\]

回答：

> 这个人长期采用什么行为模式？

---

## 对手变化学习

通过：

\[
\boxed{
Strategy\ Switch\ Detection
}
\]

回答：

> 他现在是不是已经改变打法？

---

## 对手不确定性

通过：

\[
\boxed{
Bootstrap\ Ensemble
+
Calibration
}
\]

回答：

> 我到底有多相信我的 opponent model？

---

## 针对性价值学习

通过：

\[
\boxed{
Opponent\text{-}Conditioned\ Bellman
+
Soft\ Best\ Response
}
\]

学习：

\[
Q_A
\]

回答：

> 如果这个 opponent 真像我预测的一样，我应该如何针对他？

---

# 结论

最终的学习系统不是：

\[
\boxed{\text{PPO}}
\]

也不是：

\[
\boxed{\text{Minimax-Q}}
\]

也不是：

\[
\boxed{\text{CFR}}
\]

任何一个算法单独统治整个系统。

我们的学习方法本质上是：

\[
\boxed{
\textbf{
Game-Theoretic Multi-Objective Learning
}
}
\]

其中：

\[
\text{Nash Bellman}
\]

负责学习**博弈后果**；

\[
\text{NeuRD / Regret Dynamics}
\]

负责学习**稳健混合策略**；

\[
\text{CFR / Exact / Search}
\]

负责提供**更强 game-theoretic teacher**；

\[
\text{MC + TD}(\lambda)
\]

负责学习**真实长期结果和时序一致性**；

\[
\text{Supervised Prediction}
\]

负责学习**对手下一动作**；

\[
\text{LSTM + Mamba + Contrastive Learning}
\]

负责学习**短期与长期 opponent style**；

\[
\text{Switch Detection}
\]

负责学习**非平稳对手变化**；

\[
\text{Ensemble + Calibration}
\]

负责学习/评估**自己的未知程度**；

\[
\text{Opponent-Conditioned Bellman}
\]

负责学习**针对具体对手的 exploit value**。

最高层目标仍然只有一个：

\[
\boxed{
\textbf{
更准确地预测，
更合理地优化，
最终做出更强的博弈决策。
}
}
\]

但达到这个目标并不意味着把所有知识都压进一个 reward scalar。

真正完整的智能体应该学会：

\[
\boxed{
\text{局面是什么}
}
\]

\[
\boxed{
\text{不同选择会发生什么}
}
\]

\[
\boxed{
\text{最强对手会怎么回应}
}
\]

\[
\boxed{
\text{当前这个具体对手可能怎么回应}
}
\]

\[
\boxed{
\text{这个对手长期是什么风格}
}
\]

\[
\boxed{
\text{他的风格有没有变化}
}
\]

\[
\boxed{
\text{我对这些判断有多确定}
}
\]

\[
\boxed{
\text{怎样随机化才不容易被针对}
}
\]

以及：

\[
\boxed{
\text{当对手真的有漏洞时，怎样利用它}
}
\]

这才是从我们之前五子棋体系真正继承过来的“学习方法”：不是迷信一种 RL 算法，而是让不同的学习机制分别学习智能博弈所需要的不同知识，然后保持它们的数学语义一致。