# Goofspiel-13 智能体完整训练流程实施规范
## Pre-training → Semi-Supervised → SFT → Game-Theoretic Post-training → Opponent Post-training → League → Red-Team Continual Improvement

---

# 0. 文档用途

本文规定：

> **已经设计好的 Goofspiel 模型，应该按照什么顺序、使用什么数据、调用什么学习算法、冻结哪些参数、如何生成教师标签、如何自我博弈、如何维护历史模型、如何蒸馏、如何进行红队纠错。**

本文是**训练流程执行规范**。

Codex 实现时不得自行：

- 删除训练阶段；
- 合并语义不同的数据池；
- 把所有 loss 一次 backward；
- 把整个训练流程简化成 PPO self-play；
- 跳过 pre-training；
- 跳过 semi-supervised；
- 跳过 SFT；
- 跳过 Teacher–Student；
- 跳过历史 Robust/Aggressive 模型；
- 跳过 Red Team correction；
- 将 self-play 作为唯一数据来源；
- 将 Exact Solver 仅作为 evaluator；
- 用单一 `best_model.pt` 代替不同能力 checkpoint。

---

# 1. 已冻结的上游设计

本文默认以下内容已经确定，不允许训练流程修改。

## 1.1 模型结构

公共游戏建模：

- Card Transformer
- Relational GNN
- Joint-Action Matrix CNN

对手时序建模：

- LSTM：局内短期行为
- Mamba：跨局长期行为

输出：

\[
Q_R(s,a,b)
\]

\[
\pi_R(a|s)
\]

\[
Z_R(s)
\]

\[
q_L(b)
\]

\[
q_M(b)
\]

\[
q_F(b)
\]

\[
Q_A(s,h,a,b)
\]

\[
\pi_A(a|s,h)
\]

\[
Z_A(s,h)
\]

以及 ensemble uncertainty。

---

## 1.2 已冻结的学习方法

Robust Q：

\[
\boxed{\text{Full-Matrix Nash Bellman}}
\]

Robust Actor：

\[
\boxed{\text{NeuRD + Nash/CFR Teacher Anchor}}
\]

真实 outcome：

\[
\boxed{\text{Monte Carlo + TD}(\lambda)}
\]

Off-policy trajectory：

\[
\boxed{\text{Joint V-trace}}
\]

Opponent action：

\[
\boxed{\text{Supervised next-action prediction}}
\]

Opponent style：

\[
\boxed{\text{Contrastive representation learning}}
\]

Opponent switch：

\[
\boxed{\text{Switch detection}}
\]

Adaptive Q：

\[
\boxed{\text{Opponent-conditioned Bellman}}
\]

Teacher：

- Exact
- CFR/Search
- EMA
- Ensemble
- Historical Strong Models

---

# 2. 总体训练哲学

整个系统不采用：

```text
Random initialization
→ Self-play
→ PPO
→ Done
```

而采用类似现代基础模型的：

```text
Game Corpus Construction
        ↓
Self-Supervised Pre-training
        ↓
Semi-Supervised Teacher Learning
        ↓
Strategic SFT
        ↓
Game-Theoretic RL Post-training
        ↓
Opponent-Adaptive Post-training
        ↓
League / Adversarial Co-evolution
        ↓
Red-Team Continual Correction
        ↺
```

核心思想：

\[
\boxed{
\text{先学世界与结构}
\rightarrow
\text{再学强策略长什么样}
\rightarrow
\text{再自主优化}
\rightarrow
\text{再通过对抗持续纠错}
}
\]

---

# 3. 最终训练系统必须维护的数据池

禁止把所有数据放进一个 Replay Buffer。

必须至少存在以下逻辑数据池。

---

## 3.1 `GameCorpus`

用途：

Self-supervised pre-training。

保存：

- public states；
- legal actions；
- joint actions；
- next states；
- rewards；
- full trajectory；
- opponent/session IDs；
- game boundaries。

可以来自：

- random play；
- heuristic play；
- historical agents；
- self-play；
- synthetic opponents。

它不要求具有高质量策略标签。

---

## 3.2 `ExactDataset`

用途：

最高质量数学监督。

保存：

```text
state
Q*
Nash row policy
Nash column policy
V*
solver precision
state complexity
```

来源：

Exact Solver。

---

## 3.3 `TeacherDataset`

用途：

半监督与 SFT。

保存：

```text
state
teacher_Q
teacher_policy
teacher_value
teacher_source
teacher_confidence
teacher_disagreement
```

teacher_source：

```text
EXACT
CFR
SEARCH
ENSEMBLE
EMA
HISTORICAL
```

---

## 3.4 `RobustTrajectoryBuffer`

用途：

- MC；
- TD(λ)；
- NeuRD；
- Robust self-play analysis。

必须保存：

```text
state
self_action
opp_action
reward

behavior_policy_self
behavior_policy_opp

action_prob_self
action_prob_opp

final_score_diff

model_version
opponent_version

N
round
```

---

## 3.5 `OpponentSessionBuffer`

必须保留完整 session 层次，而不是把单回合打散。

结构：

```text
OpponentSession
  ├── opponent_id
  ├── strategy_regime_id
  ├── Game 1
  │     ├── Round ...
  ├── Game 2
  ├── ...
```

用于：

- LSTM；
- Mamba；
- style；
- switch detection。

---

## 3.6 `AdaptiveTrajectoryBuffer`

只能放使用 opponent-conditioned decision 产生的数据。

禁止与 Robust trajectory 混淆。

---

## 3.7 `FailureBuffer`

保存：

- exploiter 打穿 Main 的状态；
- Q 大误差；
- Search 与裸模型强烈不一致；
- opponent prediction catastrophic failure；
- regression failures。

---

## 3.8 `ReanalysisBuffer`

保存被最新：

- Exact；
- Search；
- CFR；
- Teacher Ensemble

重新标注后的历史状态。

---

# 4. 训练阶段不是互相替代

训练阶段为：

\[
P0,P1,\dots,P7
\]

但进入后期以后：

> 前期数据不会完全消失。

例如进入 Post-training 后仍然保留：

- Exact anchors；
- pretraining consistency；
- SFT teacher；
- MC trajectories。

这是：

\[
\boxed{\text{Progressive Activation}}
\]

而不是：

\[
\boxed{\text{Train-and-Forget}}
\]

---

# Phase 0：环境、Solver 与数据系统校准

---

# 5. P0 目标

在训练任何神经网络前证明：

1. Environment 正确；
2. Exact Solver 正确；
3. Batched Nash Solver 正确；
4. Player swap 正确；
5. Dataset serialization 正确；
6. Bitmask transition 正确。

---

# 6. P0-1 Environment Verification

测试：

### Tie

相同 bid：

双方：

\[
reward=0
\]

奖品丢弃。

### Win

若：

\[
a>b
\]

：

\[
r=p/S_N
\]

。

### Lose

：

\[
-r
\]

。

### Card removal

joint action 后：

自己和对手对应牌必须消失。

### Prize removal

当前 prize 必须消失。

---

# 7. P0-2 Exact Solver Verification

对：

\[
N=1,2,3
\]

构造人工可验证游戏。

确认：

- terminal；
- chance averaging；
- matrix Nash；
- recursion；
- symmetry。

满足：

\[
F(A,B,R)
=
-F(B,A,R)
\]

。

---

# 8. P0-3 GPU Matrix Solver Verification

生成：

\[
10,000
\]

个随机：

\[
2\times2,\dots,13\times13
\]

zero-sum matrix。

Reference：

LP FP64。

GPU solver：

训练使用 solver。

测：

\[
|V_{\text{GPU}}-V_{\text{LP}}|
\]

以及 duality gap。

默认准入：

```text
mean value error < 1e-3
95% value error < 5e-3
mean duality gap < 5e-3
```

不达标：

禁止启动神经训练。

---

# 9. P0-4 Complexity Estimator

每次 Exact Solve 前必须调用：

```python
estimate(state_or_n)
```

返回：

```text
estimated_states
estimated_matrix_games
estimated_runtime
estimated_memory
risk_level
```

禁止直接：

```python
if n <= 7:
    solve()
```

。

---

# Phase 1：Game Representation Pre-training

---

# 10. P1 核心目标

这一阶段不以：

\[
win\ rate
\]

作为主要优化目标。

目标：

> 让 Transformer、GNN、Matrix CNN、LSTM、Mamba 首先理解 Goofspiel 状态、资源、关系、时间结构和对手行为。

类似基础模型：

\[
\boxed{\text{Pre-training}}
\]

---

# 11. P1 数据来源

初始 GameCorpus 按以下比例生成：

```text
35% random legal play
35% parameterized heuristic play
20% mixed heuristic-vs-random
10% exact-state trajectories
```

必须覆盖：

\[
N=3\dots13
\]

。

N=13 必须从训练初期出现。

禁止：

> “学完 N=7 以后才允许第一次看到 N=13。”

---

# 12. N Sampling

初始默认：

```text
N=3..6     35%
N=7..10    35%
N=11..12   15%
N=13       15%
```

以后逐渐向 N=13 转移。

---

# 13. P1 自监督任务 A：Player Swap

给 state：

\[
s
\]

产生：

\[
s^{swap}
\]

训练：

\[
Q(s)
\approx
-Q(swap)^T
\]

以及 representation-level consistency。

这一任务必须启用。

---

# 14. P1 自监督任务 B：Known-Transition Prediction

增加 training-only auxiliary head：

输入：

\[
s,a,b,p'
\]

预测：

- 下一 self mask；
- 下一 opponent mask；
- 下一 prize mask；
- immediate score change。

注意：

\[
\boxed{\text{这不是 learned world model}}
\]

真正训练和推理永远使用规则环境。

此 head 仅用于 representation pre-training。

Pretraining 完成后允许丢弃 auxiliary head。

---

# 15. Transition Loss

牌 mask：

Binary Cross Entropy。

score delta：

Huber。

总：

\[
L_{transition}
=
L_{self-mask}
+
L_{opp-mask}
+
L_{prize-mask}
+
0.5L_{score}
\]

。

---

# 16. P1 自监督任务 C：Immediate Joint Outcome

给：

\[
s,a,b
\]

预测：

```text
SELF_WIN
TIE
SELF_LOSE
```

以及：

\[
r_t
\]

。

目的：

强化 Matrix CNN 理解：

\[
(a,b)
\]

局部战略关系。

---

# 17. P1 自监督任务 D：Masked History Action

只针对 temporal branch。

随机选择过去 opponent action：

\[
b_\tau
\]

mask。

利用：

- public history；
- 周围序列；
- session context

预测：

\[
b_\tau
\]

。

不要对当前 public state 的确定性资源信息随意做 generic MAE。

Mask 的对象必须是：

\[
\boxed{\text{temporal behavioural information}}
\]

。

---

# 18. P1 自监督任务 E：Future Opponent Behaviour

LSTM：

预测下一 round：

\[
b_{t+1}
\]

。

Mamba：

基于历史 completed games 预测：

- 下一局平均 high-card usage；
- aggressive index；
- 当前 opponent 下一动作 prior。

这些 auxiliary targets 都必须从实际轨迹自动计算。

---

# 19. P1 对手 Style Contrastive

同一：

```text
opponent_id
strategy_regime_id
```

不同时间窗口：

Positive。

不同 regime：

Negative。

使用：

\[
InfoNCE
\]

。

---

# 20. P1 梯度

此阶段允许更新：

### Public

- Rank Encoder
- Transformer
- GNN
- Matrix CNN

### Opponent

- LSTM
- Game Summary
- Mamba
- Fusion

暂时不重点训练：

- Adaptive Q
- Adaptive Policy

Adaptive branch保持冻结。

---

# 21. P1 总 Loss

不要把权重散落代码。

默认：

```yaml
pretrain:
  transition: 1.0
  joint_outcome: 0.5
  symmetry: 0.5
  masked_history: 0.5
  future_behavior: 0.5
  style_contrastive: 0.25
```

---

# 22. P1 退出条件

不是固定训练多少 step。

至少满足：

### Transition

card mask accuracy：

\[
>99.5\%
\]

### Immediate outcome

\[
>99.5\%
\]

### Opponent synthetic validation

next-action NLL 相比 uniform 明显下降。

### Symmetry

误差稳定。

### N-generalization

N=11/13 representation tasks 不出现崩溃。

达到后进入 P2。

---

# Phase 2：Teacher-Based Semi-Supervised Learning

---

# 23. 为什么需要 P2

此时大量状态：

\[
N=13
\]

无法 Exact Solve。

但我们已经有：

- Exact labels；
- EMA；
- Ensemble；
- Search；
- CFR。

所以将状态分成：

```text
strong-label
weak-label
unlabelled
```

自然进行：

\[
\boxed{\text{Semi-Supervised Learning}}
\]

。

---

# 24. Teacher 等级

严格：

### T0

Exact Solver。

最高级。

### T1

高精度 CFR / Search。

### T2

Teacher Ensemble。

### T3

EMA teacher。

禁止低等级覆盖高等级。

---

# 25. Teacher Ensemble

至少：

\[
K=4
\]

Teacher snapshots。

来源可以包括：

- EMA；
- 当前模型不同 ensemble heads；
- Historical strong snapshot。

计算：

\[
\bar Q
=
\frac1K
\sum Q_k
\]

以及：

\[
Var(Q)
\]

。

Policy 同理。

---

# 26. Pseudo-label 接收规则

只有同时满足：

\[
U_Q<\tau_Q
\]

\[
KL_{\text{teacher}}<\tau_\pi
\]

才接受 pseudo-label。

默认初始化：

```yaml
pseudo_label:
  q_disagreement_max: 0.02
  policy_jsd_max: 0.05
```

超过阈值：

不要产生伪真值。

样本标记：

```text
HARD_UNCERTAIN
```

以后交给 Search/CFR。

---

# 27. 半监督 Consistency

没有 teacher label 的状态仍然可以训练：

### swap consistency

### padding invariance

### dropout/ensemble consistency

### temporal consistency

但禁止：

> “教师不确定时仍把教师平均值当真”。

---

# 28. P2 Teacher–Student 结构

Teacher：

不通过 Student 当前梯度直接更新。

Student：

正常反向传播。

EMA：

\[
\theta_T
\leftarrow
\beta\theta_T+
(1-\beta)\theta_S
\]

默认：

\[
\beta=0.999
\]

。

---

# 29. Teacher 不能无限自我强化错误

必须执行三个保护机制。

## 29.1 Exact Anchoring

每一个 teacher training cycle 中必须混入 ExactDataset。

---

## 29.2 Disagreement Filtering

教师不一致：

不 pseudo-label。

---

## 29.3 Regression against Exact

每次 Teacher 更新后：

重新在 fixed exact validation set 上评测。

如果 Exact Q error 明显退化：

Teacher checkpoint 不晋级。

---

# 30. P2 输出

训练得到：

\[
\boxed{\text{Pretrained Strategic Student}}
\]

它应该已经能：

- 近似小 N Q；
- 输出合理 mixed policy；
- 理解 N=13 状态；
- 具备基础 opponent representation。

但此时还没有完成真正 game-theoretic RL。

---

# Phase 3：Strategic Supervised Fine-Tuning

---

# 31. P3 对应 LLM 的什么

对应：

\[
\boxed{\text{SFT}}
\]

。

Pretraining：

学习“游戏语言”。

SFT：

明确告诉模型：

> **高质量博弈行为是什么样。**

---

# 32. SFT Dataset

必须包含四类。

---

## 32.1 Exact Strategic SFT

数据：

\[
s
\rightarrow
Q^*,\pi^*,V^*
\]

。

---

## 32.2 Search/CFR SFT

高质量：

\[
Q^{search}
\]

\[
\pi^{search/CFR}
\]

。

---

## 32.3 Opponent Behaviour SFT

：

\[
(s,h)\rightarrow b_{opp}
\]

以及：

- style metadata；
- switch metadata。

---

## 32.4 High-Confidence Pseudo SFT

仅使用 P2 通过 confidence filtering 的数据。

---

# 33. P3 模块更新

## Robust SFT

更新：

- public backbone；
- robust Q；
- robust policy；
- distribution head。

## Opponent SFT

独立 update。

不能与 Robust 一个总 loss。

## Adaptive

此时仍保持冻结或只做极轻量预热。

正式 Adaptive Training 在 P5。

---

# 34. Teacher 优先级

如果一个 state 同时存在：

Exact + Search + pseudo：

使用：

\[
Exact
\]

。

Search + pseudo：

使用：

\[
Search
\]

。

禁止平均。

---

# 35. SFT 中仍然保留 Pretraining Anchors

每个 SFT batch mixture 默认：

```text
70% Strategic SFT
20% Exact anchors
10% Self-supervised consistency
```

防 representation 被策略标签完全挤坏。

---

# 36. P3 退出条件

至少满足：

### Small-N

Exact Q error 达标。

### Policy

对 exact/search teacher KL 达标。

### Opponent

在 synthetic held-out opponent 上：

- NLL；
- Brier；
- calibration；

达到可用水平。

### N=13

面对基础 heuristic 不再接近随机水平。

然后进入 P4。

---

# Phase 4：Game-Theoretic RL Post-training

---

# 37. P4 才是真正主 RL 阶段

启动：

\[
\boxed{\text{Full-Matrix Nash Bellman}}
\]

\[
\boxed{\text{NeuRD}}
\]

\[
\boxed{\text{MC}}
\]

\[
\boxed{\text{TD}(\lambda)}
\]

以及真正的 self-play。

---

# 38. P4 Self-Play Mode

双方使用：

\[
\pi_R
\]

mixed sampling。

禁止：

```python
argmax(policy)
```

作为训练默认行为。

Mixed strategy 是博弈策略本身，不只是 exploration。

---

# 39. P4 State 来源

初期：

```text
50% current robust self-play
20% exact/small-N
15% teacher/search
15% random reachable coverage
```

随着训练成熟：

逐渐提高真实 N=13 self-play。

---

# 40. Progressive Mixture Curriculum

禁止：

```text
N=3 完成
→ N=4
→ ...
→ N=13
```

完全封锁式课程。

从 P1 开始一直允许 N=13。

但比例逐渐提高。

例如：

### Early

```text
easy/exact      50%
intermediate    30%
N=13            20%
```

### Mid

```text
easy/exact      25%
intermediate    30%
N=13            45%
```

### Mature

```text
easy/exact      10–20%
N=9..12         20–30%
N=13            50–65%
```

具体由 evaluator 动态调整。

---

# 41. Remaining-Horizon Curriculum

同样维护：

\[
error(k)
\]

其中：

\[
k=1\dots13
\]

。

采样：

\[
P(k)
\propto
(error(k)+\epsilon)^\alpha
\]

默认：

\[
\alpha=0.7
\]

。

同时设置最小概率：

\[
P(k)\ge0.02
\]

防止遗忘某 horizon。

---

# 42. Competence-Based Progression

不要按：

```text
100k updates 后增加难度
```

。

使用：

- Bellman residual；
- exact Q error；
- exploitability；
- policy teacher KL；

决定是否提高困难状态比例。

---

# 43. Exact Solver 在 P4 仍然持续存在

任何训练 state：

先调用 complexity estimator。

若 exact solve 在预算内：

优先使用 Exact target。

所以：

\[
\boxed{\text{Exact Teacher 永不下线}}
\]

。

---

# Phase 5：Opponent-Adaptive Post-training

---

# 44. P5 启动条件

Adaptive branch 不能过早开启。

必须满足 opponent predictor：

### NLL

显著优于 uniform。

### Brier

达到设定阈值。

### ECE

默认：

\[
<0.05
\]

。

### Switch benchmark

明显优于随机。

否则：

\[
Q_A
\]

继续冻结。

---

# 45. P5 Opponent Curriculum

必须从可控到困难。

---

## O1：Fixed Synthetic Styles

例如：

- aggressive；
- conservative；
- proportional；
- low-bid；
- high-bid；
- threshold；
- stochastic。

---

## O2：Parameterized Styles

定义参数：

\[
\theta_{opp}
\]

随机生成大量不同 policy。

不要只写 6 个固定 bot。

---

## O3：Historical Neural Agents

使用不同 checkpoint。

---

## O4：Cross-game Adaptive Opponents

对手根据前局结果调整。

---

## O5：Within-game Switching

一局中改变策略。

---

# 46. Opponent Session Construction

训练 Mamba 时必须提供：

\[
\text{多个连续 games}
\]

。

不能把游戏随机 shuffle 后再声称是在学长期 opponent style。

---

# 47. LSTM/Mamba 分工训练

LSTM：

当前局 round sequence。

Mamba：

只接收 completed-game summaries。

一轮结束：

更新 LSTM。

整局结束：

生成 game summary → Mamba。

然后 reset LSTM。

禁止每轮更新 Mamba long-term memory。

---

# 48. Robust 与 Adaptive Trajectory 分池

每局创建前确定：

```text
mode=ROBUST
```

或者：

```text
mode=ADAPTIVE
```

。

ROBUST：

不能训练 \(Z_A\)。

ADAPTIVE：

不能直接作为 Robust MC trajectory。

---

# 49. Oracle Opponent Experiment

对于 synthetic opponent：

系统知道真实：

\[
q^*(b|s)
\]

。

训练/评估时同时计算两种 adaptive performance：

### Predicted belief

使用：

\[
q_\phi
\]

### Oracle belief

使用：

\[
q^*
\]

。

两者差距：

\[
Gap_{\text{belief}}
\]

用来判断：

> Adaptive 不强到底是 opponent model 错，还是 policy/value 错。

必须保留这个诊断。

---

# Phase 6：League、历史版本与对抗共同进化

---

# 50. P6 不是一个简单 Historical Pool

正式建立三条 lineage。

---

# 51. Robust Lineage

：

\[
R_1,R_2,\dots,R_t
\]

准入标准：

主要看：

- exploitability；
- exact error；
- cross-play robustness。

目标：

\[
\boxed{\text{越来越难被针对}}
\]

。

---

# 52. Aggressive Lineage

：

\[
A_1,A_2,\dots,A_t
\]

目标：

针对：

- 某 opponent class；
- 某 historical robust；
- 某 style cluster；

最大化 exploitation。

它不要求 global Nash-safe。

---

# 53. Exploiter / Red-Team Lineage

：

\[
E_1,E_2,\dots
\]

唯一目标：

\[
\boxed{
\max
U(E,R_{main})
}
\]

。

允许：

- 极端策略；
- 只针对一个漏洞；
- 对其他 agent 很弱。

---

# 54. 三者不能混淆

Robust：

> 我要整体不容易输。

Aggressive：

> 我要善于占已知 opponent 的便宜。

Exploiter：

> 我要专门寻找 Main 的漏洞。

Codex 必须使用：

```text
agent_role = ROBUST
agent_role = AGGRESSIVE
agent_role = EXPLOITER
```

明确区分。

---

# 55. Historical Snapshot Admission

不能每隔固定 step 全部保存。

候选 checkpoint 必须满足至少一个：

### Robust

exploitability 明显提升。

### Strategic Novelty

与现有历史 policy 距离足够大。

### Aggressive Specialty

对某 opponent class 有明显新优势。

### Exploiter

能找到 Main 已知 pool 中没人找到的 exploit。

---

# 56. Policy Distance

可使用 held-out state set：

\[
D_{probe}
\]

计算：

\[
D(\pi_i,\pi_j)
=
E_s[
JSD(\pi_i(s),\pi_j(s))
]
\]

。

如果：

\[
D<\epsilon
\]

且性能无明显提升：

不加入永久历史池。

---

# 57. Opponent Sampling

禁止 uniform historical sampling。

应使用优先度。

默认综合：

\[
Priority_i
=
w_1Challenge_i+
w_2Novelty_i+
w_3MetaWeight_i+
w_4Staleness_i
\]

。

其中：

### Challenge

当前 Main 对它不是碾压。

### Novelty

策略差异。

### MetaWeight

meta-game relevance。

### Staleness

很久没打，适当增加概率。

---

# 58. PFSP 风格采样

对 win rate：

\[
w_i
\]

可以定义：

\[
f(w_i)
=
w_i(1-w_i)
\]

这样：

\[
w_i\approx0.5
\]

的 opponent 权重最大。

但保留一定概率采：

- 很强 opponent；
- historical nemesis；
- exploiter。

---

# 59. Robust 与 Aggressive 交叉教学

允许：

### Aggressive → Robust

发现新 tactic 后：

交给 Search/CFR 验证。

若该 tactic 在 robust constraints 下仍合理：

进入 TeacherDataset。

---

### Robust → Aggressive

Robust 的游戏基础 representation/Q 可以作为 Aggressive teacher。

但 Aggressive 仍允许针对性偏离。

---

# 60. Teacher–Student 持续存在

后期 teacher pool 包含：

```text
Exact Solver
CFR/Search
EMA Main
Robust historical
Aggressive specialist
Ensemble
```

---

# 61. Strong Student

主/full-size student。

蒸馏：

- Q；
- Policy；
- Value distribution；
- Opponent prediction。

追求：

最高纯网络性能。

---

# 62. Fast Student

小型低延迟模型。

Teacher：

Strong Student + Search targets。

蒸馏至少：

\[
L=
L_Q+
L_\pi+
L_Z+
L_{opp}
\]

。

Fast Student 不参与改变 Main 的核心知识。

它是部署/快速推理模型。

---

# 63. Distillation 不只在训练最后做一次

蒸馏必须是持续过程。

每次 Main/Search/Teacher 明显升级以后：

产生新的 distilled student。

---

# 64. Adversarial Opponent Generator

最终满血训练可以启用。

生成器：

\[
G_\omega(z)
\rightarrow
\theta_{opp}
\]

其中：

\[
\theta_{opp}
\]

控制参数化 opponent policy。

Generator 目标：

\[
\max_\omega
Loss(Main,\pi_{\theta_{opp}})
\]

同时加入 diversity：

\[
L_{div}
\]

防止所有 z 生成同一个 exploiter。

---

# 65. Generator 安全约束

生成 opponent 必须：

- 只使用合法 action；
- 不窥视当前 simultaneous hidden action；
- 不访问 Main 内部私有 logits；
- 只能通过公开 state/history 行动。

禁止 adversary 获得现实游戏中不存在的信息。

---

# Phase 7：Red-Team Continual Correction

---

# 66. Red Team 的最终职责

不是：

> “打败 Main 一次。”

而是形成：

\[
\boxed{
Attack
\rightarrow
Diagnose
\rightarrow
Relabel
\rightarrow
Correct
\rightarrow
Regression
}
\]

闭环。

---

# 67. Step R1：Attack

Red Team / Exploiter 与 Main 对局。

记录：

- trajectory；
- Q predictions；
- actor policies；
- uncertainty；
- opponent belief；
- final outcome。

---

# 68. Step R2：Failure Detection

Failure 不只看输没输。

定义候选：

### Unexpected Loss

预测很乐观但实际输。

### Exploit Repetition

同一 strategy pattern 反复击败 Main。

### Q Mismatch

Search/Exact：

\[
|Q_{\text{Main}}-Q_{\text{Teacher}}|
\]

大。

### Policy Regret

Teacher 表明 Main 某动作 regret 很高。

### Opponent-model Failure

预测极度自信但连续判断错。

---

# 69. Step R3：Failure Localization

不要整局 13 个 state 全部同优先级。

从最后往前寻找：

\[
t^*
\]

使 Main 与 Teacher 第一次显著分叉。

例如：

\[
KL(\pi_{Main},\pi_{Teacher})>\tau
\]

或：

\[
|V_{Main}-V_{Teacher}|>\tau_V
\]

。

以：

\[
s_{t^*}
\]

附近为核心 failure window。

---

# 70. Step R4：Strong Relabel

按顺序：

### 能 Exact

Exact。

### 不能 Exact

高预算 CFR/Search。

### Search 仍不稳定

Teacher Ensemble。

记录：

```text
failure_type
teacher_source
teacher_confidence
```

。

---

# 71. Step R5：Correction Dataset

将 failure 及邻近 state 放入：

\[
D_{\text{correction}}
\]

。

同时进行 player-swap augmentation。

不要只保存打败 Main 的对手动作。

要保存完整：

\[
Q/\pi/V
\]

teacher target。

---

# 72. Step R6：Focused Correction

训练 mixture：

```text
50% correction samples
25% ordinary robust data
15% exact anchors
10% generic teacher/SFT
```

短暂 focused fine-tuning。

目的：

修漏洞但防止 catastrophic regression。

---

# 73. Step R7：Original Attack Regression

训练完成后，原 Exploiter 必须重打。

要求：

原 failure 指标明显下降。

否则：

该 correction 不算成功。

---

# 74. Step R8：General Regression

再测试：

- exact benchmark；
- cross-play；
- historical robust；
- standard heuristic；
- opponent calibration。

如果修一个漏洞导致广泛能力退化：

拒绝该 checkpoint。

---

# 75. Failure 不能训练完就删

每个已修复 failure：

进入：

\[
\boxed{\text{Permanent Regression Suite}}
\]

。

未来每个 Main 候选都必须重新过。

这和软件工程单元测试思想完全一致。

---

# 76. Reanalyse

旧数据定期使用最新：

- Main；
- Target；
- Search；
- CFR；
- Exact

重新标记。

优先 reanalyse：

### High Priority

- failures；
- uncertainty 高；
- high visitation；
- early game；
- historical disagreement。

---

# 77. Reanalyse 不重新执行真实环境轨迹

对于 Robust Q：

只需要 compact state。

重新展开：

\[
(a,b,p')
\]

即可生成新 Bellman target。

对于 MC outcome：

必须保留原真实终局结果。

不要用新模型覆盖真实 MC 标签。

---

# 78. Mature Training：所有机制进入长期混合

成熟阶段不是：

```text
P1 off
P2 off
P3 off
P4 on
```

而是：

```text
Self-supervised anchors        ON
Exact teacher                 ON
Semi-supervised teacher       ON
Strategic SFT                 ON
Game-theoretic RL             ON
Opponent learning             ON
Adaptive learning             ON
League                        ON
Red team                      ON
Reanalyse                     ON
Distillation                  ON
```

但采样比例不同。

---

# 79. Mature 默认 Update Mix

作为初始配置：

```yaml
mature_training:

  robust_rl: 0.35
  strategic_sft: 0.15
  exact_anchor: 0.10
  self_supervised: 0.05

  opponent_learning: 0.10
  adaptive_learning: 0.10

  failure_correction: 0.10
  distillation: 0.05
```

这表示训练更新预算比例，不是 dataset 内样本比例。

Evaluator 可以动态调节。

---

# 80. 动态调整原则

如果：

### Exact Q 退化

提高：

`exact_anchor`

### Representation generalization 退化

提高：

`self_supervised`

### Exploitability 高

提高：

`robust_rl + failure_correction`

### Opponent NLL 高

提高：

`opponent_learning`

### Adaptive Oracle Gap 高

提高：

`adaptive_learning`

### Student 落后 Teacher

提高：

`distillation`

---

# 81. Search 什么时候运行

Search 不应该平均浪费在所有 state。

定义 Search Priority：

\[
P_{search}
=
w_UU_Q
+w_DD_{\pi,Q}
+w_FF(s)
+w_II(s)
\]

其中：

### \(U_Q\)

Q ensemble uncertainty。

### \(D_{\pi,Q}\)

Actor 和 Nash(Q) disagreement。

### \(F(s)\)

failure relevance。

### \(I(s)\)

strategic importance。

---

# 82. Strategic Importance

可以包括：

- early game；
- high prize；
- high future prize mass；
-高 policy entropy；
- cross-play divergence state。

---

# 83. Teacher Search 预算

例如：

```text
LOW      128 simulations/iterations
MEDIUM   512
HIGH     2048
CRITICAL 8192 / Exact if possible
```

具体 Search 算法后续单独规定。

---

# 84. Checkpoint 类型

禁止只有：

```text
best.pt
```

必须至少：

### `latest`

当前 learner。

### `best_robust`

最低 exploitability。

### `best_raw`

无 search 纯网络综合最强。

### `best_search`

带 search 最强。

### `best_adaptive`

adaptive benchmark 最强。

### `best_generalization`

cross-N 最强。

### `best_opponent_model`

opponent calibration 最好。

### `teacher_ema`

EMA Teacher。

---

# 85. Checkpoint 内容

必须保存：

```text
model_state
target_model_state
opponent_target_state

optimizer_states

training_stage
global_step
policy_version

configs

league_metadata

exact_eval
exploitability_eval
crossplay_eval
opponent_eval

RNG_state

git_commit
```

---

# 86. Evaluator 与 Learner 必须分开

Learner 不得根据 training loss 自己宣布：

> “我是 best model。”

Evaluator 使用冻结 checkpoint。

---

# 87. Evaluator 固定任务

## Exact

小 N /可解残局。

## Cross-N

\[
N=3\dots13
\]

。

## Held-Out N Generalization

例如训练阶段临时保留某些 N/state distribution。

## Historical Cross-play

与历史 Robust/Aggressive。

## Exploiter

Best response approximations。

## Heuristic

sanity benchmark。

## Opponent Prediction

held-out opponents。

## Switch Detection

held-out switching sessions。

## Adaptive

predicted-belief 与 oracle-belief。

## Search Gain

raw vs search。

---

# 88. Main Candidate 晋级条件

Main checkpoint 只有满足：

1. Exact 不 regression；
2. exploitability 不 regression；
3. cross-play 有改善；
4. regression suite 通过；
5. 无数值异常；

才可晋级：

\[
R_{t+1}
\]

。

---

# 89. 不允许仅凭 Elo 晋级

Elo 可以记录。

但不是唯一指标。

因为：

\[
A>B,\quad B>C,\quad C>A
\]

时单标量 Elo 会隐藏 non-transitivity。

必须同时维护 payoff matrix。

---

# 90. 最终训练主循环

伪代码：

```text
initialize_environment()
verify_exact_solver()
verify_matrix_solver()

build_initial_game_corpus()

# --------------------------------
# P1 PRETRAIN
# --------------------------------

while not pretrain_ready():

    batch = sample_game_corpus()

    train_public_representation(batch)
    train_temporal_representation(batch)

    evaluate_pretrain_tasks()

save_pretrained_checkpoint()


# --------------------------------
# P2 SEMI-SUPERVISED
# --------------------------------

while not semi_supervised_ready():

    states = sample_states()

    strong_labels =
        query_exact_when_feasible(states)

    weak_labels =
        query_teacher_ensemble(states)

    accept only high-confidence pseudo labels

    train_student(
        strong_labels,
        pseudo_labels,
        consistency_data
    )

    update_ema_teacher()

    verify_against_exact_anchor()


# --------------------------------
# P3 STRATEGIC SFT
# --------------------------------

while not strategic_sft_ready():

    exact_batch = sample_exact()
    search_batch = sample_search_teacher()
    opponent_batch = sample_opponent_supervision()

    update_robust_sft()
    update_opponent_sft()

    retain_pretraining_anchors()


# --------------------------------
# P4 GAME-THEORETIC RL
# --------------------------------

activate_self_play()

while robust_training_active:

    trajectories =
        generate_robust_self_play()

    states =
        extract_states(trajectories)

    q_targets =
        exact_or_nash_bellman(states)

    update_robust_q(q_targets)

    update_neurd_actor(states)

    update_mc_td_outcome(trajectories)

    evaluate()


# --------------------------------
# P5 OPPONENT ADAPTATION
# --------------------------------

if opponent_model_is_calibrated():

    activate_adaptive_branch()

    generate_opponent_sessions()

    update_opponent_model()
    update_adaptive_q()
    update_adaptive_policy()
    update_adaptive_distribution()


# --------------------------------
# P6 LEAGUE
# --------------------------------

maintain:
    Robust lineage
    Aggressive lineage
    Exploiter lineage

sample strategically relevant opponents

perform:
    self-play
    cross-play
    specialist training
    distillation


# --------------------------------
# P7 RED TEAM
# --------------------------------

for each discovered failure:

    localize_failure()

    relabel_with(
        Exact
        or Search/CFR
        or Teacher Ensemble
    )

    add_to_correction_dataset()

    focused_correction()

    rerun_original_attack()

    run_general_regression()

    if all_pass:
        promote_checkpoint()
```

---

# 91. 多 GPU 情况下建议的逻辑角色

硬件具体拓扑后续工程文档再定。

但程序架构应允许：

```text
Actor Workers
Exact Workers
Search Workers
Reanalyse Workers
Learner
Opponent Learner
Evaluator
League Manager
```

彼此独立。

不要把所有逻辑塞进：

```python
train.py
```

一个大循环。

---

# 92. 训练系统代码结构

建议：

```text
training/
│
├── stages/
│   ├── stage0_verify.py
│   ├── stage1_pretrain.py
│   ├── stage2_semi_supervised.py
│   ├── stage3_sft.py
│   ├── stage4_robust_rl.py
│   ├── stage5_adaptive.py
│   ├── stage6_league.py
│   └── stage7_redteam.py
│
├── data/
│   ├── game_corpus.py
│   ├── exact_dataset.py
│   ├── teacher_dataset.py
│   ├── robust_trajectory.py
│   ├── opponent_session.py
│   ├── adaptive_trajectory.py
│   ├── failure_buffer.py
│   └── reanalysis_buffer.py
│
├── teachers/
│   ├── exact_teacher.py
│   ├── ema_teacher.py
│   ├── ensemble_teacher.py
│   ├── search_teacher.py
│   └── teacher_router.py
│
├── league/
│   ├── registry.py
│   ├── robust_lineage.py
│   ├── aggressive_lineage.py
│   ├── exploiter_lineage.py
│   ├── opponent_sampler.py
│   └── payoff_matrix.py
│
├── redteam/
│   ├── attacker.py
│   ├── failure_detector.py
│   ├── failure_localizer.py
│   ├── correction_dataset.py
│   └── regression_suite.py
│
├── distill/
│   ├── strong_student.py
│   └── fast_student.py
│
├── curriculum/
│   ├── n_scheduler.py
│   ├── horizon_scheduler.py
│   └── competence.py
│
├── evaluation/
│   ├── exact_eval.py
│   ├── crossplay.py
│   ├── exploitability.py
│   ├── opponent_eval.py
│   ├── adaptive_eval.py
│   └── regression.py
│
└── coordinator.py
```

---

# 93. 配置必须显式

建议顶层：

```yaml
training_pipeline:

  stage0_verify:
    enabled: true

  pretrain:
    enabled: true

  semi_supervised:
    enabled: true

  strategic_sft:
    enabled: true

  robust_posttrain:
    enabled: true

  opponent_posttrain:
    enabled: true

  league:
    enabled: true

  redteam:
    enabled: true

  reanalysis:
    enabled: true

  distillation:
    enabled: true
```

这些默认全部：

```text
true
```

。

不能为了“先跑通”默认把后六项关闭。

开发时允许单独测试，但完整配置必须全部存在。

---

# 94. Codex 禁止事项

## 禁止 1

禁止把 Pre-training 删除并直接随机初始化 self-play。

---

## 禁止 2

禁止用一个 `ReplayBuffer` 混所有数据。

---

## 禁止 3

禁止把 Exact Solver 只用于测试。

它必须参与：

- SFT；
- anchors；
- relabel；
- correction。

---

## 禁止 4

禁止 Semi-supervised 阶段无条件接受 pseudo-label。

必须使用：

\[
\text{confidence/disagreement filtering}
\]

。

---

## 禁止 5

禁止 Teacher 自己给自己打标签然后完全没有 Exact regression。

---

## 禁止 6

禁止 SFT 后永久关闭自监督 anchor。

---

## 禁止 7

禁止把 self-play 简化为：

```text
current vs current only
```

。

---

## 禁止 8

禁止历史模型全部作为同一类型。

必须区分：

- Robust
- Aggressive
- Exploiter

---

## 禁止 9

禁止 historical opponent uniform sampling 作为最终方案。

必须有 priority / PFSP / meta relevance。

---

## 禁止 10

禁止将 Aggressive 和 Exploiter 合并。

---

## 禁止 11

禁止 Adaptive branch 在 opponent model 未达到 calibration gate 前大规模训练。

---

## 禁止 12

禁止 Robust 与 Adaptive trajectory 不做标记。

---

## 禁止 13

禁止把 Red Team 输局简单塞 replay 就结束。

必须：

\[
Attack
\rightarrow
Diagnose
\rightarrow
Relabel
\rightarrow
Correction
\rightarrow
Regression
\]

。

---

## 禁止 14

禁止修复 failure 后删除。

必须永久加入 regression suite。

---

## 禁止 15

禁止只有一个 `best_model.pt`。

---

## 禁止 16

禁止只看胜率/Elo。

必须同时看：

- exact error；
- exploitability；
- payoff matrix；
- opponent calibration；
- cross-N generalization。

---

## 禁止 17

禁止将 Strong/Fast Student 视为“以后有空再做”。

完整训练体系必须保留 distillation 接口。

---

## 禁止 18

禁止 learned world model 替代真实环境。

Transition prediction 只允许作为：

\[
\boxed{\text{pretraining auxiliary task}}
\]

。

---

# 95. 最终训练体系的逻辑

整个系统最终形成两个持续转动的飞轮。

## Knowledge Flywheel

\[
\boxed{
Exact
\rightarrow
CFR/Search
\rightarrow
Teacher
\rightarrow
Student
\rightarrow
Better\ Model
\rightarrow
Better\ Search
}
\]

负责：

> **越来越聪明。**

---

## Adversarial Flywheel

\[
\boxed{
Main
\rightarrow
Aggressive/Exploiter
\rightarrow
Failure
\rightarrow
Strong\ Relabel
\rightarrow
Correction
\rightarrow
New\ Main
}
\]

负责：

> **越来越难被打穿。**

---

同时还有一个 Opponent Adaptation Flywheel：

\[
\boxed{
Opponent\ Session
\rightarrow
LSTM/Mamba
\rightarrow
Better\ Prediction
\rightarrow
Better\ Exploitation
\rightarrow
Harder\ Opponent
\rightarrow
Better\ Opponent\ Model
}
\]

负责：

> **越来越会读懂具体对手。**

---

# 96. 最终训练生命周期一句话定义

完整 Goofspiel Agent 不应该被训练成：

> “一个靠 self-play 强化学习慢慢学会出牌的网络。”

它应该经历：

\[
\boxed{
\textbf{
自监督学习游戏世界和行为结构
}
}
\]

↓

\[
\boxed{
\textbf{
利用 Exact / CFR / Search / Ensemble 进行半监督学习
}
}
\]

↓

\[
\boxed{
\textbf{
通过高质量数学与搜索教师进行 Strategic SFT
}
}
\]

↓

\[
\boxed{
\textbf{
通过 Nash Bellman、NeuRD、MC、TD(\lambda) 完成 Game-Theoretic Post-training
}
}
\]

↓

\[
\boxed{
\textbf{
通过 LSTM + Mamba 学习短期和长期对手行为并完成 Adaptive Post-training
}
}
\]

↓

\[
\boxed{
\textbf{
通过 Robust、Aggressive、Exploiter 三条历史谱系进行自我博弈与对抗共同进化
}
}
\]

↓

\[
\boxed{
\textbf{
通过 Teacher–Student 蒸馏获得 Strong Student 和 Fast Student
}
}
\]

↓

\[
\boxed{
\textbf{
通过 Red-Team Attack → Relabel → Correction → Regression 持续发现并修复失败模式
}
}
\]

并最终进入一个不会简单“训练结束”的：

\[
\boxed{
\textbf{
Continual Game-Intelligence Post-Training System
}
}
\]

这才是当前完整设计中“训练流程”的最终含义。