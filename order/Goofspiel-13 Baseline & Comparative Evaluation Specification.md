# Goofspiel-13 Baseline & Comparative Evaluation Specification
## 基线体系、算法分组、公平预算、实现规范与比较协议

---

# 0. 文档目的

本文冻结整个 Goofspiel 项目的 Baseline 体系。

Baseline 的作用不是：

> 找几个弱算法让主模型赢。

而是回答不同层次的问题：

\[
\boxed{\text{我们的系统究竟比什么强？为什么强？}}
\]

因此 Baseline 必须覆盖：

1. 非学习策略；
2. 传统 model-free RL；
3. joint-action / minimax RL；
4. game-theoretic learning；
5. neural game-theoretic learning；
6. search；
7. opponent adaptation；
8. 系统自身消融。

---

# 1. Baseline 分为两级

所有 baseline 必须标记：

```text
PRIMARY
REFERENCE
```

---

## 1.1 Primary Baseline

满足至少一个条件：

- 与我们的核心研究问题直接竞争；
- 是该技术路线最有代表性的算法；
- 能对我们的主要 claim 构成真正挑战；
- 必须进入论文/最终报告主表。

Primary Baseline 必须：

- 完整实现；
- 调参；
- 多 seed；
- 同预算；
- 做 exploitability；
- 做学习曲线；
- 保存 checkpoint；
- 可复现。

---

## 1.2 Reference Baseline

作用：

- sanity check；
- 历史参照；
- 特定组件比较；
- 辅助解释为什么某种 inductive bias 不适合。

可以进入 appendix / 次表。

但仍必须正确实现。

不得故意弱化。

---

# 2. Baseline 不能只放在一张表

整个比较体系划分为四个 Arena：

\[
\boxed{
\text{Arena A: Exact Small-N}
}
\]

\[
\boxed{
\text{Arena B: Neural N=13}
}
\]

\[
\boxed{
\text{Arena C: Search / Compute}
}
\]

\[
\boxed{
\text{Arena D: Opponent Adaptation}
}
\]

原因：

Tabular CFR 在 N=3 和完整 N=13 上的意义不同。

不能简单写：

```text
CFR       X
Our Model Y
```

而不说明：

- N；
-状态规模；
-训练预算；
-是否 tabular；
-是否使用 Search。

---

# 第一部分：Arena A —— Exact Small-N

# 3. 目的

这里回答：

> 在我们能够知道真正答案的情况下，各算法到底多快接近 Nash equilibrium？

主要：

\[
N=3,4,5
\]

条件允许时增加：

\[
N=6
\]

。

所有算法接受严格数学评测。

---

# 4. Arena A Primary Baselines

必须包括：

```text
Random
Exact Nash
Minimax-Q
CFR
CFR+
NeuRD
R-NaD
```

建议：

```text
DCFR
```

也进入主比较。

---

# 5. Arena A Reference Baselines

```text
Heuristic Suite
Independent DQN
PPO
IPPO
NFSP
Deep CFR
```

Deep CFR 在小 N 上并不是最有优势的场景，但可验证实现正确性。

---

# 第二部分：非学习 Baseline

# 6. Random

## 等级

\[
\boxed{\text{PRIMARY SANITY BASELINE}}
\]

不是因为它很强。

而是因为任何学习方法必须显著优于它。

---

# 7. Random Policy

在所有合法动作上：

\[
\pi(a|s)
=
\frac{1}{|A(s)|}
\]

。

必须真正 uniform。

禁止：

```text
random integer 1..N then retry
```

造成微妙 sampling bias。

---

# 8. Random 的作用

测：

-环境正确性；
-基础胜率；
- opponent predictor uniform baseline；
- search 最低参照。

---

# 9. Heuristic Suite

不能只有一个 heuristic。

至少建立以下固定集合：

### H1 Random

均匀随机。

### H2 Prize Matching

尽可能：

\[
a\approx p
\]

。

### H3 Proportional

根据当前 prize relative rank 映射到剩余 bid quantile。

### H4 Aggressive

高 prize 和中高 prize 都偏向使用更高牌。

### H5 Conservative

保留高牌，对多数局面低投入。

### H6 High-for-High

对当前最大剩余 prize 使用最大剩余牌。

### H7 Low-Card Saver / Sacrifice

主动放弃部分低价值 prize。

### H8 Parameterized Stochastic

例如：

\[
P(a|s)
\propto
\exp(\beta f(a,p,s))
\]

改变参数产生不同 style。

---

# 10. Heuristic 的等级

单个 heuristic：

\[
\boxed{\text{REFERENCE}}
\]

Heuristic Suite 作为 opponent diversity benchmark：

\[
\boxed{\text{PRIMARY EVALUATION SET}}
\]

。

也就是说：

不把“Prize Matching”作为论文主算法，

但是“模型能否应对多种固定行为模式”是正式评测。

---

# 11. Exact Nash

## 等级

\[
\boxed{\text{ORACLE}}
\]

不是普通 baseline。

它代表：

\[
\boxed{\text{数学真值}}
\]

。

只在 complexity 可承受状态使用。

---

# 12. Exact 的用途

计算：

- \(Q^*\) error；
- policy distance；
- value error；
- exploitability；
- convergence；
- sample efficiency。

Exact 不与 N=13 神经模型硬比较“训练速度”。

它是：

\[
\boxed{\text{Upper/Truth Reference}}
\]

。

---

# 第三部分：传统 RL

# 13. Independent DQN

## 等级

\[
\boxed{\text{REFERENCE}}
\]

。

---

# 14. 为什么要有

Independent DQN 学：

\[
Q(s,a)
\]

把 opponent/environment nonstationarity 当作环境的一部分。

这正好和我们的：

\[
Q(s,a,b)
\]

形成重要 inductive-bias 对照。

---

# 15. Independent DQN 实现

双方共享网络或独立网络都可以做。

主 baseline：

共享参数 self-play。

输入：

仅 public state。

输出：

\[
Q(s,a)
\]

合法 action mask。

行为：

epsilon-greedy。

---

# 16. DQN 不允许获得 opponent current action

否则已经不是 Independent DQN。

---

# 17. DQN 的研究问题

它回答：

> 对 simultaneous strategic interaction，显式建模 opponent joint action 是否真的比普通单动作 Q-learning 更合理？

---

# 18. PPO

## 等级

\[
\boxed{\text{PRIMARY}}
\]

。

这是必须认真做的通用 model-free RL baseline。

---

# 19. PPO 输入

只能：

- public state；
-合法 action mask。

不得：

-对手当前 action；
-Exact target；
-Search target；
-opponent history，除非另做 PPO+Memory reference。

---

# 20. PPO Policy

Categorical policy：

\[
\pi_\theta(a|s)
\]

合法 mask。

---

# 21. PPO Value

\[
V_\phi(s)
\]

。

Reward：

必须与主系统一致：

\[
r_t=
\frac{p_t}{S_N}
\operatorname{sgn}(a-b)
\]

\[
\gamma=1
\]

。

不能 PPO 用 WDL，而主系统用 score difference。

---

# 22. PPO Self-Play

必须同时测试：

### PPO Current-vs-Current

最朴素 self-play。

### PPO Historical Self-Play

如果要测试训练生态影响。

但是论文主 PPO baseline 应优先保持算法本身清晰。

---

# 23. PPO Backbone 公平性

提供两个版本：

### PPO-Small

标准 MLP/轻量 network。

### PPO-Matched

使用和主系统参数量大致相同的 public backbone。

这样可以区分：

> 是算法强，还是网络大。

---

# 24. MAPPO / IPPO

当前 TorchRL 已提供 `MAPPOLoss` 和 `IPPOLoss`，其中 MAPPO 使用 centralized critic，IPPO 是 independent critic，可直接作为实现参考。

等级：

\[
\boxed{\text{REFERENCE}}
\]

。

---

# 25. 为什么 MAPPO 不列最核心 Primary

Goofspiel 是：

- 两人；
-零和；
- simultaneous；
-公开剩余资源。

我们真正关心的是：

\[
\boxed{\text{equilibrium learning}}
\]

而不是合作型 MARL 中 centralized-training/decentralized-execution 的收益。

因此：

PPO 必须 Primary。

MAPPO/IPPO：

作为 MARL reference 即可。

---

# 第四部分：Minimax-Q

# 26. Minimax-Q

## 等级

\[
\boxed{\text{PRIMARY}}
\]

而且它可能是整个 baseline 中最重要的之一。

---

# 27. 原因

它和我们共享最核心的表示：

\[
Q(s,a,b)
\]

并且 Bellman continuation 使用：

\[
V(s')
=
\max_x\min_yx^TQ(s')y
\]

。

所以它直接回答：

> 我们复杂体系相比经典 Minimax-Q 到底多获得了什么？

---

# 28. Minimax-Q Baseline 必须保持纯粹

不得加入：

- Transformer+GNN+CNN 全部复杂模块；
-MC distribution；
-TD(\(\lambda\))；
-NeuRD；
-CFR teacher；
-opponent modeling；
-Search；
-Red Team。

否则已经开始变成主方法。

---

# 29. Minimax-Q 推荐网络

简单但足够强：

```text
Public State Encoder
→ MLP
→ Joint Q[13,13]
```

参数量可：

\[
0.5\sim2M
\]

。

另提供：

```text
Minimax-Q Matched Backbone
```

作为消融。

---

# 30. Minimax-Q 行动

必须由：

\[
MatrixNash(Q)
\]

产生。

禁止：

\[
Q.mean(-1).argmax
\]

。

---

# 31. Minimax-Q 两个版本

### M-Q Sampled

只更新执行 joint action。

更接近传统 model-free。

### M-Q Full-Matrix

利用已知 transition 对所有：

\[
(a,b)
\]

生成 target。

这是非常重要的比较。

它直接测：

\[
\boxed{
\text{Full-Matrix Model-Based Target}
}
\]

本身贡献多大。

---

# 第五部分：CFR Family

# 32. Vanilla CFR

## 等级

Arena A：

\[
\boxed{\text{PRIMARY}}
\]

N=13：

\[
\boxed{\text{REFERENCE / ORACLE-LIMITED}}
\]

。

---

# 33. CFR 必须在正确 information-set representation 上运行

Goofspiel 当前 action simultaneous。

如果使用 turn-based CFR implementation：

必须转换成等价 imperfect-information sequential game：

```text
P0 chooses hidden action
↓
P1 chooses without observing P0 action
↓
joint reveal
```

不能直接 sequentialize 后让 P1 看见 P0 动作。

OpenSpiel明确支持把 simultaneous game 转换成等价 turn-based representation；其文档给出了 `LoadGameAsTurnBased` / `turn_based_simultaneous_game(...)`。

---

# 34. 为什么这么做

OpenSpiel 的 CFR/MCCFR 等很多实现面向 sequential extensive-form game；例如 MCCFR 对 simultaneous state 会明确要求先转换成 turn-based simultaneous representation。

因此：

\[
\boxed{
\text{Sequential representation}
\neq
\text{sequential information leakage}
}
\]

必须保持 information set 正确。

---

# 35. CFR 输出

正式使用：

\[
\boxed{\text{Average Strategy}}
\]

而不是最后 iteration current strategy。

---

# 36. CFR 记录

横轴至少：

- iterations；
- traversed nodes；
- wall time。

纵轴：

\[
NashConv/Exploitability
\]

。

---

# 37. CFR+

## 等级

\[
\boxed{\text{PRIMARY}}
\]

。

这是传统 CFR family 的核心 baseline。

---

# 38. CFR+ 实现

优先直接复用/验证 OpenSpiel `CFRPlusSolver`。

OpenSpiel 的当前实现明确使用：

- Regret Matching+；
- alternating updates；
- linear averaging。

不要 Codex 根据博客重新手写一个“类似 CFR+”。

---

# 39. CFR vs CFR+

必须使用：

-相同 game representation；
-相同 stopping metrics；
-相同 exploitability evaluator。

这样比较才有意义。

---

# 40. DCFR

## 等级

\[
\boxed{\text{REFERENCE-STRONG}}
\]

并建议进入 Arena A 主表。

---

# 41. DCFR 实现注意

OpenSpiel 有 `DCFRSolver` / `LCFRSolver` 实现，但其源码明确提醒该贡献版本未确认复现论文结果。

因此：

如果直接使用：

必须：

1. pin commit；
2. small-game benchmark；
3. 与论文/独立实现数值检查。

不能因为“OpenSpiel 有”就完全不验证。

---

# 42. 为什么保留 DCFR

它可以回答：

> 主系统的 improved learning 是否只是因为使用了更先进 regret weighting？

---

# 第六部分：Deep CFR

# 43. Deep CFR

## 等级

\[
\boxed{\text{PRIMARY LARGE-GAME GAME-THEORETIC BASELINE}}
\]

。

---

# 44. 原因

Tabular CFR 大规模不可行以后，

Deep CFR 是非常自然的 neural approximation baseline。

它学习：

- advantage；
- average strategy；

而不是直接 \(Q_R\)。

---

# 45. OpenSpiel 可以直接参考

当前 OpenSpiel 有 PyTorch Deep CFR implementation，采用 advantage network + strategy network，并使用 memory buffer 存储训练样本。

优先：

- fork/reference；
-适配我们规则；
-验证信息集。

---

# 46. Deep CFR 不允许使用我们的 Exact/Search teacher

否则不再是 Deep CFR baseline。

它自己的 traversal/sampling 就是数据来源。

---

# 47. Deep CFR 参数预算

提供：

```text
DeepCFR-Small
DeepCFR-Matched
```

至少 Matched 版本 neural parameter count 与主网络同量级。

---

# 第七部分：NeuRD

# 48. NeuRD

## 等级

\[
\boxed{\text{PRIMARY}}
\]

而且必须有。

因为主系统本身使用 NeuRD actor dynamics。

---

# 49. NeuRD Baseline 的意义

不是证明：

> NeuRD 比我们的系统弱。

而是回答：

> 我们最终性能中，到底有多少只是 NeuRD 本身带来的？

---

# 50. NeuRD-Only

定义：

```text
Public State Encoder
→ Policy logits
→ NeuRD
```

不给：

- Exact teacher；
-Nash Bellman joint Q；
-MC distribution；
-Search；
-Opponent model；
-League 增强。

---

# 51. NeuRD + Simple Critic

为了得到 action advantages，可以实现论文/参考代码对应的合理 critic。

不能偷偷把我们的整个 Full-Matrix Nash Q 当给 NeuRD baseline。

---

# 52. OpenSpiel Reference

OpenSpiel 当前仓库仍有 PyTorch NeuRD implementation，可以直接作为算法实现与测试参考。

---

# 53. 另一个重要消融

除了“外部 NeuRD baseline”，还必须有：

```text
OUR_Q + ordinary policy gradient
OUR_Q + NeuRD
```

。

这才能证明：

\[
\boxed{\text{NeuRD actor choice}}
\]

自身是否有贡献。

---

# 第八部分：NFSP

# 54. NFSP

## 等级

\[
\boxed{\text{REFERENCE}}
\]

。

---

# 55. 为什么值得保留

NFSP 代表：

\[
\boxed{
\text{best-response RL}
+
\text{average-policy supervised learning}
}
\]

这一经典 neural fictitious-play 路线。

它和我们的：

- League；
-Historical；
-Robust/Exploit；

在思想上有联系。

---

# 56. OpenSpiel

OpenSpiel 当前有 PyTorch NFSP implementation，可作为 reference。

---

# 57. 为什么不是最高优先级

我们已经有：

- CFR；
-Deep CFR；
-NeuRD；
-R-NaD。

NFSP 的边际信息价值较低。

所以如果算力/开发时间有限：

先完成 Primary，再做 NFSP。

---

# 第九部分：R-NaD

# 58. R-NaD

## 等级

\[
\boxed{\text{PRIMARY}}
\]

。

这是必须认真比较的现代 game-theoretic neural baseline。

---

# 59. R-NaD 角色

它代表：

> 利用 regularization + Nash dynamics 进行神经 equilibrium learning 的另一条完整路线。

因此它与：

\[
\text{NeuRD}
\]

不是完全同一个 baseline。

---

# 60. R-NaD 实现策略

优先使用/适配：

OpenSpiel R-NaD reference implementation。

但是：

必须 pin 一个明确 commit。

原因是 OpenSpiel 历史 issue 中确实出现过 R-NaD/NeuRD loss、valid-action averaging 等实现细节修复记录，因此绝不能只写“装最新版然后跑”。

---

# 61. R-NaD Reproducibility

保存：

```text
open_spiel_commit
rnad_config
legal_action_handling
reward_transform
vtrace parameters
neurd clip
```

。

否则几年后几乎无法复现实验。

---

# 62. R-NaD 两档

### R-NaD Reference

尽量接近 reference implementation。

### R-NaD Matched

把 neural capacity 调整到和主系统相近。

主表优先报告：

Matched。

Appendix 同时报告 Reference。

---

# 第十部分：Search Baselines

# 63. Search 必须完全单独比较

不能：

```text
Our Agent + 5000 search
vs
PPO no search
```

然后说算法强。

---

# 64. Search Arena 固定 Base Model

必须选一个冻结：

\[
\theta_{base}
\]

然后比较：

```text
Network Only
Matrix Nash Only
Reference Simultaneous Search
SM-MCTS
GT-CFR Search
SM-MCTS + Exact Leaves
GT-CFR + Exact Leaves
```

。

所有 Search：

使用同一个 neural evaluator。

---

# 65. Network Only

## Primary

\[
\boxed{\text{YES}}
\]

。

它代表：

> 完全不花在线 planning compute 的能力。

---

# 66. Matrix Nash Only

输入：

\[
Q_R^\theta
\]

仅求：

\[
MatrixNash(Q)
\]

。

这应该是 Tool Agent 的：

\[
\boxed{\text{compute=minimum baseline}}
\]

。

---

# 67. Reference Simultaneous Search

等级：

\[
\boxed{\text{REFERENCE}}
\]

。

要求：

-正确 simultaneous semantics；
-不使用 Exact leaf；
-简单 leaf evaluator；
-固定预算。

目的：

证明提升不是因为“任何树搜索都行”。

---

# 68. SM-MCTS

在“我们完整系统”的搜索组件里属于 proposed component。

在 Search Ablation 中则作为：

\[
\boxed{\text{TARGET METHOD}}
\]

而不是 baseline。

---

# 69. CFR Search / GT-CFR

同理。

比较：

```text
Reference Search
vs
SM-MCTS
vs
GT-CFR
```

。

---

# 70. Exact Leaf Ablation

必须比较：

```text
SM-MCTS neural leaves only
SM-MCTS + Exact leaf override

GT-CFR neural leaves only
GT-CFR + Exact leaf override
```

。

这样回答：

\[
\boxed{\text{数学工具到底贡献多少}}
\]

。

---

# 71. Search Budget 公平性

必须至少报告两种预算：

## Node-Matched

相同：

\[
expanded\ nodes
\]

。

## Wall-Time-Matched

相同：

\[
milliseconds
\]

。

真正最重要的是：

\[
\boxed{\text{Wall-Time Matched}}
\]

。

---

# 72. 为什么两个都需要

一个算法单节点昂贵，

另一个便宜。

只比节点数不公平。

但只比 wall time 又看不出算法 sample efficiency。

所以都报告。

---

# 73. Search Compute Curve

至少：

```text
0
128
512
2048
8192
```

或对应 wall time：

```text
10ms
50ms
100ms
500ms
2s
```

。

得到：

\[
\boxed{\text{Strength vs Compute}}
\]

曲线。

---

# 第十一部分：Primary Baseline 最终名单

# 74. 必须完成

正式冻结：

### Non-learning

1. Random
2. Heuristic Suite
3. Exact Nash small-N oracle

### Traditional RL

4. PPO

### Joint-Action RL

5. Minimax-Q

### Game-Theoretic

6. CFR
7. CFR+
8. NeuRD
9. Deep CFR
10. R-NaD

### Search comparison

11. Network/Matrix-Nash only
12. Reference simultaneous search
13. SM-MCTS variants
14. GT-CFR variants

---

# 75. 强烈建议加入 Primary/Strong Reference

如果工程时间允许：

15. DCFR

因为它是很有价值的强 regret-minimization reference。

---

# 第十二部分：Reference Baseline 最终名单

# 76. Reference

```text
Independent DQN
IPPO
MAPPO
NFSP
MCCFR
DCFR if not promoted Primary
simple heuristic individual bots
```

。

---

# 77. 不需要一开始做的东西

例如：

- QMIX；
-VDN；
-MADDPG；
-SAC；

在这个问题上的研究信息价值相对低。

可以以后追加。

不要为了 baseline 数量而失焦。

---

# 第十三部分：Baseline 公平比较的第一原则——信息公平

# 78. Robust Track

所有 Robust baseline 只能看：

\[
\boxed{\text{Public State}}
\]

。

不得看：

- opponent ID；
-历史 session；
-当前 hidden action。

---

# 79. Adaptive Track

Opponent adaptation 必须另开一张表。

不能：

```text
Our adaptive agent
vs
PPO robust agent
```

然后说主方法赢。

---

# 80. Adaptive Baselines

以后 Arena D 至少：

```text
Robust No-Memory
LSTM Opponent Model
Mamba Only
LSTM+Mamba
Oracle Opponent Policy
```

以及简单：

```text
Bayesian / frequency opponent model
```

。

这属于 opponent baseline，而不是本文件主要 game-solving baseline。

---

# 第十四部分：Representation Fairness

# 81. Baseline 分两种比较

## Algorithm-Native

使用该算法典型网络。

例如：

PPO MLP。

Deep CFR 原生 advantage/policy nets。

---

## Backbone-Matched

尽量使用相同/类似 representation capacity。

目的分别是：

### Native

这个算法实际通常能做到什么。

### Matched

控制模型容量后，算法本身有什么差异。

---

# 82. 不允许只报告对我们有利的一档

例如：

PPO-Small 输了，

PPO-Matched 很强。

必须都记录。

主表可以选预注册的一档。

---

# 第十五部分：参数量公平

# 83. Matched Baseline

目标参数量：

\[
0.8P_{\text{main}}
\le
P_{\text{baseline}}
\le
1.25P_{\text{main}}
\]

。

---

# 84. 例外

算法结构天生需要多个网络：

比如 Deep CFR：

- advantage；
-strategy。

此时报告：

```text
total trainable params
inference params
```

分别比较。

---

# 第十六部分：训练 Compute 公平

# 85. 不能只比较 environment steps

不同算法每个 state 计算量差别巨大。

至少同时记录：

### Environment interactions

\[
G
\]

games / states。

### Model updates

### FLOP proxy

### GPU-hours

### CPU-hours

### wall-clock

---

# 86. 两种正式公平预算

## Sample-Matched

相同环境 interaction。

回答：

> 谁更 sample-efficient？

## Compute-Matched

相同总硬件预算。

回答：

> 给相同现实算力，谁更强？

两者都必须报告。

---

# 87. Exact / Search 算力计入预算

如果我们的训练使用：

- Exact Solver；
-CFR Teacher；
-Search；
-Reanalysis；

这些 CPU/GPU 时间：

\[
\boxed{\text{全部计入训练 compute}}
\]

。

不能称“free teacher”。

---

# 第十七部分：Hyperparameter Tuning 公平

# 88. 每个 Primary Baseline 都必须调参

禁止：

```text
PPO default config
vs
Our model 3 months tuning
```

。

---

# 89. Tuning Budget

建议每个 Primary：

相同数量：

\[
K
\]

config trials。

例如：

\[
K=20
\]

初步搜索。

然后 top 3 多 seed。

---

# 90. 调参数据

只能使用：

training/validation benchmark。

最终 test/golden set：

不能用来选 hyperparameter。

---

# 91. Baseline Owner Rule

每个 baseline 建立：

```text
baseline_card.yaml
```

记录：

```text
paper/reference
implementation
commit
hyperparameters
parameter count
training budget
seed
modifications
known deviations
```

。

---

# 第十八部分：Seed

# 92. Primary

最终：

至少：

\[
5
\]

个独立 training seeds。

如果训练昂贵：

最低：

\[
3
\]

，但必须注明。

---

# 93. Small-N

成本低：

建议：

\[
10
\]

seeds。

---

# 94. Reporting

报告：

\[
mean\pm std
\]

并尽量给：

bootstrap 95% CI。

---

# 第十九部分：Evaluation Protocol

# 95. 每个 Robust Primary 必测

### Score Difference

\[
E[
(Score_A-Score_B)/S_N
]
\]

### WDL

辅助。

### Exploitability / NashConv

核心。

### Cross-play

### Sample efficiency

### Compute efficiency

### Variable-N generalization

---

# 96. Exploitability 特别重要

OpenSpiel 提供 NashConv / best-response 工具；其现有 exploitability implementation要求 sequential constant-sum representation，因此如果用它评估我们的 simultaneous Goofspiel，必须先使用信息等价的 turn-based-simultaneous transformation，而不能直接把 simultaneity 改成可观察顺序动作。

---

# 97. Small-N

尽可能：

\[
\boxed{\text{Exact Exploitability}}
\]

。

---

# 98. N=13

如果 exact BR 太贵：

必须报告：

```text
Approximate Exploitability
BR algorithm
BR compute
BR convergence curve
```

。

不能直接写：

```text
exploitability = 0.02
```

让人以为是真值。

---

# 第二十部分：Cross-Play

# 99. 所有主要模型建立 payoff matrix

例如：

\[
G_{ij}
=
E[U(i,j)]
\]

包括：

```text
PPO
Minimax-Q
CFR-derived
Deep CFR
NeuRD
R-NaD
Our Model
Historical Main
```

。

---

# 100. 为什么必须 Cross-Play

因为零和多智能体学习可能出现：

\[
A>B,\;
B>C,\;
C>A
\]

。

单一 Elo 隐藏 non-transitivity。

---

# 第二十一部分：Learning Curves

# 101. 所有 Primary 必须保存

横轴：

### Environment steps

和：

### compute time

各一张。

纵轴至少：

- exploitability；
-score difference vs fixed suite；
-exact Q error where possible。

---

# 102. 不能只报告最终模型

因为我们的主要 claim 之一可能是：

\[
\boxed{\text{sample efficiency}}
\]

。

---

# 第二十二部分：Exact Small-N 表

# 103. 推荐主表

| Algorithm | N | States/Steps | Wall Time | Value Error | Policy Error | Exploitability |
|---|---:|---:|---:|---:|---:|---:|
| CFR | | | | | | |
| CFR+ | | | | | | |
| DCFR | | | | | | |
| Minimax-Q | | | | | | |
| NeuRD | | | | | | |
| R-NaD | | | | | | |
| Ours | | | | | | |

---

# 第二十三部分：N=13 主表

# 104. 推荐

| Algorithm | Train Games | GPU h | CPU h | Search Train? | Exploitability* | Score vs Suite | Cross-play |
|---|---:|---:|---:|---|---:|---:|---:|
| PPO | | | | No | | | |
| Minimax-Q | | | | No | | | |
| Deep CFR | | | | internal traversal | | | |
| NeuRD | | | | No | | | |
| R-NaD | | | | No | | | |
| Ours network-only | | | | training teachers noted | | | |
| Ours full | | | | Yes | | | |

其中：

\[
*
\]

明确标识 approximate BR 时的预算。

---

# 第二十四部分：Search 主表

# 105. 固定同一个 Base Network

| Search | Time | Nodes | Exact Leaves | Policy Quality | Score | Exploitability |
|---|---:|---:|---:|---:|---:|---:|
| None | 0 | 0 | 0 | | | |
| Matrix Nash | | | 0 | | | |
| Reference SM Search | | | 0 | | | |
| SM-MCTS | | | 0 | | | |
| SM-MCTS + Exact | | | ✓ | | | |
| GT-CFR | | | 0 | | | |
| GT-CFR + Exact | | | ✓ | | | |

---

# 第二十五部分：Main System 消融不是 Baseline 的替代品

# 106. 两种问题不同

Baseline：

> 和外部已有方法相比怎么样？

Ablation：

> 自己哪些组件有效？

必须同时存在。

---

# 107. 主系统必须做的核心 Ablation

至少：

```text
Full System

- Self-supervised pretraining
- SFT
- Semi-supervised teacher
- Exact teacher
- Search teacher
- MC
- TD(lambda)
- NeuRD
- GNN
- Matrix CNN
- LSTM
- Mamba
- Historical league
- Aggressive lineage
- Exploiter
- Red-team correction
- Reanalysis
```

。

不要求所有 \(2^{18}\) combinations。

使用有意义的分组 ablation。

---

# 第二十六部分：推荐 Baseline 实现来源

# 108. OpenSpiel

这是核心 reference framework。

当前 OpenSpiel：

-支持 simultaneous-move games；
-包含 game-theoretic algorithms 和 evaluation tools；
-有 CFR/CFR+；
-有 Deep CFR；
-有 NeuRD/NFSP；
-有 NashConv/best-response infrastructure。

因此：

\[
\boxed{\text{优先复用/适配，而不是从论文重新手搓}}
\]

。

---

# 109. TorchRL

PPO/IPPO/MAPPO baseline 可以参考 TorchRL 当前 multi-agent objectives 与 recipes。

---

# 110. RLlib

如果需要 historical self-play / league 版本的 PPO baseline，可以借鉴 RLlib 的 OpenSpiel league self-play 示例；当前示例显式区分 main policies 和 exploiters。

但不要因此把 baseline 和主系统全部迁入 RLlib。

---

# 第二十七部分：Baseline Repository Structure

# 111. 推荐

```text
baselines/
├── common/
│   ├── env_adapter.py
│   ├── evaluation.py
│   ├── budget.py
│   └── reporting.py
│
├── nonlearning/
│   ├── random.py
│   └── heuristics.py
│
├── dqn/
│   └── independent_dqn.py
│
├── ppo/
│   ├── ppo.py
│   ├── ippo.py
│   └── mappo.py
│
├── minimax_q/
│   ├── tabular.py
│   └── neural.py
│
├── cfr/
│   ├── vanilla.py
│   ├── cfr_plus.py
│   ├── dcfr.py
│   └── adapters.py
│
├── deep_cfr/
│   └── open_spiel_adapter.py
│
├── neurd/
│   └── open_spiel_adapter.py
│
├── nfsp/
│   └── open_spiel_adapter.py
│
├── rnad/
│   └── open_spiel_adapter.py
│
└── search/
    ├── network_only.py
    ├── reference_sm_search.py
    └── budget_runner.py
```

---

# 第二十八部分：所有 Baseline 使用统一 Evaluator

# 112. Baseline 不允许自己算自己的分数

必须：

```text
PolicyAdapter
      ↓
UnifiedEvaluator
```

。

---

# 113. Policy Interface

```python
class EvaluationPolicy:

    def reset_game(self):
        ...

    def action_distribution(
        self,
        public_state,
        legal_actions,
    ) -> np.ndarray:
        ...
```

Adaptive benchmark 另用：

```python
SessionPolicy
```

。

---

# 114. 所有输出必须是概率分布

即使 DQN/PPO 通常 greedy：

评测时 adapter 明确返回其执行 distribution。

---

# 第二十九部分：Snapshot Freeze

# 115. 评测时模型冻结

禁止：

> baseline 在 cross-play 过程中继续在线学习，而我们的 model 不学习。

除非 benchmark 明确叫：

```text
ONLINE_ADAPTATION
```

。

---

# 第三十部分：训练环境完全统一

# 116. 全部使用同一个 authoritative environment

不得：

```text
PPO uses PettingZoo implementation
CFR uses slightly different OpenSpiel rules
Ours uses custom Env
```

然后不做 parity。

---

# 117. OpenSpiel Baseline

如果必须使用 OpenSpiel game representation：

先通过：

\[
\boxed{\text{Rule Parity Test}}
\]

确认：

- tie；
-prize order；
-payoff；
-N；
-information state；

完全一致。

---

# 第三十一部分：禁止事项

## 禁止 1

禁止把 Random 当作唯一 baseline。

---

## 禁止 2

禁止只有 PPO，不比较 game-theoretic methods。

---

## 禁止 3

禁止只有 CFR，不比较现代 neural methods。

---

## 禁止 4

禁止主系统用 opponent history，而 Robust baseline 不允许，之后放同一 Robust 表。

---

## 禁止 5

禁止主系统带 Search 与无 Search baseline 直接比较后把提升全部归因于 learning。

---

## 禁止 6

禁止主系统使用 Exact teacher 但不记录其 compute。

---

## 禁止 7

禁止默认 PPO 参数不调。

---

## 禁止 8

禁止只用单 seed。

---

## 禁止 9

禁止 CFR sequentialization 泄露第一个玩家当前 action。

---

## 禁止 10

禁止把 CFR 的 current strategy 和我们的 final policy比较，而不是 CFR average strategy。

---

## 禁止 11

禁止 Deep CFR 使用我们的 Search/Exact teacher。

---

## 禁止 12

禁止 Minimax-Q 偷偷加入 NeuRD/CFR teacher。

---

## 禁止 13

禁止 baseline 使用不同 utility。

全部：

\[
normalized\ score\ difference
\]

。

---

## 禁止 14

禁止只比 WDL。

---

## 禁止 15

禁止用单一 Elo 代替 payoff matrix。

---

## 禁止 16

禁止 approximate exploitability 不标 BR budget。

---

## 禁止 17

禁止 baseline 失败就删除。

如果正确实现但表现差：

仍然报告。

---

## 禁止 18

禁止针对测试集调 baseline 或主模型。

---

## 禁止 19

禁止不同硬件 wall time 直接比较。

---

## 禁止 20

禁止把自家 ablation 冒充 external baseline。

---

# 第三十二部分：开发顺序

# 118. Baseline 不要全部最后才实现

建议：

### B0

Random + Heuristics。

和 Environment 一起完成。

### B1

Exact。

和数学核心一起完成。

### B2

CFR / CFR+。

用于验证博弈表示。

### B3

PPO。

主 RL 系统出现前完成。

### B4

Minimax-Q。

主 Nash Bellman 出现前完成。

### B5

NeuRD-only。

主 NeuRD actor 出现时完成。

### B6

Deep CFR。

进入大规模 game-theoretic comparison。

### B7

R-NaD。

进入完整 N=13 comparison 前完成。

### B8

NFSP/DCFR/IPPO/MAPPO 等 secondary。

---

# 第三十三部分：Baseline Acceptance Test

# 119. 每个 Primary Baseline 在正式比赛前必须证明“实现没坏”

例如：

### PPO

简单 heuristic opponent 上能显著学习。

### Minimax-Q

N=2/3 接近 Exact。

### CFR

exploitability 随 iterations 下降。

### CFR+

小 N 至少不出现明显实现错误。

### NeuRD

Matching Pennies / small game 产生合理 mixed policy。

### Deep CFR

小 game 逼近 CFR reference。

### R-NaD

在 OpenSpiel 已知 toy/reference game 上先复现合理学习行为。

然后才允许进入 Goofspiel 主表。

---

# 第三十四部分：Baseline Card

# 120. 每一个 baseline 都必须有

```yaml
baseline:
  name: cfr_plus
  tier: primary

  reference:
    paper: ...
    implementation: open_spiel
    commit: ...

  information:
    public_state_only: true
    opponent_history: false

  utility:
    normalized_score_difference: true

  representation:
    type: tabular

  training:
    iterations: ...
    games: ...
    gpu_hours: ...
    cpu_hours: ...

  evaluation:
    seeds: ...
    checkpoint: ...

  deviations:
    - ...
```

---

# 第三十五部分：最重要的比较矩阵

最终不要问：

> “哪个算法最好？”

而要回答不同问题。

| 科研问题 | 最关键对照 |
|---|---|
| 普通 RL 是否足够？ | PPO vs Ours |
| 是否必须显式 joint action？ | DQN/PPO vs Minimax-Q |
| Minimax Bellman 本身贡献？ | Minimax-Q vs Ours |
| Regret dynamics 是否重要？ | ordinary PG vs NeuRD |
| CFR 路线是否更适合？ | CFR/CFR+/Deep CFR vs Ours |
| 现代 equilibrium RL 如何？ | R-NaD vs Ours |
| Pretrain/SFT 是否有效？ | direct self-play vs staged training |
| Search 是否有效？ | Network Only vs SM-MCTS/GT-CFR |
| Exact tools 是否有效？ | Search ± Exact leaves |
| League 是否有效？ | current self-play vs historical league |
| Red Team 是否有效？ | league ± red-team correction |
| Opponent adaptation 是否有效？ | Robust vs Adaptive vs Oracle belief |

这才是 Baseline 体系真正要服务的目标。

---

# 最终冻结

## Primary Baselines

\[
\boxed{
\begin{aligned}
&Random\\
&Exact\ small\text{-}N\\
&PPO\\
&Minimax\text{-}Q\\
&CFR\\
&CFR+\\
&NeuRD\\
&Deep\ CFR\\
&R\text{-}NaD
\end{aligned}
}
\]

以及 Search Arena：

\[
\boxed{
Network\ Only
+
Reference\ Simultaneous\ Search
+
SM\text{-}MCTS
+
GT\text{-}CFR
}
\]

---

## Strong Reference / Secondary

\[
\boxed{
DCFR
+
NFSP
+
Independent\ DQN
+
IPPO
+
MAPPO
+
MCCFR
+
Heuristic\ Suite
}
\]

---

# 最后原则

整个 Baseline 体系必须坚持四句话：

\[
\boxed{
\textbf{信息必须公平。}
}
\]

\[
\boxed{
\textbf{训练和推理算力必须透明。}
}
\]

\[
\boxed{
\textbf{Baseline 必须被认真调好，而不是故意做弱。}
}
\]

以及最重要的：

\[
\boxed{
\textbf{
每一个 Baseline 都必须对应一个明确的科研问题，
而不是为了让表格看起来算法很多。
}
}
\]

最终，我们希望得到的不是：

> “Full System 第一，其他都比它弱。”

而是一条能够解释系统演进的证据链：

\[
PPO
\rightarrow
Minimax\text{-}Q
\rightarrow
NeuRD/CFR
\rightarrow
Neural\ Game\ Theory
\rightarrow
Pretrain/SFT/Posttrain
\rightarrow
Search
\rightarrow
Exact\ Tools
\rightarrow
League
\rightarrow
Opponent\ Adaptation
\rightarrow
Red\ Team
\]

从而真正回答：

\[
\boxed{
\textbf{每增加一种能力，究竟解决了前一种方法的什么问题。}
}
\]