# Goofspiel-13 Evaluation & Benchmark Specification
## 评测、Benchmark、晋级门槛与统计报告规范

---

# 0. 文档目的

本文定义整个 Goofspiel 智能体项目的统一评测体系。

目标不是回答：

> “这个模型赢了多少局？”

而是系统回答：

\[
\boxed{
\text{它是否真正理解游戏？}
}
\]

\[
\boxed{
\text{它距离 Nash equilibrium 多远？}
}
\]

\[
\boxed{
\text{它是否容易被针对？}
}
\]

\[
\boxed{
\text{它能否适应具体对手？}
}
\]

\[
\boxed{
\text{Search / Exact / Tool Use 到底带来了多少收益？}
}
\]

\[
\boxed{
\text{这些收益花了多少训练和推理算力？}
}
\]

以及最终：

\[
\boxed{
\text{一个新 checkpoint 是否有资格晋升为新的 Main Agent？}
}
\]

---

# 1. 评测总原则

整个 Benchmark 固定分为七个主 Arena：

```text
E0  Mathematical Correctness
E1  Exact Small-N Strategic Benchmark
E2  N=13 Robust Strategic Benchmark
E3  Opponent Modeling Benchmark
E4  Adaptive Exploitation & Safety Benchmark
E5  Search / Tool / Compute Scaling Benchmark
E6  Generalization & Robustness Benchmark
E7  Continual / League / Red-Team Benchmark
```

任何“完整模型”都必须至少通过 E0–E6。

Main Agent promotion 还必须通过 E7。

---

# 2. 评测时严禁训练

除非 benchmark 明确标记：

```text
ONLINE_ADAPTATION
```

否则评测期间：

- model parameters frozen；
- optimizer disabled；
- EMA disabled；
- replay 不写入训练；
- league opponent 不更新；
- opponent long-term memory 是否允许更新必须由 benchmark 明确规定。

评测不能边考边学。

---

# 3. Evaluation Profile

每次评测必须声明：

```text
evaluation_profile
```

固定枚举：

```text
QUICK
STANDARD
FULL
RELEASE
```

---

# 4. QUICK

开发过程中快速检查。

目标：

\[
<10\text{ min}
\]

内容：

- small-N exact subset；
- basic heuristic suite；
- 1–2 historical opponents；
- short search test；
- opponent synthetic subset。

不得用于论文最终结论。

---

# 5. STANDARD

常规 checkpoint evaluation。

目标：

约几十分钟到数小时。

覆盖：

- Exact benchmark；
- N=13 fixed suite；
- approximate exploitability；
- opponent benchmark；
- adaptive benchmark；
- search 3 档预算。

用于训练过程中 candidate 排序。

---

# 6. FULL

Major checkpoint。

覆盖全部 Arena。

必须多 seed / 大量对局。

用于：

- ablation；
- baseline；
- major model selection。

---

# 7. RELEASE

Main promotion / 论文最终模型。

FULL 基础上增加：

- permanent regression suite；
- red-team stress；
- long search curve；
- historical full cross-play；
- performance benchmark；
- statistical confidence interval；
- checkpoint reproducibility validation。

---

# 第一部分：Golden Benchmark Dataset

# 8. Benchmark 数据禁止动态漂移

必须建立固定：

```text
benchmarks/golden/
```

包括：

```text
exact_states/
robust_states/
opponent_sessions/
switch_sessions/
adaptive_scenarios/
search_states/
generalization/
redteam_regressions/
```

所有 major 模型使用相同数据。

---

# 9. Golden Dataset 不参与训练

不得：

- replay；
- SFT；
- teacher generation；
- hyperparameter tuning。

如果某 failure 必须进入训练 correction：

创建训练副本。

Golden benchmark 原件保持独立。

---

# 10. Benchmark Split

统一：

```text
TRAIN
VALIDATION
TEST
GOLDEN
```

最终论文报告：

只允许 TEST/GOLDEN。

VALIDATION：

用于 hyperparameter selection。

---

# 第二部分：E0 Mathematical Correctness

# 11. E0 不是能力评测，是资格考试

如果 E0 失败：

\[
\boxed{\text{禁止报告任何后续性能}}
\]

---

# 12. Environment Invariants

必须 100% 通过：

- card conservation；
- prize conservation；
- tie discard；
- simultaneous privacy；
- terminal；
- score；
- action legality。

---

# 13. Matrix Solver

随机至少：

\[
10^4
\]

矩阵和 SciPy LP reference 比较。

正式 Release：

建议：

\[
10^5
\]

。

指标：

\[
MAE_V
\]

\[
P95(|\Delta V|)
\]

\[
P99(|\Delta V|)
\]

duality gap。

---

# 14. Exact Solver

N≤4：

尽量 exhaustive。

N=5/6：

随机大量 reachable states。

检查：

- Python/C++ parity；
- player-swap；
- chance；
- cache；
- LP gap。

---

# 15. E0 Pass

任何核心数学 invariant failure：

\[
\boxed{FAIL}
\]

不能用平均分掩盖。

---

# 第三部分：E1 Exact Small-N Strategic Benchmark

这是整个项目最重要的“真值 Benchmark”。

---

# 16. N 范围

默认：

\[
N=3,4,5,6
\]

N=7 如果当前机器可承担足够 Exact states，则加入。

---

# 17. State Sampling 必须按 Horizon 分层

令：

\[
k=\text{remaining bid cards}
\]

。

不能随机状态后让中盘样本淹没残局。

每个：

\[
(N,k)
\]

独立 bucket。

例如：

```text
N=5,k=1
N=5,k=2
...
N=5,k=5
```

---

# 18. 每 bucket 状态数量

FULL 建议：

至少：

\[
1000
\]

states/bucket。

如果全部状态少于 1000：

全部使用。

---

# 19. 状态来源

混合：

```text
25% uniformly generated reachable states
25% random-policy visited states
25% strong-policy visited states
25% adversarial / difficult states
```

避免 benchmark 只覆盖模型自己常见 distribution。

---

# 20. E1 指标 1：Q Matrix Error

对于合法 cells：

\[
MAE_Q
=
\frac1{|M|}
\sum_{(a,b)\in M}
|Q_\theta(a,b)-Q^*(a,b)|
\]

同时记录：

\[
RMSE_Q
\]

和：

\[
P90/P99
\]

。

---

# 21. Q Error 必须分桶

至少：

- by N；
- by horizon k；
- by current prize；
- by uncertainty quartile。

不能只报告一个 global mean。

---

# 22. E1 指标 2：Value Error

对模型：

\[
V_\theta=
Val(Q_\theta)
\]

Exact：

\[
V^*
\]

记录：

\[
MAE_V
=
E|V_\theta-V^*|
\]

。

---

# 23. E1 指标 3：Policy Distance

Exact Nash policy：

\[
\pi^*
\]

Model matrix Nash：

\[
\pi_\theta
\]

记录：

\[
JSD(\pi_\theta,\pi^*)
\]

以及：

\[
L_1
\]

。

若 Exact equilibrium 不唯一：

必须使用统一 canonical max-entropy equilibrium 或 equilibrium-set-aware 指标。

---

# 24. 不得简单用 Argmax Accuracy

因为 Nash policy 可能是 mixed。

因此：

```text
top1 action accuracy
```

只能是辅助 metric。

---

# 25. E1 指标 4：Exact Exploitability

对于 policy：

\[
\pi
\]

计算其 best response loss。

在 two-player zero-sum：

可表示为：

\[
Exploitability(\pi)
\]

或 NashConv。

必须统一定义，并在报告中固定公式。

---

# 26. 建议统一归一化

所有 exploitability：

使用 normalized score-difference：

\[
[-1,1]
\]

单位。

---

# 27. E1 指标 5：Calibration of Q Uncertainty

ensemble 给：

\[
U_Q
\]

。

按 uncertainty 分 bucket：

比较：

\[
U_Q
\]

与实际：

\[
|Q-Q^*|
\]

。

指标：

- Spearman correlation；
- AUROC for high-error states；
- calibration curve。

目标：

> 模型不仅要错得少，还要知道自己什么时候可能错。

---

# 28. Exact Benchmark 主报告

必须生成：

```text
exact_eval_summary.json
exact_by_n.csv
exact_by_horizon.csv
exact_q_error_histogram
uncertainty_error_curve
```

---

# 第四部分：E2 N=13 Robust Strategic Benchmark

N=13 Exact 不可全面使用。

因此需要多重证据。

---

# 29. E2 禁止只测 Self-Play Win Rate

同模型打自己：

理论上永远约：

\[
50\%
\]

。

没有意义。

---

# 30. E2 Opponent Suite

固定至少包括：

### Non-learning

- Random；
-全部 Heuristic Suite。

### Historical

- old Robust；
- old Aggressive；
- known nemesis。

### External Baselines

- PPO；
- Minimax-Q；
- Deep CFR；
- NeuRD；
- R-NaD；
-其他 Primary Baselines。

### Adversarial

- approximate BR；
- Red-Team exploiters。

---

# 31. 每组对局必须 Seat Balanced

虽然游戏理论上对称，

工程上仍必须：

P0/P1 各一半。

使用相同 prize seeds paired evaluation。

例如：

同一个 seed：

```text
Agent A=P0 vs B=P1
Agent B=P0 vs A=P1
```

形成 paired game。

减少 chance variance。

---

# 32. E2 主 Utility

最重要：

\[
U=
\frac{Score_{self}-Score_{opp}}{S_N}
\]

报告：

\[
mean
\]

\[
std
\]

\[
95\% CI
\]

。

---

# 33. WDL 是次要指标

同时报告：

- Win；
- Draw；
- Loss。

但不得替代 score difference。

---

# 34. Opponent Suite Aggregate Score

定义：

\[
RobustSuiteScore
=
\sum_i w_i
E[U(\pi,\pi_i)]
\]

默认权重不能由模型表现后调整。

必须预注册。

---

# 35. 不建议简单所有 opponent uniform

可以分组：

```text
Heuristic      20%
Historical     25%
Primary Model  30%
Adversarial    25%
```

组内 uniform。

最终具体权重写入 benchmark config。

---

# 36. Cross-Play Payoff Matrix

所有重要算法：

构造：

\[
G_{ij}
=
E[U(i,j)]
\]

保存完整矩阵。

指标：

- antisymmetry error；
- cycle count；
- meta-game distribution；
- worst-opponent score。

---

# 37. Worst-Case Historical Score

定义：

\[
W_{hist}
=
\min_{j\in Historical}
E[U(Main,j)]
\]

。

这是判断 catastrophic forgetting 非常重要的指标。

---

# 38. Approximate Exploitability

N=13 无法完整 exact BR 时：

必须用一组越来越强的 BR attacks。

例如：

```text
BR-1 heuristic optimizer
BR-2 policy-gradient exploiter
BR-3 Minimax/response learner
BR-4 high-budget search exploiter
BR-5 red-team population
```

最终：

\[
ApproxExploit=
\max_j
ExploitAdvantage(BR_j)
\]

。

---

# 39. Approximate 不得写成 Exact

报告必须标：

```text
Approximate Exploitability
```

并附：

- BR algorithm；
- training games；
- GPU h；
- search budget；
- convergence curve。

---

# 40. Exploitability Curve

随着 BR attacker compute：

\[
C
\]

增加，绘制：

\[
ExploitAdvantage(C)
\]

。

一个模型如果：

> 小攻击找不到漏洞，大攻击马上打穿，

和真正 robust 不一样。

---

# 第五部分：E3 Opponent Modeling Benchmark

这一部分完全独立于最终胜率。

---

# 41. 四类 opponent scenario

### O1 Stationary Known Family

固定 synthetic style。

### O2 Parameterized Unseen Style

测试参数在训练分布之外或 held-out 参数。

### O3 Cross-Game Adaptation

行为随前几局变化。

### O4 Within-Game Switch

局内突然改变策略。

---

# 42. O1 Stationary

至少：

- aggressive；
- conservative；
- proportional；
- sacrifice；
- stochastic mixed；
- high-for-high；
- random-temperature variants。

---

# 43. O2 Unseen

训练 style 参数：

例如：

\[
\beta\in\{0.5,1,2\}
\]

测试：

\[
0.8,1.5,3
\]

等未见参数。

---

# 44. O3 Cross-Game

例如：

Opponent：

```text
Game 1–3 conservative
If losing:
Game 4+ aggressive
```

测试 Mamba 是否捕捉长期变化。

---

# 45. O4 Within-Game

例如：

第：

\[
t_s
\]

轮切换。

必须随机：

\[
t_s
\]

防止模型学固定第 6 轮切换。

---

# 46. Opponent Action Metrics

核心：

\[
NLL
\]

\[
Brier
\]

\[
ECE
\]

\[
Top1
\]

\[
Top3
\]

。

NLL/Brier 比 accuracy 更重要。

---

# 47. Short / Long / Fused 都必须独立报告

```text
q_L
q_M
q_F
```

分别：

- NLL；
- Brier；
- accuracy。

这样能发现 Fusion 是否只忽略其中一个。

---

# 48. Style Embedding Benchmark

如果 synthetic opponent 有真实 regime：

可以做：

- linear probe；
- nearest-neighbor retrieval；
- clustering ARI/NMI。

但这些是辅助。

真正重要仍是：

\[
\boxed{\text{prediction usefulness}}
\]

。

---

# 49. Switch Detection

必须报告：

- AUROC；
- AUPRC；
- F1；
- Detection Delay。

---

# 50. Detection Delay

真实 switch：

\[
t_s
\]

第一次：

\[
P_{switch}>\tau
\]

：

\[
Delay=t_{detect}-t_s
\]

。

平均、median、P90 都报告。

---

# 51. False Alarm

没有切换的 session：

记录：

\[
false\ switches/game
\]

。

不能只追求高 recall。

---

# 第六部分：E4 Adaptive Exploitation & Safety

这是 Robust/Adaptive 双分支最核心 Benchmark。

---

# 52. 必须有三条 agent

同一个基础模型：

### Robust

完全不使用 opponent history。

### Predicted Adaptive

使用：

\[
q_\phi
\]

。

### Oracle Adaptive

直接给真实 opponent policy：

\[
q^*
\]

。

---

# 53. 为什么 Oracle 很重要

定义：

\[
Gain_{oracle}
=
U_{oracle}-U_{robust}
\]

代表：

> 如果对手模型完美，Adaptive decision 最多有多少收益。

定义：

\[
Gain_{pred}
=
U_{pred}-U_{robust}
\]

代表实际收益。

两者差：

\[
BeliefGap
=
Gain_{oracle}-Gain_{pred}
\]

。

---

# 54. 如果 Oracle 也没收益

说明：

Adaptive policy / Q / Safe LP 有问题。

不是 opponent prediction 问题。

---

# 55. 如果 Oracle 很强但 Predicted 很差

说明：

主要瓶颈是 opponent modeling。

---

# 56. Adaptive Gain

按 opponent style 分别报告：

\[
Gain_i
=
U_{adaptive,i}
-
U_{robust,i}
\]

。

不能只报告 aggregate。

---

# 57. Safety Loss

对 Adaptive final policy：

使用 highest-quality robust evaluator，测：

\[
SafetyViolation
=
\max(
0,
V_R-\epsilon
-
V_{worst}(\pi_A)
)
\]

理想：

\[
0
\]

。

---

# 58. Empirical Reverse-Exploit Test

还要真实训练 adversarial attacker 去攻击 Adaptive policy。

看它相比 Robust：

增加了多少 exploitability。

定义：

\[
AdaptiveRiskIncrease
=
Exploit_{adaptive}
-
Exploit_{robust}
\]

。

---

# 59. Gain-Risk Frontier

改变：

\[
\epsilon_{max}
\]

例如：

```text
0
0.005
0.01
0.02
0.05
```

画：

\[
ExploitationGain
\]

vs：

\[
WorstCaseRisk
\]

。

这可能成为非常漂亮的核心实验。

---

# 60. Confidence Gate Benchmark

把 opponent confidence 分 bucket：

```text
0-.2
.2-.4
.4-.6
.6-.8
.8-1
```

看：

- prediction error；
- adaptive gain；
- safety violation。

如果 confidence 设计合理：

confidence 越高，

prediction 通常越准。

---

# 第七部分：E5 Search / Tool / Compute Benchmark

这是 Tool-Using Agent 核心评测。

---

# 61. 必须冻结同一 Base Network

搜索比较时：

\[
\boxed{\theta_{base}\text{完全相同}}
\]

否则无法判断 Search 贡献。

---

# 62. Search Variants

至少：

```text
Raw Actor
Matrix Nash(Q)
SM-MCTS
SM-MCTS + Exact Leaves
GT-CFR
GT-CFR + Exact Leaves
Full Router
```

。

---

# 63. Compute Axis 1：Simulation / Iteration

例如：

SM-MCTS：

```text
128
512
2048
8192
```

GT-CFR：

相应 iteration/nodes。

---

# 64. Compute Axis 2：Wall Time

至少：

```text
10 ms
50 ms
100 ms
500 ms
2 s
10 s
30 s
```

适配当前网页 30s think 场景。

---

# 65. 为什么必须 Wall-Time

真实部署关心：

> 给我 100ms / 1s / 30s，它到底多强？

不是只关心 simulations。

---

# 66. Search Benchmark States

不能只从 self-play 取。

分四组：

### Low uncertainty

模型本来就自信。

### High uncertainty

ensemble disagreement 高。

### Failure states

历史漏洞。

### Strategic high-impact

高 prize / early resource commitment。

---

# 67. Search Improvement Metrics

如果 Exact 可解：

直接：

\[
\Delta VError
\]

\[
\Delta PolicyError
\]

\[
\Delta Exploitability
\]

。

---

# 68. N=13 Search

使用：

- fixed opponent suite；
- approximate BR；
- cross-play。

比较 search/no-search。

---

# 69. Search Confirmation Rate

定义：

\[
JSD(\pi_{search},\pi_{base})<\tau
\]

但 Search quality 高。

表示：

> 搜索确认网络本身已经正确。

统计：

\[
ConfirmationRate
\]

。

---

# 70. Search Correction Rate

Search 显著修改 policy：

\[
JSD>\tau
\]

且 Exact/strong teacher 验证搜索更好。

定义：

\[
CorrectionRate
\]

。

---

# 71. Harm Rate

Search 改了策略，但变差：

\[
HarmRate
\]

。

这是 Search 质量非常重要的指标。

---

# 72. Exact Leaf Benefit

固定 Search 算法和预算：

比较：

```text
neural leaves
vs
neural + exact leaves
```

记录：

\[
\Delta performance
\]

以及：

\[
ExactLeafRatio
\]

。

---

# 73. Tool Router Benchmark

比较：

### Always Network

### Always Search

### Fixed Search Budget

### Deterministic Router

### Learned VOC Router（以后）

指标：

\[
Strength
\]

\[
MeanLatency
\]

\[
P95Latency
\]

\[
GPU/CPU Cost
\]

。

---

# 74. Value of Compute

定义：

\[
VOC=
\frac{
PerformanceGain
}{
ComputeCost
}
\]

可以用：

- normalized value per second；
- exploitability reduction per second。

---

# 第八部分：E6 Generalization Benchmark

---

# 75. Variable-N

训练支持：

\[
N=3\dots13
\]

。

必须逐 N 报告。

---

# 76. Seen / Underweighted N

即使所有 N 都见过，

可以人为：

某些 N 在训练采样很少，

检查 generalization。

---

# 77. Held-Out State Distribution

例如训练主要 self-play states。

测试：

- random reachable；
- heuristic-induced；
- red-team-induced；
- rare endgames。

---

# 78. Opponent Generalization

Test opponent：

必须包含：

\[
\boxed{\text{训练时从未出现的参数/策略族}}
\]

。

---

# 79. Search Generalization

Router 的 search trigger 也必须在 unseen states 上测。

不能只在训练采样 distribution 上效果好。

---

# 80. Distribution Shift Robustness

可以人为改变：

- opponent mix；
- strategy entropy；
- switching frequency。

看性能曲线。

---

# 第九部分：E7 League / Continual / Red-Team

---

# 81. Historical Regression

每一个新 Main：

必须打固定 historical set。

至少：

```text
R_{t-1}
R_{t-5}
R_{t-10}
historical nemesis
top aggressive specialist
top exploiter
```

。

---

# 82. Forgetting Metric

定义：

\[
Forgetting
=
\max_j
[
U(old,j)-U(new,j)
]
\]

对 permanent historical benchmark。

---

# 83. Red-Team Discovery Rate

给固定 compute：

Red Team 在：

\[
K
\]

小时/episodes 内发现多少：

\[
new\ valid\ failures
\]

。

---

# 84. Correction Success Rate

某 failure 修复后：

原 attack exploit advantage：

下降多少。

---

# 85. Recurrence

历史 fixed failure：

未来 checkpoint 是否重新失败。

定义：

\[
RegressionRate
\]

。

目标应随系统成熟持续下降。

---

# 86. Generalization after Correction

必须验证：

> 修一个 specific exploiter 后，并不是只针对它 overfit。

因此 correction 以后：

测：

- nearby strategy variants；
- unseen attackers；
- normal opponent suite。

---

# 第十部分：Main Promotion Gate

这是 Evaluation 文档最重要的一部分。

---

# 87. Candidate

当前：

\[
R_t
\]

产生候选：

\[
R_{cand}
\]

。

候选不得自动替换 Main。

---

# 88. Hard Gates

以下任何一项失败：

\[
\boxed{\text{REJECT}}
\]

---

# 89. Gate G0：Integrity

必须：

- checkpoint checksum；
- schema compatible；
- no NaN；
- reproducible inference；
- E0 all pass。

---

# 90. Gate G1：Exact Regression

在 Golden Exact Set：

候选：

\[
MAE_Q
\]

不能比 Main 恶化超过：

\[
\delta_Q
\]

。

建议初始：

\[
\delta_Q=2\%
\]

relative。

---

# 91. Gate G2：Exploitability

Approximate/exact exploitability：

不得显著恶化。

理想：

\[
Exploit_{cand}
\le
Exploit_{main}
\]

允许 statistical noise：

使用 CI 判断。

---

# 92. Gate G3：Historical

不得出现严重 historical catastrophic loss。

例如：

\[
W_{hist,cand}
\ge
W_{hist,main}-\epsilon_{hist}
\]

。

---

# 93. Gate G4：Regression Suite

所有 CRITICAL historical failures：

必须：

\[
100\%\ pass
\]

。

普通 failure：

至少：

\[
99\%
\]

或配置阈值。

---

# 94. Gate G5：Opponent Calibration

如果候选改变 opponent model：

要求：

ECE/Brier 不显著恶化。

若只更新 robust：

可以跳 opponent promotion。

---

# 95. Gate G6：Adaptive Safety

如果候选改变 Adaptive：

Safe LP violation：

必须：

\[
0
\]

numerical tolerance 内。

Empirical adaptive risk 不得突破设定 ceiling。

---

# 96. Gate G7：Numerical / Performance

不允许：

-大量 solver failure；
- search failure；
-严重 latency regression；
- GPU memory regression。

---

# 97. Soft Ranking

通过所有 Hard Gate 后，

才比较综合 improvement。

建议：

\[
PromotionScore
=
w_E\Delta Exploitability
+
w_R\Delta RobustSuite
+
w_X\Delta Exact
+
w_G\Delta Generalization
\]

。

具体权重提前冻结。

---

# 98. Promotion 不允许只靠 Elo

再次强调：

\[
\boxed{\text{Elo 只能辅助}}
\]

。

---

# 99. Candidate 结论

只有：

```text
PASS_HARD_GATES
AND
PROMOTION_SCORE > threshold
```

才能：

\[
R_{cand}\rightarrow R_{main}
\]

。

---

# 第十一部分：Statistical Protocol

---

# 100. 固定 Seeds

每个 benchmark：

有固定 seed set。

例如：

```text
benchmark_seeds_v1.json
```

。

---

# 101. Paired Evaluation

两个 agent 比较：

尽量使用同一 prize order seed。

这样做 paired comparison。

---

# 102. 报告均值与置信区间

至少：

\[
mean
\]

\[
std
\]

\[
95\%CI
\]

。

---

# 103. Bootstrap

对 game-level outcomes：

推荐 bootstrap：

\[
10,000
\]

次。

---

# 104. Multiple Seeds

Primary model：

至少：

\[
5
\]

training seeds。

很昂贵时最低：

\[
3
\]

。

---

# 105. 不得挑最好 Seed

最终主报告：

\[
mean\pm CI
\]

。

可以单独报告 best checkpoint，

但必须明确：

```text
best-seed model
```

。

---

# 106. Significance 不是唯一标准

一个 improvement：

即使 statistically significant，

但数值极小：

也可能没有工程意义。

同时报告：

\[
EffectSize
\]

。

---

# 第十二部分：Compute Accounting

---

# 107. 训练成本必须完整计入

包括：

- learner GPU；
-actor GPU；
-search GPU；
-teacher GPU；
-Exact CPU；
-Reanalysis；
-Red Team；
-hyperparameter tuning。

---

# 108. 记录

```text
GPU-hours by device class
CPU-core-hours
wall-clock hours
games generated
states evaluated
search nodes
exact states
```

。

---

# 109. Search / Exact 不是免费

任何借助 Exact/Search 提升的模型：

必须显式报告：

\[
TrainingCompute_{teacher}
\]

。

---

# 110. Inference Cost

每个 Agent 配置：

报告：

- mean latency；
- P50/P95/P99；
- CPU utilization；
- GPU utilization；
- memory；
- search nodes；
- exact hits。

---

# 111. Strength–Compute Pareto

最终最好画：

横轴：

\[
Compute
\]

纵轴：

\[
Strength/Exploitability
\]

。

比较不同方法的 Pareto frontier。

---

# 第十三部分：Benchmark Config

---

# 112. 所有评测配置必须版本化

例如：

```text
benchmark_profile_v1.yaml
```

。

内容：

```yaml
benchmark:
  version: "1.0.0"

  exact:
    n: [3,4,5,6]
    states_per_bucket: 1000

  robust_n13:
    games_per_matchup: 5000

  exploitability:
    attackers:
      - br_pg
      - br_search
      - redteam_population

  opponent:
    sessions_per_style: 500

  search:
    wall_times_ms: [10, 50, 100, 500, 2000, 10000]

  seeds_file: benchmark_seeds_v1.json
```

。

---

# 113. Benchmark Version

如果更换：

- opponent suite；
- exact state set；
- search budget；
- promotion threshold；

必须新 benchmark version。

不能覆盖旧结果。

---

# 第十四部分：报告输出

每次 FULL/RELEASE 必须自动生成：

```text
reports/<checkpoint>/
├── summary.json
├── summary.md
│
├── exact/
├── robust/
├── exploitability/
├── opponent/
├── adaptive/
├── search/
├── generalization/
├── league/
├── regression/
└── compute/
```

---

# 114. `summary.json`

机器可读。

必须包括所有关键指标。

---

# 115. `summary.md`

人类可读：

必须给出：

### Model

checkpoint/version。

### Benchmark

benchmark version。

### Main improvements

### Main regressions

### Hard Gate result

### Promotion decision

---

# 第十五部分：核心主表

---

# 116. Main Robust Table

建议：

| Model | Train Compute | Exact Q MAE | Approx Exploitability | Robust Suite | Worst Historical | N=13 Score |
|---|---:|---:|---:|---:|---:|---:|

---

# 117. Opponent Table

| Model | Fused NLL | Brier | ECE | Switch AUROC | Switch Delay | Unseen NLL |
|---|---:|---:|---:|---:|---:|---:|

---

# 118. Adaptive Table

| Model | Robust Utility | Adaptive Utility | Oracle Utility | Pred Gain | Oracle Gain | Risk Increase |
|---|---:|---:|---:|---:|---:|---:|

---

# 119. Search Table

| Method | Time | Nodes | Exact Leaf % | Exploitability | Score Gain | Harm Rate |
|---|---:|---:|---:|---:|---:|---:|

---

# 120. Generalization Table

| N / Shift | Robust | Adaptive | Exploitability | Opp NLL |
|---|---:|---:|---:|---:|

---

# 第十六部分：Baseline 与 Main 使用完全同一 Evaluator

Baseline 不允许自己定义评测。

必须：

```text
PolicyAdapter
      ↓
Unified Evaluation Harness
```

。

所有算法：

同：

- Game Rules；
- Seeds；
- Opponent Suite；
- Utility；
- Statistical Protocol。

---

# 第十七部分：训练中在线 Evaluator

---

# 121. Lightweight Evaluator

每：

\[
K
\]

updates：

运行 QUICK。

---

# 122. STANDARD

例如每：

\[
10\times K
\]

updates。

---

# 123. FULL

只对：

- milestone checkpoint；
- candidate promotion；
- best candidate。

避免 evaluator 吃掉过多训练算力。

---

# 124. Evaluation 不应阻塞 Learner

Evaluator：

使用 frozen checkpoint。

独立 worker / GPU。

---

# 第十八部分：Early Stopping

训练阶段可以使用：

\[
validation\ exploitability
\]

等，而不是 training reward。

---

# 125. Robust Early Stop

如果：

连续：

\[
M
\]

次 evaluation：

- exploitability 不降；
- exact error 不降；
- robust suite 不升；

则进入 plateau。

可以：

-调 LR；
-提升 teacher/search；
-停止该实验。

---

# 126. Opponent Early Stop

用：

- NLL；
- Brier；
- ECE；

而不是 adaptive win rate。

---

# 第十九部分：Research Claim ↔ Benchmark 对应

每一个 claim 必须预先对应 metric。

---

# 127. Claim：Pre-training 提高 Sample Efficiency

必须看：

\[
Exploitability\ vs\ EnvironmentSteps
\]

和：

\[
Exploitability\ vs\ Compute
\]

。

---

# 128. Claim：Joint Q 更适合 simultaneous games

比较：

- PPO/DQN；
- Minimax-Q；
- Full Matrix Nash Q。

指标：

- exact Q error；
- exploitability；
- sample efficiency。

---

# 129. Claim：Opponent Modeling 有效

看：

- NLL；
- Brier；
- Adaptive Gain；
- Oracle Gap。

---

# 130. Claim：Search 有效

看：

\[
Strength\text{-}Compute
\]

和：

\[
HarmRate
\]

。

---

# 131. Claim：Exact Tool 有效

看：

Search：

\[
\pm ExactLeaves
\]

以及 teacher：

\[
\pm ExactTeacher
\]

。

---

# 132. Claim：League 提高 Robustness

看：

- approximate exploitability；
- worst historical；
- cross-play cycles；
- red-team attack success。

---

# 133. Claim：Red-Team Correction 有效

看：

- failure recurrence；
- correction success；
- unseen exploiter robustness。

---

# 第二十部分：禁止事项

## 禁止 1

禁止只报告 Win Rate。

---

## 禁止 2

禁止只报告 Elo。

---

## 禁止 3

禁止在 Exact 可算时不用 Exact 做验证。

---

## 禁止 4

禁止 N=13 approximate exploitability 冒充 exact。

---

## 禁止 5

禁止不同模型使用不同 opponent suite。

---

## 禁止 6

禁止不同模型使用不同 prize seeds。

---

## 禁止 7

禁止 Search benchmark 使用不同 base network。

---

## 禁止 8

禁止 main 用 30s search、baseline 用 no-search 后将全部差距归给训练算法。

---

## 禁止 9

禁止使用 Golden benchmark 调 hyperparameter。

---

## 禁止 10

禁止只挑对自己有利的 opponent。

---

## 禁止 11

禁止只报告平均值而隐藏 worst-case。

---

## 禁止 12

禁止 Adaptive 只报告 gain 不报告 risk。

---

## 禁止 13

禁止 opponent model 只报告 accuracy。

---

## 禁止 14

禁止 switch detector 只报告 AUROC，不报告 detection delay/false alarms。

---

## 禁止 15

禁止 Search 只报告“赢得更多”，不报告 latency/compute。

---

## 禁止 16

禁止模型升级后不跑 historical regression。

---

## 禁止 17

禁止 Red-Team failure 修复后从 benchmark 删除。

---

## 禁止 18

禁止训练 seed 挑最好结果作为平均性能。

---

## 禁止 19

禁止修改 benchmark 后继续沿用旧 benchmark version。

---

## 禁止 20

禁止 Main promotion 由人工“感觉更强”决定。

---

# 第二十一部分：Main Promotion 最终决策表

| Gate | 要求 | Failure |
|---|---|---|
| E0 Math | 全部通过 | Reject |
| Exact | 无显著 regression | Reject |
| Exploitability | 无显著 regression | Reject |
| Historical | 无 catastrophic forgetting | Reject |
| Regression Suite | Critical 100% | Reject |
| Opponent | 若更新则 calibration pass | Reject |
| Adaptive | safety pass | Reject |
| Numerical | solver/search stable | Reject |
| Performance | 无严重 latency regression | Reject |
| Overall | 综合显著改善 | Promote |

---

# 第二十二部分：最重要的六个顶级指标

如果最终只能盯六个数字：

### 1.

\[
\boxed{\text{Exact Q / Value Error}}
\]

回答：

> 模型理解游戏数学有多准？

### 2.

\[
\boxed{\text{Exploitability}}
\]

回答：

> 它到底多容易被针对？

### 3.

\[
\boxed{\text{Robust Opponent Suite Score}}
\]

回答：

> 面对真实多种对手是否整体强？

### 4.

\[
\boxed{\text{Opponent NLL + Calibration}}
\]

回答：

> 它真的读懂对手了吗？

### 5.

\[
\boxed{\text{Adaptive Gain vs Risk}}
\]

回答：

> 它利用对手是否值得，而且安全吗？

### 6.

\[
\boxed{\text{Strength vs Compute}}
\]

回答：

> 多花推理算力到底能换来多少能力？

---

# 第二十三部分：这个 Benchmark 最终要证明什么

我们最终不是想得到：

> “Full System 赢了 PPO 63%。”

而是形成一条完整证据链：

\[
\boxed{
\text{Math Accuracy}
\rightarrow
\text{Equilibrium Quality}
\rightarrow
\text{Robustness}
\rightarrow
\text{Opponent Understanding}
\rightarrow
\text{Safe Exploitation}
\rightarrow
\text{Search Scaling}
\rightarrow
\text{Continual Robustness}
}
\]

也就是说，一个真正强的 Goofspiel Agent 应该同时满足：

\[
\boxed{
\textbf{知道什么策略在数学上合理}
}
\]

\[
\boxed{
\textbf{不容易被新的对手针对}
}
\]

\[
\boxed{
\textbf{能识别具体对手的行为规律}
}
\]

\[
\boxed{
\textbf{有把握时可以利用对方，但不会轻易牺牲自身安全性}
}
\]

\[
\boxed{
\textbf{给更多计算预算时能够通过 Search / Exact 工具继续变强}
}
\]

以及：

\[
\boxed{
\textbf{在长期 League 和 Red-Team 攻击下，能力不是循环退化，而是持续减少已知失败模式。}
}
\]

---

# 最终冻结

整个项目的正式评测层最终采用：

\[
\boxed{
E0\ Mathematical
+
E1\ Exact
+
E2\ Robust
+
E3\ Opponent
+
E4\ Adaptive
+
E5\ Search
+
E6\ Generalization
+
E7\ Continual
}
\]

任何新算法、新模型、新训练机制如果不能明确说明：

> **它预计改善哪一个 Arena、哪一个 metric、用什么对照证明，**

就不应该直接进入 Full System。

这是整个项目防止“模块越来越多，但不知道哪些真正有用”的最后一道约束。