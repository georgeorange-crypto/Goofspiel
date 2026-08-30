# Goofspiel-13 智能体测试、验证、日志与可观测性详细设计书
## Testing · Verification · Regression · Logging · Metrics · Tracing · Provenance

---

# 0. 文档定位

本文定义整个 Goofspiel 智能体项目的：

\[
\boxed{\text{测试体系}}
\]

\[
\boxed{\text{数学验证体系}}
\]

\[
\boxed{\text{回归体系}}
\]

\[
\boxed{\text{性能测试体系}}
\]

\[
\boxed{\text{日志体系}}
\]

\[
\boxed{\text{指标体系}}
\]

\[
\boxed{\text{Trace / Event / Provenance 体系}}
\]

目的不是“方便 debug”。

而是确保整个系统满足：

\[
\boxed{
\text{Correct}
\rightarrow
\text{Reproducible}
\rightarrow
\text{Observable}
\rightarrow
\text{Comparable}
\rightarrow
\text{Optimizable}
}
\]

---

# 1. 为什么这个项目必须有比普通 RL 更严格的测试体系

本项目同时包含：

- simultaneous-action environment；
- Exact dynamic programming；
- LP/Nash solver；
- neural joint-action Q；
- NeuRD；
- MC / TD(\(\lambda\))；
- CFR / Search；
- LSTM/Mamba opponent modeling；
- Adaptive branch；
- Safe Exploit LP；
- self-play；
- league；
- red-team；
- semi-supervised teacher；
- distillation；
- multi-GPU / asynchronous workers。

如果只测试：

```text
loss decreases
```

或者：

```text
agent can finish a game
```

几乎没有意义。

系统完全可能出现：

> loss 很漂亮，但 Nash value 符号写反。

或者：

> self-play 胜率 50%，实际上双方一起学成一个非常容易被第三方 exploit 的策略。

甚至：

> Search 看起来更强，实际是第二个玩家偷偷看到了第一个玩家当前 action。

因此所有模块必须满足：

\[
\boxed{
\text{Reference Oracle}
+
\text{Unit Test}
+
\text{Property Test}
+
\text{Integration Test}
+
\text{Regression Test}
}
\]

---

# 2. 测试体系总分层

整个测试体系固定分为七层：

```text
L0  Static / Schema Tests
L1  Unit Tests
L2  Mathematical Property Tests
L3  Cross-Backend / Oracle Parity Tests
L4  Integration Tests
L5  Training Convergence Tests
L6  Regression / Red-Team Tests
L7  Performance / Stress / Soak Tests
```

每一层回答不同问题。

---

# 3. L0：Static / Schema Tests

回答：

> 接口、shape、dtype、字段有没有被开发者随意改掉？

包括：

- dataclass schema；
- configuration schema；
- tensor shape；
- enum；
- serialization；
- dataset version；
- checkpoint version。

---

# 4. L1：Unit Tests

回答：

> 一个函数单独拿出来是否正确？

例如：

- reward；
- bitmask；
- Nash solver；
- NeuRD gradient；
- TD(\(\lambda\))；
- safe LP。

---

# 5. L2：Mathematical Property Tests

回答：

> 是否满足理论上必须成立的不变量？

例如：

\[
F(A,B,R)=-F(B,A,R)
\]

以及：

\[
\lambda=1
\Rightarrow
TD(\lambda)=MC
\]

。

---

# 6. L3：Oracle Parity Tests

回答：

> 我们的高速/复杂实现和可信 reference 是否一致？

例如：

```text
GPU RM+ ↔ SciPy LP
C++ Exact ↔ Python Exact
GPU Transition ↔ Python Transition
Our CFR ↔ OpenSpiel CFR
```

。

这是性能优化不破坏数学正确性的核心保障。

---

# 7. L4：Integration Tests

回答：

> 多模块连起来以后是否仍正确？

例如：

```text
State
→ Model
→ Matrix Nash
→ Search
→ Router
→ Final Policy
```

。

---

# 8. L5：Training Convergence Tests

回答：

> 学习算法是否真的能够学会一个我们知道答案的小问题？

例如：

\[
N=3
\]

必须能够逼近 Exact Nash。

这是比“loss 下降”强得多的验收。

---

# 9. L6：Regression / Red-Team Tests

回答：

> 以前修好的错误有没有回来？

所有历史 failure 都永久保存。

---

# 10. L7：Performance / Stress / Soak

回答：

> 系统快不快、稳定不稳定、跑几天会不会内存泄漏？

包括：

- GPU throughput；
- search nodes/sec；
- exact states/sec；
- queue latency；
- memory；
- long-running stability。

---

# 第一部分：目录结构

# 11. 测试目录

必须使用：

```text
tests/
├── unit/
│   ├── game/
│   ├── math/
│   ├── models/
│   ├── learning/
│   ├── reasoning/
│   ├── data/
│   └── training/
│
├── property/
│   ├── game_invariants/
│   ├── symmetry/
│   ├── solver_properties/
│   └── learning_identities/
│
├── parity/
│   ├── openspiel/
│   ├── scipy/
│   ├── python_cpp/
│   └── cpu_gpu/
│
├── integration/
│   ├── agent/
│   ├── search/
│   ├── training/
│   ├── league/
│   └── distributed/
│
├── convergence/
│   ├── n1/
│   ├── n2/
│   ├── n3/
│   └── synthetic_opponents/
│
├── regression/
│   ├── bugs/
│   ├── redteam/
│   ├── numerical/
│   └── checkpoints/
│
├── performance/
│   ├── model/
│   ├── solver/
│   ├── exact/
│   ├── search/
│   ├── data/
│   └── end_to_end/
│
└── fixtures/
```

---

# 12. 测试命名

统一：

```text
test_<module>_<behavior>_<condition>()
```

例如：

```python
test_transition_tie_discards_prize()
test_exact_solver_player_swap_sign()
test_neurd_gradient_matches_negative_regret()
test_router_exact_overrides_search()
```

禁止：

```text
test1
test_misc
test_agent
```

。

---

# 第二部分：Game Core 测试

# 13. Environment 是整个项目第一 Oracle

如果规则错，后面全部没有意义。

Game Core 必须拥有最高覆盖率。

---

# 14. Initial State

对于 N：

必须满足：

\[
self\_cards
=
opp\_cards
=
prizes
=
\{1,\dots,N\}
\]

。

总 prize：

\[
S_N=\frac{N(N+1)}2
\]

。

---

# 15. Legal Action

任意 state：

\[
legal(self)=remaining(self)
\]

\[
legal(opp)=remaining(opp)
\]

。

已经出过的牌：

绝不能再次合法。

---

# 16. Tie Rule

若：

\[
a=b
\]

必须：

\[
\Delta score_{self}=0
\]

\[
\Delta score_{opp}=0
\]

当前 prize：

删除。

---

# 17. Win / Loss

若：

\[
a>b
\]

：

self 获得当前 prize。

若：

\[
a<b
\]

：

opponent 获得当前 prize。

---

# 18. Conservation

任意终局：

所有 prize：

要么属于 self，

要么属于 opponent，

要么由于 tie discarded。

必须满足：

\[
Score_{self}
+
Score_{opp}
+
Discarded
=
S_N
\]

。

---

# 19. Card Conservation

第 t 轮后：

双方使用牌数量：

\[
t
\]

。

剩余：

\[
N-t
\]

。

必须一致。

---

# 20. Game Length

所有合法完整游戏：

必须恰好：

\[
N
\]

轮结束。

不得提前结束。

不得多一轮。

---

# 21. Property-Based Random Game Tests

使用 Hypothesis/randomized tests。

对于：

\[
N=1\dots13
\]

随机生成数千条合法 joint-action sequence。

每一步检查：

- mask；
- score；
- current prize；
- remaining count；
- invariants。

---

# 22. Illegal Action Tests

必须测试：

- action < 1；
- action > N；
- 已使用；
- done 后继续 step；
- 当前不存在 action。

必须显式抛异常。

禁止 silently clamp。

---

# 23. Deterministic RNG Test

使用固定：

```text
seed = 20260830
```

运行游戏两次。

prize order 与结果必须完全一致。

---

# 第三部分：State / Encoding / Data Tests

# 24. Bitmask Roundtrip

任意 card set：

```text
set
→ mask
→ set
```

必须相同。

---

# 25. Rank Mapping

必须冻结：

### Game semantic rank

\[
1\dots N
\]

### Tensor index

\[
0\dots N-1
\]

只有 encoding boundary 允许转换。

测试所有 rank。

防止 off-by-one。

---

# 26. Dense Tensor Encoding

同一个 GameState：

编码两次：

Tensor 必须 bitwise deterministic。

---

# 27. N Padding

N=7 state：

单独编码到 Nmax=13。

与混在 N=13 batch 中编码：

前 7 个有效位置必须完全一致。

---

# 28. Serialization Roundtrip

所有：

- GameState；
- trajectory；
- TeacherTarget；
- SearchResult；
- Checkpoint metadata；

必须满足：

```text
object
→ serialize
→ deserialize
→ equality
```

。

---

# 29. Schema Version

所有 persistent data 必须带：

```text
schema_version
```

。

测试旧 version：

必须：

- 明确 migrate；
或
- 明确拒绝。

禁止 silent misread。

---

# 第四部分：Reference Matrix Solver 测试

# 30. Matching Pennies

\[
Q=
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}
\]

必须：

\[
V\approx0
\]

\[
\pi_A\approx(0.5,0.5)
\]

\[
\pi_B\approx(0.5,0.5)
\]

。

---

# 31. Pure Saddle Point

例如：

\[
Q=
\begin{bmatrix}
3&2\\
1&0
\end{bmatrix}
\]

验证 pure minimax solution。

---

# 32. Dominated Strategy

人为加入 strictly dominated action。

该 action equilibrium probability：

\[
<10^{-8}
\]

Reference FP64 下。

---

# 33. Constant Shift

矩阵：

\[
Q'=Q+c
\]

必须：

策略不变，

value：

\[
V'=V+c
\]

。

---

# 34. Positive Scaling

\[
Q'=cQ,\quad c>0
\]

策略应相同。

value：

\[
V'=cV
\]

。

---

# 35. Player Swap

\[
Q'
=
-Q^T
\]

则：

row/column 角色交换，

value：

\[
V'=-V
\]

。

---

# 36. Probability Validity

所有 solver：

\[
p_i\ge0
\]

\[
\sum p_i=1
\]

。

---

# 37. Duality Gap

Reference LP：

要求：

\[
gap<10^{-8}
\]

正常小矩阵测试。

---

# 第五部分：GPU Nash Solver 测试

# 38. Random Matrix Parity

固定随机种子。

生成至少：

\[
10,000
\]

个随机矩阵。

N：

\[
2\dots13
\]

。

GPU solver 对 SciPy Reference：

统计：

- value MAE；
- P95 error；
- P99 error；
- duality gap。

---

# 39. Acceptance Threshold

初始：

```text
mean |ΔV| < 1e-3
P95 |ΔV| < 5e-3
P99 |ΔV| < 1e-2
```

。

如果后来收紧：

配置版本化。

---

# 40. Mask Tests

随机 mask 掉部分行列。

结果必须等价于：

把非法行列真正删除后求 Reference Nash。

---

# 41. Batch Independence

一个 batch 中修改 sample 0。

sample 1..B-1 结果不得变化。

---

# 42. Compile Parity

`torch.compile` 前后：

相同输入。

输出误差在设定 tolerance 内。

---

# 43. Triton Parity

未来 Triton backend：

必须与 PyTorch backend parity。

---

# 第六部分：Exact Solver 测试

# 44. N=1

所有情况直接人工可算。

必须完全匹配。

---

# 45. N=2

枚举所有状态人工/reference 比较。

---

# 46. Full Enumeration Small N

对：

\[
N\le4
\]

尽量遍历全部 reachable states。

验证：

- value；
- symmetry；
- cache；
- terminal。

---

# 47. Player Swap

随机 states：

\[
F(A,B,R)+F(B,A,R)
\]

FP64 tolerance 内：

\[
\approx0
\]

。

---

# 48. Symmetric Remaining Cards

若：

\[
A=B
\]

before prize reveal：

必须：

\[
F(A,A,R)=0
\]

。

---

# 49. Chance Permutation

Remaining prize set 顺序改变：

Exact value 不变。

Exact Solver 不得依赖预先 shuffle 的未来 prize 顺序。

---

# 50. Cache Test

第一次：

miss。

第二次相同 state：

hit。

结果完全相同。

---

# 51. Canonicalization

swap canonicalization 开启与关闭：

输出必须等价。

---

# 52. Python / C++ Parity

性能版 C++ 上线后：

随机：

\[
10^4
\]

小 N states。

value / Q / policy 比 Python reference。

---

# 53. Complexity Estimator Calibration

对若干 N/state：

记录：

\[
estimated\ time
\]

与：

\[
actual\ time
\]

。

必须输出：

\[
ratio
=
actual/estimated
\]

。

P50/P90/P95。

防止 estimator 长期失真。

---

# 54. Exact Budget Refusal

给一个明显不可计算 state。

`solve(force=False)`：

必须拒绝。

不能真的开始爆内存。

---

# 第七部分：模型结构测试

# 55. Shape Contract

对：

\[
B\in\{1,8,32\}
\]

\[
N\in\{3,5,7,10,13\}
\]

测试所有输出 shape。

例如：

\[
Q_R:[B,13,13]
\]

即使 N 小，也统一 Nmax。

---

# 56. Finite Values

除 masked logits 外：

不允许：

- NaN；
- Inf。

---

# 57. Opponent Leakage Test

最重要测试之一。

构造：

public state 完全相同。

opponent history：

不同。

要求：

\[
Q_R^{(1)}
=
Q_R^{(2)}
\]

\[
\pi_R^{(1)}
=
\pi_R^{(2)}
\]

\[
Z_R^{(1)}
=
Z_R^{(2)}
\]

在 numerical tolerance 内完全一致。

Adaptive 输出：

允许不同。

---

# 58. LSTM Isolation

固定：

- public state；
- long-term memory。

改变 current-game history。

要求：

LSTM hidden 变化。

Mamba hidden：

不变化。

---

# 59. Mamba Isolation

固定 current game。

改变 previous completed games。

要求：

Mamba hidden 变化。

LSTM hidden：

不变化。

---

# 60. Stateful LSTM Parity

方式 A：

完整 sequence forward。

方式 B：

逐 round `step()`。

最终 hidden：

必须一致。

---

# 61. Mamba Game Boundary

当前局发生一轮 hypothetical event：

Mamba state 不应变化。

完成真实 game summary：

Mamba 才更新。

---

# 62. Padding Invariance

N=7：

单独 batch。

与 N=13 padded batch。

valid region 输出一致。

---

# 63. Player View

交换 self/opp：

确保 public features 不依赖固定 player ID。

---

# 64. Parameter Count Regression

保存每个模块参数量。

CI 比较：

如果参数量变化超过阈值：

要求明确 approval。

防止 Codex 偷偷加/删模块。

---

# 第八部分：Learning Algorithm Tests

# 65. Nash Bellman Terminal

最后一轮：

\[
Y_{ab}=r_{ab}
\]

必须精确成立。

---

# 66. Full Matrix Target

若合法动作：

\[
k
\]

必须生成：

\[
k^2
\]

target cells。

不是只生成实际 action。

---

# 67. Chance Exact Threshold

当：

\[
|R|\le threshold
\]

必须完整枚举。

---

# 68. Chance Sampling Unbiased Test

大量重复 sampled chance target。

样本均值应接近 exact chance enumeration。

---

# 69. Target Network Isolation

Bellman target backward：

target model 参数：

无 gradient。

---

# 70. Q-MC Semantic Isolation

MC outcome loss backward：

Robust Q Head：

无 gradient。

这是硬测试。

---

# 71. TD(\(\lambda\)) Endpoint

\[
\lambda=0
\]

等于 TD(0)。

\[
\lambda=1
\]

等于 MC return。

---

# 72. V-trace On-Policy

若：

\[
target=behavior
\]

ratio：

\[
1
\]

应退化到对应 on-policy trace。

---

# 73. V-trace Ratio Clip

人为构造：

\[
\rho=100
\]

确保 clipping 正确。

---

# 74. NeuRD Gradient

给：

\[
g=[0.4,-0.1,-0.3]
\]

要求：

raw logits gradient：

与：

\[
-g
\]

对应。

---

# 75. NeuRD 禁止 LogSoftmax

可以直接通过 gradient fixture 验证。

如果实现改成普通 policy gradient：

测试应该失败。

---

# 76. Illegal Actor Mask

非法 action logit 不产生策略概率和 gradient。

---

# 77. Nash Anchor

Teacher policy one-hot/mixed：

KL 方向正确。

---

# 78. Teacher Priority

同时存在：

```text
Exact
Search
Nash Bellman
```

必须选择 Exact。

---

# 79. CFR Teacher Priority

Policy 同时有：

```text
Exact
GT-CFR
Matrix Nash
```

必须选 Exact。

---

# 80. Opponent Label Integrity

Opponent CE label：

必须等于真实观察 action。

不能来自：

- Nash；
- Search；
- Teacher。

---

# 81. Style Positive Pair

same opponent + same regime：

positive。

same opponent + different regime：

不是 positive。

---

# 82. No Positive Pair

InfoNCE batch 中没有合法 positive：

必须安全 skip。

不能 NaN。

---

# 83. Switch BCE

人工构造切换点。

label 与 loss 正确。

---

# 84. Adaptive Gradient Isolation

Adaptive loss backward：

- public robust backbone：无 grad；
- LSTM/Mamba：无 grad；
- adaptive branch：有 grad。

---

# 85. Adaptive Q Residual

设置：

\[
\Delta Q=0
\]

时：

\[
Q_A=Q_R
\]

。

---

# 86. Unknown Opponent Regularization

confidence：

0。

Residual regularization 最大。

---

# 第九部分：Search 测试

# 87. Simultaneous Privacy

这是最高优先级 Search test。

Instrument search：

记录双方 action selection timestamp/data dependencies。

必须证明：

\[
b_t
\]

是在不知道：

\[
a_t
\]

的情况下产生。

禁止 sequential leakage。

---

# 88. Matching Pennies SM-MCTS

足够 simulation：

average strategy：

接近：

\[
(0.5,0.5)
\]

。

---

# 89. Search Average Policy

最终正式 policy：

必须是 average strategy。

current strategy 只用于 diagnostics。

---

# 90. Exact Leaf Override

构造 root 不可 Exact，但 child 可 Exact。

必须：

```text
exact_leaf_hits > 0
```

。

---

# 91. Neural Leaf Fallback

Exact 不可用：

必须正常使用 neural \(Q_R\rightarrow Nash\)。

---

# 92. Search Budget Stop

分别测试：

- max simulation；
- max node；
- wall time。

任何一个达到：

停止。

---

# 93. Wall-Time Tolerance

例如 100ms budget。

允许小 scheduler tolerance。

但不能运行 2s。

---

# 94. Search NaN Fallback

人工制造 Search failure。

Router 必须使用 Matrix Nash baseline。

---

# 95. Tree Reuse

真实 joint action + prize 后：

正确 child 提升成 root。

---

# 96. Tree Version

模型版本变化：

旧 tree 不可无条件复用。

---

# 97. GT-CFR Small Game Parity

N=2/3：

与 full CFR / Exact equilibrium 比较。

---

# 98. GT-CFR Expansion

提高 search budget：

expanded nodes 应非递减。

质量通常改善或保持。

---

# 99. Search Quality Gate

人为返回：

高 duality gap。

必须拒绝成为 Teacher。

---

# 第十部分：Safe Exploit / Router Tests

# 100. Exact Priority

Exact success：

必须覆盖：

- SM-MCTS；
- Actor；
- Matrix Nash。

---

# 101. Matrix Nash Baseline Always Exists

即使所有 Search/Exact 失败：

仍必须有 valid robust policy。

---

# 102. Unknown Opponent

confidence=0：

final policy 应接近 Robust。

---

# 103. Safe LP Constraint

最终：

对于所有 opponent pure action：

\[
\pi^TQ_R[:,b]
\ge
V_R-\epsilon
\]

必须逐列验证。

---

# 104. Adaptive Gain

构造已知 exploitable opponent。

Adaptive expected utility 应高于 robust。

---

# 105. Safety Tradeoff

提高：

\[
\epsilon
\]

允许更 aggressive。

测试 exploitation gain 非减趋势。

---

# 106. Router Determinism

同：

- state；
- model；
- config；
- budget；

Router tool choice：

必须确定性一致。

PLAY 最终 action sampling 可随机。

Router 选择本身不随机。

---

# 107. Search Trigger Boundary

测试阈值：

```text
0.249
0.250
0.499
0.500
...
```

明确哪一侧进入哪个 budget。

---

# 第十一部分：Training Pipeline Tests

# 108. Stage Gate

若 P1 未达标：

不能自动进入 P2。

---

# 109. Exact Anchor Persistence

进入 P4/P5 后：

Exact sample probability 不得意外变成 0。

---

# 110. Adaptive Gate

Opponent calibration 不达标：

Adaptive branch 必须保持冻结。

---

# 111. Robust/Adaptive Buffer Isolation

ROBUST trajectory：

不能进入 Adaptive MC target。

ADAPTIVE：

不能作为 Robust MC target。

---

# 112. Teacher Confidence Filtering

高 disagreement pseudo-label：

必须拒绝。

---

# 113. EMA Teacher

更新公式正确。

Online teacher gradient：

无。

---

# 114. Checkpoint Resume

训练：

100 steps → save → resume 100 steps。

与连续 200 steps 在 deterministic small test 下：

结果相同/数值近似。

---

# 115. RNG Resume

必须保存：

- Python RNG；
- NumPy RNG；
- Torch CPU RNG；
- CUDA RNG。

---

# 116. Optimizer Resume

AdamW momentum state 必须恢复。

---

# 第十二部分：League Tests

# 117. Role Integrity

每个 agent 必须有：

```text
ROBUST
AGGRESSIVE
EXPLOITER
```

。

不能 null。

---

# 118. Historical Freeze

加入 historical pool 后：

参数不可再被训练修改。

---

# 119. PFSP Sampling

给固定 win rates。

采样大量次数。

经验分布应接近理论 priority。

---

# 120. Novelty Rejection

两个策略距离极小且性能无改进：

candidate 不应加入永久 pool。

---

# 121. Payoff Antisymmetry

zero-sum：

\[
G_{ij}\approx-G_{ji}
\]

。

统计误差范围内必须成立。

---

# 122. Duplicate Policy

相同 checkpoint：

不能重复注册多个永久 ID。

---

# 第十三部分：Red-Team Regression Tests

# 123. Failure ID 永久化

每个 failure：

```text
failure_id
```

。

永不复用。

---

# 124. Failure Reproduction

加入 regression suite 前：

必须能够固定 seed/config 复现。

---

# 125. Correction Test

修复后：

原 attacker 在同测试条件下 exploit advantage 必须下降。

---

# 126. General Regression

任何 focused correction checkpoint：

必须同时跑：

- Exact；
- Historical；
- Calibration；
- generic benchmark。

---

# 127. Permanent Suite

历史 failure 不能删除。

只允许：

```text
ACTIVE
FIXED
OBSOLETE_BY_RULE_CHANGE
```

其中最后一种必须有明确规则版本变化。

---

# 第十四部分：Training Convergence Tests

这是整个项目最关键的一类测试之一。

---

# 128. N=1 Convergence

网络应极快学到 exact Q。

如果 N=1 都学不会：

禁止 N=13。

---

# 129. N=2 Convergence

训练小模型。

要求：

- Q error；
- policy exploitability；

明显趋近 Exact。

---

# 130. N=3 Convergence

这是第一个真正有意义的 end-to-end learning CI。

固定：

- model size；
- dataset；
- seed；
- training steps。

输出：

\[
Q\ MAE
\]

\[
policy\ KL
\]

\[
exploitability
\]

。

---

# 131. Convergence Regression

保存 baseline curve。

例如：

```text
step 1000 exploitability <= X
step 5000 <= Y
step 10000 <= Z
```

。

如果某 PR 后明显变慢：

CI/weekly test 报警。

---

# 132. Synthetic Opponent Learning

构造一个简单 opponent：

```text
prize > threshold → high bid
else → low bid
```

。

要求 opponent predictor：

显著优于 uniform。

---

# 133. Strategy Switch Convergence

第 6 轮切换。

Switch head：

必须学到明显高于随机。

---

# 第十五部分：Performance Tests

# 134. Model Forward Throughput

固定 hardware/config。

benchmark：

\[
B=
1,32,64,128,256,512,1024
\]

记录：

- latency；
- samples/sec；
- GPU utilization；
- memory。

---

# 135. Model Backward

记录：

- update/sec；
- tokens/states/sec；
- peak VRAM。

---

# 136. GPU Nash Solver

记录：

\[
matrices/sec
\]

按：

- batch；
- N；
- iterations。

---

# 137. Exact Solver

记录：

- states/sec；
- LP/sec；
- cache hit；
- memory/state。

---

# 138. SM-MCTS

记录：

- sims/sec；
- nodes/sec；
- leaf eval/sec；
- exact leaf ratio。

---

# 139. GT-CFR

记录：

- CFR iterations/sec；
- expansions/sec；
- policy quality per second。

---

# 140. Data Pipeline

记录：

- rows/sec；
- H2D latency；
- DataLoader wait；
- queue depth。

---

# 141. Search Leaf Batch

测试 batch：

\[
32\dots2048
\]

找：

- maximum throughput；
- latency knee。

---

# 142. Compile Speedup

记录：

```text
eager throughput
compiled throughput
speedup
compile overhead
```

。

---

# 143. Performance Regression Threshold

例如：

若同硬件：

\[
throughput\下降>10\%
\]

自动报警。

但不能自动 fail PR，除非是关键路径。

---

# 第十六部分：Stress / Soak Tests

# 144. 24h Training Soak

定期跑。

检查：

- RAM 是否增长；
- VRAM 是否增长；
- queue 是否积压；
- NaN；
- deadlock；
- checkpoint 是否正常。

---

# 145. Search Soak

连续运行：

\[
10^6
\]

次小 search。

检查：

- tree cleanup；
- cache；
- memory。

---

# 146. Worker Restart

杀死一个 Actor/Search worker。

系统是否能：

- detect；
- restart；
-继续训练。

成熟分布式阶段必须测试。

---

# 第十七部分：Fuzz Testing

# 147. State Fuzzing

随机生成：

- 合法 state；
- 非法 state；
-边界 bitmask。

确保：

合法正常处理，

非法明确报错。

---

# 148. Numerical Fuzzing

随机 Q：

包含：

- 极小值；
- 极大值；
-相同值；
-近退化 saddle。

检查 solver stability。

---

# 149. Mask Fuzzing

随机 legal mask。

至少一个合法动作。

检查所有算法。

---

# 第十八部分：日志体系总架构

测试解决：

> “系统应该正确。”

日志解决：

> “系统实际发生了什么？”

整个项目禁止依赖：

```python
print(...)
```

作为正式日志。

---

# 150. 日志分四类

必须严格区分：

## Event Log

发生了什么。

## Metrics

量化指标是多少。

## Trace

一次复杂操作经过了哪些步骤、花了多久。

## Artifact

需要保存的较大结果。

---

# 151. Event Log

例如：

```text
CHECKPOINT_PROMOTED
SEARCH_STARTED
SEARCH_FAILED
EXACT_REFUSED
FAILURE_DISCOVERED
PSEUDO_LABEL_REJECTED
ADAPTIVE_GATE_OPENED
WORKER_RESTARTED
```

。

是离散事件。

---

# 152. Metrics

例如：

```text
robust/q_loss
solver/duality_gap
search/nodes_per_sec
opponent/nll
gpu/utilization
```

。

用于画曲线。

---

# 153. Trace

例如一次：

```text
agent.think()
```

内部：

```text
model_forward
matrix_nash
exact_estimate
sm_mcts
adaptive_search
safe_lp
```

。

每一步带：

- start；
- end；
- duration；
- status。

---

# 154. Artifact

例如：

- checkpoint；
- payoff matrix；
- profiler trace；
- failed trajectory；
- Q heatmap；
- search tree dump；
- exact dataset shard。

---

# 第十九部分：统一 Event Schema

# 155. 所有事件必须有基础字段

```python
@dataclass
class BaseEvent:
    timestamp_ns: int

    run_id: str
    experiment_id: str

    process_id: int
    worker_id: str
    worker_role: str

    event_type: str
    severity: str

    git_commit: str
    config_hash: str

    model_version: str | None
    policy_version: str | None

    game_id: str | None
    session_id: str | None
    state_hash: str | None

    trace_id: str | None
    parent_trace_id: str | None
```

---

# 156. 为什么必须有 `state_hash`

当发现：

> 第 7 轮这个状态 Search 崩了。

开发者应该能够通过：

```text
state_hash
```

定位：

- trajectory；
- model prediction；
- search log；
- teacher；
- failure；
- exact result。

---

# 157. 为什么必须有 `model_version`

没有它：

Search cache、training log、game result 都无法解释。

同一个 state：

不同模型可能行为完全不同。

---

# 158. `game_id`

每一局唯一。

格式可：

```text
<run>-<worker>-<counter>-<uuid-short>
```

。

---

# 159. `session_id`

对手长期建模必须跨多局关联。

所以：

多个 `game_id` 可以共享：

```text
session_id
```

。

---

# 160. `trace_id`

一次 `agent.think()`：

一个 trace。

内部 Search / Exact：

child spans。

---

# 第二十部分：结构化日志格式

# 161. 本地日志

使用：

\[
\boxed{\text{JSONL}}
\]

不是自由文本。

例如：

```json
{
  "event_type": "SEARCH_COMPLETED",
  "run_id": "run_20260830_001",
  "worker_id": "search_03",
  "game_id": "g_18291",
  "state_hash": "8a7f...",
  "model_version": "R_0017",
  "algorithm": "SM_MCTS",
  "simulations": 2048,
  "runtime_ms": 91.4,
  "root_gap": 0.013,
  "exact_leaf_hits": 84,
  "valid": true
}
```

---

# 162. 禁止高频自由字符串拼接

例如：

```python
logger.info(f"search good {x} {y}")
```

不作为正式 telemetry。

应该：

```python
emit_event(
    SearchCompletedEvent(...)
)
```

。

---

# 163. 人类可读 Console

可以另外输出 compact console：

```text
[SEARCH] g=18291 sims=2048 nodes=1324 gap=.013 91ms exact=84
```

但它只是 JSON event 的渲染。

不是独立信息源。

---

# 第二十一部分：Severity

统一：

```text
DEBUG
INFO
WARNING
ERROR
CRITICAL
```

---

# 164. DEBUG

高频细节。

默认文件可关闭。

例如：

- individual target stats；
- queue batch。

---

# 165. INFO

正常生命周期：

- stage start；
- checkpoint；
- evaluation；
- search result。

---

# 166. WARNING

系统还能继续，但值得关注：

- solver gap 高；
- search fallback；
- exact estimate underestimate；
- queue saturation。

---

# 167. ERROR

本次操作失败：

- Search NaN；
- dataset corrupt；
- worker task failed。

---

# 168. CRITICAL

整体运行应该停止：

- rules invariant broken；
- exact/reference mismatch；
- checkpoint corrupt；
- widespread NaN。

---

# 第二十二部分：训练 Metrics 命名

所有 metric 使用：

```text
namespace/subsystem/name
```

。

---

# 169. Robust Q

```text
train/robust_q/loss
train/robust_q/bellman_residual
train/robust_q/q_mean
train/robust_q/q_std
train/robust_q/target_mean
train/robust_q/ensemble_jsd
```

---

# 170. Actor

```text
train/actor/neurd_loss
train/actor/nash_anchor_kl
train/actor/policy_entropy
train/actor/positive_regret
train/actor/negative_regret
train/actor/actor_nash_jsd
```

---

# 171. Outcome

```text
train/outcome/mc_ce
train/outcome/td_lambda_loss
train/outcome/value_mae
train/outcome/win_prob
train/outcome/draw_prob
train/outcome/loss_prob
```

---

# 172. Opponent

```text
train/opponent/fused_nll
train/opponent/short_nll
train/opponent/long_nll
train/opponent/top1
train/opponent/brier
train/opponent/ece
train/opponent/style_loss
train/opponent/switch_auc
train/opponent/ensemble_jsd
```

---

# 173. Adaptive

```text
train/adaptive/q_loss
train/adaptive/policy_kl
train/adaptive/distribution_loss
train/adaptive/delta_q_norm
train/adaptive/opponent_confidence
train/adaptive/expected_gain
```

---

# 第二十三部分：Solver Metrics

# 174. Matrix Nash

```text
solver/matrix/value
solver/matrix/duality_gap
solver/matrix/iterations
solver/matrix/runtime_ms
solver/matrix/failure_rate
```

---

# 175. Exact

```text
solver/exact/runtime_ms
solver/exact/states
solver/exact/matrix_games
solver/exact/cache_hit_rate
solver/exact/memory_bytes
solver/exact/estimate_ratio
```

---

# 176. Exact Risk

记录：

```text
solver/exact/risk_green
solver/exact/risk_yellow
solver/exact/risk_orange
solver/exact/risk_red
solver/exact/risk_black
```

调用次数。

---

# 第二十四部分：Search Metrics

# 177. SM-MCTS

```text
search/sm_mcts/simulations
search/sm_mcts/nodes
search/sm_mcts/nodes_per_sec
search/sm_mcts/root_gap
search/sm_mcts/policy_shift_jsd
search/sm_mcts/strategy_stability
search/sm_mcts/exact_leaf_ratio
search/sm_mcts/runtime_ms
search/sm_mcts/fallback_rate
```

---

# 178. GT-CFR

```text
search/gt_cfr/iterations
search/gt_cfr/nodes
search/gt_cfr/frontier_count
search/gt_cfr/regret
search/gt_cfr/root_gap
search/gt_cfr/runtime_ms
search/gt_cfr/teacher_accept_rate
```

---

# 第二十五部分：Router Metrics

# 179. Tool Choice

```text
router/direct_matrix_rate
router/exact_rate
router/sm_mcts_rate
router/gt_cfr_rate
router/adaptive_rate
```

---

# 180. Trigger

```text
router/search_score
router/uncertainty_component
router/disagreement_component
router/importance_component
router/failure_component
```

---

# 181. Tool Value

```text
router/search_policy_shift
router/search_value_shift
router/search_confirmation_rate
router/value_gain_per_ms
```

后期用于 learned VOC Router。

---

# 第二十六部分：Training Pipeline Metrics

# 182. Dataset Mix

每个 update window：

```text
data/exact_fraction
data/selfplay_fraction
data/sft_fraction
data/failure_fraction
data/reanalysis_fraction
data/opponent_fraction
```

。

---

# 183. Teacher

```text
teacher/exact_fraction
teacher/search_fraction
teacher/cfr_fraction
teacher/pseudo_fraction
teacher/pseudo_accept_rate
teacher/disagreement
```

---

# 184. Stage

```text
pipeline/current_stage
pipeline/global_step
pipeline/stage_step
pipeline/games_generated
pipeline/states_generated
```

。

---

# 第二十七部分：League Metrics

# 185. Population

```text
league/robust_count
league/aggressive_count
league/exploiter_count
league/payoff_antisymmetry_error
league/novelty_mean
league/main_exploitability
```

。

---

# 186. Cross-Play Matrix

不能只记录标量。

每次 major evaluation：

保存完整：

\[
G_{ij}
\]

作为 artifact。

---

# 187. Nemesis

记录当前 Main 最危险历史 opponent：

```text
league/nemesis_id
league/nemesis_advantage
```

。

---

# 第二十八部分：Red-Team Metrics

# 188. Failure Discovery

```text
redteam/failures_found
redteam/new_failure_types
redteam/exploit_advantage
redteam/localization_depth
```

---

# 189. Correction

```text
redteam/correction_success_rate
redteam/original_attack_after_fix
redteam/general_regression_count
```

---

# 190. Regression Suite

```text
regression/total
regression/pass
regression/fail
regression/newly_broken
```

。

---

# 第二十九部分：System / GPU Metrics

# 191. GPU

每 1–5 秒采样：

```text
system/gpu/utilization
system/gpu/memory_used
system/gpu/power
system/gpu/temperature
system/gpu/sm_clock
```

多 GPU 加 device ID。

---

# 192. CPU

```text
system/cpu/utilization
system/cpu/rss
system/cpu/load
```

---

# 193. Queue

```text
system/queue/actor_depth
system/queue/search_leaf_depth
system/queue/exact_depth
system/queue/reanalysis_depth
```

。

---

# 194. Throughput

```text
system/throughput/games_per_sec
system/throughput/states_per_sec
system/throughput/updates_per_sec
system/throughput/leaf_evals_per_sec
```

。

---

# 第三十部分：Trace 体系

# 195. Agent Think Trace

一次决策：

```text
agent.think
├── model.forward
├── matrix_nash.solve
├── exact.estimate
├── exact.solve        optional
├── search.sm_mcts     optional
│   ├── leaf_batch...
│   └── exact_leaf...
├── adaptive.predict   optional
├── adaptive.search    optional
└── safe_lp.solve
```

。

---

# 196. Training Update Trace

```text
learner.update
├── load_batch
├── collate
├── h2d
├── forward
├── target_build
│   └── matrix_solver
├── loss
├── backward
├── optimizer_step
└── logging
```

。

---

# 197. Search Trace Sampling

不能每个 simulation 都 trace。

太重。

只 trace：

- root；
- leaf batch；
- exact task；
- search summary。

---

# 198. Trace Sampling Rate

正常训练：

例如：

\[
1\%
\]

完整 traces。

ERROR：

100%。

DEBUG session：

100%。

---

# 第三十一部分：MLflow 使用规范

# 199. 一个 Experiment 对应什么

例如：

```text
Goofspiel-Robust-RL
Goofspiel-Pretraining
Goofspiel-Search-Ablation
```

。

---

# 200. 一个 Run 对应

一次完整：

- 配置；
- seed；
-代码 commit；
-训练/评测执行。

---

# 201. MLflow 记录

### Params

resolved Hydra config 摘要。

### Metrics

低频聚合 metrics。

### Artifacts

- full config；
- checkpoints；
- payoff matrix；
- plots；
- regression report；
- profiler traces。

---

# 202. 高频指标不要每一步全部发 MLflow

例如 learner：

每 step 产生几十指标。

本地先聚合：

```text
mean
std
min
max
p50
p95
```

每：

\[
100
\]

updates 写一次 MLflow。

原始高频事件留 JSONL。

---

# 第三十二部分：Log Storage

# 203. 推荐结构

```text
artifacts/
└── runs/
    └── <run_id>/
        ├── events/
        │   ├── learner.jsonl
        │   ├── actors.jsonl
        │   ├── search.jsonl
        │   └── evaluator.jsonl
        │
        ├── traces/
        ├── metrics/
        ├── profiler/
        ├── failures/
        ├── configs/
        └── reports/
```

---

# 204. Log Rotation

JSONL：

按：

- 文件大小；
或
- 时间；

rotate。

不要单个 800GB 文件。

---

# 205. Compression

完成后的历史 JSONL：

可 gzip/zstd。

实时文件不压缩。

---

# 第三十三部分：错误日志必须包含什么

# 206. Exception Event

至少：

```text
exception_type
message
stack_trace
state_hash
game_id
model_version
worker_id
last_successful_operation
```

。

---

# 207. Numerical Failure

额外保存：

- offending tensor statistics；
- min/max/mean/std；
- NaN count；
- relevant masks。

不要直接 dump 整个巨大 tensor 到 console。

---

# 208. Search Failure

保存：

- root state；
- model version；
- search config；
- RNG seed；
- partial result。

确保能够重放。

---

# 第三十四部分：Replay / Reproduction

# 209. 每个严重 bug 都应生成 Reproducer

例如：

```bash
python tools/reproduce_failure.py \
  --failure-id F_00182
```

应该能：

1. load state；
2. load model；
3. load config；
4. fixed seed；
5. rerun。

---

# 210. Reproducer 是 Regression Test 的来源

bug 修好后：

将 failure fixture 加：

```text
tests/regression/bugs/
```

。

---

# 第三十五部分：Checkpoint Validation

# 211. 保存前检查

checkpoint：

- 无 NaN；
- parameter count；
- config；
- version；
- optimizer；
- RNG。

---

# 212. 保存后重新读取

必须：

```text
save
→ new process load
→ checksum
→ inference smoke test
```

。

---

# 213. Checksum

checkpoint 文件保存：

```text
SHA256
```

。

防止 silent corruption。

---

# 214. Promotion Test

候选 Main：

必须先跑固定 promotion suite。

禁止训练进程直接：

```text
best = latest
```

。

---

# 第三十六部分：CI 分层

# 215. Pull Request CI

目标：

<10 分钟左右。

运行：

- L0；
- L1；
-关键 L2；
- CPU parity small；
- Model shape；
- Gradient routing；
- tiny integration。

---

# 216. GPU PR CI

如果有 GPU CI：

额外：

- GPU Nash small parity；
- model CUDA forward；
- compiled/eager smoke。

---

# 217. Nightly CI

运行：

- 10k random Nash parity；
- N≤4 Exact exhaustive；
- C++/Python parity；
- CPU/GPU transition parity；
- small search convergence；
- N=2/N=3 mini training。

---

# 218. Weekly CI

运行：

- performance benchmark；
- 1–4h training；
- search scaling；
- league smoke；
- data stress；
- memory tests。

---

# 219. Release / Main Promotion Validation

完整：

- Exact benchmark；
- exploitability；
- cross-play；
- opponent benchmark；
- adaptive benchmark；
- regression suite；
- performance；
- checkpoint reload。

---

# 第三十七部分：固定 Golden Test Set

# 220. 必须建立不可随意变化的 `golden/`

包括：

### Golden States

几百个手工/随机固定 state。

### Golden Exact

小 N 精确结果。

### Golden Matrices

matrix Nash test。

### Golden Opponents

固定 synthetic styles。

### Golden Failures

历史漏洞。

---

# 221. Golden Set 不能拿来训练

它是：

\[
\boxed{\text{测试集}}
\]

除非明确创建副本进入 correction dataset。

测试版本仍保持独立。

---

# 第三十八部分：数值容差规范

# 222. 不允许测试里乱写 tolerance

建立：

```python
tests/tolerances.py
```

例如：

```text
FP64_EXACT = 1e-9
FP32_SOLVER = 1e-4
BF16_MODEL = 5e-3
SEARCH_POLICY = 2e-2
```

。

每个 tolerance 有语义。

---

# 223. Relax Tolerance 必须 Review

如果 test fail：

不能直接：

```text
1e-4 → 0.1
```

让它通过。

必须解释数值误差来源。

---

# 第三十九部分：随机性测试

# 224. 两种模式

### Deterministic Test Mode

固定：

- RNG；
- deterministic algorithms 尽可能开启；
- worker count。

### Production Training Mode

允许高性能 non-deterministic kernel。

---

# 225. Statistical Test

对于随机 mixed policy：

不能要求每次 action 一样。

应该跑：

\[
10000
\]

samples，

验证经验分布接近目标。

---

# 第四十部分：测试覆盖率

# 226. 不迷信单一 Code Coverage

但核心模块建议：

```text
game      >95%
math      >95%
learning  >90%
reasoning >90%
```

。

UI / glue code 可以低一些。

---

# 227. 更重要的是 Semantic Coverage

必须有：

-边界；
-非法输入；
-随机；
-退化 matrix；
- exact；
- approximate；
- fallback；
- timeout；
- cache；
- version mismatch。

---

# 第四十一部分：科研日志

除了工程日志，还必须保存科研可解释数据。

---

# 228. 每次 Major Evaluation 生成 Report

例如：

```text
evaluation_report.json
```

包含：

- exact error；
- exploitability；
- cross-play；
- variable-N；
- opponent；
- adaptive；
- search scaling。

---

# 229. Q Error 分布

不仅记录 mean。

必须：

- median；
- P90；
- P99；
- by N；
- by horizon；
- by prize；
- by uncertainty。

---

# 230. Exploitability 分解

如果 approximate：

记录：

- oracle type；
- BR training budget；
- confidence interval。

不要只写一个数字不说怎么算的。

---

# 231. Opponent Metrics 分解

按：

- style；
- session length；
- switch/no-switch；
- seen/unseen opponent；

分别统计。

---

# 232. Search Gain 分解

按：

- uncertainty bucket；
- horizon；
- N；
- state importance；

看搜索哪里有效。

---

# 第四十二部分：Dashboard

成熟系统建议至少三个 dashboard。

---

# 233. Training Dashboard

显示：

- loss；
- exact error；
- exploitability；
- throughput；
- GPU；
- dataset mix。

---

# 234. Agent Reasoning Dashboard

单局逐手：

- Q；
- actor；
- Nash；
- search；
- opponent；
- adaptive；
- final policy；
- tool provenance。

---

# 235. System Dashboard

显示：

- workers；
- queues；
- GPU；
- CPU；
- errors；
- search latency；
- exact workers。

---

# 第四十三部分：Detector 日志

你已经有实时 Detector 的概念。

最终 Detector 应消费统一 `ReasoningEvent`。

例如：

```text
━━ GAME 00182 · N=13 · ROBUST+ADAPTIVE ━━

#07 Prize=Q

MODEL
Qμ=+0.081  UQ=.034
Actor↔Nash JSD=.072

NASH
V=+.067  gap=.002

EXACT
risk=RED  estimated=42.8s  skipped

SEARCH
SM-MCTS sims=2048 nodes=1381
gap=.011 stability=.008
exact-leaves=94/1017
Δπ=.063 runtime=116ms

OPP
top=9  p=.41
entropy=1.21 epistemic=.033
switch=.08

SAFE
ε=.015
adaptive-gain=.031
robust-floor=.052

FINAL
policy=[...]
sample=10
```

所有字段来自结构化 events。

Detector 本身不重新计算。

---

# 第四十四部分：日志不能改变算法行为

这是硬规则。

打开 DEBUG logging：

模型行为必须不变。

禁止：

- logger 调 RNG；
- diagnostics 修改 state；
- debug path 改 search budget。

---

# 第四十五部分：敏感性能路径日志

Search simulation loop：

禁止每 simulation 写日志。

训练每 tensor：

禁止日志。

必须：

\[
\boxed{\text{aggregate then emit}}
\]

。

---

# 第四十六部分：日志背压

异步 log queue 满：

不能阻塞 learner 几秒。

策略：

- ERROR/CRITICAL 不丢；
- DEBUG 可 drop；
- INFO 可采样。

同时记录：

```text
logging/dropped_events
```

。

---

# 第四十七部分：最终必须实现的基础工具

建议：

```text
observability/
├── events.py
├── event_bus.py
├── jsonl_sink.py
├── console_sink.py
├── metric_aggregator.py
├── mlflow_sink.py
├── trace.py
├── profiler.py
├── system_metrics.py
└── replay.py
```

---

# 第四十八部分：测试工具

```text
testing/
├── fixtures/
├── generators/
│   ├── random_state.py
│   ├── random_matrix.py
│   └── synthetic_opponent.py
│
├── oracles/
│   ├── scipy_nash.py
│   ├── python_exact.py
│   └── openspiel.py
│
├── assertions/
│   ├── probability.py
│   ├── symmetry.py
│   └── gradients.py
│
└── benchmarks/
```

---

# 第四十九部分：Codex 禁止事项

## 禁止 1

禁止以“训练 loss 在下降”为正确性证明。

---

## 禁止 2

禁止删除 Reference Backend。

---

## 禁止 3

禁止为了让 test 通过随意放宽 tolerance。

---

## 禁止 4

禁止 Search 测试只看“是否返回 action”。

---

## 禁止 5

禁止不测试 simultaneous privacy。

---

## 禁止 6

禁止 Adaptive test 修改 Robust history 后 Robust 输出变化。

---

## 禁止 7

禁止将所有测试塞进一个文件。

---

## 禁止 8

禁止使用随机测试但不保存失败 seed。

---

## 禁止 9

任何 randomized test fail：

必须打印：

```text
seed
state
config
```

。

---

## 禁止 10

禁止日志依赖 print。

---

## 禁止 11

禁止训练日志只有：

```text
loss
reward
```

。

---

## 禁止 12

禁止没有：

```text
run_id
model_version
state_hash
```

的 Search/Failure event。

---

## 禁止 13

禁止只保存最终 aggregate，不保存严重 failure 原始复现信息。

---

## 禁止 14

禁止 Red-Team 修复后删除失败 case。

---

## 禁止 15

禁止性能优化没有前后 benchmark。

---

## 禁止 16

禁止 benchmark 混用不同硬件然后直接比较。

---

## 禁止 17

禁止 checkpoint promotion 只看 Elo。

---

## 禁止 18

禁止 high-frequency logging 阻塞 GPU learner。

---

## 禁止 19

禁止错误被 catch 后 silent continue。

至少 WARNING/ERROR event。

---

## 禁止 20

禁止在测试中 monkeypatch 掉核心数学逻辑然后声称通过 end-to-end。

---

# 第五十部分：开发者日常工作流

每次开发：

```text
修改代码
↓
运行相关 Unit Tests
↓
运行 Property Tests
↓
运行 Reference Parity
↓
运行 Integration Test
↓
若涉及数学/学习
    → N=2/3 Convergence
↓
若涉及性能
    → Benchmark Before/After
↓
提交
↓
CI
```

---

# 第五十一部分：Bug 工作流

发现 bug：

```text
Observe
↓
Capture Event
↓
Create Reproducer
↓
Write Failing Test
↓
Fix
↓
Test Pass
↓
Add Regression Fixture
↓
Run General Regression
```

顺序必须是：

\[
\boxed{\text{先让测试复现，再修}}
\]

。

---

# 第五十二部分：性能优化工作流

发现慢：

```text
Measure
↓
Profiler Trace
↓
Identify Bottleneck
↓
Create Performance Benchmark
↓
Optimize
↓
Correctness Parity
↓
Re-benchmark
↓
Accept / Reject
```

。

如果：

> 快 20%，但数学误差明显变大，

默认拒绝。

除非算法规格允许 approximate tradeoff。

---

# 第五十三部分：模型晋级工作流

候选：

\[
R_{candidate}
\]

必须经过：

```text
Checkpoint Integrity
↓
Exact Benchmark
↓
N=3/5 Regression
↓
Cross-play
↓
Exploitability
↓
Historical Regression
↓
Red-Team Regression
↓
Opponent Calibration
↓
Adaptive Safety
↓
Performance
```

全部满足才成为：

\[
R_{main}
\]

。

---

# 第五十四部分：最终测试金字塔

整个项目应该形成：

```text
                         ┌──────────────┐
                         │ Full Release │
                         │ Evaluation   │
                         └──────┬───────┘
                      ┌─────────▼─────────┐
                      │ Regression / Red │
                      │ Team / Converge  │
                      └─────────┬─────────┘
                   ┌────────────▼────────────┐
                   │ End-to-End Integration │
                   └────────────┬────────────┘
                ┌───────────────▼──────────────┐
                │ Oracle / Mathematical Parity │
                └───────────────┬──────────────┘
             ┌──────────────────▼──────────────────┐
             │ Property / Randomized / Fuzz Tests │
             └──────────────────┬──────────────────┘
                          ┌─────▼─────┐
                          │ Unit Test │
                          └───────────┘
```

底层运行频繁。

越上层越昂贵，但越接近真正能力。

---

# 第五十五部分：最终可观测性结构

同时形成：

```text
                         RUN
                          │
         ┌────────────────┼────────────────┐
         │                │                │
      Metrics           Events           Traces
         │                │                │
      MLflow           JSONL          Trace Store
         │                │                │
         └───────────┬────┴─────┬──────────┘
                     │          │
                 Dashboard   Reproducer
                     │          │
                     └────┬─────┘
                          │
                       Developer
```

---

# 结论

这套测试与日志体系的最终目标不是：

> “程序不报错。”

而是让我们在任何时刻都能够回答下面这些问题：

### 数学正确吗？

可以用：

\[
Reference\ Solver
\]

证明。

### GPU 快版本算错了吗？

可以用：

\[
Cross\text{-}Backend\ Parity
\]

证明。

### 模型真的学到了吗？

可以用：

\[
Small\text{-}N\ Exact\ Convergence
\]

证明。

### Self-play 是不是自嗨？

可以用：

\[
Exploitability
+
Cross\text{-}Play
\]

检查。

### Search 真的更好吗？

可以用：

\[
Network\rightarrow Search
\]

strength/compute curve 检查。

### Opponent model 真的读懂对手了吗？

可以用：

\[
NLL+Brier+ECE+Switch
\]

验证。

### Adaptive exploit 是不是把自己暴露了？

可以用：

\[
Robust\ Safety\ Constraint
\]

验证。

### Red Team 修好的漏洞又回来了吗？

可以用：

\[
Permanent\ Regression\ Suite
\]

发现。

### 某一手为什么这么下？

可以从：

\[
game_id
\rightarrow
state_hash
\rightarrow
trace
\rightarrow
tool\ provenance
\]

完整还原。

### 某次科研结果能不能复现？

可以从：

\[
run_id
\rightarrow
config
\rightarrow
git\ commit
\rightarrow
dataset
\rightarrow
checkpoint
\rightarrow
seed
\]

重新运行。

因此最终原则是：

\[
\boxed{
\textbf{
任何重要结果都必须可验证，
任何重要失败都必须可复现，
任何重要优化都必须可比较，
任何重要决策都必须能追溯来源。
}
}
\]

只有做到这一点，这个项目才不是一个“复杂但黑箱的 RL 工程”，而是一个真正可以长期迭代、做科研、做消融、做性能优化，并且能够相信实验结论的完整博弈智能系统。