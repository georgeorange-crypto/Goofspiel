# Goofspiel-13 Final Decision Protocol
## 最终决策协议：Robust Tool Selection · Adaptive Gate · Safe LP · Budget Fallback · Mixed-Policy Sampling

---

# 0. 唯一目标

本文只规定一件事：

> 当 Model、Matrix Nash、Exact、SM-MCTS、GT-CFR、Opponent Model、Adaptive Search 都可能给出结果时，最终到底相信谁，以及怎样产生这一手真实 action。

任何实现不得自行改变本协议。

最终流程固定为：

\[
\boxed{
\text{Robust Result}
\rightarrow
\text{Adaptive Gate}
\rightarrow
\text{Safe LP}
\rightarrow
\text{Final Mixed Policy}
\rightarrow
\text{Categorical Sample}
}
\]

---

# 1. Robust Result 的基础优先级

首先得到：

\[
R_0=\text{MatrixNash}(Q_R^\theta)
\]

它是**永远存在的 fallback**。

Robust Tool 的理论优先级：

\[
\boxed{
Exact
>
Accepted\ GT\text{-}CFR
>
Accepted\ SM\text{-}MCTS
>
MatrixNash(Q_R)
>
RawActor
}
\]

Raw Actor：

\[
\pi_R^{actor}
\]

不得直接作为正常 PLAY 的 Robust 最终结果。

它主要用于：

- NeuRD actor；
- search trigger；
- diagnostics；
- emergency fallback when Matrix Solver itself fails。

---

# 2. Tool Result 必须先通过 Validity Gate

任何 tool result 在参与优先级比较前必须满足：

### 通用条件

- 无 NaN；
- 无 Inf；
- legal action probability 非负；
- illegal probability = 0；
- probability sum ≈ 1；
- state hash 一致；
- model version 一致；
- 未读取非法 simultaneous information。

否则：

\[
\boxed{INVALID}
\]

立即丢弃。

---

# 3. Exact 的特殊规则

如果 Exact 返回：

```text
valid = true
exactness = NUMERICAL_EXACT
```

或：

```text
RATIONAL_EXACT
```

则：

\[
\boxed{
R_{\text{robust}}=R_{\text{exact}}
}
\]

立即停止 Robust Tool 竞争。

SM-MCTS、GT-CFR、Matrix Nash：

均不得覆盖 Exact。

它们可以在 EVALUATION 模式运行做比较，但不能改变 PLAY 的 Robust 结果。

---

# 4. Search Quality Gate

## SM-MCTS

默认接受条件：

```yaml
sm_mcts_accept:
  min_simulations: 128
  max_root_duality_gap: 0.03
  max_strategy_instability_jsd: 0.03
```

任何一项失败：

```text
SM_MCTS_UNRELIABLE
```

不得覆盖 Matrix Nash。

---

## GT-CFR

默认接受条件：

```yaml
gt_cfr_accept:
  min_iterations: 256
  max_root_duality_gap: 0.01
  max_strategy_instability_jsd: 0.02
```

失败：

不得作为最终 Robust result，也不得作为 Teacher。

---

# 5. 多个 Approximate Search 冲突怎么办

若 Exact 不存在，同时有多个有效 Search result：

禁止：

\[
\boxed{\text{平均多个 Search policy}}
\]

。

先比较统一：

```text
quality_score
```

选择：

\[
\arg\max_i quality_i
\]

。

如果：

\[
|quality_1-quality_2|<0.02
\]

则 tie-break：

\[
\boxed{
GT\text{-}CFR
>
SM\text{-}MCTS
}
\]

再相同时：

选择计算预算更高者。

因此：

```text
quality first
→ algorithm class
→ compute budget
```

而不是看到 `GT-CFR` 名字就无条件选择。

---

# 6. Search 不稳定怎么办

如果 Search：

- gap 超阈值；
- policy 在最后窗口持续震荡；
- NaN；
- timeout 前没有稳定结果；
- leaf evaluator failure；

则：

\[
\boxed{\text{不要混合 Search 与 Baseline}}
\]

直接 fallback：

\[
R_{\text{robust}}
\leftarrow
\text{上一个已经 certified 的 Robust Result}
\]

一般就是：

\[
MatrixNash(Q_R)
\]

。

如果 Search 在 timeout 前已经产生 `last_certified_snapshot`：

可以返回该 snapshot。

否则完全忽略该 Search。

---

# 7. Matrix Nash 自己失败怎么办

若 GPU Matrix Solver 不可靠：

首先尝试 Reference Solver，只要时间允许。

若仍失败：

使用：

\[
\pi_R^{actor}
\]

作为 emergency fallback。

同时产生：

```text
CRITICAL_SOLVER_FALLBACK
```

日志。

这不属于正常运行路径。

---

# 8. Opponent Model 必须先通过全局资格门

Adaptive 能力不是模型一存在就允许使用。

当前 opponent-model checkpoint 必须被 Evaluator 标记：

```text
opponent_model_usable = true
```

最低要求：

\[
ECE\le0.05
\]

并且：

- NLL 显著优于 uniform；
- Brier 显著优于 baseline；
- switch benchmark 通过；
- 没有 calibration regression。

若：

```text
opponent_model_usable = false
```

则：

\[
\boxed{\text{Adaptive 完全关闭}}
\]

最终直接使用 Robust policy。

---

# 9. 单局实时 Opponent Confidence

全局模型合格后，每一个 decision 再计算：

\[
c_{\text{opp}}\in[0,1]
\]

定义：

\[
c_{\text{evidence}}
=
1-e^{-n_{\text{obs}}/4}
\]

其中：

\[
n_{\text{obs}}
\]

是当前 session 已获得的有效对手行为观察数。

定义 epistemic confidence：

\[
c_{\text{epi}}
=
e^{-U_{\text{opp}}/0.1}
\]

其中：

\[
U_{\text{opp}}
\]

是 opponent ensemble JSD。

定义 switch confidence：

\[
c_{\text{switch}}
=
1-P_{\text{switch}}
\]

最终：

\[
\boxed{
c_{\text{opp}}
=
c_{\text{evidence}}
\cdot
c_{\text{epi}}
\cdot
c_{\text{switch}}
}
\]

并 clamp：

\[
[0,1]
\]

。

---

# 10. Adaptive Confidence Gate

固定三档：

| \(c_{\text{opp}}\) | 决策 |
|---:|---|
| \(<0.60\) | **禁止 Adaptive** |
| \(0.60\le c<0.80\) | 允许 Conservative Adaptive |
| \(c\ge0.80\) | 允许 Full Adaptive Candidate |

“Full Adaptive”仍然：

\[
\boxed{\text{必须经过 Safe LP}}
\]

绝不意味着直接执行 adaptive policy。

---

# 11. Unknown Opponent

若没有任何历史：

\[
n_{\text{obs}}=0
\]

则：

\[
c_{\text{opp}}=0
\]

因此：

\[
\boxed{\pi_{final}=\pi_R}
\]

。

这是默认安全行为。

---

# 12. Strategy Switch 的处理

若：

\[
P_{\text{switch}}
\]

突然升高，

则：

\[
c_{\text{opp}}
\]

自动下降。

因此系统会自动：

\[
Adaptive
\rightarrow
Conservative Adaptive
\rightarrow
Robust
\]

而不是继续相信已经过时的 Mamba 长期风格。

---

# 13. Adaptive Result 的优先级

如果允许 Adaptive：

Adaptive candidate 优先级：

\[
\boxed{
ExactBR(q_{\text{opp}})
>
Accepted\ AdaptiveSearch
>
Q_A\text{-based SoftBR}
>
AdaptivePolicyHead
}
\]

其中：

`ExactBR` 的含义是：

\[
\boxed{\text{EXACT\_WRT\_OPPONENT\_MODEL}}
\]

不是整个游戏的无条件 Exact。

---

# 14. Exact 与 Adaptive 怎么结合

这是硬规则：

> **Exact Nash 决定安全底线；Adaptive 决定是否在这个安全底线上进一步 exploit。**

如果 Robust Exact 成功：

得到：

\[
Q_R^*
\]

\[
\pi_R^*
\]

\[
V_R^*
\]

。

若：

\[
c_{\text{opp}}<0.60
\]

：

\[
\boxed{
\pi_{final}=\pi_R^*
}
\]

。

若：

\[
c_{\text{opp}}\ge0.60
\]

：

允许 Adaptive candidate，

但 Safe LP 必须使用：

\[
\boxed{Q_R^*}
\]

作为 robust safety matrix。

因此 Exact 并不自动禁止 exploit。

它提供的是**最高可信的安全约束**。

---

# 15. Exact BR 与 Exact Nash 同时存在

如果：

- Robust Exact Nash 可解；
- 对当前 opponent model 的 Exact BR 也可解；

则：

Robust：

\[
Q_R^*,V_R^*
\]

Adaptive objective：

使用 Exact BR 所提供的 opponent-conditioned value。

然后仍然求：

\[
\boxed{\text{Safe LP}}
\]

。

禁止：

> “Exact BR 更赚钱，所以直接执行 Exact BR。”

因为它只对：

\[
q_{\text{opp}}
\]

正确时成立。

---

# 16. Safe Exploit Budget

默认最大允许 Robust value sacrifice：

\[
\epsilon_{\max}=0.02
\]

这里 utility 已 normalized 到：

\[
[-1,1]
\]

。

实际允许：

\[
\epsilon_{\text{eff}}
=
\epsilon_{\max}
\cdot
\operatorname{clip}
\left(
\frac{c_{\text{opp}}-0.60}{0.40},
0,
1
\right)
\]

所以：

### \(c=0.60\)

\[
\epsilon_{\text{eff}}=0
\]

### \(c=0.80\)

\[
\epsilon_{\text{eff}}=0.01
\]

### \(c=1.00\)

\[
\epsilon_{\text{eff}}=0.02
\]

。

---

# 17. Safe LP 是最终 Adaptive Controller

计算 adaptive action utility：

\[
c_a
=
\sum_b
q_{\text{opp}}(b)
Q_A(a,b)
\]

求：

\[
\max_\pi c^T\pi
\]

subject to：

\[
\sum_a
\pi(a)
Q_R(a,b)
\ge
V_R-\epsilon_{\text{eff}}
\qquad
\forall b
\]

以及：

\[
\sum_a\pi(a)=1
\]

\[
\pi(a)\ge0
\]

。

---

# 18. Safe LP 失败怎么办

若：

- infeasible；
- numerical error；
- timeout；
- NaN；

则：

\[
\boxed{
\pi_{final}=\pi_R
}
\]

。

禁止 fallback 到 unconstrained Adaptive policy。

---

# 19. Robust Floor 的来源

Safe LP 使用**当前最高质量 Robust Result**：

\[
Exact
>
Accepted Search
>
Matrix Nash
\]

。

不得使用低质量 Raw Actor 来设置 robust safety floor。

---

# 20. Time Budget 总原则

每次 `think()` 都有 deadline：

\[
T_{\text{deadline}}
\]

。

必须预留 finalization 时间：

\[
T_{\text{reserve}}
=
\max
\left(
2\text{ms},
\min(
50\text{ms},
0.05T_{\text{total}}
)
\right)
\]

。

当剩余：

\[
T_{\text{remaining}}
\le
T_{\text{reserve}}
\]

时：

\[
\boxed{\text{禁止启动任何新 Tool}}
\]

立即进入 Finalize。

---

# 21. 时间不足时的降级顺序

固定：

```text
Exact
↓
GT-CFR
↓
SM-MCTS Large
↓
SM-MCTS Medium
↓
SM-MCTS Small
↓
Matrix Nash
↓
Raw Actor emergency only
```

但这是**预算降级顺序**，不是结果优先级。

---

# 22. Exact 调用预算

PLAY 中：

只有 estimator 满足：

\[
T_{\text{exact-est}}
\le
0.30
\times
T_{\text{usable}}
\]

并且 risk：

```text
GREEN or YELLOW
```

才启动完整 Exact。

否则直接进入 Search。

Teacher/Evaluation 可以使用更高比例。

---

# 23. Search 到 deadline 怎么办

Search 必须周期性维护：

```text
last_certified_snapshot
```

。

deadline 到达：

### 有 certified snapshot

返回它。

### 没有

返回：

```text
SEARCH_TIMEOUT_UNCERTIFIED
```

Router fallback 当前 Robust baseline。

绝不允许为了等 Search 多跑几秒。

---

# 24. Adaptive Search 的时间不足处理

如果 Robust 已完成但剩余时间不足进行 Adaptive Search：

仍可使用：

\[
Q_A + q_{\text{opp}}
\]

构造 cheap Soft Best Response candidate。

然后 Safe LP。

如果连 Safe LP 的 finalization reserve 都不足：

完全跳过 Adaptive：

\[
\boxed{\pi_{final}=\pi_R}
\]

。

---

# 25. 任何时候都必须优先保证有 Robust Policy

因此工具预算使用原则：

\[
\boxed{
\text{先保证 Robust 可执行}
\rightarrow
\text{再花时间提升 Robust}
\rightarrow
\text{最后才花时间 Adaptive}
}
\]

不能因为 Adaptive Search 吃掉时间导致没有安全结果。

---

# 26. 最终 Case Table

| Case | Robust | Opponent | Adaptive | 最终 |
|---|---|---|---|---|
| Exact 可用，confidence < .60 | Exact | 不可信 | OFF | Exact Nash |
| Exact 可用，confidence ≥ .60 | Exact | 可用 | Candidate | Exact Q + Safe LP |
| Exact + Exact BR | Exact | 高可信 | Exact BR | Exact Nash safety + Safe LP |
| Exact 不可用，Search 稳定 | Search | 任意 | 按 gate | Search Q + optional Safe LP |
| Search 不稳定 | Matrix Nash | 任意 | 按 gate | fallback Matrix Nash + optional Safe LP |
| 所有 Search timeout | 已认证 Robust | 任意 | 视剩余时间 | 使用最后 certified Robust |
| Opponent global gate fail | 任意 Robust | 禁用 | OFF | Robust |
| \(c<.60\) | 任意 Robust | 低可信 | OFF | Robust |
| \(.60\le c<.80\) | 任意 Robust | 中可信 | Conservative | Safe LP，较小 \(\epsilon\) |
| \(c\ge.80\) | 任意 Robust | 高可信 | Full Candidate | Safe LP |
| Safe LP fail | 任意 Robust | 任意 | fail | Robust |
| Matrix Solver fail | Raw Actor emergency | 任意 | OFF | Actor + CRITICAL event |

---

# 27. Final Policy Sanitization

得到：

\[
\pi_{final}
\]

后必须执行：

### 1. Illegal zeroing

非法动作：

\[
p=0
\]

。

### 2. Numerical clipping

仅允许把微小浮点负数：

\[
-10^{-8}<p<0
\]

clip 到 0。

如果：

\[
p<-10^{-8}
\]

视为算法错误。

### 3. Normalize

\[
\pi
\leftarrow
\frac{\pi}{\sum_a\pi_a}
\]

。

### 4. Validate

必须：

\[
|\sum\pi-1|<10^{-6}
\]

。

---

# 28. 禁止 Top-K / Temperature 修改最终 Nash Policy

最终策略不得擅自：

- top-k；
- top-p；
- temperature；
- probability floor；
- greedy sharpening。

因为 mixed probabilities 是策略本身。

极小但合法的 equilibrium probability：

不能为了“看起来干净”删除。

---

# 29. 最终 Action 必须 Sampling

正式 PLAY：

\[
\boxed{
a\sim Categorical(\pi_{final})
}
\]

禁止：

\[
argmax(\pi)
\]

。

因为对零和 simultaneous game：

随机化本身可能是 Nash strategy 的组成部分。

---

# 30. RNG 规则

### 正常真实 PLAY

使用：

```text
secrets.SystemRandom
```

按照 cumulative probability sampling。

### Training / Reproducible Evaluation

使用：

```text
seeded torch.Generator
```

或统一 seeded RNG。

必须保存 seed/state。

---

# 31. Deterministic Debug Mode

可以提供：

```text
decision_mode=ARGMAX_DEBUG
```

。

它只能用于：

- UI debug；
-人工分析。

不得：

- 作为正式 benchmark；
- 作为 self-play 默认策略；
- 宣称是 Agent 正式性能。

---

# 32. 每次 Action 必须记录

最终 trajectory 至少保存：

```text
robust_source
adaptive_source
opponent_confidence
epsilon_eff

robust_policy
adaptive_policy
final_policy

selected_action
selected_action_probability

tool_quality
tool_runtime

rng_mode
```

从而能够完整回答：

> 为什么这一手是这张牌？

---

# 33. 最终伪代码

```python
def final_decision(ctx):

    # --------------------------------
    # 1. BASE ROBUST
    # --------------------------------

    robust = matrix_nash(ctx.model.q_robust)

    # --------------------------------
    # 2. EXACT
    # --------------------------------

    if exact_is_budget_feasible(ctx):

        exact = exact_nash(ctx.state)

        if exact.valid:
            robust = exact

    # --------------------------------
    # 3. APPROXIMATE SEARCH
    # --------------------------------

    if not robust.is_exact:

        search_candidates = run_allowed_search(ctx)

        accepted = [
            r for r in search_candidates
            if passes_quality_gate(r)
        ]

        if accepted:
            robust = select_best_quality(accepted)

    # --------------------------------
    # 4. OPPONENT GATE
    # --------------------------------

    if not ctx.opponent_model_usable:
        return sample(sanitize(robust.policy))

    c_opp = compute_opponent_confidence(ctx)

    if c_opp < 0.60:
        return sample(sanitize(robust.policy))

    # --------------------------------
    # 5. ADAPTIVE CANDIDATE
    # --------------------------------

    adaptive = best_available_adaptive_result(ctx)

    if adaptive is None:
        return sample(sanitize(robust.policy))

    # --------------------------------
    # 6. SAFE LP
    # --------------------------------

    epsilon_eff = (
        epsilon_max *
        clip((c_opp - 0.60) / 0.40, 0, 1)
    )

    final = safe_lp(
        robust_q=robust.q_matrix,
        robust_value=robust.value,
        adaptive_q=adaptive.q_matrix,
        opponent_policy=ctx.q_opponent,
        epsilon=epsilon_eff,
    )

    if not final.valid:
        final_policy = robust.policy
    else:
        final_policy = final.policy

    # --------------------------------
    # 7. SAMPLE
    # --------------------------------

    final_policy = sanitize(final_policy)

    return categorical_sample(
        final_policy,
        rng=ctx.rng,
    )
```

---

# 34. Codex 禁止事项

1. **禁止** Exact 成功后让 Search 覆盖 Exact。
2. **禁止** 多个 tool policy 简单平均。
3. **禁止** Search 不稳定时“折中混一点 Search”。
4. **禁止** opponent model 未过 calibration gate 就 Adaptive。
5. **禁止** \(c_{\text{opp}}<0.60\) 时 Adaptive。
6. **禁止** Adaptive result 不经过 Safe LP 直接执行。
7. **禁止** Exact BR 被误称整个游戏 Exact。
8. **禁止** Safe LP 失败后 fallback 到 unconstrained Adaptive。
9. **禁止** Search 超时后继续阻塞 final decision。
10. **禁止** Adaptive computation 占掉 Robust finalization budget。
11. **禁止** 最终使用 argmax 代替 mixed-policy sampling。
12. **禁止**对最终 Nash policy做 temperature/top-k。
13. **禁止**删除合法的小概率动作。
14. **禁止**使用 opponent history 修改 Robust result。
15. **禁止**正式决策没有 provenance/log。

---

# 最终一句话规则

整个 Final Decision Protocol 可以压缩为：

\[
\boxed{
\textbf{
先找到当前最高可信的 Robust 策略；
只有在对手模型足够可信时才考虑利用；
任何利用都必须受最高可信 Robust Q 的安全约束；
任何搜索失败都退回已认证结果；
任何预算不足都优先保证 Robust；
最后永远从合法 mixed policy 中随机采样，而不是贪心 argmax。
}
}
\]

其中最核心的优先级只有两条：

\[
\boxed{
\text{Correctness / Robustness}
>
\text{Exploitation}
}
\]

以及：

\[
\boxed{
\text{Exact mathematical knowledge}
>
\text{validated search}
>
\text{neural approximation}
}
\]