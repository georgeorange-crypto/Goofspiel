# VEIL：匿名信息竞价游戏设计方案
**Version 0.1 — Game & Research Environment Specification**

## 1. 核心概念

VEIL 是一个多人、同时行动、有限资源、部分可观测、跨局学习的策略游戏。

每名玩家拥有同样的一组有限牌。每轮所有玩家被随机分配一个临时花色/位置，同时秘密选择一张牌进行竞价。

游戏中存在两种核心收益：

\[
\boxed{\text{高牌争分数，低牌争信息}}
\]

最高出牌者获得当前轮的 **Score Reward**；

最低出牌者获得关于未来奖励的 **Information Reward**。

与此同时，玩家的真实身份可以被隐藏。每轮临时座位重新随机，使其他玩家必须通过历史行为推断：

\[
\boxed{\text{“这个匿名位置上的人到底是谁？”}}
\]

每局结束后，所有隐藏身份、奖励与私人信息全部揭晓并进入公共历史。

因此整个游戏的长期循环为：

\[
\boxed{
Play
\rightarrow
Infer
\rightarrow
Reveal
\rightarrow
Learn
\rightarrow
Remember
\rightarrow
Play\ Again
}
\]

---

# 2. 核心设计原则

### 2.1 结构公平

所有玩家在规则上完全对称。

任何玩家标签置换：

\[
(P_1,P_2,\ldots,P_M)
\rightarrow
(P_{\sigma(1)},P_{\sigma(2)},\ldots,P_{\sigma(M)})
\]

都不会改变游戏规则。

没有：

- 固定先手；
- 固定强势位置；
- 永久高花色；
- 玩家专属能力；
- 初始资源差异。

临时优势全部通过随机位置重新分配。

因此：

\[
\boxed{\text{Fairness by construction}}
\]

---

## 2.2 同时行动

每轮所有玩家同时秘密出牌。

不存在：

\[
Player\ A\rightarrow Player\ B\rightarrow Player\ C
\]

这样的行动顺序。

因此不存在传统棋牌游戏中的先手优势。

---

## 2.3 有限资源

标准模式中每名玩家拥有：

\[
A,2,3,\ldots,10,J,Q,K
\]

共：

\[
13
\]

张牌。

每张牌一局只能使用一次。

因此每次行动既影响当前轮，也影响未来所有轮。

---

## 2.4 分数与信息是两种不同资源

最高牌获得即时收益：

\[
\boxed{\text{Score}}
\]

最低牌获得未来收益：

\[
\boxed{\text{Information}}
\]

从而避免：

\[
\text{牌面越大}\Rightarrow\text{价值永远越高}
\]

这种单调机制。

---

# 3. 参数系统

游戏所有重要机制均参数化。

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `players` | 玩家数量 | **4** |
| `ranks` | 每人牌数 | **13** |
| `match_games` | 每场 Match 局数 | **7** |
| `seat_shuffle` | 是否随机位置 | **每轮重新随机** |
| `suit_mode` | 花色/位置优先级 | **开启** |
| `prize_visibility` | 当前奖励是否提前公开 | **隐藏** |
| `identity_visibility` | 对手真实身份 | **每轮匿名** |
| `identity_reveal` | 身份何时揭晓 | **每局结束** |
| `info_reward` | 最低者是否获信息 | **开启** |
| `info_bits` | 信息量 | **1 bit** |
| `info_target` | 情报针对哪轮 | **下一轮奖励** |
| `history_mode` | 历史是否跨局可用 | **开启** |
| `game_tie_rule` | 单局并列规则 | 可配置 |
| `match_tie_rule` | Match 并列规则 | 胜局→总分→并列 |

---

# 4. 玩家数量

定义：

\[
M=\text{players}
\]

推荐支持：

\[
2\le M\le 4
\]

其中：

### 2 人

最接近传统 Goofspiel，可作为基础研究模式。

### 3 人

最小多人 general-sum 模式。

### 4 人

\[
\boxed{\text{旗舰模式}}
\]

因为刚好对应扑克牌四种花色：

\[
\clubsuit,\diamondsuit,\heartsuit,\spadesuit
\]

同时多人博弈已经足够丰富，而计算复杂度仍然可控。

未来如果需要扩展到：

\[
M>4
\]

可以把“花色”推广为抽象的 Round Priority Role，而不再局限于标准扑克牌花色。

---

# 5. 花色 / 临时位置机制

四人模式固定四个 Round Roles：

\[
\clubsuit<\diamondsuit<\heartsuit<\spadesuit
\]

每轮开始时，将四个真实玩家随机映射到四个位置：

\[
\sigma_t:
Player\rightarrow Suit
\]

例如：

### Round 1

\[
P_1\rightarrow\spadesuit
\]

\[
P_2\rightarrow\clubsuit
\]

\[
P_3\rightarrow\heartsuit
\]

\[
P_4\rightarrow\diamondsuit
\]

Round 2 重新随机。

因此：

\[
\boxed{\text{Suit不是玩家属性}}
\]

而是：

\[
\boxed{\text{Temporary Round Role}}
\]

---

# 6. 出牌大小

玩家实际出的牌可以表示为：

\[
(r,s)
\]

其中：

\[
r=\text{rank}
\]

\[
s=\text{suit}
\]

按照字典序比较：

首先比较点数：

\[
K>Q>J>\cdots>2>A
\]

点数相同时比较花色：

\[
\spadesuit>\heartsuit>\diamondsuit>\clubsuit
\]

因此任何一轮都一定存在唯一：

\[
\boxed{\text{Highest Bid}}
\]

和唯一：

\[
\boxed{\text{Lowest Bid}}
\]

无需额外处理最高并列。

---

# 7. 花色的双向价值

花色不是简单的“♠最好”。

如果争最高：

\[
\spadesuit
\]

同点情况下最有利。

但如果争最低：

\[
\clubsuit
\]

同点情况下最有利。

因此：

\[
\boxed{
\spadesuit
\text{偏向 Score Competition}
}
\]

而：

\[
\boxed{
\clubsuit
\text{偏向 Information Competition}
}
\]

花色价值取决于当前战略目标，而不存在绝对强花色。

---

# 8. 奖励牌系统

标准完整版使用独立的 Prize Deck：

\[
\{1,2,\ldots,13\}
\]

一局开始时随机打乱：

\[
p_1,p_2,\ldots,p_{13}
\]

每轮使用一张。

---

# 9. Hidden Prize 模式

旗舰规则中：

\[
\boxed{\text{玩家出牌前不知道当前奖励}}
\]

第 \(t\) 轮行动阶段：

\[
p_t
\]

仍然隐藏。

所有玩家完成 simultaneous bid 后：

\[
p_t
\]

才公开。

然后 Highest Bid 获得：

\[
p_t
\]

点。

---

# 10. Prize Belief

虽然当前奖励隐藏，但已出现奖励公开。

假设未出现集合：

\[
R_t
\]

则所有没有额外情报的玩家知道：

\[
P(p_t=p)=\frac1{|R_t|}
\]

其中：

\[
p\in R_t
\]

所以隐藏奖励并不是无规则猜测，而是一个明确的概率 belief。

---

# 11. Information Reward

这是完整版的核心机制之一。

每轮：

\[
\boxed{\text{最低出牌者获得下一轮奖励的一条私人信息}}
\]

最高者获得 Score Reward；

最低者获得 Information Reward。

二者分别对应：

\[
\boxed{\text{Immediate Utility}}
\]

和：

\[
\boxed{\text{Future Information Value}}
\]

---

# 12. 标准 1-bit 情报

当前轮结束、奖励揭晓以后，得到下一轮剩余 Prize Set：

\[
R_{t+1}
\]

按数值排序：

\[
r_{(1)}<r_{(2)}<\cdots<r_{(k)}
\]

按照中位区域分成：

\[
R^{LOW}
\]

和：

\[
R^{HIGH}
\]

系统检查已经预先确定的下一张 Prize：

\[
p_{t+1}
\]

属于哪一半。

最低出牌者私人收到：

\[
\boxed{LOW}
\]

或者：

\[
\boxed{HIGH}
\]

其他玩家不知道该 bit 的值。

---

# 13. 奇数剩余牌处理

如果：

\[
|R_{t+1}|
\]

为奇数，则按照公开固定规则进行尽可能均衡的二分。

推荐：

\[
R^{LOW}
=
\text{前 }\lceil k/2\rceil\text{ 张}
\]

\[
R^{HIGH}
=
\text{其余牌}
\]

这样始终只产生：

\[
1\text{ bit}
\]

的 signal 类型。

所有玩家提前知道划分规则。

---

# 14. 最后一轮信息奖励

最后一轮没有下一张 Prize。

因此默认：

\[
\boxed{\text{最后一轮 Lowest Bid 不产生情报}}
\]

这使低牌的战略价值在游戏末期自然下降。

这是游戏的一部分，而不是异常情况。

也可以通过参数配置最后一轮给予其他补偿，但旗舰规则默认关闭。

---

# 15. Information Reward 可配置等级

标准：

\[
\boxed{Binary\ Half}
\]

即 HIGH / LOW。

扩展模式可以支持：

| 模式 | 信息 |
|---|---|
| `none` | 无情报 |
| `half` | 上半区 / 下半区 |
| `quartile` | 四分位 |
| `parity` | 奇 / 偶 |
| `exact` | 精确下一张奖励 |
| `noisy_half` | 带概率错误的 HIGH / LOW |

旗舰版：

\[
\boxed{\texttt{half}}
\]

---

# 16. 匿名身份机制

Match 中存在固定真实玩家：

\[
P_1,P_2,P_3,P_4
\]

这些身份跨局保持不变。

但是游戏过程中，其他玩家不能看到当前：

\[
Suit\leftrightarrow Player
\]

映射。

公开界面只显示：

\[
\spadesuit,\heartsuit,\diamondsuit,\clubsuit
\]

以及每个匿名位置的公开行为。

---

# 17. 玩家自己的身份

每个玩家始终知道：

\[
\boxed{\text{自己是谁}}
\]

以及：

\[
\boxed{\text{自己当前拿到什么 Suit}}
\]

因此对于一个四人玩家而言，每轮需要推断的只是另外三个玩家的身份。

可能 permutation：

\[
3!=6
\]

种。

这使四人模式具有非常漂亮的 identity belief structure。

---

# 18. 每轮重新匿名

旗舰规则：

\[
\boxed{\text{每轮重新随机位置}}
\]

因此某一轮的：

\[
\spadesuit
\]

不一定是下一轮的：

\[
\spadesuit
\]

对应的同一个真实玩家。

玩家只能根据：

- 历史动作；
- 已使用牌；
- 花色条件下的行为；
- 信息获得情况；
- 长期玩家风格；

进行身份推断。

---

# 19. Identity Belief

对于观察者 \(i\)，可以维护：

\[
P_i(Player_j\mid Seat_s,H_t)
\]

例如：

\[
P(\spadesuit=P_2)=0.72
\]

\[
P(\spadesuit=P_3)=0.21
\]

等等。

四人模式甚至可以直接维护完整的：

\[
P(\sigma_t\mid H_t)
\]

因为只有六种对手身份排列。

---

# 20. 身份与资源耦合

每个真实玩家自己的 13 张 rank 一局只能使用一次。

因此身份推断必须满足资源一致性。

例如模型认为某匿名位置曾经是：

\[
P_2
\]

并观察到：

\[
P_2
\]

出了 K。

如果之后另一个匿名位置再次出 K，那么两个位置不可能都属于同一个：

\[
P_2
\]

。

所以身份推断天然包含：

\[
\boxed{
Behavioral Evidence
+
Resource Constraints
}
\]

---

# 21. Information Ownership 也可以隐藏

假设上一轮：

\[
\clubsuit
\]

是 Lowest Bid。

所有人知道：

\[
\clubsuit
\]

位置的真实玩家获得了一条 PRIVATE HIGH/LOW signal。

但是由于：

\[
\clubsuit\leftrightarrow Player
\]

身份仍然隐藏，

其他玩家未必知道：

\[
\boxed{\text{到底哪个长期玩家掌握了情报}}
\]

下一轮再次随机位置后，他们甚至还必须推断：

> 当前哪个匿名位置可能是上一轮信息拥有者？

于是游戏同时具有：

\[
\boxed{\text{Hidden Identity}}
\]

和：

\[
\boxed{\text{Hidden Information Ownership}}
\]

---

# 22. 一轮完整流程

旗舰规则下一轮完整执行顺序：

### Phase A — Seat Shuffle

随机生成：

\[
Player\rightarrow Suit
\]

映射。

每名玩家只知道自己的映射。

---

### Phase B — Public State

公开：

- 当前轮数；
- 当前累计分数；
- 已经揭晓的 Prize；
- 当前剩余 Prize 集合；
- 四种 Suit；
- 所有过去匿名行动；
- 上一轮哪个 Suit 获得 Information Reward。

---

### Phase C — Private State

每名玩家额外知道：

- 自己真实身份；
- 自己当前 Suit；
- 自己剩余手牌；
- 自己获得过且仍有效的私人情报；
- 过去 Match 中已经公开的历史。

---

### Phase D — Hidden Prize

当前：

\[
p_t
\]

不公开。

---

### Phase E — Simultaneous Bid

所有玩家从自己剩余 rank 中秘密选择一张。

---

### Phase F — Joint Reveal

同时公开：

\[
Suit + Rank
\]

但仍然不公开真实玩家身份。

---

### Phase G — Prize Reveal

公开：

\[
p_t
\]

---

### Phase H — Score Reward

Highest Bid 获得：

\[
p_t
\]

点。

---

### Phase I — Information Reward

Lowest Bid 获得关于：

\[
p_{t+1}
\]

的 HIGH / LOW 私人 signal。

---

### Phase J — Resource Removal

所有玩家使用的 rank 从其真实手牌中永久移除。

进入下一轮。

---

# 23. 一局结束

标准：

\[
13
\]

轮后，每个人全部使用完自己的 13 张 rank。

统计：

\[
S_i
\]

为玩家 \(i\) 的单局总分。

单局排名按照：

\[
S_i
\]

决定。

最高者获得：

\[
1\text{ Game Win}
\]

---

# 24. 单局并列

参数：

`game_tie_rule`

可以选择：

### `shared_win`

所有并列最高者各记一次胜局。

### `fractional_win`

若 \(k\) 人并列，每人获得：

\[
1/k
\]

Win Point。

### `draw`

该局无人记胜。

旗舰版建议：

\[
\boxed{\texttt{shared\_win}}
\]

因为规则最直观。

---

# 25. 局末全透明揭晓

每局结束后系统公布完整 Truth Log：

- 每轮 Prize；
- 每轮真实 Player → Suit 映射；
- 每轮所有玩家出牌；
- 每轮 Highest Bid；
- 每轮 Lowest Bid；
- 每轮私人 HIGH / LOW signal；
- 每个玩家最终得分。

过去的隐藏变量全部变为：

\[
\boxed{\text{Public History}}
\]

---

# 26. 跨局学习

局末揭晓的完整信息可以进入后续游戏。

因此玩家可以逐步形成长期画像：

\[
m_j
\]

例如：

\[
m_2=
\text{“拿♠时更激进、落后时喜欢抢情报……”}
\]

下一局重新匿名后，玩家可以利用这些历史重新识别：

\[
P(Player_j\mid CurrentBehavior)
\]

所以：

\[
\boxed{\text{每局不是独立游戏}}
\]

而是一场连续学习过程。

---

# 27. Match 制

完整比赛不是一局，而是：

\[
\boxed{G\text{ 局组成一个 Match}}
\]

旗舰：

\[
\boxed{G=7}
\]

即：

\[
Best\text{-}of\text{-}7\ style
\]

但并非两人淘汰式 Bo7，而是四人累计排名赛。

所有七局中的玩家身份保持不变，因此跨局 opponent modeling 有意义。

---

# 28. Match 排名

定义：

\[
W_i=\text{玩家 }i\text{ 的胜局数}
\]

\[
T_i=\text{七局累计总分}
\]

Match 最终排名按照字典序：

\[
\boxed{(W_i,T_i)}
\]

首先比较：

\[
W_i
\]

若胜局数相同，再比较：

\[
T_i
\]

若仍完全相同：

\[
\boxed{\text{并列}}
\]

不再强行增加额外 tie-break。

---

# 29. Tournament Pool

游戏可以有一个纯积分化奖池：

四名玩家赛前各投入相同：

\[
B
\]

枚 Tournament Chips。

总池：

\[
4B
\]

七局结束：

\[
\arg\max_i(W_i,T_i)
\]

获得全部奖池。

完全并列则均分。

这可以作为娱乐模式，而核心规则本身不依赖真实金钱。

---

# 30. 三个时间尺度

整个游戏天然形成三个 decision horizon。

## Round

\[
\boxed{\text{现在出哪张牌？}}
\]

争分还是争信息？

---

## Game

\[
\boxed{\text{13张有限资源如何规划？}}
\]

同时进行：

- 奖励推断；
- 身份推断；
- 情报推断；
- 对手预测。

---

## Match

\[
\boxed{\text{如何在七局中逐渐认识和利用三个对手？}}
\]

包括：

- 长期风格学习；
- adaptation；
- counter-adaptation；
- match standing strategy。

---

# 31. 旗舰完整版配置

推荐正式完整版：

```text
Players                = 4
Ranks                  = 13
Match Games            = 7

Suit Roles             = ♣ ♦ ♥ ♠
Suit Order             = ♣ < ♦ < ♥ < ♠
Seat Shuffle           = Every Round

Current Prize          = Hidden
Prize Reveal           = After Joint Bid

Identity               = Hidden Per Round
Identity Reveal        = End of Game
History                = Persistent Across Match

Highest Bid Reward     = Current Prize Score
Lowest Bid Reward      = 1-bit Next-Prize Information

Information Type       = LOW / HIGH
Information Visibility = Private

Game Ranking           = Total Score
Match Ranking          = Wins First, Total Score Second
Final Exact Tie        = Shared Championship
```

这套配置定义：

\[
\boxed{\text{VEIL Full}}
\]

---

# 32. 简化模式族

同一个环境可以生成多个难度层级。

| 模式 | Prize | Identity | Info Reward | Match Memory |
|---|---|---|---|---|
| Classic | 公开 | 公开 | 无 | 无 |
| Hidden Prize | 隐藏 | 公开 | 无 | 无 |
| Information | 隐藏 | 公开 | 有 | 无 |
| Anonymous | 公开/隐藏 | 匿名 | 无 | 有 |
| Anonymous Info | 隐藏 | 匿名 | 有 | 有 |
| **VEIL Full** | **隐藏** | **每轮匿名** | **1-bit** | **7局持续** |

因此研究者不需要一次处理所有复杂度。

---

# 33. 强化学习形式化

VEIL Full 可以视为：

\[
\boxed{\text{Partially Observable Multi-Agent Stochastic Game}}
\]

真实隐藏状态包括：

\[
x_t=
(
Prize_t,
Permutation_t,
Hands_t,
PrivateSignals_t,
PlayerStyles_t
)
\]

玩家 \(i\) 的观察：

\[
o_t^i
\]

只是其中允许公开以及属于自己的部分。

行动：

\[
a_t^i\in RemainingHand_i
\]

所有行动：

\[
a_t^1,\ldots,a_t^M
\]

同时产生。

---

# 34. 可以测试的 AI 能力

该环境不是只能测最终胜率。

它可以单独测：

### Strategic Learning

是否学会管理 13 张有限资源。

### Prize Belief

是否正确处理 hidden reward uncertainty。

### Value of Information

是否知道什么时候值得故意争最低。

### Identity Inference

是否能识别匿名位置背后的长期玩家。

### Opponent Modeling

是否能预测具体玩家策略。

### Information-State Inference

是否能推断谁可能拥有额外私人情报。

### Long-Term Memory

是否能利用上一局、上几局的历史。

### Adaptation

面对固定玩家是否越来越会打。

### Strategy-Switch Detection

当玩家突然改变打法时是否能发现。

### Deception Resistance

当玩家故意模仿其他人时是否仍能正确维护 uncertainty。

---

# 35. 核心研究曲线

与其只报告最终胜率，更重要的是：

\[
Performance(g)
\]

其中：

\[
g=1,\ldots,7
\]

表示 Match 中第几局。

可以观察模型随着经验增长：

\[
IdentityAccuracy(g)
\]

\[
OpponentNLL(g)
\]

\[
Score(g)
\]

\[
ExploitGain(g)
\]

是否持续改善。

这真正测试的是：

\[
\boxed{\text{Learning Ability}}
\]

而不只是预训练后的静态游戏强度。

---

# 36. 环境公平性测试

如果四个位置使用完全相同策略：

\[
\pi_1=\pi_2=\pi_3=\pi_4
\]

则经过足够多 permutation-balanced 对局后应满足：

\[
E[U_1]
=
E[U_2]
=
E[U_3]
=
E[U_4]
\]

若显著不成立，应优先视为：

\[
\boxed{\text{Environment / RNG / Implementation Bug}}
\]

而不是策略差异。

---

# 37. 游戏设计的核心张力

VEIL 不应该继续无限增加奖励类型。

当前核心已经形成两个相反方向：

\[
\boxed{\text{Top Competition}}
\]

和：

\[
\boxed{\text{Bottom Competition}}
\]

即：

\[
\boxed{\textbf{High cards buy points. Low cards buy information.}}
\]

中文核心表达：

\[
\boxed{\textbf{高牌争当下，低牌买未来。}}
\]

身份匿名则进一步加入：

\[
\boxed{\textbf{隐藏自己，也识破别人。}}
\]

因此整个游戏可以概括为：

> **争分，也争情报；管理你的牌，隐藏你的身份，学习你的对手。**

---

# 38. 设计边界

旗舰规则第一阶段不建议加入：

- 第二高额外奖励；
- 第三名补偿；
- 玩家专属技能；
- 特殊功能牌；
- 主动换花色；
- 改写他人手牌；
- 多种货币；
- 多层随机事件。

原因不是这些机制一定不好，而是当前核心机制已经能够产生：

\[
\text{Resource Management}
\]

\[
\text{Information Acquisition}
\]

\[
\text{Hidden Identity}
\]

\[
\text{Bluffing}
\]

\[
\text{Opponent Modeling}
\]

\[
\text{Cross-Game Learning}
\]

新规则只有在实际 playtest 证明存在明确缺陷时才加入。

---

# 39. 一句话定义

VEIL 是一个：

\[
\boxed{
\textbf{规则对称、同时行动、有限资源、信息可争夺、身份可推断、历史可学习的多人策略游戏。}
}
\]

作为桌游，它要求玩家：

> **决定何时赢、何时输、何时获取信息，以及眼前匿名的人究竟是谁。**

作为强化学习环境，它测试：

> **智能体能否在公平而部分可观测的多人世界中，通过长期交互形成 belief、学习个体、评估信息价值，并把学习真正转化为战略优势。**