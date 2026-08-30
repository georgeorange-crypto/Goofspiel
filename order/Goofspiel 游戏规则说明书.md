# Goofspiel 游戏规则说明书

**英文名称：** Goofspiel  
**常用简称：** GOPS  
**别名：** Game of Pure Strategy  
**推荐玩家数：** 2 人  
**标准局长度：** 13 轮  
**游戏类型：** 同时行动博弈 / 竞价博弈 / 有限资源博弈

---

# 1. 游戏目标

Goofspiel 是一个两人竞价游戏。

游戏中存在一组 **奖励牌（Prize Cards）**。

每一轮都会出现一张奖励牌。两名玩家需要从自己尚未使用的竞价牌中秘密选择一张进行竞价。

双方同时公开竞价牌：

- 出价较高者赢得当前奖励；
- 双方出价相同时，奖励暂时无人获得，并进入下一轮的累计奖池。

游戏结束后：

> **获得奖励牌总价值更高的玩家获胜。**

---

# 2. 游戏组件

标准游戏使用一副普通的 52 张扑克牌。

从中选择三个花色：

- 一个花色作为 **玩家 A 的竞价牌**
- 一个花色作为 **玩家 B 的竞价牌**
- 一个花色作为 **奖励牌**

第四个花色不使用。

例如：

| 花色 | 用途 |
|---|---|
| ♠ 黑桃 | 玩家 A |
| ♣ 梅花 | 玩家 B |
| ♦ 方块 | 奖励牌 |
| ♥ 红桃 | 不使用 |

花色本身没有任何强弱关系。

因此也完全可以不用扑克牌，而直接使用数字：

### 玩家 A

```text
A = {1,2,3,...,13}
```

### 玩家 B

```text
B = {1,2,3,...,13}
```

### 奖励牌

```text
P = {1,2,3,...,13}
```

这实际上是 AI 环境中最推荐的表示方式。

---

# 3. 卡牌数值

所有牌按照如下方式赋值：

| 扑克牌 | 数值 |
|---|---:|
| A | 1 |
| 2 | 2 |
| 3 | 3 |
| ... | ... |
| 10 | 10 |
| J | 11 |
| Q | 12 |
| K | 13 |

因此：

```text
A < 2 < 3 < ... < 10 < J < Q < K
```

不存在扑克牌游戏中常见的：

```text
A > K
```

规则。

这里 **A 永远等于 1，是最小牌。**

---

# 4. 游戏初始化

游戏开始时：

### Step 1：分配竞价牌

玩家 A 获得：

```text
{1,2,3,...,13}
```

玩家 B 同样获得：

```text
{1,2,3,...,13}
```

双方的竞价能力完全对称。

---

### Step 2：准备奖励牌

奖励牌为：

```text
{1,2,3,...,13}
```

将 13 张奖励牌随机打乱：

```text
shuffle(P)
```

形成一个随机排列：

```text
P = [p1,p2,...,p13]
```

例如：

```text
[5, 13, 2, 8, 1, 11, 6, 3, 12, 7, 10, 4, 9]
```

---

### Step 3：初始化分数

```text
score_A = 0
score_B = 0
```

同时初始化：

```text
pot = 0
```

`pot` 表示当前待争夺奖励池。

---

# 5. 一轮游戏的完整流程

每一轮由四个阶段组成：

```text
奖励揭示
   ↓
秘密选择竞价牌
   ↓
同时公开
   ↓
判定结果
```

---

# 6. 阶段一：揭示奖励牌

从奖励牌堆顶部翻开一张奖励牌。

假设：

```text
prize = 8
```

则本轮基础奖励价值为：

```text
8
```

将其加入当前奖池：

```text
pot += 8
```

如果之前不存在平局：

```text
pot = 8
```

如果上一轮发生平局，假设已有：

```text
pot = 10
```

则本轮翻出 8 后：

```text
pot = 10 + 8 = 18
```

双方实际上是在争夺整个：

```text
18 分
```

的奖励池。

---

# 7. 阶段二：玩家秘密竞价

两名玩家分别从自己的 **剩余竞价牌** 中选择一张。

例如：

玩家 A 剩余：

```text
{1,2,3,4,6,7,9,11,12}
```

玩家 B 剩余：

```text
{1,2,4,5,6,8,9,10,13}
```

双方分别选择：

```text
A → 7
B → 9
```

选择必须秘密进行。

也就是说，在玩家 A 决策的时候：

```text
A 不知道 B 本轮选择了什么。
```

同样：

```text
B 不知道 A 本轮选择了什么。
```

---

# 8. 同时行动规则

Goofspiel 最关键的一条规则是：

> **双方不是轮流行动，而是同时行动。**

因此不存在：

```text
A 出牌 → B 看见 → B 再出牌
```

这样的过程。

正确的过程是：

```text
A secretly chooses a
B secretly chooses b

        ↓

simultaneously reveal

        ↓

compare(a,b)
```

从博弈论角度，可以表示为：

\[
a_t \sim \pi_A(s_t)
\]

\[
b_t \sim \pi_B(s_t)
\]

双方在不知道对方当前动作的情况下产生动作。

随后系统同时接收：

\[
(a_t,b_t)
\]

并计算结果。

---

# 9. 阶段三：竞价牌公开

双方选择完成后，同时公开。

设：

```text
bid_A = a
bid_B = b
```

共有三种情况。

---

# 10. 情况一：玩家 A 出价更高

如果：

\[
a>b
\]

则玩家 A 赢得整个当前奖励池。

例如：

```text
Prize = 8
A bid = 10
B bid = 6
```

则：

```text
score_A += 8
```

然后：

```text
pot = 0
```

---

# 11. 情况二：玩家 B 出价更高

如果：

\[
b>a
\]

则玩家 B 赢得整个奖励池。

例如：

```text
Prize = 12
A bid = 4
B bid = 11
```

则：

```text
score_B += 12
```

随后：

```text
pot = 0
```

---

# 12. 情况三：双方出价相同

如果：

\[
a=b
\]

则发生 **Tie（平局）**。

例如：

```text
Prize = 11

A bid = 9
B bid = 9
```

此时：

```text
A 不得分
B 不得分
```

但奖励 **不会消失**。

当前：

```text
pot = 11
```

保留到下一轮。

下一轮假设翻出：

```text
Prize = 7
```

那么：

```text
pot = 11 + 7 = 18
```

双方新一轮实际上争夺：

```text
18
```

分。

例如：

```text
A bid = 5
B bid = 12
```

则 B 赢得：

```text
11 + 7 = 18
```

分。

因此：

```text
score_B += 18
pot = 0
```

---

# 13. 连续平局

奖励池可以连续累积。

例如：

## Round 1

```text
Prize = 10
A = 7
B = 7
```

平局。

```text
pot = 10
```

---

## Round 2

```text
Prize = 12
A = 5
B = 5
```

再次平局：

```text
pot = 10 + 12 = 22
```

---

## Round 3

```text
Prize = 6
A = 13
B = 9
```

A 获胜。

于是：

```text
score_A += 10 + 12 + 6
```

即：

```text
score_A += 28
```

随后：

```text
pot = 0
```

这也是 Goofspiel 一个非常有意思的战略来源：

> 一个原本只有 6 分的奖励牌，因为前面的连续平局，可能突然变成价值 28 分的关键局。

---

# 14. 竞价牌属于一次性资源

任何已经使用的竞价牌都会永久移除。

例如玩家 A 首轮使用：

```text
A = 10
```

那么接下来：

```text
remaining_A =
{1,2,3,4,5,6,7,8,9,11,12,13}
```

整个游戏期间：

```text
10
```

不能再次使用。

因此每名玩家恰好：

```text
13 张牌
13 次行动
```

---

# 15. 竞价牌本身不计分

这一点尤其容易混淆。

假设：

```text
Prize = 4

A bid = 13
B bid = 1
```

A 赢得的是：

```text
4 分
```

而不是：

```text
13 分
```

13 只是 A 为争夺奖励投入的竞价资源。

因此可以理解成：

```text
Prize Card = 商品价值
Bid Card   = 一次性竞价货币
```

---

# 16. 一个完整示例

考虑简化版：

```text
Cards = {1,2,3,4,5}
```

双方竞价牌：

```text
A = {1,2,3,4,5}
B = {1,2,3,4,5}
```

奖励顺序：

```text
[3,5,1,4,2]
```

---

## Round 1

奖励：

```text
3
```

竞价：

```text
A → 2
B → 4
```

因为：

```text
4 > 2
```

B 获得 3 分。

当前：

```text
A = 0
B = 3
```

---

## Round 2

奖励：

```text
5
```

竞价：

```text
A → 5
B → 5
```

平局。

所以：

```text
pot = 5
```

比分不变。

---

## Round 3

新奖励：

```text
1
```

奖池：

```text
5 + 1 = 6
```

竞价：

```text
A → 4
B → 3
```

A 获胜。

因此：

```text
A += 6
```

当前：

```text
A = 6
B = 3
```

---

## Round 4

奖励：

```text
4
```

竞价：

```text
A → 1
B → 2
```

B 获胜：

```text
B += 4
```

当前：

```text
A = 6
B = 7
```

---

## Round 5

奖励：

```text
2
```

双方只剩：

```text
A → 3
B → 1
```

A 获胜：

```text
A += 2
```

最终：

```text
A = 8
B = 7
```

所以：

```text
Player A wins.
```

---

# 17. 最后一轮出现平局

这是实现时必须明确规定的边界情况。

如果最后一轮：

```text
A bid = B bid
```

并且不存在下一轮可以继续争夺奖池，那么：

> **剩余奖池无人获得。**

例如：

```text
pot = 15

最后一轮：
A = 8
B = 8
```

则这 15 分：

```text
discarded
```

不会分配给任何玩家。

游戏直接结束。

---

# 18. 游戏结束条件

当：

```text
13 张奖励牌全部揭示
```

并且：

```text
双方 13 张竞价牌全部使用
```

游戏结束。

计算：

```text
score_A
score_B
```

---

# 19. 胜负判定

如果：

\[
Score_A > Score_B
\]

则：

```text
Player A wins
```

如果：

\[
Score_B > Score_A
\]

则：

```text
Player B wins
```

如果：

\[
Score_A = Score_B
\]

则整局：

```text
Draw
```

---

# 20. 总奖励分

标准 13 张奖励牌总价值为：

\[
1+2+\cdots+13
\]

利用等差数列：

\[
\frac{13\times14}{2}=91
\]

因此总奖励价值：

```text
91
```

如果不存在最终未结算奖池：

\[
Score_A+Score_B=91
\]

---

# 21. 信息结构

在标准 Goofspiel 中，玩家知道：

### 公开信息

- 当前奖励牌；
- 当前奖池价值；
- 自己剩余哪些竞价牌；
- 对方剩余哪些竞价牌；
- 自己过去出过什么；
- 对手过去出过什么；
- 所有已经出现的奖励牌；
- 当前比分；
- 当前轮数。

因为双方过去的行动都已经公开，所以：

> **双方的剩余竞价牌也是完全可推导的。**

---

## 玩家不知道的信息

本轮决策时：

```text
不知道对手当前选择的竞价牌。
```

同时，如果奖励牌堆经过随机洗牌，则玩家通常也：

```text
不知道未来奖励牌的具体出现顺序。
```

但他们知道：

```text
未来还剩哪些奖励牌。
```

---

# 22. Goofspiel 属于什么类型的博弈？

这是理解这个游戏最重要的部分之一。

Goofspiel 可以被描述为：

### ① 两人博弈

```text
2-player game
```

### ② 有限时域博弈

标准版本：

```text
Horizon = 13
```

### ③ 同时行动博弈

每轮：

\[
(a_t,b_t)
\]

同时决定。

### ④ 对称博弈

双方：

```text
初始资源相同
动作空间相同
规则相同
目标相反
```

不存在五子棋那种天然的先手 / 后手角色差异。

### ⑤ 重复竞价博弈

玩家每轮都在决定：

> “为了当前奖励，我愿意消耗多少未来竞价能力？”

### ⑥ 有限资源管理博弈

每一张竞价牌只能使用一次。

所以出掉：

```text
13
```

不仅代表：

> “这一轮我出了 13。”

更意味着：

> “未来所有轮次，我再也没有 13。”

因此行动具有明显的长期机会成本。

---

# 23. 标准状态定义

如果实现 AI 环境，可以把游戏状态定义为：

\[
s_t =
(
H_A,
H_B,
R,
p_t,
P_t,
S_A,
S_B,
t
)
\]

其中：

### \(H_A\)

Player A 剩余竞价牌：

```text
remaining_bid_cards_A
```

### \(H_B\)

Player B 剩余竞价牌：

```text
remaining_bid_cards_B
```

### \(R\)

尚未出现的奖励牌集合：

```text
remaining_prize_cards
```

### \(p_t\)

当前公开奖励牌：

```text
current_prize
```

### \(P_t\)

累计奖池：

```text
current_pot
```

### \(S_A,S_B\)

当前比分。

### \(t\)

当前轮次：

```text
1 ... 13
```

---

# 24. 动作空间

某玩家当前合法动作：

\[
A_i(s)=H_i
\]

也就是：

> 玩家所有尚未使用的竞价牌。

第一轮：

```text
13 actions
```

第二轮：

```text
12 actions
```

……

最后一轮：

```text
1 action
```

动作空间不断缩小。

---

# 25. 状态转移

假设：

```text
current prize = p
current pot = P
A bids a
B bids b
```

首先：

```text
P ← P + p
```

然后比较。

---

### 如果

\[
a>b
\]

则：

```text
score_A += P
P = 0
```

---

### 如果

\[
b>a
\]

则：

```text
score_B += P
P = 0
```

---

### 如果

\[
a=b
\]

则：

```text
P 保留
```

接着：

```text
remove a from hand_A
remove b from hand_B
```

进入下一轮。

---

# 26. 推荐的 RL Reward

最简单的方法是使用即时分数差。

如果 A 赢得奖池 \(P\)：

\[
r_A=P
\]

\[
r_B=-P
\]

如果 B 赢：

\[
r_A=-P
\]

\[
r_B=P
\]

平局：

\[
r_A=r_B=0
\]

于是：

\[
r_A=-r_B
\]

整个环境成为严格的：

> **two-player zero-sum game**

---

# 27. 另一种终局 Reward

也可以只在游戏结束时给奖励：

\[
R_A =
Score_A-Score_B
\]

\[
R_B =
Score_B-Score_A
\]

所以：

\[
R_A=-R_B
\]

或者如果只关心胜负：

\[
R_A =
\begin{cases}
+1 & Score_A>Score_B\\
0 & Score_A=Score_B\\
-1 & Score_A<Score_B
\end{cases}
\]

B 的 reward 为：

\[
R_B=-R_A
\]

---

# 28. 推荐 AI 环境采用的正式规则

为了避免不同 Goofspiel 文献中的规则变体造成实验不可复现，建议项目明确使用以下版本：

## Goofspiel-13 Standard

```text
Players:           2

Bid cards:
Player A = {1,...,13}
Player B = {1,...,13}

Prize cards:
{1,...,13}

Prize ordering:
Uniform random permutation at game start

Information:
Current prize public
Past actions public
Past prizes public
Remaining cards inferable
Future prize order hidden

Action:
Each player secretly selects one unused bid card

Move:
Simultaneous

Higher bid:
Wins entire current prize pot

Equal bid:
Prize pot rolls over

Final-round equal bid:
Remaining pot discarded

Used bid cards:
Removed permanently

Game length:
Exactly 13 rounds

Final score:
Total prize value won

Winner:
Higher score

Draw:
Equal score

Training payoff:
score_A - score_B
```

这应该作为整个项目的 **唯一 canonical ruleset**。

---

# 29. 常见规则变体

Goofspiel 并不存在唯一的平局处理方式，因此阅读论文或者不同代码仓库时必须注意规则差异。

主要有三种。

---

## Variant A：Rollover

即本规则文档采用的版本：

```text
tie → prize moves into next pot
```

例如：

```text
10 tie
next prize 8
→ compete for 18
```

这是非常有战略趣味的版本。

---

## Variant B：Split

平局时：

```text
A gets prize / 2
B gets prize / 2
```

例如：

```text
Prize = 9
```

双方各：

```text
4.5
```

此版本允许小数分数。

---

## Variant C：Discard

平局：

```text
nobody gets prize
```

奖励牌直接丢弃。

---

这三种规则：

```text
Rollover
Split
Discard
```

会产生不同的博弈结构和最优策略。

因此进行 AI Benchmark 时绝不能混用。

---

# 30. 为什么 Goofspiel 很适合 AI / 强化学习研究

虽然规则只有几行，但每轮玩家实际上同时在解决三个问题：

### 当前价值

```text
这张奖励牌值多少？
```

### 对手建模

```text
对方可能愿意为它出多少？
```

### 长期资源配置

```text
我现在用了这张牌，
未来还有什么牌可以用？
```

因此，一个动作实际上具有：

\[
Immediate\ Value
+
Opponent\ Prediction
+
Future\ Opportunity\ Cost
\]

例如当前：

```text
Prize = 13
```

最简单策略当然是：

```text
bid 13
```

但是对手知道你可能这么做。

于是对手可能：

```text
bid 1
```

故意放弃这局。

那么你实际上：

```text
用最强资源 13
换了奖励 13
```

而对手只损失：

```text
1
```

之后对手在剩余 12 轮中的竞价资源整体比你强。

于是你又可能预测：

```text
“他知道我会出 13，所以他会出 1。”
```

然后你可以：

```text
bid 2
```

但对方也可能预测到这一点……

于是自然形成：

> **混合策略、随机化、欺骗、反欺骗、对手建模与策略循环。**

---

# 31. Goofspiel 与五子棋最大的不同

五子棋中：

```text
A move
↓
B observes A
↓
B move
```

Goofspiel 中：

```text
A chooses ─┐
           ├→ simultaneous reveal
B chooses ─┘
```

因此五子棋更接近：

> **Sequential Game**

而 Goofspiel 包含大量：

> **Simultaneous-Move Game**

结构。

这意味着简单的：

```text
Minimax
```

已经不能直接按照普通棋类那种：

```text
MAX → MIN → MAX → MIN
```

的方式理解每一轮。

一轮本身更接近一个矩阵游戏：

\[
M_{ij}
\]

其中：

```text
row = A bid
column = B bid
```

策略通常需要考虑：

\[
\pi_A(a|s)
\]

和：

\[
\pi_B(b|s)
\]

这样的 **概率分布**，而不仅仅是找到一个确定的“最佳动作”。

---

# 32. 一句话规则

如果要把整个 Goofspiel 压缩成一句话：

> **双方各持有 1–13 共十三张一次性竞价牌；系统每轮公开一张价值 1–13 的奖励牌，双方秘密选择一张剩余竞价牌同时公开，高者获得奖励，平局则奖励累积至下一轮；13 轮后奖励总分更高者获胜。**

---

# 33. 环境规则核心伪代码

```text
initialize:
    hand_A = {1,...,13}
    hand_B = {1,...,13}

    prizes = shuffle({1,...,13})

    score_A = 0
    score_B = 0

    pot = 0


for prize in prizes:

    pot += prize

    bid_A = player_A.choose(hand_A)
    bid_B = player_B.choose(hand_B)

    assert bid_A in hand_A
    assert bid_B in hand_B

    hand_A.remove(bid_A)
    hand_B.remove(bid_B)

    reveal(bid_A, bid_B)

    if bid_A > bid_B:

        score_A += pot
        pot = 0

    elif bid_B > bid_A:

        score_B += pot
        pot = 0

    else:

        # Tie:
        # pot carries into next round
        pass


# If the final round was tied:
pot = 0


if score_A > score_B:
    winner = A

elif score_B > score_A:
    winner = B

else:
    winner = DRAW
```

---

# 34. 核心规则总结

Goofspiel 的核心只有六条：

1. **双方各有完全相同的 1–13 竞价牌。**
2. **奖励牌 1–13 随机排列，每轮公开一张。**
3. **双方每轮秘密选择一张尚未使用的竞价牌。**
4. **双方同时公开，数字较大者获得奖励。**
5. **竞价牌使用后永久消耗；平局时奖励累积到下一轮。**
6. **13 轮结束后，获得奖励总价值最高者获胜。**

真正复杂的不是规则。

复杂的是：

> **你知道我知道你知道我可能会出什么。**

这正是 Goofspiel 从一个简单纸牌游戏变成一个很漂亮的博弈论与强化学习问题的地方。