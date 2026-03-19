# 18.3 策略优化算法详解：PPO、DPO 与 GRPO

在 [18.1 节](./01_agentic_rl_overview.md) 中，我们介绍了 Agentic-RL 的两阶段训练范式（SFT → RL）。RL 阶段的核心问题是：**如何根据奖励信号来更新模型参数？** 这正是策略优化算法要解决的问题。

本节将从零开始，**系统性地讲解三种主流策略优化算法**：PPO（工业界经典）、DPO（学术界新秀）、GRPO（DeepSeek 创新）。我们将从最基本的直觉出发，逐步推导数学公式，并通过大量图示帮助理解。

![三大策略优化算法架构对比](../svg/chapter_agentic_rl_03_three_algorithms.svg)

---

## 预备知识：策略梯度的基本思想

在深入三种算法之前，我们需要理解一个共同的起点——**策略梯度（Policy Gradient）** [1]。

### 核心直觉

想象你在练习投篮。每次投篮后，你会得到一个反馈：进了（奖励 +1）或没进（奖励 0）。策略梯度的思想极其朴素：

> **如果某个动作获得了高奖励，就增加该动作的概率；如果获得了低奖励，就降低该动作的概率。**

形式化地，策略梯度定理给出的梯度方向为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]$$

下面我们**逐项拆解**这个公式，并用一个具体的语言模型例子来帮助理解。

---

#### ① $\nabla_\theta J(\theta)$ —— "我该往哪个方向调参数？"

- $J(\theta)$ 是我们的**总目标**：模型在所有可能输入上的期望累积奖励。$J$ 越大，模型整体表现越好
- $\nabla_\theta$ 是对模型参数 $\theta$（即模型中数十亿个权重值）求梯度
- $\nabla_\theta J(\theta)$ 就是一个与 $\theta$ 同维度的向量，**它告诉我们：如果把每个参数往哪个方向微调一点点，$J$ 会增大最快**
- 训练时我们做的就是：$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$（$\alpha$ 是学习率），即沿梯度方向"上坡"

> **类比**：你蒙着眼站在山坡上，梯度就是"脚下最陡的上坡方向"。每走一步（更新一次参数），你就往山顶（最大奖励）靠近一点。

---

#### ② $\nabla_\theta \log \pi_\theta(a_t | s_t)$ —— "怎样调参数才能让这个动作更可能发生？"

这是公式中最核心也最难理解的部分，我们分层解释：

**第一层：$\pi_\theta(a_t | s_t)$ 是什么？**

$\pi_\theta$ 就是我们的语言模型。给定当前状态 $s_t$（对话历史 + 已生成的 token），它输出一个概率分布，表示"下一个 token 是什么"的概率。例如：

| 下一个 token ($a_t$) | 概率 $\pi_\theta(a_t \| s_t)$ |
|---------------------|-------------------------------|
| "搜索" | 0.35 |
| "回答" | 0.25 |
| "计算" | 0.20 |
| "我" | 0.10 |
| ... | ... |

$\pi_\theta(\text{"搜索"} | s_t) = 0.35$ 表示：在当前上下文下，模型认为下一步输出"搜索"的概率是 35%。

**第二层：$\log \pi_\theta(a_t | s_t)$ 为什么取对数？**

取对数有两个好处：
1. **数值稳定**：概率值在 0 到 1 之间，连乘多个 token 的概率会变得极小（如 $0.35 \times 0.20 \times 0.15 = 0.0105$），取对数后变为加法（$\log 0.35 + \log 0.20 + \log 0.15 = -4.56$），避免数值下溢
2. **梯度形式简洁**：$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)}$，这个比率形式恰好是我们需要的

**第三层：$\nabla_\theta \log \pi_\theta(a_t | s_t)$ 到底是什么？**

这被称为**得分函数（score function）**。它是一个与模型参数 $\theta$ 同维度的向量，指出：

> **"如果想让动作 $a_t$ 在状态 $s_t$ 下的概率增大，模型的每个参数应该分别往哪个方向调？"**

- 它不直接改变概率，而是给出一个**方向**
- 沿这个方向调整参数 → $\pi_\theta(a_t | s_t)$ 增大（这个动作变得更可能）
- 沿相反方向调整参数 → $\pi_\theta(a_t | s_t)$ 减小（这个动作变得更不可能）

> **类比**：得分函数就像动作 $a_t$ 的"方向盘"——转动它可以增加或减少这个动作被选中的概率。但光有方向盘不够，你还需要知道**该转多少**——这就是下面 $R(\tau)$ 的作用。

---

#### ③ $R(\tau)$ —— "这个方向盘该转多少？"

$R(\tau) = \sum_{t=0}^{T} r_t$ 是整条轨迹（从开始到结束的完整交互过程）的**累积奖励**，它充当**权重**：

- **$R(\tau) > 0$（正奖励）**：说明这条轨迹整体表现不错
  - 梯度 = 正权重 × 得分函数 → 沿得分函数方向更新 → **增加**轨迹中每个动作的概率
  - 直觉：这次表现好，下次要更多地做类似的事情
  
- **$R(\tau) < 0$（负奖励）**：说明这条轨迹整体表现很差
  - 梯度 = 负权重 × 得分函数 → 沿得分函数**反方向**更新 → **减少**轨迹中每个动作的概率
  - 直觉：这次表现差，下次要避免做类似的事情

- **$R(\tau) = 0$（零奖励）**：这条轨迹对梯度无贡献

- **$|R(\tau)|$ 越大**：权重越大，这条轨迹对参数更新的影响越大。奖励/惩罚越极端，模型"记忆"越深刻

> **类比**：$R(\tau)$ 就像教练的评分。得分函数指明了方向盘，$R(\tau)$ 决定了转多大的角度。教练打高分（$R > 0$）→ 大力转向"增加该动作概率"；教练打低分（$R < 0$）→ 大力转向"减少该动作概率"。

---

#### ④ $\sum_{t=0}^{T}$ —— "轨迹中每一步都要算"

一条轨迹包含 $T+1$ 个时间步（从 $t=0$ 到 $t=T$），每一步都有一个 $(s_t, a_t)$ 对。求和意味着：轨迹中**每一步的得分函数都被同一个 $R(\tau)$ 加权**。

在语言模型中，一步 = 生成一个 token。如果模型生成了一个 50 token 的回答，$T = 49$，那么这 50 个 token 中每一个的生成概率都会被同一个总奖励加权更新。

> **注意**：这其实是一个粗糙的做法——用整条轨迹的总奖励来加权每一步。如果轨迹中前 30 个 token 是正确推理，后 20 个 token 是错误结论，它们都会被总奖励同等对待。这就是"**信用分配问题（credit assignment）**"——PPO 的优势函数 $A_t$ 正是为了解决这个问题（见 §1.3）。

---

#### ⑤ $\mathbb{E}_{\tau \sim \pi_\theta}$ —— "对很多次尝试取平均"

$\mathbb{E}$ 是**期望运算符**，$\tau \sim \pi_\theta$ 表示轨迹 $\tau$ 是按策略 $\pi_\theta$ 随机采样的。

- 因为语言模型的生成是**随机的**（通过 temperature 采样），同一个输入可能产生不同的输出
- 每次采样得到一条轨迹 $\tau$，对应一个 $R(\tau)$ 值
- 期望就是**对所有可能轨迹加权平均**——概率越高的轨迹权重越大

**实际操作中**：我们无法枚举所有可能轨迹（语言模型的输出空间是天文数字级的），因此用**蒙特卡洛近似**——采样 $N$ 条轨迹，取平均作为期望的估计：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{n=1}^{N} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(n)} | s_t^{(n)}) \cdot R(\tau^{(n)}) \right]$$

$N$ 越大，估计越准确，但计算成本也越高。这就是训练中 batch size 的本质。

---

#### 完整例子：语言模型 Agent 的一次策略梯度更新

假设我们正在训练一个能调用搜索工具的 Agent，用户提问"北京今天天气如何？"

**采样两条轨迹：**

| | 轨迹 A（好的回答） | 轨迹 B（差的回答） |
|---|---|---|
| $s_0$ | 用户："北京今天天气如何？" | 用户："北京今天天气如何？" |
| $a_0$ | `<think>` | `<think>` |
| $a_1$ | 需要查询实时天气 | 我直接回答吧 |
| $a_2$ | `</think>` | `</think>` |
| $a_3$ | `<tool_call>search("北京天气")</tool_call>` | 北京今天晴天，25°C |
| ... | （获取结果后给出准确回答） | （瞎编的，可能完全错误） |
| $R(\tau)$ | **+0.8**（调用了工具，回答准确） | **-0.2**（没调用工具，回答错误） |

**梯度更新效果：**

- **轨迹 A**（$R = +0.8$）：模型会**增加**"遇到实时信息问题 → 调用搜索工具"这一系列动作的概率
- **轨迹 B**（$R = -0.2$）：模型会**减少**"遇到实时信息问题 → 直接瞎编回答"这一系列动作的概率

经过成千上万次这样的更新，模型逐渐学会：**遇到需要实时信息的问题，应该先调用工具，而不是直接编造答案。**

---

### 原始策略梯度的缺陷

虽然直觉清晰，但原始策略梯度有两个严重问题：

| 问题 | 具体表现 | 后果 |
|------|---------|------|
| **高方差** | $R(\tau)$ 可能在不同轨迹间差异极大 | 梯度估计不稳定，训练收敛极慢 |
| **步长不可控** | 没有约束单步更新的大小 | 一次"大跳步"就可能毁掉整个策略 |

**PPO、DPO、GRPO 各自用不同方式解决了这两个问题。** 下面逐一详解。

---

## 第一部分：PPO（Proximal Policy Optimization）

### 1.1 PPO 解决了什么问题？

PPO [2] 是 OpenAI 于 2017 年提出的策略优化算法，是 InstructGPT [3] 和 ChatGPT 的核心训练算法。PPO 的设计目标是：

> **在保证训练稳定性的前提下，尽可能高效地利用已采样数据来更新策略。**

PPO 通过两个关键机制实现这一目标：
1. **重要性采样**：允许用"旧策略"采集的数据来训练"当前策略"（数据复用）
2. **Clip 裁剪**：限制策略更新的步长，防止策略崩溃

### 1.2 重要性采样比率 $\rho_t$：离策训练的核心

在策略梯度中，我们需要从当前策略 $\pi_\theta$ 中采样轨迹来计算梯度。但如果每次更新参数后都重新采样，效率极低。**重要性采样** 允许我们用旧策略 $\pi_{\theta_{old}}$ 的样本来估计新策略 $\pi_\theta$ 的梯度。

核心是引入**重要性采样比率**：

$$\rho_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

逐项解读：

- **分子** $\pi_\theta(a_t | s_t)$：当前策略在状态 $s_t$ 下选择动作 $a_t$ 的概率
- **分母** $\pi_{\theta_{old}}(a_t | s_t)$：采样时的旧策略选择该动作的概率
- $\rho_t = 1$：当前策略与旧策略对这个动作的偏好完全一致
- $\rho_t > 1$：当前策略比旧策略更倾向于这个动作（即"当前策略认为这个动作更好了"）
- $\rho_t < 1$：当前策略比旧策略更不倾向于这个动作（"当前策略认为这个动作变差了"）
- $\rho_t = 2$：当前策略选择该动作的概率是旧策略的 2 倍

**重要性采样的直觉**：假设旧策略采样到某个动作的概率是 10%（$\pi_{old} = 0.1$），而当前策略认为该动作概率应该是 30%（$\pi_\theta = 0.3$），则 $\rho = 3$。这意味着如果用旧策略的数据来估计新策略的期望，每条这样的数据应该被赋予 3 倍权重——因为新策略"本应"更频繁地采到它。

### 1.3 优势函数 $A_t$：判断动作的"好坏"

策略梯度中，用累积奖励 $R(\tau)$ 作为权重会导致高方差。**优势函数（Advantage Function）** 通过引入一个基准线来解决这个问题：

$$A_t = Q(s_t, a_t) - V(s_t)$$

- $Q(s_t, a_t)$：在状态 $s_t$ 下执行动作 $a_t$ 后，能获得的期望累积奖励（**动作价值**）
- $V(s_t)$：在状态 $s_t$ 下，按当前策略执行所能获得的期望累积奖励（**状态价值**，即"基准线"）
- $A_t > 0$：动作 $a_t$ 比"平均水平"好 → 应当**强化**
- $A_t < 0$：动作 $a_t$ 比"平均水平"差 → 应当**抑制**
- $A_t = 0$：动作 $a_t$ 与平均水平持平 → 无需调整

**为什么减去基准线能降低方差？** 一个形象的比喻：假设你考试得了 85 分。如果全班平均 60 分，你会觉得"考得不错"（$A = +25$）；如果全班平均 90 分，你会觉得"发挥失常"（$A = -5$）。**把绝对分数转为相对分数，消除了分数尺度的干扰**，让信号更加稳定。

### 1.4 GAE：广义优势估计

实际训练中，$Q(s_t, a_t)$ 和 $V(s_t)$ 都不是精确已知的，需要用一个 **Critic 模型** $V_\phi(s)$ 来估计。**GAE（Generalized Advantage Estimation）** [4] 是一种融合多步估计的方法，在偏差和方差之间取得平衡：

$$A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

其中 **TD 误差（Temporal Difference Error）** 定义为：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

逐项解读 TD 误差：

- $r_t$：在时间步 $t$ 实际获得的即时奖励
- $\gamma V_\phi(s_{t+1})$：Critic 对下一状态的价值估计，乘以折扣因子 $\gamma$
- $V_\phi(s_t)$：Critic 对当前状态的价值估计
- **直觉**：$\delta_t$ 衡量的是 "**实际发生的**"（$r_t + \gamma V_\phi(s_{t+1})$）与 "**Critic 预期的**"（$V_\phi(s_t)$）之间的差异。如果 $\delta_t > 0$，说明实际结果超出预期（惊喜！）；$\delta_t < 0$ 则说明实际结果不如预期（失望！）

逐项解读 GAE：

- $(\gamma\lambda)^l$：**指数衰减权重**——越远的时间步，对当前优势的贡献越小
- $\lambda \in [0, 1]$：**GAE 折衷参数**，控制偏差-方差权衡：

| $\lambda$ 值 | GAE 退化为 | 偏差 | 方差 | 直觉 |
|-------------|-----------|------|------|------|
| $\lambda = 0$ | 单步 TD：$A_t = \delta_t$ | 高（完全依赖 Critic 精度）| 低 | 只看一步的"惊喜" |
| $\lambda = 1$ | 蒙特卡洛：$A_t = \sum_l \gamma^l \delta_{t+l}$ | 低（用完整轨迹）| 高 | 看完整轨迹的表现 |
| $\lambda = 0.95$ | **推荐值** | 适中 | 适中 | 兼顾近期和远期信息 |

> **📌 关键问题**：GAE 需要 Critic 模型 $V_\phi(s)$ 来估计状态价值。对于大语言模型（如 7B 参数），这意味着**需要额外加载一个同等规模的 Critic 模型**——这是 PPO 在大模型训练中最大的资源瓶颈。

### 1.5 PPO Clip 机制：策略更新的"安全绳"

有了优势 $A_t$ 和比率 $\rho_t$，PPO 的损失函数为：

$$\mathcal{L}_{PPO}(\theta) = -\mathbb{E}_t \left[ \min\left( \rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

这个公式看起来复杂，但核心思想很简单。让我们分两种情况理解：

**情况 ①：$A_t > 0$（好动作，应强化）**

- 无 Clip 时：$\rho_t$ 越大（即越增加该动作概率），损失越小 → 梯度会推动策略不断增加该动作概率
- 有 Clip 时：$\rho_t$ 超过 $1+\epsilon$ 后，$\text{clip}(\rho_t) \cdot A_t$ 不再增大 → $\min$ 取到裁剪值 → **梯度变为零**
- **效果**：即使动作很好，也不允许概率增加太多（防止"过度自信"）

**情况 ②：$A_t < 0$（差动作，应抑制）**

- 无 Clip 时：$\rho_t$ 越小（即越降低该动作概率），损失越小 → 梯度会推动策略大幅降低该动作概率
- 有 Clip 时：$\rho_t$ 低于 $1-\epsilon$ 后，**梯度变为零**
- **效果**：即使动作很差，也不允许概率降低太多（防止"矫枉过正"）

![PPO Clip 机制图解](../svg/chapter_agentic_rl_03_ppo_clip.svg)

**Clip 参数 $\epsilon$ 的含义**：$\epsilon$（通常取 0.1–0.3）定义了"信任域"的大小——每次策略更新，每个动作的概率变化不得超过旧策略的 $(1 \pm \epsilon)$ 倍。$\epsilon$ 越小越保守，$\epsilon$ 越大越激进。

### 1.6 PPO 完整训练流程

![PPO 完整训练迭代流程](../svg/chapter_agentic_rl_03_ppo_training_flow.svg)

PPO 的训练需要同时维护以下模型：

![PPO 训练架构](../svg/chapter_agentic_rl_03_ppo_architecture.svg)

### 1.7 PPO 的优缺点总结

| 维度 | 评价 |
|------|------|
| ✅ **通用性** | 几乎适用于所有 RL 场景，不限于语言模型 |
| ✅ **稳定性** | Clip 机制提供了可靠的训练稳定性保障 |
| ✅ **理论基础** | 有成熟的理论支撑（信任域方法的简化） |
| ❌ **显存需求** | 需要 Critic 模型，显存占用 ≈ 3× 模型大小 |
| ❌ **训练复杂** | Critic 与 Policy 互相依赖，联合训练不稳定 |
| ❌ **超参数多** | GAE λ、Critic 学习率、clip ε、KL β 等需精心调节 |

---

## 第二部分：DPO（Direct Preference Optimization）

### 2.1 DPO 的核心洞察

DPO [5] 是 2023 年 Stanford 团队提出的算法。它的核心洞察可以用一句话概括：

> **既然 RLHF 的最终目标是让模型的输出分布符合人类偏好，那能否跳过"训练奖励模型 → 用 PPO 优化"的两步流程，直接从偏好数据中优化策略？**

答案是——**可以！** DPO 通过一个精巧的数学推导，证明了 RLHF 的最优策略可以用一个**闭式解**表示，从而将 RL 问题转化为简单的**监督学习**问题。

![DPO 核心直觉](../svg/chapter_agentic_rl_03_dpo_intuition.svg)

### 2.2 数学推导：从 RLHF 到 DPO

这一推导是 DPO 论文最精妙的部分，我们逐步展开。

**Step 1：RLHF 的优化目标**

标准的 RLHF 优化目标是：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot D_{KL}\left(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)\right)$$

其中 $r(x, y)$ 是奖励模型的输出，$\beta$ 控制 KL 约束强度。

**Step 2：推导最优策略的闭式解**

对上述目标求解（使用变分法），可以得到最优策略的**闭式表达**：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$ 是配分函数（归一化常数）。

逐项解读：

- $\pi_{ref}(y|x)$：参考策略（SFT 模型）的概率——最优策略以 SFT 策略为"先验"
- $\exp\left(\frac{r(x,y)}{\beta}\right)$：奖励的指数函数——高奖励的输出概率被放大，低奖励的被缩小
- $\frac{1}{Z(x)}$：归一化因子——确保概率之和为 1
- $\beta$：**温度参数**——$\beta$ 越小，最优策略越集中在高奖励输出上；$\beta$ 越大，越接近参考策略
- **直觉**：最优策略 = 参考策略 × 奖励的指数调制。好的输出被放大，差的输出被缩小

**Step 3：反解出隐式奖励**

从 Step 2 的闭式解中，可以反解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

这是一个关键的发现：**奖励可以用策略的对数概率比来表示！** 虽然我们不知道 $Z(x)$ 的值，但它只依赖于 $x$，不依赖于 $y$——在比较两个输出时会被消去。

**Step 4：代入 Bradley-Terry 偏好模型**

在 RLHF 中，人类偏好建模使用 **Bradley-Terry 模型** [6]：

$$P(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

其中 $\sigma$ 是 sigmoid 函数，$y_w$ 是偏好的（winning）输出，$y_l$ 是不偏好的（losing）输出。

将 Step 3 的隐式奖励代入，$\beta \log Z(x)$ 项在做差时相消：

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

**Step 5：得到 DPO 损失函数**

最终的 DPO 损失函数就是上述偏好概率的负对数似然：

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right] \right) \right]$$

逐项解读（由内到外）：

- $\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$：**好输出的隐式奖励**——当前策略相对参考策略对"好输出"的对数概率比。值越大 → 当前策略越偏好好输出
- $\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$：**差输出的隐式奖励**——同理，但是对"差输出"
- $\Delta = \beta \cdot [\text{好输出隐式奖励} - \text{差输出隐式奖励}]$：**隐式奖励差**。我们希望 $\Delta > 0$ 且尽可能大
- $\sigma(\Delta)$：将奖励差映射为 [0, 1] 的概率
- $-\log \sigma(\Delta)$：负对数似然损失。$\Delta$ 越大，损失越小
- $\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}$：在偏好数据集上求期望

**一句话总结**：DPO 让模型学会"给好输出更高的隐式奖励，给差输出更低的隐式奖励"——不需要显式训练奖励模型，也不需要在线采样。

### 2.3 DPO 的训练架构

![DPO 训练架构](../svg/chapter_agentic_rl_03_dpo_architecture.svg)

DPO 的训练只需要：
1. **Policy 模型 $\pi_\theta$**：可训练参数（初始化为 SFT 模型）
2. **Reference 模型 $\pi_{ref}$**：冻结的 SFT 模型副本，用于计算对数概率比
3. **偏好数据集 $\mathcal{D}$**：每条数据包含 (输入 $x$, 好输出 $y_w$, 差输出 $y_l$)

**不需要**：
- ❌ 奖励模型
- ❌ Critic 模型
- ❌ 在线采样（完全离线训练）

### 2.4 DPO 的代码实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # π_θ(y_w|x) 的 log 概率 [batch]
    policy_rejected_logps: torch.Tensor,  # π_θ(y_l|x) 的 log 概率 [batch]
    ref_chosen_logps: torch.Tensor,       # π_ref(y_w|x) 的 log 概率 [batch]
    ref_rejected_logps: torch.Tensor,     # π_ref(y_l|x) 的 log 概率 [batch]
    beta: float = 0.1,                    # 温度参数
) -> tuple[torch.Tensor, dict]:
    """
    计算 DPO 损失函数
    
    核心公式：
    L = -log σ(β · [log(π_θ/π_ref)(y_w) - log(π_θ/π_ref)(y_l)])
    
    Args:
        policy_chosen_logps:   当前策略对好输出的对数概率
        policy_rejected_logps: 当前策略对差输出的对数概率
        ref_chosen_logps:      参考策略对好输出的对数概率
        ref_rejected_logps:    参考策略对差输出的对数概率
        beta: 温度参数，控制对数概率差的缩放
    
    Returns:
        loss: 标量损失值
        metrics: 监控指标字典
    """
    # ── 计算隐式奖励 ──────────────────────────────────────────────
    # 好输出的隐式奖励：log(π_θ/π_ref)(y_w)
    chosen_rewards = policy_chosen_logps - ref_chosen_logps      # [batch]
    
    # 差输出的隐式奖励：log(π_θ/π_ref)(y_l)
    rejected_rewards = policy_rejected_logps - ref_rejected_logps # [batch]
    
    # ── 计算隐式奖励差 ────────────────────────────────────────────
    # Δ = β · [好输出隐式奖励 - 差输出隐式奖励]
    reward_margin = beta * (chosen_rewards - rejected_rewards)    # [batch]
    
    # ── DPO 损失 = -log σ(Δ) ─────────────────────────────────────
    loss = -F.logsigmoid(reward_margin).mean()
    
    # ── 监控指标 ──────────────────────────────────────────────────
    metrics = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
        "reward_margin": reward_margin.mean().item(),
        # 准确率：隐式奖励差 > 0 的比例（模型正确区分好差输出的比例）
        "accuracy": (reward_margin > 0).float().mean().item(),
    }
    
    return loss, metrics
```

### 2.5 DPO 的优缺点总结

| 维度 | 评价 |
|------|------|
| ✅ **极简架构** | 无需 Reward 模型、Critic 模型，显存 ≈ 2× 模型大小 |
| ✅ **训练稳定** | 本质是监督学习，不存在 RL 特有的不稳定性 |
| ✅ **易于实现** | 核心代码不到 20 行，超参数仅 $\beta$ 一个 |
| ❌ **需偏好数据** | 依赖高质量的 $(y_w, y_l)$ 偏好对，标注成本高 |
| ❌ **离线局限** | 完全离线训练，无法利用在线探索发现新策略 |
| ❌ **泛化有限** | 只能学到偏好数据中已有的"好"模式，难以超越数据上界 |

> **📌 DPO vs PPO 的核心差异**
> 
> - PPO 是**在线 RL**：模型边生成边学习，能探索数据中未见过的行为模式
> - DPO 是**离线监督学习**：只从已有的偏好对中学习，无法超越数据质量
> 
> 这意味着：**如果任务需要模型涌现全新的推理策略（如 DeepSeek-R1 的长链推理），DPO 不是最佳选择；但如果已有高质量偏好数据，DPO 是最简单高效的对齐方案。**

---

## 第三部分：GRPO（Group Relative Policy Optimization）

### 3.1 GRPO 的核心洞察

GRPO [7] 是 DeepSeek 团队为大模型 RL 训练量身打造的算法。它的核心洞察是：

> **PPO 的 Critic 模型本质上只是提供一个"基准线"来减小优势估计的方差。对于语言模型，有更简单的方式获得基准线——对同一问题采样多个回答，用组内均值作为基准线。**

这个洞察带来了巨大的实践价值：

| 维度 | PPO | GRPO | 改善 |
|------|-----|------|------|
| **模型数量** | Policy + Critic + Reference | Policy + Reference | **少一个 Critic** |
| **显存需求** | ≈ 3× 模型大小 | ≈ 1.5× 模型大小 | **节省约 50%** |
| **训练稳定性** | Critic 误差会传播到 Policy | 无 Critic 误差传播 | **更稳定** |
| **超参数** | 多（GAE λ, Critic lr, ...） | 少（clip ε, KL β, G） | **更易调参** |

### 3.2 组内采样与标准化：用"同组比较"替代 Critic

GRPO 的核心操作如下：

对每个输入 $x$，使用当前策略（的旧版本 $\pi_{\theta_{old}}$）采样 $G$ 个回答：

$$\{y_1, y_2, \ldots, y_G\} \sim \pi_{\theta_{old}}(\cdot | x)$$

然后分别计算每个回答的奖励 $r_i = R(x, y_i)$，并进行**组内标准化**：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

其中：

$$\mu_r = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_r)^2}$$

逐项解读：

- $\mu_r$：**组内奖励均值**——同一问题 $G$ 个回答的平均奖励，充当"基准线"（Critic 的替代品）
- $\sigma_r$：**组内奖励标准差**——用于归一化，消除奖励绝对尺度的影响
- $\epsilon$：数值稳定性常数（通常取 $10^{-8}$），防止除零
- $\hat{A}_i > 0$：第 $i$ 个回答比组内平均更好 → 应当**强化**
- $\hat{A}_i < 0$：第 $i$ 个回答比组内平均更差 → 应当**抑制**

**标准化的统计性质**：

1. **零均值**：$\sum_i \hat{A}_i \approx 0$——一半回答被强化，一半被抑制（相对比较）
2. **单位方差**：$\text{Var}(\hat{A}_i) \approx 1$——梯度大小不受奖励尺度影响

**为什么组内均值可以替代 Critic？** 核心论证：
- Critic 的作用 = 提供基准线 → 将绝对奖励转为相对优势 → 减小梯度方差
- 组内均值同样提供了一个基准线 → 同样将绝对奖励转为相对优势 → 同样减小梯度方差
- **区别**：Critic 是一个参数化的函数逼近器（需要训练，可能有估计误差）；组内均值是一个非参数的统计量（无需训练，但依赖采样质量）
- **代价**：GRPO 需要对每个问题采样 $G$ 个回答（增加采样成本），而 PPO 只需 1 个

```python
import numpy as np

def compute_grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """
    计算 GRPO 组内标准化优势函数
    
    Args:
        rewards: 同一问题 G 个回答的奖励值 [r₁, r₂, ..., r_G]
        eps: 数值稳定性常数
    
    Returns:
        标准化优势值列表 [Â₁, Â₂, ..., Â_G]
    
    性质：
        - Σ Â_i ≈ 0（零均值）
        - Var(Â_i) ≈ 1（单位方差）
    """
    rewards = np.array(rewards, dtype=np.float64)
    mu = rewards.mean()
    sigma = rewards.std()
    
    if sigma < eps:
        # 所有回答奖励相同 → 无法区分好坏 → 优势为零
        return [0.0] * len(rewards)
    
    advantages = (rewards - mu) / (sigma + eps)
    return advantages.tolist()


# ── 示例 ──────────────────────────────────────────────────────────────
# 同一数学题，模型生成 8 个回答：5 个正确，3 个错误
rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
advantages = compute_grpo_advantages(rewards)

print("奖励值:  ", rewards)
print("优势值:  ", [f"{a:+.3f}" for a in advantages])
# 正确答案（r=1.0）→ 优势 ≈ +0.667 → 强化这些推理路径
# 错误答案（r=0.0）→ 优势 ≈ -1.333 → 抑制这些推理路径
# 注意：|负优势| > |正优势|，错误答案受到的抑制力度更大
```

### 3.3 GRPO 完整目标函数

GRPO 的优化目标结合了 PPO 的 Clip 机制和 KL 散度约束：

$$\mathcal{L}_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left[ \min\left( \rho_{i,t} \hat{A}_i,\ \text{clip}\left(\rho_{i,t}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) - \beta \cdot \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{ref}\right] \right]$$

逐项解读：

- $\frac{1}{G} \sum_{i=1}^{G}$：对 $G$ 个回答取平均——每个回答对梯度的贡献相等
- $\frac{1}{|y_i|} \sum_{t=1}^{|y_i|}$：对第 $i$ 个回答的 token 取平均——防止长回答主导梯度（长度归一化）
- $\rho_{i,t} = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$：第 $i$ 个回答第 $t$ 个 token 的重要性采样比率
- $\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, ...) \hat{A}_i)$：PPO Clip 策略损失——继承自 PPO，防止单步更新过大
- $\beta \cdot \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$：KL 散度惩罚——防止策略偏离 SFT 模型太远，避免奖励黑客和语言退化。关于 KL 散度的详细解释，请参阅 [附录 E：KL 散度详解](../appendix/kl_divergence.md)

### 3.4 GRPO 训练架构与流程

![GRPO 训练架构](../svg/chapter_agentic_rl_03_grpo_architecture.svg)

![GRPO 单次训练迭代流程](../svg/chapter_agentic_rl_03_grpo_iteration.svg)

### 3.5 基于 TRL 的 GRPO 完整实现

```python
"""
GRPO 训练的完整实现
基于 Hugging Face TRL 库的 GRPOTrainer
"""

from trl import GRPOConfig, GRPOTrainer

# ── GRPO 训练配置 ─────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir="./checkpoints/grpo",

    # GRPO 核心参数
    num_generations=8,               # G=8：平衡优势估计质量与采样成本
                                     # G 太小 → 方差大；G 太大 → 计算成本高

    # 训练超参数
    num_train_epochs=2,
    per_device_train_batch_size=1,   # 因需生成 G 个回答，batch size 要小
    gradient_accumulation_steps=8,   # 有效 batch size = 1 × 8 = 8
    learning_rate=5e-6,              # RL 阶段学习率 ≈ SFT 学习率的 1/40
                                     # 过大会导致策略崩溃，过小则收敛极慢
    warmup_ratio=0.1,
    max_grad_norm=0.5,               # 梯度裁剪，防止 RL 训练中的梯度爆炸

    # 生成参数
    max_new_tokens=512,
    temperature=0.7,                 # 保证 G 个回答的多样性
                                     # temperature 过低 → 回答趋同 → 优势全为 0

    # GRPO 算法参数
    kl_coef=0.01,                    # β：KL 散度惩罚系数
                                     # 过大 → 策略无法充分优化；过小 → 策略偏离过远

    # 精度与性能
    bf16=True,

    # 日志与检查点
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
)


# ── 奖励函数定义 ──────────────────────────────────────────────────────────
def reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Agent 行为质量的综合奖励函数（示例实现）
    
    详细的奖励函数设计方法参见 18.4 节
    """
    rewards = []
    for completion in completions:
        reward = 0.0

        # 维度 1：格式正确性
        has_think = "<think>" in completion and "</think>" in completion
        if has_think:
            reward += 0.2
            think_content = completion.split("<think>")[1].split("</think>")[0].strip()
            if len(think_content) > 20:
                reward += 0.1   # 有实质性推理内容

        # 维度 2：工具调用合理性
        if "<tool_call>" in completion and "</tool_call>" in completion:
            reward += 0.3
            try:
                tool_str = completion.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                if "(" in tool_str and ")" in tool_str:
                    reward += 0.2   # 函数调用语法正确
            except IndexError:
                reward -= 0.1   # 标签不配对

        # 维度 3：效率惩罚
        num_tool_calls = completion.count("<tool_call>")
        if num_tool_calls > 5:
            reward -= 0.1 * (num_tool_calls - 5)

        rewards.append(max(0.0, reward))

    return rewards


# ── 初始化并启动训练 ──────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,                     # SFT 阶段训练好的模型
    config=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

print("🚀 开始 GRPO 训练...")
trainer.train()
trainer.save_model("./checkpoints/grpo-final")
print("✅ GRPO 训练完成！")
```

---

## 第四部分：三大算法系统性对比

### 4.1 架构对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **所需模型** | Policy + Critic + Reference | Policy + Reference | Policy + Reference |
| **显存需求** | ≈ 3× 模型大小 | ≈ 2× 模型大小 | ≈ 1.5× 模型大小 |
| **训练数据** | 在线采样 + 奖励模型 | 离线偏好对 | 在线采样 + 奖励函数 |
| **优势估计** | GAE（依赖 Critic）| 无（隐式奖励差）| 组内标准化（无 Critic）|
| **更新约束** | Clip + KL | 隐式 KL（通过 $\beta$） | Clip + KL |

### 4.2 训练特性对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **训练稳定性** | 中（Critic 误差传播）| 高（监督学习）| 高（无 Critic 误差）|
| **超参数数量** | 多（≥6 个）| 极少（$\beta$ 1 个）| 少（≤4 个）|
| **数据效率** | 低（需在线采样）| 高（离线复用）| 中（需 G× 采样）|
| **可探索性** | 强（在线 RL）| 无（纯离线）| 强（在线 RL）|
| **能力上界** | 可超越数据 | 受限于偏好数据质量 | 可超越数据 |

### 4.3 选型决策指南

```
你的任务是否有客观可验证的评估标准？
├── 否 → 任务评估主要依赖人类偏好？
│         ├── 是 → 有足够的偏好标注数据？
│         │         ├── 是 → 选择 DPO ✅（最简单高效）
│         │         └── 否 → 先收集偏好数据，或选择 PPO + 奖励模型
│         └── 否 → 考虑是否真的需要 RL（也许 SFT 就够了）
└── 是 → 模型规模 > 7B？
          ├── 是 → 选择 GRPO ✅（显存友好，DeepSeek-R1 验证）
          └── 否 → PPO 或 GRPO 均可
                    ├── 追求通用性和成熟工具链 → PPO
                    └── 追求简洁和训练效率 → GRPO
```

### 4.4 实证表现

| 项目 | 算法 | 核心成果 |
|------|------|---------|
| **InstructGPT** [3] | PPO | 证明 RLHF 可大幅提升指令遵循能力 |
| **Llama 2** [8] | PPO | 70B 模型的安全对齐 |
| **Zephyr** [9] | DPO | 7B 模型用 DPO 超越 PPO 基线 |
| **DeepSeek-R1** [10] | GRPO | 涌现长链推理，数学/代码能力媲美 o1 |
| **DeepSWE** [11] | GRPO | SWE-bench Verified 59%（开源 SOTA）|

---

## 关键监控指标与调参指南

在 RL 训练过程中（PPO 或 GRPO），以下指标是判断训练健康状态的核心依据：

| 指标 | 健康范围 | 异常信号 | 处理方法 |
|------|---------|---------|---------| 
| `mean_reward` | 应稳步上升 | 长期不变或下降 | 检查奖励函数设计，降低 KL 系数 |
| `kl_divergence` | < 10–15 nats | 持续增大 | 增大 KL 系数 $\beta$ |
| `clip_fraction` | 0.1–0.3 | > 0.5 | 降低学习率或增大 clip $\epsilon$ |
| `mean_ratio` | 接近 1.0 | 持续偏离 1.0 | 减小学习率，增加 warmup |
| `reward_std` | > 0（组内有差异）| ≈ 0 | 增大 temperature，检查奖励函数 |

> **📌 工程实践要点**
>
> - **组大小 $G$ 的选择**（GRPO）：$G = 4$–$16$ 是常见范围。$G$ 太小则优势估计方差大，$G$ 太大则采样成本高。建议从 $G = 8$ 开始。
> - **温度参数**（GRPO）：建议 0.6–0.8。若 temperature 过低，$G$ 个回答可能完全相同，导致 $\sigma_r \approx 0$，优势全为零。
> - **学习率**：RL 阶段的学习率通常是 SFT 阶段的 $\frac{1}{10}$ 到 $\frac{1}{50}$。过大的学习率会导致策略在几步内崩溃。
> - **梯度裁剪**：建议 `max_grad_norm=0.5`，RL 训练中梯度爆炸比 SFT 更常见。
> - **$\beta$ 调节**（DPO）：$\beta$ 通常取 0.1–0.5。$\beta$ 太小 → 训练不稳定；$\beta$ 太大 → 策略几乎不更新。

---

*奖励函数是 GRPO 训练的核心驱动力——它定义了"什么是好的 Agent 行为"。下一节将系统介绍奖励函数的设计原则、多维度组合方法，以及如何防御奖励黑客攻击。*

---

## 参考文献

[1] WILLIAMS R J. Simple statistical gradient-following algorithms for connectionist reinforcement learning[J]. Machine Learning, 1992, 8(3): 229-256.

[2] SCHULMAN J, WOLSKI F, DHARIWAL P, et al. Proximal policy optimization algorithms[R]. arXiv preprint arXiv:1707.06347, 2017.

[3] OUYANG L, WU J, JIANG X, et al. Training language models to follow instructions with human feedback[C]//Advances in Neural Information Processing Systems (NeurIPS). 2022.

[4] SCHULMAN J, MORITZ P, LEVINE S, et al. High-dimensional continuous control using generalized advantage estimation[C]//International Conference on Learning Representations (ICLR). 2016.

[5] RAFAILOV R, SHARMA A, MITCHELL E, et al. Direct preference optimization: Your language model is secretly a reward model[C]//Advances in Neural Information Processing Systems (NeurIPS). 2023.

[6] BRADLEY R A, TERRY M E. Rank analysis of incomplete block designs: I. The method of paired comparisons[J]. Biometrika, 1952, 39(3/4): 324-345.

[7] SHAO Z, WANG P, ZHU Q, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models[R]. arXiv preprint arXiv:2402.03300, 2024.

[8] TOUVRON H, MARTIN L, STONE K, et al. Llama 2: Open foundation and fine-tuned chat models[R]. arXiv preprint arXiv:2307.09288, 2023.

[9] TUNSTALL L, BEECHING E, LAMBERT N, et al. Zephyr: Direct distillation of LM alignment[R]. arXiv preprint arXiv:2310.16944, 2023.

[10] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[11] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
