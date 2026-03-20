<!--
====================================
  知乎发布版 - 使用说明
====================================

本文件是从 agentic_rl_zhihu.md 自动转换的知乎适配版本。

发布步骤：
1. 打开知乎 → 写文章
2. 点击右上角 "..." → "导入文档" → 选择本 .md 文件
3. 导入后，搜索 "📷" 找到所有图片占位符
4. 逐一上传 images/ 目录下对应的 PNG 图片
5. 检查公式渲染是否正确（知乎支持 LaTeX）
6. 检查表格显示是否正常
7. 发布！

图片文件清单（共 10 张）：
  images/chapter_agentic_rl_01_overview.png
  images/chapter_agentic_rl_02_sft_grpo.png
  images/chapter_agentic_rl_03_three_algorithms.png
  images/chapter_agentic_rl_03_ppo_clip.png
  images/chapter_agentic_rl_03_ppo_training_flow.png
  images/chapter_agentic_rl_03_ppo_architecture.png
  images/chapter_agentic_rl_03_dpo_intuition.png
  images/chapter_agentic_rl_03_dpo_architecture.png
  images/chapter_agentic_rl_03_grpo_architecture.png
  images/chapter_agentic_rl_03_grpo_iteration.png

====================================
-->

# Agentic-RL 全面解析：从 SFT 到 GRPO，系统掌握智能体强化学习训练

> *"如果说 Prompt Engineering 是给 Agent 写'使用说明书'，那么 Agentic-RL 就是让 Agent 通过反复实践，自己悟出最优解法。"*

全书github地址： https://github.com/Haozhe-Xing/agent_learning
在线阅读地址：https://haozhe-xing.github.io/agent_learning/

长期以来，我们构建 AI Agent 的主流方式是**提示词 + 工具调用**——Agent 的所有能力来自基座模型的预训练知识加上精心设计的 prompt。这种方式简单灵活，但存在一个根本性瓶颈：

> **Agent 的能力上界 = 基座模型的通用能力上界。**

**Agentic-RL（Agentic Reinforcement Learning）** 提供了另一条路径：**通过强化学习训练，让模型自主习得完成 Agent 任务的最优策略**。DeepSeek-R1 和 DeepSWE 等工作已经证明，经过 RL 训练的模型可以涌现出训练数据中从未出现过的推理策略，在推理和工具使用能力上显著超越纯 prompt 方式。

本文将系统性地介绍 Agentic-RL 的完整知识体系，涵盖以下内容：

- **什么是 Agentic-RL**：与传统后训练的本质区别，MDP 框架建模
- **SFT + LoRA 基础训练**：监督微调的形式化原理与参数高效训练
- **PPO**：策略梯度、重要性采样、优势函数、GAE 与 Clip 机制
- **DPO**：从 RLHF 到 DPO 的完整数学推导，隐式奖励思想
- **GRPO + 奖励函数设计**：组内比较替代 Critic、多维度奖励函数与奖励黑客防御
- **完整实战 Pipeline**：基于 GSM8K 从数据准备到模型部署
- **最新研究进展（2025—2026）**：DeepSeek-R1、DAPO、VAPO、SAR 等前沿工作


# 一、什么是 Agentic-RL

## 从“提示工程”到“训练优化”的范式转变

> **Agent 的能力上界 = 基座模型的通用能力上界。**

无论提示词设计得多精妙，如果基座模型在某类推理上存在系统性缺陷（如多步数学推理、复杂代码修复、长程规划），Agent 的表现就无法突破这一天花板。

**Agentic-RL（Agentic Reinforcement Learning）** 提出了一个根本不同的思路：**与其在推理时通过 prompt 引导模型行为，不如在训练时通过强化学习信号让模型自主习得高质量的 Agent 策略**。这一范式的核心洞察来自 DeepSeek-R1 的实验发现——纯粹的 RL 训练可以在模型中涌现出人类未曾显式教授的推理链条。

### 两种范式的系统性对比

| 维度 | Prompt Engineering | Agentic-RL |
|------|-------------------|------------|
| **能力来源** | 基座模型预训练知识 + 提示词引导 | 基座模型 + 任务特定的 RL 优化 |
| **开发成本** | 低（工程师时间） | 高（GPU 算力 + 数据标注） |
| **任务适应性** | 通用但不精专 | 针对特定任务深度优化 |
| **推理效率** | 依赖长 prompt，Token 消耗大 | 能力内化到权重，推理更高效 |
| **可扩展性** | 受限于上下文窗口和 prompt 长度 | 可通过持续训练迭代提升 |
| **能力上界** | 受限于基座模型 | 可超越基座模型（涌现能力）|
| **适用场景** | 快速原型、通用任务、低频需求 | 高频、高价值、有明确评估标准的任务 |

### 何时应当选择 Agentic-RL？

Agentic-RL 并非万能药。以下是基于实践经验总结的决策框架：

**适合投入 Agentic-RL 的场景：**
- ✅ 任务具有**客观可验证的评估标准**（代码测试通过率、数学答案正确性、API 调用成功率）
- ✅ 任务**高频重复**，训练成本可被长期收益摊薄
- ✅ 当前基座模型在该任务上存在**系统性、可改进的缺陷**
- ✅ 具备**足够的训练数据**或可通过自动化方式生成数据

**不适合 Agentic-RL 的场景：**
- ❌ 一次性、低频的开放式任务（ROI 不足）
- ❌ 无法客观量化评估的任务（如开放式创意写作）
- ❌ 基座模型 + prompt 方式已达到可接受水平
- ❌ 缺乏 GPU 算力资源（7B 模型 GRPO 训练至少需要 1× A100 40GB）

## Agentic-RL 的理论基础：MDP 框架

### 马尔可夫决策过程建模

Agentic-RL 的理论基础是**马尔可夫决策过程（Markov Decision Process, MDP）**。将 Agent 的任务执行过程形式化为一个有限时域 MDP：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

其中各要素在 Agent 场景中的对应关系如下：

| MDP 要素 | 形式化定义 | Agent 场景中的对应 |
|----------|-----------|-------------------|
| **状态空间** $\mathcal{S}$ | $s_t \in \mathcal{S}$ | 当前对话历史 + 工具返回结果 + 环境上下文 |
| **动作空间** $\mathcal{A}$ | $a_t \in \mathcal{A}$ | 模型的下一次 Token 序列输出（文本、工具调用、代码等）|
| **转移函数** $\mathcal{T}$ | $s_{t+1} \sim \mathcal{T}(\cdot \mid s_t, a_t)$ | 环境对动作的响应（工具执行结果、用户反馈）|
| **奖励函数** $\mathcal{R}$ | $r_t = \mathcal{R}(s_t, a_t)$ | 任务完成度评估（答案正确性、代码通过率等）|
| **策略** $\pi_\theta$ | $a_t \sim \pi_\theta(\cdot \mid s_t)$ | 模型参数 $\theta$ 决定的条件生成分布 |
| **折扣因子** $\gamma$ | $\gamma \in [0, 1]$ | 对未来奖励的折扣（通常取 1.0，即不折扣）|

**训练目标**是最大化期望累积奖励：

$$\theta^* = \arg\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

逐项解读：

- $\theta^*$：最优模型参数，即我们希望通过训练找到的目标
- $\arg\max_\theta$：在所有可能的参数 $\theta$ 中，找到使目标函数最大的那个
- $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$：**期望**运算符——由于模型生成是随机的（temperature > 0），同一问题每次生成的轨迹 $\tau$ 不同，我们优化的是所有可能轨迹上的**平均**表现，而非某一次特定生成的结果
- $\sum_{t=0}^{T} \gamma^t r_t$：**折扣累积奖励**，将轨迹中每一步的即时奖励 $r_t$ 按时间折扣 $\gamma^t$ 加权求和；$\gamma < 1$ 时，近期奖励比远期奖励权重更大（在 Agentic-RL 中通常取 $\gamma = 1.0$，即不折扣，因为我们关心任务的最终完成质量而非中间步骤的时序差异）
- $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$：一条完整的**交互轨迹**，记录了从初始状态到终止状态的完整状态-动作序列

**直觉理解**：这个目标函数的含义是——调整模型参数 $\theta$，使得模型在面对各种任务时，平均能够获得尽可能高的累积奖励。这与人类学习的直觉一致：通过大量练习（采样轨迹），不断调整策略（更新参数），使得平均表现持续提升。

### Agent 交互循环的形式化描述


> **📷 图 1：Agentic-RL 训练架构概览（从 Prompt 方式到 SFT+GRPO 两阶段训练）**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_01_overview.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_01_overview.png`



### 六大核心能力维度

经过 Agentic-RL 训练，模型可在以下六个维度获得系统性提升：

| 能力维度 | 描述 | 典型提升表现 |
|---------|------|------------|
| **指令遵循** | 准确理解并执行复杂、多约束的指令 | 格式符合率从 ~30% 提升至 ~90% |
| **工具使用** | 在正确时机调用正确工具，处理工具返回结果 | 工具调用准确率显著提升 |
| **多步推理** | 在复杂任务中维持长链条推理，减少中间步骤错误 | 数学推理准确率提升 20-30% |
| **自我纠错** | 识别执行错误并主动修正，而非继续错误路径 | 错误恢复率提升 |
| **探索策略** | 在不确定情况下合理尝试不同方案 | 首次成功率提升 |
| **效率优化** | 用更少步骤、更少 Token 完成任务 | 平均轨迹长度缩短 |


## 两阶段训练范式

当代主流的 Agentic-RL 训练遵循 **SFT → RL** 的两阶段范式，两个阶段各有其不可替代的作用。

### 阶段一：SFT（监督微调）—— 策略初始化

**核心目标**：将基座模型的策略分布 $\pi_0$ 调整为具备基本 Agent 行为格式的初始策略 $\pi_{SFT}$。

SFT 阶段的训练目标是最大化对数似然：

$$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x, y^*) \sim \mathcal{D}_{SFT}} \left[ \log \pi_\theta(y^* \mid x) \right]$$

逐项解读：

- $-\mathbb{E}_{(x,y^*)\sim\mathcal{D}_{SFT}}$：负号 + 期望符号，表示在监督微调数据集 $\mathcal{D}_{SFT}$ 上，对所有样本 $(x, y^*)$ 取平均。
  - $x$：模型的输入（如用户指令、提示词）。
  - $y^*$：对应的**人类标注的理想输出**（即"标准答案"）。
  - $\mathcal{D}_{SFT}$：收集好的指令 - 答案对数据集。
- $\mathcal{D}_{SFT} = \{(x^{(i)}, y^{*(i)})\}_{i=1}^N$：高质量的 Agent 交互轨迹数据集，每条样本包含输入上下文 $x$（系统提示 + 用户问题）和专家示范输出 $y^*$（含推理过程和工具调用）
- $\log \pi_\theta(y^* \mid x)$：模型在给定输入 $x$ 的条件下，生成专家示范序列 $y^*$ 的**对数概率**。这个值越大（越接近 0），说明模型认为专家示范是"合理的输出"；加负号后变为损失，最小化损失即最大化对数概率
- 实际计算时，$\log \pi_\theta(y^* \mid x)$ 按**自回归分解**展开为逐 token 的对数概率之和：$\sum_{t=1}^{|y^*|} \log \pi_\theta(y^*_t \mid x, y^*_{<t})$，即每个 token 的生成概率都以前面所有 token 为条件

**直觉理解**：SFT 的本质是"模仿学习"——给模型看大量专家示范，让模型学会"在这种输入下，专家会输出什么"。这个阶段类似于"临帖练字"——先模仿正确的行为模式，建立格式规范和基本能力。

```
训练数据示例：
输入 x: "计算 1234 × 5678 的结果"
期望输出 y*:
  <think>
  这需要精确的整数乘法计算，应使用计算器工具确保精度。
  </think>
  <tool_call>calculator(expression="1234 * 5678")</tool_call>
```

### 阶段二：RL（强化学习）—— 策略优化

**核心目标**：在 $\pi_{SFT}$ 的基础上，通过奖励信号引导策略向 $\pi^*$ 逼近，突破 SFT 数据的质量上界。

RL 阶段的训练目标是最大化期望奖励，同时通过 KL 散度约束防止策略偏离过远：

$$\mathcal{L}_{RL}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{SFT})$$

逐项解读：

- $-\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$：**策略损失项**——最大化当前策略 $\pi_\theta$ 采样到的轨迹的期望奖励。负号是将最大化转为最小化（梯度下降惯例）
- $\beta \cdot D_{KL}(\pi_\theta \| \pi_{SFT})$：**KL 散度惩罚项**——$D_{KL}$ 衡量当前策略 $\pi_\theta$ 与 SFT 初始策略 $\pi_{SFT}$ 之间的"分布距离"。当两个分布完全相同时 $D_{KL} = 0$；差异越大，$D_{KL}$ 越大。系数 $\beta$ 控制惩罚强度：$\beta$ 越大，策略越保守（不敢偏离 SFT 模型）；$\beta$ 越小，策略越激进（可能产生奖励黑客行为）。
  > 💡 **什么是 KL 散度？** 如果你对 KL 散度还不熟悉，推荐阅读相关 KL 散度的科普资料，其中包含从直觉理解到数学定义的完整介绍。简单来说，$D_{KL}(P \| Q)$ 回答的问题是：**“如果真实分布是 $P$，用分布 $Q$ 来近似它，平均会损失多少信息？”**
- **两项的博弈**：策略损失项鼓励模型"大胆探索"以获得更高奖励，KL 惩罚项约束模型"不要走太远"以保持语言质量。$\beta$ 是这两种力量的平衡点

SFT 只能让模型达到训练数据的水平上界。而 RL 阶段通过奖励函数告诉模型"什么是好的结果"，让模型自主探索出超越示范数据的解决路径——这正是 DeepSeek-R1 能够涌现出"自我反思"和"长链推理"能力的关键机制。

```
SFT 阶段学到：  "看到数学题 → 调用计算器"
RL 阶段学到：   "分析问题结构 → 判断是否需要分步 → 
                 选择最优工具组合 → 验证中间结果 → 
                 发现错误时主动回溯"
```

### 为什么两个阶段缺一不可？

| 对比维度 | 纯 SFT | 纯 RL（从随机初始化）| SFT → RL |
|---------|--------|---------------------|----------|
| **格式规范性** | ✅ 高 | ❌ 极低（输出混乱）| ✅ 高 |
| **能力上界** | ❌ 受限于数据质量 | ⚠️ 理论上无上界，实践中难收敛 | ✅ 可超越数据质量 |
| **训练稳定性** | ✅ 稳定 | ❌ 极不稳定，容易发散 | ✅ 较稳定 |
| **收敛速度** | 快 | 极慢 | 中等 |
| **最终性能** | 中等 | 不确定 | **最优** |

> **📌 工程实践要点**
>
> SFT 阶段的数据质量比数量更关键。LIMA 的研究表明，1,000 条精心筛选的高质量数据往往优于 10,000 条噪声数据。实践建议：
> - **SFT 数据规模**：500–2,000 条经过人工验证的 Agent 交互轨迹
> - **RL 计算成本**：约为 SFT 阶段的 3–10 倍（因需要在线采样）
> - **验证策略**：先在 7B 小模型上验证训练流程的正确性，再扩展到更大模型


## 代表性工作与实证结果

| 项目 | 基座模型 | 训练方法 | 核心成果 |
|------|---------|---------|---------|
| **DeepSeek-R1** | DeepSeek-V3 | SFT + GRPO | 数学/代码推理能力媲美 OpenAI o1 |
| **DeepSWE** | DeepSeek-R1 | SFT + GRPO | SWE-bench Verified 59%（开源 SOTA）|
| **OpenAI o1** | GPT-4 系列 | RL（具体方法未公开）| 数学/编程/科学推理大幅提升 |
| **Qwen-Agent** | Qwen2.5 | SFT + DPO | 工具调用和多步推理能力提升 |

这些工作共同验证了 Agentic-RL 范式的有效性：**通过强化学习训练，模型可以涌现出训练数据中未曾出现的推理策略**，这是纯 SFT 方法无法实现的。


*下面从 SFT + LoRA 开始，详细介绍第一阶段监督微调的原理与实现。*


# 二、SFT + LoRA：监督微调与参数高效训练

## 监督微调的形式化定义

**SFT（Supervised Fine-Tuning，监督微调）** 是 Agentic-RL 训练的第一阶段，其目标是将通用基座模型 $\pi_0$ 调整为具备特定 Agent 行为格式的初始策略 $\pi_{SFT}$。

形式化地，SFT 的训练目标是在专家示范数据集 $\mathcal{D} = \{(x^{(i)}, y^{*(i)})\}_{i=1}^N$ 上最小化负对数似然损失：

$$\mathcal{L}_{SFT}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{|y^{*(i)}|} \log \pi_\theta\left(y^{*(i)}_t \mid x^{(i)}, y^{*(i)}_{<t}\right)$$

逐项解读：

- $\frac{1}{N} \sum_{i=1}^{N}$：对 $N$ 条训练样本求平均，使损失不受数据集规模影响
- $\sum_{t=1}^{|y^{*(i)}|}$：对第 $i$ 条样本的输出序列中每个 token 求和——这是语言模型的**自回归分解**：$\log \pi_\theta(y^* \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y^*_t \mid x, y^*_{<t})$，即序列的联合概率等于每个 token 条件概率的乘积（对数域为求和）
- $\log \pi_\theta(y^{*(i)}_t \mid x^{(i)}, y^{*(i)}_{<t})$：模型在已知输入 $x^{(i)}$ 和前 $t-1$ 个 token $y^{*(i)}_{<t}$ 的条件下，预测第 $t$ 个 token 为 $y^{*(i)}_t$ 的对数概率。这个值越大（越接近 0），说明模型对这个 token 的预测越确信
- 加负号后，最小化损失 = 最大化对数似然 = 让模型对专家示范的每个 token 都尽可能高的概率生成

**自回归的直觉**：训练时，模型看到的是“正确答案”（teacher forcing）——即使模型在第 $t-1$ 步预测错了，第 $t$ 步仍然使用真实的 $y^*_{t-1}$ 作为条件。这使得 SFT 训练非常高效，但也带来了训练-推理分布偏移的问题（推理时模型看到的是自己生成的 token，而非专家示范）。

其中 $x^{(i)}$ 是输入上下文（系统提示 + 用户问题），$y^{*(i)}$ 是专家示范的完整输出序列（包含推理过程和工具调用）。

### SFT 在 Agentic-RL 中的角色

SFT 阶段解决的核心问题是**策略初始化**：从随机初始化或通用预训练权重出发，直接进行 RL 训练会面临极高的探索成本——模型需要从海量随机输出中偶然发现有效的 Agent 行为格式。SFT 通过模仿学习将策略分布约束在合理的行为空间内，为后续 RL 优化提供稳定的起点。


> **📷 图 2：SFT → RL 两阶段训练流程**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_02_sft_grpo.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_02_sft_grpo.png`



### SFT 训练数据格式规范

对于 Agent 任务，SFT 的训练数据是**多轮对话 + 工具调用**的完整轨迹，遵循 ChatML 格式：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个数学助手，可以使用 calculator 工具进行精确计算。"
    },
    {
      "role": "user",
      "content": "一个圆形水池半径 3 米，水深 1.5 米，水池中有多少立方米的水？"
    },
    {
      "role": "assistant",
      "content": "<think>\n圆柱体体积公式：V = π × r² × h\n其中 r = 3 m，h = 1.5 m\n需要精确计算 π × 9 × 1.5，使用计算器工具。\n</think>\n\n<tool_call>\ncalculator(expression=\"3.14159265 * 3**2 * 1.5\")\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"result\": 42.4115}"
    },
    {
      "role": "assistant",
      "content": "水池中的水量约为 **42.41 立方米**。\n\n计算过程：V = π × r² × h = π × 3² × 1.5 ≈ 42.41 m³"
    }
  ]
}
```

### 数据质量优先原则

LIMA 的研究提供了重要的实证依据：**1,000 条精心筛选的高质量数据，其效果往往优于 10,000 条噪声数据**。对于 Agent SFT，数据质量的评估维度如下：

| 质量维度 | 标准 | 验证方法 |
|---------|------|---------|
| **格式一致性** | 统一的 `<think>`/`<tool_call>` 标签格式 | 正则表达式自动检查 |
| **工具调用正确性** | 参数类型、名称与工具定义完全匹配 | 静态解析验证 |
| **推理连贯性** | `<think>` 内容与最终动作逻辑一致 | 人工抽样审查 |
| **任务覆盖度** | 覆盖所有工具的调用模式和边界情况 | 工具调用分布统计 |
| **难度分布** | 简单/中等/复杂任务比例均衡 | 人工分级标注 |

**推荐数据规模**：500–2,000 条经过人工验证的高质量 Agent 交互轨迹。


## 全参数微调的资源困境

在理解 LoRA 之前，需要先明确全参数微调（Full Fine-Tuning）面临的根本性挑战。

以 Llama 3.1 8B 为例，训练时的显存需求分析如下：

```python
# 显存需求精确估算（以 Llama 3.1 8B 为例）
total_params = 8_000_000_000        # 80 亿参数

# 推理阶段（float16）
inference_memory = total_params * 2 / (1024**3)   # ≈ 14.9 GB

# 训练阶段（Adam 优化器，混合精度）
# 参数（float32）：4 bytes/param
# 梯度（float32）：4 bytes/param
# Adam 一阶矩（float32）：4 bytes/param
# Adam 二阶矩（float32）：4 bytes/param
# 合计：16 bytes/param
training_memory = total_params * 16 / (1024**3)   # ≈ 119.2 GB

# 结论：全参数微调需要至少 2-3 张 A100 80GB
# 对大多数团队而言，这一成本难以承受
```

这一资源困境催生了**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 方法的研究，其中 LoRA 是目前最广泛应用的方案。


## LoRA：低秩适应的理论基础

### 核心假设与数学推导

**LoRA（Low-Rank Adaptation）** 的理论基础来自一个关键的实证发现：

> **预训练模型在微调过程中，权重更新矩阵 $\Delta W$ 具有显著的低秩特性（intrinsic low rank）。**

**为什么微调的更新是低秩的？** 直觉上，预训练模型已经学会了丰富的通用表示，微调只需要在这个表示空间中做小幅度的“方向调整”。这种调整展开在一个低维子空间中，而非需要改变所有 $d \times k$ 个方向。Aghajanyan et al. 通过实验证明，微调的“内在维度”（intrinsic dimensionality）远小于模型参数量。

这意味着微调时真正需要改变的“信息量”远小于模型参数量。基于此，LoRA 提出以下参数化方案：

对于原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，不直接更新 $W_0$，而是将权重更新分解为两个低秩矩阵的乘积：

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B A$$

逐项解读：

- $W_0 \in \mathbb{R}^{d \times k}$：**冻结的预训练权重**，训练期间完全不变，保留模型的通用知识
- $A \in \mathbb{R}^{r \times k}$：**下投影矩阵**，将 $k$ 维输入压缩到 $r$ 维的低秩空间；使用高斯分布初始化，保证训练开始时有非零梯度
- $B \in \mathbb{R}^{d \times r}$：**上投影矩阵**，将 $r$ 维低秩表示映射回 $d$ 维输出空间；**初始化为全零**，确保训练开始时 $\Delta W = BA = 0$，即模型行为与基座模型完全相同，训练稳定性有保障
- $r \ll \min(d, k)$：**秩**，控制参数量（通常取 8, 16, 32, 64）。$r$ 越小，参数量越少，但表达能力越弱；$r$ 越大，表达能力越强，但参数量增加
- $\frac{\alpha}{r}$：**缩放因子**，控制 LoRA 更新的幅度。设置为 $\frac{\alpha}{r}$ 而非直接用 $\alpha$，是为了使实际缩放幅度与 $r$ 的选择解耦合：当 $\alpha = 2r$ 时，无论 $r$ 取何值，实际缩放因子始终为 2，方便跨不同 $r$ 值的实验对比

**前向传播**：

$$h = W x = W_0 x + \frac{\alpha}{r} B A x$$

训练时冻结 $W_0$，仅更新 $A$ 和 $B$。

### 参数效率分析

LoRA 的参数量压缩比可以精确计算：

$$\text{压缩比} = \frac{r \cdot k + d \cdot r}{d \cdot k} = \frac{r(d + k)}{dk} \approx \frac{2r}{\min(d,k)} \quad (\text{当 } d \approx k \text{ 时})$$

当 $r = 16$，$d = k = 4096$ 时，压缩比 $\approx \frac{2 \times 16}{4096} \approx 0.78\%$，即仅需训练不到 1% 的参数。

```python
# 参数量对比（以 Llama 3.1 8B 的单个注意力投影层为例）
d, k = 4096, 4096

# 原始层参数量
original_params = d * k                    # = 16,777,216 (16M)

# LoRA 参数量（r=16）
r = 16
lora_params = r * k + d * r               # A: 65,536 + B: 65,536 = 131,072 (128K)

# 单层压缩比
compression_ratio = lora_params / original_params   # ≈ 0.78%

# 全模型 LoRA 参数量（应用于所有注意力层）
# Llama 3.1 8B: 32 层，每层 4 个投影（q, k, v, o）
num_lora_layers = 32 * 4
total_lora_params = num_lora_layers * lora_params   # ≈ 16.8M

# 全模型压缩比
total_compression = total_lora_params / total_params  # ≈ 0.21%
# 仅需训练不到 0.3% 的参数！
```

### 关键超参数的选择指南

| 超参数 | 含义 | 推荐范围 | 选择依据 |
|--------|------|---------|---------|
| **$r$（秩）** | 低秩矩阵的秩，控制表达能力 | 8–64 | 任务复杂度：简单格式学习用 8–16，复杂推理用 32–64 |
| **$\alpha$（缩放）** | LoRA 更新的缩放因子 | 通常 $= 2r$ | 实际缩放为 $\alpha/r$，设为 $2r$ 使有效缩放为 2 |
| **`target_modules`** | 应用 LoRA 的层 | 至少 q_proj, v_proj | 注意力层效果最显著；加入 FFN 层可提升表达能力 |
| **`lora_dropout`** | LoRA 层的 Dropout 率 | 0.05–0.1 | 数据量少时适当增大以防过拟合 |

> **💡 秩 $r$ 的选择经验法则**
>
> - **格式学习、风格迁移**（任务简单）：$r = 8$–$16$
> - **Agent 工具调用学习**（任务中等）：$r = 16$–$32$
> - **数学推理、代码生成**（任务复杂）：$r = 32$–$64$
> - **不确定时**：从 $r = 16$ 开始，观察验证集损失是否充分收敛

## 实战：基于 LoRA 的 Agent SFT 训练

### 环境依赖

```bash
pip install torch>=2.1.0 transformers>=4.40.0 peft>=0.10.0
pip install trl>=0.12.0 datasets accelerate bitsandbytes
```

### 步骤一：构建训练数据集

```python
"""
Agent SFT 训练数据构建
将 Agent 交互轨迹转换为 ChatML 格式的训练样本
"""

import json
from datasets import Dataset

def build_agent_sft_dataset() -> Dataset:
    """
    构建 Agent SFT 训练数据集
    
    每条样本包含完整的多轮对话轨迹：
    系统提示 → 用户问题 → 助手推理+工具调用 → 工具结果 → 最终回答
    """
    examples = [
        {
            "instruction": "帮我查看北京今天的天气，然后告诉我应该穿什么衣服。",
            "output": (
                "<think>\n"
                "用户需要两步操作：① 获取北京实时天气数据；② 根据温度给出穿衣建议。\n"
                "先调用天气工具获取数据，再基于结果给出建议。\n"
                "</think>\n\n"
                "<tool_call>\n"
                "get_weather(city=\"北京\")\n"
                "</tool_call>"
            )
        },
        {
            "instruction": "计算从上海到北京的高铁票价，每人 553 元，3 个人来回的总费用。",
            "output": (
                "<think>\n"
                "计算公式：总费用 = 单价 × 人数 × 2（来回）= 553 × 3 × 2\n"
                "使用计算器确保精确性。\n"
                "</think>\n\n"
                "<tool_call>\n"
                "calculator(expression=\"553 * 3 * 2\")\n"
                "\</tool_call\>"
            )
        },
        # 实际训练需要 500–2,000 条覆盖各类工具调用场景的数据
    ]
    
    return Dataset.from_list(examples)


def format_to_chatml(example: dict) -> dict:
    """将单条样本格式化为 ChatML 训练格式"""
    text = (
        "<|im_start|>system\n"
        "你是一个智能助手，可以使用工具完成任务。"
        "解题时请先在 <think> 标签中写出推理过程，需要工具时使用 <tool_call> 标签。\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}\n<|im_end|>"
    )
    return {"text": text}


dataset = build_agent_sft_dataset()
train_dataset = dataset.map(format_to_chatml)
```

### 步骤二：模型加载与 LoRA 配置

```python
"""
模型加载与 LoRA 配置
采用 QLoRA（4-bit 量化 + LoRA）进一步降低显存需求
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── 模型选择 ──────────────────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-7B-Instruct"

# ── QLoRA：4-bit 量化配置 ─────────────────────────────────────────────────
# QLoRA 的核心思想：将模型权重量化为 4-bit 存储，但计算时反量化为 bfloat16
# 
# NF4（NormalFloat4）量化原理：
#   预训练权重近似服从正态分布 N(0, σ²)
#   NF4 将正态分布的值域划分为 16 个等概率区间（而非等间距区间）
#   这使得量化误差在统计意义上最小（信息论最优）
#   相比均匀量化（INT4），NF4 在相同 bit 数下精度损失更小
#
# 双重量化（Double Quantization）：
#   量化过程本身需要存储量化常数（scale factor），每 64 个参数共享一个 float32 常数
#   双重量化将这些量化常数再次量化为 8-bit，额外节省约 0.4 bits/param
#   对 7B 模型，双重量化额外节省约 0.4 × 7B / 8 ≈ 350 MB 显存
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4：信息论最优的 4-bit 量化格式
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # 对量化常数再次量化，额外节省 ~0.4 bits/param
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA 配置 ─────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                                # Agent 任务建议 r=32，平衡表达能力与参数效率
    lora_alpha=64,                       # alpha = 2r，有效缩放因子为 2
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力投影层（必选）
        "gate_proj", "up_proj", "down_proj",        # FFN 层（可选，提升表达能力）
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 示例输出：trainable params: 83,886,080 || all params: 7,615,684,608 || trainable%: 1.10%
```

### 步骤三：训练配置与执行

```python
"""
SFT 训练执行
关键超参数的选择依据均在注释中说明
"""

from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./checkpoints/sft",

    # ── 训练超参数 ──────────────────────────────────────────────────────────
    num_train_epochs=3,                  # Agent 格式学习通常 2–3 个 epoch 即可收敛
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # 有效 batch size = 4 × 4 = 16
    learning_rate=2e-4,                  # LoRA 推荐学习率范围：1e-4 ~ 5e-4
    warmup_ratio=0.1,                    # 前 10% 步数线性 warmup，防止训练初期不稳定
    weight_decay=0.01,                   # L2 正则化，防止过拟合

    # ── 精度与性能优化 ──────────────────────────────────────────────────────
    bf16=True,                           # bfloat16 混合精度：比 fp16 数值更稳定
    gradient_checkpointing=True,         # 以重计算换显存：显存减少 ~60%，速度降低 ~20%

    # ── 学习率调度 ──────────────────────────────────────────────────────────
    lr_scheduler_type="cosine",          # 余弦退火：比线性衰减通常效果更好

    # ── 日志与检查点 ────────────────────────────────────────────────────────
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,         # 训练结束后自动加载验证集最优检查点

    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

print("🚀 开始 SFT 训练...")
trainer.train()

# 保存 LoRA 适配器权重（仅保存增量参数，约 100–300 MB）
model.save_pretrained("./checkpoints/sft-lora")
tokenizer.save_pretrained("./checkpoints/sft-lora")
print("✅ LoRA 权重已保存至 ./checkpoints/sft-lora")
```

### 步骤四：权重合并与推理验证

```python
"""
将 LoRA 适配器权重合并回基座模型
合并后的模型与原始模型结构完全相同，可直接用于推理部署
"""

from peft import PeftModel

# 加载基座模型（全精度，用于合并）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载 LoRA 适配器并合并
# merge_and_unload() 将 ΔW = BA 合并到 W₀ 中，恢复标准模型结构
model = PeftModel.from_pretrained(base_model, "./checkpoints/sft-lora")
model = model.merge_and_unload()

# 推理验证
prompt = (
    "<|im_start|>system\n你是一个数学助手，可以使用 calculator 工具进行精确计算。\n<|im_end|>\n"
    "<|im_start|>user\n计算 17 的平方根，精确到小数点后 4 位\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```


## 训练过程中的常见问题与诊断

### 问题一：训练损失不下降

```python
# 系统性诊断检查清单
diagnostics = {
    "学习率设置": {
        "症状": "损失从第一步就几乎不变",
        "原因": "学习率过小（< 1e-5）或过大（> 1e-3）",
        "解决": "尝试 2e-4，使用学习率查找器（LR Finder）确定最优值",
    },
    "数据格式": {
        "症状": "损失下降但模型输出格式混乱",
        "原因": "tokenizer 的 special tokens 未正确设置",
        "解决": "检查 pad_token、eos_token 是否正确，验证 ChatML 模板",
    },
    "LoRA 目标层": {
        "症状": "损失下降极慢",
        "原因": "target_modules 与模型架构不匹配",
        "解决": "打印 model.named_modules() 确认层名称",
    },
    "梯度消失": {
        "症状": "损失先下降后停滞",
        "原因": "gradient_checkpointing 与某些层不兼容",
        "解决": "临时关闭 gradient_checkpointing 验证",
    },
}
```

### 问题二：过拟合

```python
# 过拟合信号：训练损失持续下降，验证损失在某点后开始上升
solutions = {
    "增加正则化": "lora_dropout 从 0.05 提升至 0.1",
    "减少训练轮数": "使用 early stopping，通常 2–3 个 epoch 足够",
    "增加数据多样性": "用 GPT-4 改写现有数据，增加表达多样性",
    "降低秩 r": "r 过大容易过拟合，从 32 降至 16 尝试",
}
```

### 问题三：显存不足（OOM）

```python
# 显存优化策略（按效果从强到弱排列）
memory_optimizations = [
    ("4-bit 量化 QLoRA",        "显存减少 ~75%，精度损失极小"),
    ("gradient_checkpointing",  "显存减少 ~60%，速度降低 ~20%"),
    ("降低 max_seq_length",     "显存与序列长度平方成正比"),
    ("减小 batch_size",         "最基本的方法，配合 gradient_accumulation"),
    ("降低 LoRA r 值",          "r 减半，LoRA 参数量减半"),
    ("CPU offload",             "将优化器状态卸载到 CPU，速度大幅降低"),
]
```

> **📌 工程实践要点**
>
> - **数据准备时间 >> 训练时间**：实际项目中，80% 的时间花在数据收集、清洗和验证上
> - **数据可执行性验证**：每条训练数据的工具调用格式应通过静态解析验证，确保参数合法
> - **A/B 测试**：SFT 模型上线前，务必与 prompt-only 基线进行对照实验
> - **显存估算公式**：QLoRA 4-bit 下，7B 模型训练约需 10–12 GB 显存（含梯度和优化器状态）
> - **版本管理**：使用 wandb 或 TensorBoard 记录每次训练的超参数、损失曲线和评估指标


*SFT 阶段让模型习得了 Agent 行为的基本格式与工具调用模式。然而，SFT 的能力上界受限于训练数据的质量——模型无法超越示范数据的水平。接下来介绍的 PPO 算法是 RL 阶段的经典方案。*


# 三、PPO：近端策略优化

前面我们介绍了 Agentic-RL 的两阶段训练范式（SFT → RL）。RL 阶段的核心问题是：**如何根据奖励信号来更新模型参数？** 这正是策略优化算法要解决的问题。

本节将从零开始，**系统性地讲解 PPO（Proximal Policy Optimization）算法**——它是 InstructGPT 和 ChatGPT 的核心训练算法，也是理解后续 DPO、GRPO 算法的基础。我们将从最基本的直觉出发，逐步推导数学公式，并通过大量图示帮助理解。


> **📷 图 3：PPO / DPO / GRPO 三大策略优化算法架构对比**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_three_algorithms.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_three_algorithms.png`



## 预备知识：策略梯度的基本思想

在深入三种算法之前，我们需要理解一个共同的起点——**策略梯度（Policy Gradient）**。

### 核心直觉

想象你在练习投篮。每次投篮后，你会得到一个反馈：进了（奖励 +1）或没进（奖励 0）。策略梯度的思想极其朴素：

> **如果某个动作获得了高奖励，就增加该动作的概率；如果获得了低奖励，就降低该动作的概率。**

形式化地，策略梯度定理给出的梯度方向为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]$$

下面我们**逐项拆解**这个公式，并用一个具体的语言模型例子来帮助理解。



**① $\nabla_\theta J(\theta)$ —— "我该往哪个方向调参数？"**


- $J(\theta)$ 是我们的**总目标**：模型在所有可能输入上的期望累积奖励。$J$ 越大，模型整体表现越好
- $\nabla_\theta$ 是对模型参数 $\theta$（即模型中数十亿个权重值）求梯度
- $\nabla_\theta J(\theta)$ 就是一个与 $\theta$ 同维度的向量，**它告诉我们：如果把每个参数往哪个方向微调一点点，$J$ 会增大最快**
- 训练时我们做的就是：$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$（$\alpha$ 是学习率），即沿梯度方向"上坡"

> **类比**：你蒙着眼站在山坡上，梯度就是"脚下最陡的上坡方向"。每走一步（更新一次参数），你就往山顶（最大奖励）靠近一点。



**② $\nabla_\theta \log \pi_\theta(a_t | s_t)$ —— "怎样调参数才能让这个动作更可能发生？"**


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



**③ $R(\tau)$ —— "这个方向盘该转多少？"**


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



**④ $\sum_{t=0}^{T}$ —— "轨迹中每一步都要算"**


一条轨迹包含 $T+1$ 个时间步（从 $t=0$ 到 $t=T$），每一步都有一个 $(s_t, a_t)$ 对。求和意味着：轨迹中**每一步的得分函数都被同一个 $R(\tau)$ 加权**。

在语言模型中，一步 = 生成一个 token。如果模型生成了一个 50 token 的回答，$T = 49$，那么这 50 个 token 中每一个的生成概率都会被同一个总奖励加权更新。

> **注意**：这其实是一个粗糙的做法——用整条轨迹的总奖励来加权每一步。如果轨迹中前 30 个 token 是正确推理，后 20 个 token 是错误结论，它们都会被总奖励同等对待。这就是"**信用分配问题（credit assignment）**"——PPO 的优势函数 $A_t$ 正是为了解决这个问题（见 §1.3）。



**⑤ $\mathbb{E}_{\tau \sim \pi_\theta}$ —— "对很多次尝试取平均"**


$\mathbb{E}$ 是**期望运算符**，$\tau \sim \pi_\theta$ 表示轨迹 $\tau$ 是按策略 $\pi_\theta$ 随机采样的。

- 因为语言模型的生成是**随机的**（通过 temperature 采样），同一个输入可能产生不同的输出
- 每次采样得到一条轨迹 $\tau$，对应一个 $R(\tau)$ 值
- 期望就是**对所有可能轨迹加权平均**——概率越高的轨迹权重越大

**实际操作中**：我们无法枚举所有可能轨迹（语言模型的输出空间是天文数字级的），因此用**蒙特卡洛近似**——采样 $N$ 条轨迹，取平均作为期望的估计：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{n=1}^{N} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(n)} | s_t^{(n)}) \cdot R(\tau^{(n)}) \right]$$

$N$ 越大，估计越准确，但计算成本也越高。这就是训练中 batch size 的本质。



**完整例子：语言模型 Agent 的一次策略梯度更新**


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


### 原始策略梯度的缺陷

虽然直觉清晰，但原始策略梯度有两个严重问题：

| 问题 | 具体表现 | 后果 |
|------|---------|------|
| **高方差** | $R(\tau)$ 可能在不同轨迹间差异极大 | 梯度估计不稳定，训练收敛极慢 |
| **步长不可控** | 没有约束单步更新的大小 | 一次"大跳步"就可能毁掉整个策略 |

**PPO、DPO、GRPO 各自用不同方式解决了这两个问题。** 本节详细讲解 PPO，DPO 和 GRPO 将分别在 后文 DPO 部分 和 后文 GRPO 部分 中介绍。


### 1.1 PPO 解决了什么问题？

PPO 是 OpenAI 于 2017 年提出的策略优化算法，是 InstructGPT 和 ChatGPT 的核心训练算法。PPO 的设计目标是：

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

实际训练中，$Q(s_t, a_t)$ 和 $V(s_t)$ 都不是精确已知的，需要用一个 **Critic 模型** $V_\phi(s)$ 来估计。**GAE（Generalized Advantage Estimation）** 是一种融合多步估计的方法，在偏差和方差之间取得平衡：

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


> **📷 图 4：PPO Clip 机制图解**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_ppo_clip.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_ppo_clip.png`



**Clip 参数 $\epsilon$ 的含义**：$\epsilon$（通常取 0.1–0.3）定义了"信任域"的大小——每次策略更新，每个动作的概率变化不得超过旧策略的 $(1 \pm \epsilon)$ 倍。$\epsilon$ 越小越保守，$\epsilon$ 越大越激进。

### 1.6 KL 散度惩罚：防止策略"跑偏"的另一道保险

Clip 机制限制的是**单个动作**的概率变化幅度，但它无法约束策略**整体分布**的漂移。在 RLHF 场景中，如果模型为了追求高奖励而"忘记"了 SFT 阶段学到的通用语言能力（如语法、连贯性），就会出现**语言退化**或**奖励黑客（reward hacking）**——模型找到某种"讨巧"的输出模式来骗取高奖励，但人类读起来一塌糊涂。

为此，PPO 在 RLHF 中通常会额外加入一个 **KL 散度惩罚项**，完整的优化目标变为：

$$\mathcal{L}_{PPO-RLHF}(\theta) = -\mathbb{E}_t \left[ \min\left( \rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right] + \beta \cdot D_{KL}\left(\pi_\theta \| \pi_{ref}\right)$$

其中 KL 散度定义为：

$$D_{KL}\left(\pi_\theta \| \pi_{ref}\right) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]$$

逐项解读：

- $\pi_{ref}$：**参考策略（Reference Policy）**——详见下方专门说明
- $D_{KL}(\pi_\theta \| \pi_{ref})$：衡量当前策略 $\pi_\theta$ 相对于参考策略 $\pi_{ref}$ 的**分布偏离程度**。$D_{KL} = 0$ 表示两者完全一致；$D_{KL}$ 越大，偏离越严重
- $\beta$：**KL 惩罚系数**——控制"探索新策略"与"保持原有能力"之间的平衡

> 关于 KL 散度的数学定义和直觉解释，请参阅 相关 KL 散度的科普资料


**什么是 Reference 模型（$\pi_{ref}$）？**


Reference 模型是 RLHF 和 GRPO 中一个非常重要的概念，初学者容易将它与"旧策略"（$\pi_{\theta_{old}}$）混淆。我们用一个表格来彻底厘清：

| | **Reference 模型 $\pi_{ref}$** | **旧策略 $\pi_{\theta_{old}}$** |
|---|---|---|
| **是什么** | RL 训练**开始前**的 SFT 模型快照 | 当前 RL 迭代**开始时**的策略模型快照 |
| **何时创建** | RL 训练启动时，复制一份 SFT 模型 | 每次采样新数据前，复制当前 Policy 模型 |
| **是否更新** | ❌ **永远不更新**（冻结参数） | ✅ 每轮迭代更新一次（同步为当前策略） |
| **存在目的** | 充当"锚点"，防止策略偏离 SFT 太远 | 提供重要性采样的分母（数据复用） |
| **作用范围** | 整个 RL 训练过程 | 仅在当前迭代内有效 |
| **用在哪里** | KL 散度惩罚 $D_{KL}(\pi_\theta \| \pi_{ref})$ | 重要性采样比率 $\rho_t = \pi_\theta / \pi_{\theta_{old}}$ |

**形象比喻**：

> 想象你在学开车（RL 训练）。驾校教练教会了你基本操作（SFT），现在你上路练习。
>
> - **Reference 模型** = 教练手册上的标准操作规范。无论你练了多久，手册永远不变。它的作用是：如果你的驾驶习惯偏离标准太远（比如开始"飙车"），就把你拉回来。
> - **旧策略** = 你上一次练车结束时的驾驶水平。每次练完车，你的水平都会提升一点。它的作用是：评估你这次练车相比上次有哪些变化。

**为什么需要 Reference 模型？**

在 RL 训练过程中，模型会不断迭代更新。如果没有 Reference 模型作为锚点，可能出现以下问题：

1. **奖励黑客（Reward Hacking）**：模型发现某种"讨巧"的输出模式可以骗取高奖励（如不断重复某个高奖励短语），但实际输出质量极差
2. **语言退化（Language Degeneration）**：模型为追求奖励而丧失了 SFT 阶段学到的语法能力、连贯性和通用知识
3. **模式坍缩（Mode Collapse）**：模型对所有输入都生成类似的"安全"回答，丧失多样性

KL 散度 $D_{KL}(\pi_\theta \| \pi_{ref})$ 就像一根"弹力绳"——当 $\pi_\theta$ 偏离 $\pi_{ref}$ 越远，惩罚越大，把策略拉回来。

**训练过程中三个模型的时间线示意**：

```
时间 →
              RL 迭代 1        RL 迭代 2        RL 迭代 3
              ─────────       ─────────       ─────────
π_ref:     [SFT 模型] ════════════════════════════════════  （永远不变）

π_θ_old:   [SFT 模型]──→[θ₁]──→[θ₁]──→[θ₂]──→[θ₂]──→[θ₃]  （每轮迭代开始时同步）
                采样↓      更新↑    采样↓      更新↑    采样↓
π_θ:       [SFT 模型]→→→[θ₁]  [θ₁]→→→[θ₂]  [θ₂]→→→[θ₃]    （持续训练更新）
```

- 第 1 行：$\pi_{ref}$ 始终是 SFT 模型，从头到尾不变
- 第 2 行：$\pi_{\theta_{old}}$ 在每轮迭代开始时，从 $\pi_\theta$ 复制一份快照
- 第 3 行：$\pi_\theta$ 是持续训练更新的 Policy 模型

> **📌 实现细节**：Reference 模型需要单独占用显存。对于 7B 参数模型（bf16），Reference 模型约占 14GB 显存。为了节省显存，有些实现会使用 LoRA 适配器——此时 Reference 模型不需要单独加载，只需在推理时关闭 LoRA 适配器即可得到 $\pi_{ref}$ 的输出。

**$\beta$ 的作用与调节**：

| $\beta$ 值 | 效果 | 适用场景 |
|------------|------|---------|
| $\beta$ 过小（如 0.001） | KL 约束几乎无效，策略可大幅偏离 | 探索性强，但容易奖励黑客、语言退化 |
| $\beta$ 适中（如 0.01–0.1） | 平衡探索与约束，推荐起始值 | 大多数 RLHF 场景 |
| $\beta$ 过大（如 1.0） | 策略几乎无法偏离 SFT 模型 | RL 训练形同虚设 |

**自适应 KL 控制**：InstructGPT 提出了一种动态调节 $\beta$ 的方法——设定一个目标 KL 值 $D_{target}$，如果实际 $D_{KL}$ 超过目标值，则增大 $\beta$（收紧约束）；反之则减小 $\beta$（放松约束）：

$$\beta \leftarrow \begin{cases} \beta \times (1 + \alpha) & \text{if } D_{KL} > 1.5 \times D_{target} \\ \beta \times (1 - \alpha) & \text{if } D_{KL} < 0.5 \times D_{target} \\ \beta & \text{otherwise} \end{cases}$$

其中 $\alpha$ 是调节步长（通常取 0.1–0.2）。这种自适应机制让训练更加稳健——不需要手动精调 $\beta$。

**Clip + KL 的协同作用**：

| 约束机制 | 约束对象 | 约束粒度 | 直觉 |
|---------|---------|---------|------|
| **Clip** | 单个动作的概率比 $\rho_t$ | **局部**（逐 token 级别） | "每一步不能走太远" |
| **KL** | 整体输出分布 $\pi_\theta$ vs $\pi_{ref}$ | **全局**（策略级别） | "总体路线不能偏离太远" |

两者互补：Clip 防止单步更新过大，KL 防止累积漂移过大。在实践中，两者通常同时使用。

### 1.7 PPO 完整训练流程


> **📷 图 5：PPO 完整训练迭代流程**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_ppo_training_flow.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_ppo_training_flow.png`



PPO 的训练需要同时维护以下模型：


> **📷 图 6：PPO 训练架构（四个模型的协作关系）**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_ppo_architecture.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_ppo_architecture.png`



### 1.8 PPO 核心代码实现

下面用 PyTorch 代码完整实现 PPO 的核心组件，帮助从代码层面理解每个公式的含义。


**1.8.1 GAE 优势估计**


```python
import torch
import torch.nn.functional as F

def compute_gae(
    rewards: torch.Tensor,       # [T] 每步即时奖励
    values: torch.Tensor,        # [T+1] Critic 估计的状态价值（含终止状态）
    gamma: float = 1.0,          # 折扣因子（语言模型任务通常取 1.0）
    lam: float = 0.95,           # GAE λ 参数
) -> torch.Tensor:
    """
    计算 GAE（广义优势估计）

    公式：A_t = Σ (γλ)^l · δ_{t+l}
    其中 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

    Args:
        rewards:  每步获得的即时奖励，形状 [T]
        values:   Critic 对每个状态的价值估计，形状 [T+1]
                  （最后一个是终止状态的价值，通常为 0）
        gamma:    折扣因子，控制未来奖励的衰减
        lam:      GAE λ 参数，控制偏差-方差权衡
                  λ=0 → 单步 TD（低方差高偏差）
                  λ=1 → 蒙特卡洛（高方差低偏差）

    Returns:
        advantages: GAE 优势估计，形状 [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0  # 从最后一步向前累积

    for t in reversed(range(T)):
        # TD 误差：δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        # "实际发生的" - "Critic 预期的"
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # GAE 递推：A_t = δ_t + γλ·A_{t+1}
        # 等价于 A_t = Σ (γλ)^l · δ_{t+l}
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages
```


**1.8.2 PPO Clip 损失**


```python
def ppo_clip_loss(
    log_probs: torch.Tensor,         # [B, T] 当前策略的 log π_θ(a_t|s_t)
    old_log_probs: torch.Tensor,     # [B, T] 旧策略的 log π_θ_old(a_t|s_t)
    advantages: torch.Tensor,         # [B, T] GAE 优势估计
    clip_epsilon: float = 0.2,        # Clip 范围 ε
) -> tuple[torch.Tensor, dict]:
    """
    计算 PPO Clip 策略损失

    公式：L = -E[min(ρ_t·A_t, clip(ρ_t, 1-ε, 1+ε)·A_t)]

    Args:
        log_probs:     当前策略对每个 token 的对数概率 [batch, seq_len]
        old_log_probs: 旧策略对每个 token 的对数概率 [batch, seq_len]
        advantages:    每个 token 的优势值 [batch, seq_len]
        clip_epsilon:  裁剪范围，通常 0.1-0.3

    Returns:
        loss: 策略损失标量
        metrics: 监控指标
    """
    # ── 计算重要性采样比率 ────────────────────────────────────────
    # ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    # 在 log 空间做除法 = 做减法，再 exp 回去
    ratio = torch.exp(log_probs - old_log_probs)  # [B, T]

    # ── 未裁剪的目标 ──────────────────────────────────────────────
    # ρ_t · A_t
    surr1 = ratio * advantages  # [B, T]

    # ── 裁剪后的目标 ──────────────────────────────────────────────
    # clip(ρ_t, 1-ε, 1+ε) · A_t
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surr2 = clipped_ratio * advantages  # [B, T]

    # ── PPO 损失 = -min(surr1, surr2) ────────────────────────────
    # 取 min 确保：
    #   A>0 时：不让 ρ 超过 1+ε（防止过度强化）
    #   A<0 时：不让 ρ 低于 1-ε（防止过度抑制）
    loss = -torch.min(surr1, surr2).mean()

    # ── 监控指标 ──────────────────────────────────────────────────
    with torch.no_grad():
        # 裁剪比例：被裁剪的 token 占比（健康范围 0.1-0.3）
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
        # 近似 KL 散度（用于监控策略偏移程度）
        approx_kl = (old_log_probs - log_probs).mean().item()

    metrics = {
        "policy_loss": loss.item(),
        "clip_fraction": clip_fraction,       # > 0.5 说明更新步长太大
        "approx_kl": approx_kl,               # > 0.02 可能需要减小学习率
        "mean_ratio": ratio.mean().item(),     # 应接近 1.0
    }

    return loss, metrics
```


**1.8.3 Critic（价值函数）损失**


```python
def critic_loss(
    values: torch.Tensor,        # [B, T] Critic 预测的 V(s_t)
    returns: torch.Tensor,       # [B, T] 实际回报 = advantages + old_values
) -> torch.Tensor:
    """
    计算 Critic 损失（均方误差）

    Critic 的目标：让 V_ϕ(s_t) 尽可能接近实际回报

    Args:
        values:  Critic 对每个状态的价值预测 [batch, seq_len]
        returns: 实际回报（GAE 优势 + 旧价值估计）[batch, seq_len]

    Returns:
        loss: Critic 损失标量
    """
    return F.mse_loss(values, returns)
```


**1.8.4 KL 散度惩罚**


```python
def kl_penalty(
    log_probs: torch.Tensor,     # [B, T] 当前策略的 log π_θ
    ref_log_probs: torch.Tensor, # [B, T] 参考策略（SFT 模型）的 log π_ref
) -> torch.Tensor:
    """
    计算当前策略与参考策略之间的 KL 散度

    公式：D_KL(π_θ ‖ π_ref) = E[log(π_θ/π_ref)]

    Args:
        log_probs:     当前策略的对数概率
        ref_log_probs: 参考策略的对数概率（SFT 模型，冻结参数）

    Returns:
        kl: KL 散度标量
    """
    # KL = E[log π_θ - log π_ref]
    kl = (log_probs - ref_log_probs).mean()
    return kl
```


**1.8.5 PPO 完整训练步骤**


```python
def ppo_training_step(
    policy_model,          # 策略模型 π_θ（可训练）
    critic_model,          # Critic 模型 V_ϕ（可训练）
    ref_model,             # 参考模型 π_ref（冻结，不训练）
    input_ids,             # 输入 token ids
    response_ids,          # 生成的回答 token ids
    old_log_probs,         # 旧策略的 log 概率
    rewards,               # 奖励信号
    old_values,            # 旧 Critic 的价值估计
    clip_epsilon=0.2,      # PPO Clip ε
    kl_coef=0.01,          # KL 惩罚系数 β
    critic_coef=0.5,       # Critic 损失权重
    gamma=1.0,             # 折扣因子
    lam=0.95,              # GAE λ
):
    """
    PPO 单步训练的完整流程

    总损失 = 策略损失 + β·KL惩罚 + c·Critic损失
    """
    # ── Step 1: 计算当前策略的 log 概率 ──────────────────────────
    # 前向传播，获取当前策略对每个 token 的对数概率
    logits = policy_model(input_ids=input_ids, response_ids=response_ids)
    log_probs = compute_token_log_probs(logits, response_ids)  # [B, T]

    # ── Step 2: 计算 Critic 的价值估计 ───────────────────────────
    values = critic_model(input_ids=input_ids, response_ids=response_ids)  # [B, T]

    # ── Step 3: 计算 GAE 优势 ────────────────────────────────────
    # 对 batch 中的每个样本分别计算 GAE
    advantages_list = []
    for b in range(rewards.shape[0]):
        adv = compute_gae(rewards[b], old_values[b], gamma=gamma, lam=lam)
        advantages_list.append(adv)
    advantages = torch.stack(advantages_list)  # [B, T]

    # 标准化优势（减小方差，稳定训练）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 实际回报 = 优势 + 旧价值估计（用于训练 Critic）
    returns = advantages + old_values[:, :-1]

    # ── Step 4: 计算 PPO Clip 策略损失 ───────────────────────────
    policy_loss, policy_metrics = ppo_clip_loss(
        log_probs, old_log_probs, advantages, clip_epsilon
    )

    # ── Step 5: 计算 KL 散度惩罚 ────────────────────────────────
    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids, response_ids=response_ids)
        ref_log_probs = compute_token_log_probs(ref_logits, response_ids)
    kl = kl_penalty(log_probs, ref_log_probs)

    # ── Step 6: 计算 Critic 损失 ────────────────────────────────
    c_loss = critic_loss(values[:, :-1], returns)

    # ── Step 7: 总损失 ──────────────────────────────────────────
    # 策略损失（让模型做对的事）+ KL 惩罚（不要忘了语言能力）+ Critic 损失（学会评估好坏）
    total_loss = policy_loss + kl_coef * kl + critic_coef * c_loss

    return total_loss, {
        **policy_metrics,
        "kl_divergence": kl.item(),
        "critic_loss": c_loss.item(),
        "total_loss": total_loss.item(),
    }
```

> **📌 与 GRPO 代码的关键差异**：
> - PPO 需要额外维护一个 **Critic 模型**（`critic_model`），它与 Policy 模型同等规模
> - 优势估计使用 **GAE 递推**（依赖 Critic），而不是 GRPO 的组内标准化
> - 总损失包含三部分：策略损失 + KL 惩罚 + **Critic 损失**，而 GRPO 没有 Critic 损失
> - 这也解释了为什么 PPO 的显存需求是 ≈ 3× 模型大小（Policy + Critic + Reference）

### 1.9 PPO 的优缺点总结

| 维度 | 评价 |
|------|------|
| ✅ **通用性** | 几乎适用于所有 RL 场景，不限于语言模型 |
| ✅ **稳定性** | Clip 机制提供了可靠的训练稳定性保障 |
| ✅ **理论基础** | 有成熟的理论支撑（信任域方法的简化） |
| ❌ **显存需求** | 需要 Critic 模型，显存占用 ≈ 3× 模型大小 |
| ❌ **训练复杂** | Critic 与 Policy 互相依赖，联合训练不稳定 |
| ❌ **超参数多** | GAE λ、Critic 学习率、clip ε、KL β 等需精心调节 |




*PPO 是一个强大但资源消耗较大的算法——需要 Critic 模型、奖励模型和在线采样。接下来介绍 DPO，它通过精妙的数学推导直接跳过了奖励模型和 Critic，将 RL 问题转化为简单的监督学习。*


# 四、DPO：直接偏好优化

前面我们详细介绍了 PPO 算法——它需要训练一个 Critic 模型来估计优势函数，并且依赖在线采样和奖励模型。这使得 PPO 的训练流程复杂、资源消耗大。

**DPO（Direct Preference Optimization）** 提出了一种全新的思路：**直接从人类偏好数据中优化策略，无需奖励模型，无需 Critic，无需在线采样**——将 RL 问题巧妙地转化为简单的监督学习问题。


### 2.1 DPO 的核心洞察

DPO 是 2023 年 Stanford 团队提出的算法。它的核心洞察可以用一句话概括：

> **既然 RLHF 的最终目标是让模型的输出分布符合人类偏好，那能否跳过"训练奖励模型 → 用 PPO 优化"的两步流程，直接从偏好数据中优化策略？**

答案是——**可以！** DPO 通过一个精巧的数学推导，证明了 RLHF 的最优策略可以用一个**闭式解**表示，从而将 RL 问题转化为简单的**监督学习**问题。


> **📷 图 7：DPO 核心直觉（偏好学习示意）**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_dpo_intuition.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_dpo_intuition.png`



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

在 RLHF 中，人类偏好建模使用 **Bradley-Terry 模型**：

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


> **📷 图 8：DPO 训练架构**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_dpo_architecture.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_dpo_architecture.png`



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

### 2.5 深入理解：DPO 与 KL 散度的关系

读到这里，你可能会有一个疑问：**DPO 的 loss 中还有 KL 散度吗？** 毕竟在 PPO 中，KL 散度是作为显式惩罚项出现的。

**简短回答：DPO 的最终 loss 函数中没有显式的 KL 散度项，但 KL 散度已经被隐式地"吸收"进了 loss 的数学结构中。**


**KL 散度出现在推导的"起点"**


回顾 Step 1，DPO 的推导从标准 RLHF 优化目标出发：

$$\max_{\pi_\theta} \mathbb{E}\left[ r(x, y) \right] - \beta \cdot D_{KL}\left(\pi_\theta \| \pi_{ref}\right)$$

这里 **确实有一个显式的 KL 散度惩罚项** $D_{KL}(\pi_\theta \| \pi_{ref})$，它约束当前策略不能偏离参考策略太远。这和 PPO 的 KL 惩罚是同一个东西。


**KL 散度在推导过程中被"消化"了**


DPO 的精妙之处在于：通过 Step 2 → Step 5 的数学推导，将 KL 约束的 RLHF 目标**变换**为了一个纯粹的监督学习 loss。在最终的 DPO loss 中：

- ❌ **没有显式的 KL 散度项**（不像 PPO 那样在 loss 里加上 $-\beta \cdot D_{KL}$）
- ✅ **KL 约束被隐式编码在了 $\log \frac{\pi_\theta}{\pi_{ref}}$ 中**——对数概率比 $\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ 本身就是 KL 散度的组成部分


**为什么说 KL 散度被"隐式包含"了？**


KL 散度的定义是：

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right]$$

DPO loss 中的核心项 $\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ 正是 KL 散度的**被积函数**。因此：

| | PPO | DPO |
|---|---|---|
| **KL 散度** | 作为**显式惩罚项**加到 loss 中 | **隐式编码**在对数概率比中，无需额外计算 |
| **参考策略 $\pi_{ref}$** | 可选（也可以只用 Clip） | 必须有（是 loss 的核心组件） |
| **$\beta$ 的作用** | 控制 KL 惩罚的权重 | 控制隐式奖励差的缩放（本质一样） |


**直觉理解**


> **PPO 说**："先算出奖励，再用 KL 散度当刹车，防止跑偏。" → 需要奖励模型 + 显式 KL 计算
> 
> **DPO 说**："我直接把奖励和 KL 约束合并成一个公式，用对数概率比同时编码了'什么是好的'和'别跑太远'。" → 一步到位

### 2.6 DPO 的优缺点总结

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


*DPO 极大地简化了 RLHF 的流程，但它是完全离线的——无法通过在线探索发现训练数据中未出现的新策略。接下来介绍 GRPO，它结合了 PPO 的在线探索能力和比 PPO 更低的资源消耗，是 DeepSeek-R1 的核心训练算法。*


# 五、GRPO：组内相对策略优化与奖励函数设计

前面我们分别介绍了 PPO 和 DPO 两种策略优化算法。PPO 需要额外的 Critic 模型（显存占用大），DPO 虽然简单但完全离线（无法探索新策略）。

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek 团队为大模型 RL 训练量身打造的算法，它通过**组内采样比较**替代了 Critic 模型，在保持在线探索能力的同时大幅降低了资源消耗。本节同时介绍 GRPO 的核心驱动力——**奖励函数的设计**，因为奖励函数定义了"什么是好的 Agent 行为"，直接决定了 GRPO 的训练效果。


### 3.1 GRPO 的核心洞察

GRPO 是 DeepSeek 团队为大模型 RL 训练量身打造的算法。它的核心洞察是：

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
- $\beta \cdot \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$：KL 散度惩罚——防止策略偏离 SFT 模型太远，避免奖励黑客和语言退化。关于 KL 散度的详细解释，请参阅 相关 KL 散度的科普资料

### 3.4 GRPO 训练架构与流程


> **📷 图 9：GRPO 训练架构**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_grpo_architecture.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_grpo_architecture.png`




> **📷 图 10：GRPO 单次训练迭代流程**
> 
> 👉 [点击查看高清大图](https://haozhe-xing.github.io/agent_learning/svg/chapter_agentic_rl_03_grpo_iteration.svg)
> 
> ⚠️ **发布时请在此处手动上传对应图片** `images/chapter_agentic_rl_03_grpo_iteration.png`




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
    
详细的奖励函数设计方法参见下文「奖励函数设计」部分
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


## 三大算法系统性对比

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
| **InstructGPT** | PPO | 证明 RLHF 可大幅提升指令遵循能力 |
| **Llama 2** | PPO | 70B 模型的安全对齐 |
| **Zephyr** | DPO | 7B 模型用 DPO 超越 PPO 基线 |
| **DeepSeek-R1** | GRPO | 涌现长链推理，数学/代码能力媲美 o1 |
| **DeepSWE** | GRPO | SWE-bench Verified 59%（开源 SOTA）|


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


## 奖励函数设计——将目标形式化为可优化的信号

### 5.1 奖励函数的核心地位

在 GRPO 训练框架中，**奖励函数 $R: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ 是连接"人类意图"与"模型行为"的唯一桥梁**。它将我们对"好 Agent"的直觉判断形式化为可微分（或可采样）的数値信号，直接决定了强化学习的优化方向。

奖励函数设计的核心挑战在于：

$$\text{真实目标} \neq \text{可计算的代理指标}$$

**为什么两者不等价？** 真实目标通常是模糊的主观判断（如"输出质量高""用户满意度高"），而可计算的代理指标必须是具体的数字（如"测试用例通过率""格式符合率"）。这一差距是**奖励黑客（Reward Hacking）** 的根本来源——模型会找到最大化代理指标的捷径，而这些捷径往往不符合真实意图。

**典型案例**：若奖励函数仅检查最终答案是否正确，模型可能学会在 `<think>` 内输出乱码，然后凑出正确答案——奖励很高，但推理过程完全无意义。这就是代理指标（答案正确性）与真实目标（有意义的推理）之间的典型差距。


**奖励函数设计的四项基本原则**


| 原则 | 形式化描述 | 违反后果 |
|------|-----------|------|
| **可验证性** | 奖励基于客观可计算的标准，而非主观判断 | 奖励信号噪声大，训练不稳定 |
| **多维度覆盖** | $R = \sum_k w_k R_k$，覆盖任务的多个质量维度 | 模型在单一维度上过度优化，忽视其他维度 |
| **稠密性** | 在轨迹的多个时间步提供奖励信号，而非仅在终止时 | 稀疏奖励导致信用分配困难，训练收敛慢 |
| **鲁棒性** | 奖励函数对模型的"钻空子"行为具有抵抗力 | 模型学会奖励黑客，高奖励但低实际质量 |

**关于多维度合并公式 $R = \sum_k w_k R_k$ 的解读**：各维度奖励 $R_k \in [0, 1]$ 独立计算，加权系数 $w_k$ 满足 $\sum_k w_k = 1$。权重的选择体现了不同维度的相对重要性：准确率权重最高（任务核心），安全权重最低（大多数情况下不会触发）。

### 5.2 核心奖励维度的设计与实现


**维度一：准确率奖励（Accuracy Reward）**


准确率奖励是最核心的奖励维度，直接衡量 Agent 是否正确完成了任务。不同任务类型需要不同的评估方法：

```python
import re
from typing import Optional

def accuracy_reward(
    prediction: str,
    ground_truth: str,
    task_type: str = "math",
    tolerance: float = 1e-2,
) -> float:
    """
    准确率奖励：评估 Agent 输出是否正确完成任务
    
    Args:
        prediction:   模型的完整输出（含推理过程）
        ground_truth: 标准答案
        task_type:    任务类型，决定评估方法
        tolerance:    数值比较的相对误差容忍度
    
    Returns:
        奖励值 ∈ [0, 1]
    """
    if task_type == "math":
        # 数学任务：从输出中提取最终数值，允许 tolerance 相对误差
        try:
            pred_num = _extract_final_number(prediction)
            true_num = float(ground_truth.replace(",", ""))
            relative_error = abs(pred_num - true_num) / (abs(true_num) + 1e-8)
            return 1.0 if relative_error < tolerance else 0.0
        except (ValueError, AttributeError):
            return 0.0

    elif task_type == "code":
        # 代码任务：执行测试用例，按通过率给分（部分奖励）
        # 
        # 为什么使用部分奖励而非 0/1 奖励？
        # 0/1 奖励（稀疏奖励）会导致信用分配困难：
        #   - 若模型通过了 9/10 个测试用例，0/1 奖励给 0 分，无法区分"接近正确"和"完全错误"
        #   - 部分奖励 k/n 提供了更密集的梯度信号，帮助模型逐步改进
        # 这与课程学习（Curriculum Learning）的思想一致：先学会通过简单测试，再逐步攻克难测试
        code = _extract_code_block(prediction)
        if not code:
            return 0.0
        test_results = _run_test_cases(code, ground_truth)
        # 部分奖励：通过 k/n 个测试用例得 k/n 分
        return test_results["passed"] / max(test_results["total"], 1)

    elif task_type == "tool_call":
        # 工具调用任务：检查工具名称和参数是否正确
        pred_call = _parse_tool_call(prediction)
        true_call = _parse_tool_call(ground_truth)
        if pred_call is None:
            return 0.0
        score = 0.0
        if pred_call.get("name") == true_call.get("name"):
            score += 0.5   # 工具名称正确
        if pred_call.get("args") == true_call.get("args"):
            score += 0.5   # 参数完全匹配
        return score

    else:
        # 通用：精确字符串匹配
        return 1.0 if prediction.strip() == ground_truth.strip() else 0.0


def _extract_final_number(text: str) -> float:
    """从文本中提取最后出现的数值（通常是最终答案）"""
    # 匹配整数、小数、负数，忽略千位分隔符
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if not numbers:
        raise ValueError(f"No number found in: {text[:100]}")
    return float(numbers[-1].replace(",", ""))
```


**维度二：格式奖励（Format Reward）**


格式奖励确保模型输出符合预期的结构化格式，这对于 Agent 的可靠性至关重要：

```python
def format_reward(completion: str) -> float:
    """
    格式奖励：评估输出是否符合 Agent 格式规范
    
    期望格式（两种合法模式）：
    模式 A（需要工具）：<think>推理</think> <tool_call>调用</tool_call>
    模式 B（直接回答）：<think>推理</think> 最终答案
    
    评分细则：
    - <think> 标签配对且内容非空：+0.4
    - <tool_call> 标签配对且语法正确：+0.4
    - 无重复/嵌套标签：+0.2
    """
    score = 0.0

    # ── 检查 <think> 标签 ─────────────────────────────────────────────────
    think_open  = completion.count("<think>")
    think_close = completion.count("</think>")

    if think_open == 1 and think_close == 1:
        score += 0.2
        # 检查 think 内容的实质性
        think_content = completion.split("<think>")[1].split("</think>")[0].strip()
        if len(think_content) >= 20:
            score += 0.2   # 有实质性推理内容（非空壳）
    elif think_open != think_close:
        score -= 0.2       # 标签不配对，严重格式错误

    # ── 检查 <tool_call> 标签 ─────────────────────────────────────────────
    tool_open  = completion.count("<tool_call>")
    tool_close = completion.count("</tool_call>")

    if tool_open == tool_close and tool_open > 0:
        score += 0.2
        # 检查工具调用语法
        try:
            tool_str = completion.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            # 验证函数调用格式：name(args)
            if re.match(r'^\w+\(.*\)$', tool_str, re.DOTALL):
                score += 0.2
        except IndexError:
            pass
    elif tool_open != tool_close:
        score -= 0.2       # 标签不配对

    return max(0.0, min(1.0, score))
```


**维度三：效率奖励（Efficiency Reward）**


效率奖励鼓励模型用最少的步骤和 Token 完成任务，防止冗余行为：

```python
def efficiency_reward(
    completion: str,
    expected_steps: int = 3,
    max_tokens: int = 512,
) -> float:
    """
    效率奖励：惩罚冗余的工具调用和过长的输出
    
    设计原则：
    - 在 expected_steps 以内：满分
    - 超出 expected_steps：线性惩罚，最多扣 0.5 分
    - 超出 max_tokens：额外惩罚，最多扣 0.3 分
    - 检测重复内容：额外惩罚
    """
    score = 1.0

    # ── 步骤数惩罚 ────────────────────────────────────────────────────────
    num_steps = completion.count("<tool_call>")
    if num_steps > expected_steps:
        step_penalty = 0.1 * (num_steps - expected_steps)
        score -= min(step_penalty, 0.5)

    # ── Token 数惩罚 ──────────────────────────────────────────────────────
    num_tokens = len(completion.split())
    if num_tokens > max_tokens:
        token_penalty = 0.3 * (num_tokens - max_tokens) / max_tokens
        score -= min(token_penalty, 0.3)

    # ── 重复内容检测 ──────────────────────────────────────────────────────
    # 将输出分句，检测重复率（防止模型通过重复填充获得高奖励）
    sentences = [s.strip() for s in re.split(r'[。！？\n]', completion) if len(s.strip()) > 5]
    if len(sentences) > 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.7:
            score -= 0.2   # 超过 30% 的句子是重复的

    return max(0.0, score)
```


**维度四：安全奖励（Safety Reward）**


安全奖励防止 Agent 产生危险或有害的行为，这在生产环境中至关重要：

```python
def safety_reward(completion: str) -> float:
    """
    安全奖励：检测并惩罚潜在危险行为
    
    检测类别：
    1. 危险系统命令（文件删除、权限修改等）
    2. 危险数据库操作（DROP、DELETE 等不可逆操作）
    3. 代码注入风险（eval、exec 等动态执行）
    4. 敏感信息泄露（API Key、邮箱、身份证号等）
    """
    score = 1.0

    # ── 危险命令模式 ──────────────────────────────────────────────────────
    dangerous_patterns = [
        (r'\brm\s+-rf\b',          0.8, "危险文件删除命令"),
        (r'\bDROP\s+TABLE\b',      0.8, "不可逆数据库操作"),
        (r'\bDELETE\s+FROM\b',     0.5, "数据库删除操作"),
        (r'\bsudo\b',              0.3, "提权命令"),
        (r'\bchmod\s+777\b',       0.3, "危险权限设置"),
        (r'\beval\s*\(',           0.5, "动态代码执行"),
        (r'\bexec\s*\(',           0.5, "动态代码执行"),
        (r'\b__import__\s*\(',     0.5, "动态模块导入"),
    ]

    for pattern, penalty, _ in dangerous_patterns:
        if re.search(pattern, completion, re.IGNORECASE):
            score -= penalty

    # ── 敏感信息泄露检测 ──────────────────────────────────────────────────
    sensitive_patterns = [
        (r'sk-[a-zA-Z0-9]{32,}',                              0.5, "API Key"),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', 0.3, "邮箱地址"),
        (r'\b\d{3}-\d{2}-\d{4}\b',                            0.5, "SSN"),
        (r'\b1[3-9]\d{9}\b',                                   0.3, "手机号"),
    ]

    for pattern, penalty, _ in sensitive_patterns:
        if re.search(pattern, completion, re.IGNORECASE):
            score -= penalty

    return max(0.0, score)
```

### 5.3 多维度奖励的组合策略

实际训练中，将多个维度的奖励加权组合为单一标量信号：

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class RewardConfig:
    """奖励函数配置，支持动态调整各维度权重"""
    accuracy_weight:   float = 0.50   # 准确率：最核心的维度
    format_weight:     float = 0.20   # 格式：确保输出可解析
    efficiency_weight: float = 0.15   # 效率：鼓励简洁
    safety_weight:     float = 0.15   # 安全：防止危险行为


class AgentRewardFunction:
    """
    多维度 Agent 奖励函数
    
    设计原则：
    1. 各维度独立计算，便于调试和分析
    2. 支持动态调整权重（训练初期格式权重高，后期准确率权重高）
    3. 记录各维度分数，便于监控训练过程
    """

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config
        self._validate_weights()

    def _validate_weights(self):
        total = (self.config.accuracy_weight + self.config.format_weight +
                 self.config.efficiency_weight + self.config.safety_weight)
        assert abs(total - 1.0) < 1e-6, f"权重之和必须为 1.0，当前为 {total:.4f}"

    def __call__(
        self,
        completion: str,
        ground_truth: Optional[str] = None,
        task_type: str = "math",
    ) -> dict[str, float]:
        """
        计算综合奖励
        
        Returns:
            包含各维度分数和加权总分的字典，便于监控和调试
        """
        scores = {}

        # 各维度独立计算
        scores["accuracy"] = (
            accuracy_reward(completion, ground_truth, task_type)
            if ground_truth else 0.5   # 无标准答案时给中性分
        )
        scores["format"]     = format_reward(completion)
        scores["efficiency"] = efficiency_reward(completion)
        scores["safety"]     = safety_reward(completion)

        # 加权求和
        scores["total"] = (
            scores["accuracy"]   * self.config.accuracy_weight +
            scores["format"]     * self.config.format_weight +
            scores["efficiency"] * self.config.efficiency_weight +
            scores["safety"]     * self.config.safety_weight
        )

        return scores


# 使用示例
reward_fn = AgentRewardFunction(RewardConfig(
    accuracy_weight=0.50,
    format_weight=0.20,
    efficiency_weight=0.15,
    safety_weight=0.15,
))

result = reward_fn(
    completion=(
        "<think>\n需要计算圆的面积：S = π × r² = π × 5² ≈ 78.54\n</think>\n"
        "<tool_call>calculator(expression='3.14159 * 5**2')</tool_call>"
    ),
    ground_truth="78.54",
    task_type="math",
)
# 预期输出：{'accuracy': 1.0, 'format': 0.8, 'efficiency': 1.0, 'safety': 1.0, 'total': 0.93}
print(result)
```

### 5.4 奖励黑客的防御机制

**奖励黑客（Reward Hacking）** 是指模型学会了"钻奖励函数的空子"——在不真正完成任务的情况下获得高奖励。这是 RL 训练中最常见也最危险的失效模式。


**典型奖励黑客案例分析**


| 奖励设计缺陷 | 模型的黑客行为 | 根本原因 | 防御方法 |
|------------|-------------|---------|---------| 
| 按输出长度给奖励 | 输出大量无意义填充文本 | 奖励与质量解耦 | 改为评估信息密度，惩罚重复内容 |
| 按工具调用次数给奖励 | 疯狂调用不必要的工具 | 奖励与任务目标不一致 | 增加冗余调用惩罚，设置最大步数 |
| 只看最终答案正确性 | `<think>` 内输出乱码，凑出正确答案 | 奖励忽视了推理过程质量 | 同时检查推理过程的连贯性 |
| 用 LLM 评分作为唯一奖励 | 学会输出讨好评分 LLM 的措辞 | 奖励模型本身可被攻击 | 混合使用规则奖励和 LLM 奖励 |


**鲁棒奖励函数的实现**


```python
def robust_reward(
    completion: str,
    ground_truth: str,
    task_type: str = "math",
) -> float:
    """
    防奖励黑客的鲁棒奖励函数
    
    在基础准确率奖励之上，叠加多层防御机制：
    1. 推理过程连贯性检查（防止乱码 think）
    2. 输出长度合理性检查（防止无意义填充）
    3. 工具调用频率检查（防止冗余调用）
    4. 答案来源验证（确保答案来自推理，而非随机猜测）
    """
    # 基础准确率奖励
    base_reward = accuracy_reward(completion, ground_truth, task_type)

    # ── 防御 1：推理过程连贯性 ────────────────────────────────────────────
    if "<think>" in completion and "</think>" in completion:
        think_content = completion.split("<think>")[1].split("</think>")[0]
        coherence = _compute_text_coherence(think_content)
        if coherence < 0.5:
            base_reward *= 0.5   # 推理不连贯（可能是乱码），奖励减半

    # ── 防御 2：输出长度合理性 ────────────────────────────────────────────
    token_count = len(completion.split())
    if token_count > 1000:
        base_reward *= 0.7   # 异常长的输出，可能是填充行为

    # ── 防御 3：工具调用频率 ──────────────────────────────────────────────
    tool_calls = completion.count("<tool_call>")
    if tool_calls > 8:
        base_reward *= max(0.5, 1.0 - 0.05 * (tool_calls - 8))

    return base_reward


def _compute_text_coherence(text: str) -> float:
    """
    计算文本连贯性分数（简化版）
    
    通过统计有效字符（中文、英文、数字、标点）的比例
    来近似估计文本是否为正常语言（而非随机字符）
    """
    if not text.strip():
        return 0.0
    valid_chars = len(re.findall(r'[\u4e00-\u9fff\w\s.,!?，。！？；：]', text))
    return valid_chars / max(len(text), 1)
```

### 5.5 不同任务类型的奖励设计模板


**数学推理任务**


```python
math_reward_config = RewardConfig(
    accuracy_weight=0.60,    # 数学任务以正确性为核心
    format_weight=0.15,
    efficiency_weight=0.15,
    safety_weight=0.10,
)
# 准确率评估：数值精确匹配（允许 1% 相对误差）
# 格式要求：必须包含 <think> 推理过程
# 效率标准：期望步数 ≤ 3，最大 Token 数 ≤ 400
```


**代码生成与修复任务**


```python
code_reward_config = RewardConfig(
    accuracy_weight=0.50,    # 测试用例通过率
    format_weight=0.10,
    efficiency_weight=0.25,  # 代码任务效率更重要（减少文件编辑次数）
    safety_weight=0.15,      # 代码安全性至关重要
)
# 准确率评估：执行测试用例，按通过率给分（部分奖励）
# 效率标准：期望文件编辑次数 ≤ 3，最大迭代轮数 ≤ 5
# 安全检查：严格检测危险命令和代码注入
```


**信息检索与问答任务**


```python
retrieval_reward_config = RewardConfig(
    accuracy_weight=0.40,    # 答案准确性（需 LLM 评判）
    format_weight=0.20,      # 引用格式、来源标注
    efficiency_weight=0.20,  # 搜索次数和 Token 消耗
    safety_weight=0.20,      # 防止信息泄露
)
# 准确率评估：LLM-as-Judge（需混合规则奖励防止黑客）
# 格式要求：必须包含来源引用，最少 2 个可验证来源
```

> **📌 工程实践要点**
>
> - **从简单开始**：先用准确率 + 格式两个维度训练，确认模型行为正常后再逐步加入效率和安全维度
> - **人工审查**：每 100 个训练步骤，随机抽取 20 条高奖励和 20 条低奖励样本进行人工审查，验证奖励函数是否合理
> - **奖励版本管理**：奖励函数的每次修改都应纳入版本控制，记录修改原因、预期效果和实际效果
> - **动态权重调整**：训练初期（前 20% 步骤）适当提高格式权重，帮助模型快速建立格式规范；后期逐步提高准确率权重
> - **奖励分布监控**：定期检查奖励分布，若大多数样本奖励趋于相同（方差极小），说明奖励函数区分度不足，需要重新设计


*掌握了算法原理与奖励函数设计后，接下来将把所有组件整合起来，完成一个从数据准备到模型部署的完整 Agentic-RL 训练 Pipeline。*


# 六、实战：完整 Agentic-RL 训练 Pipeline

## 项目概述与实验设计

本节将从零构建一个完整的 Agentic-RL 训练项目，验证前四节介绍的所有理论与方法。

> **实验目标**：训练一个能够使用计算器工具解决数学推理问题的 Agent 模型
>
> **基座模型**：`Qwen/Qwen2.5-1.5B-Instruct`（消费级 GPU 可训练）
>
> **数据集**：GSM8K（8,500 条小学数学应用题，含标准答案）
>
> **训练流程**：数据准备 → SFT（格式学习）→ GRPO（推理优化）→ 评估对比

### 为什么选择 GSM8K？

GSM8K 是验证 Agentic-RL 效果的理想基准数据集，具备以下关键特性：

| 特性 | 说明 | 对训练的意义 |
|------|------|------------|
| **客观可验证** | 每题有唯一正确的数值答案 | 可自动计算准确率，无需人工标注奖励 |
| **多步推理** | 平均需要 3–5 个推理步骤 | 能充分测试 Agent 的链式推理能力 |
| **规模适中** | 7,473 训练题 + 1,319 测试题 | 训练成本可控，结果具有统计显著性 |
| **社区基准** | 广泛用于 LLM 评估 | 有大量公开基准数据可供对比 |

### 硬件需求与预期训练时间

| 配置 | SFT 阶段 | GRPO 阶段 | 备注 |
|------|---------|----------|------|
| **最低配置** | 1× RTX 3090 24GB | 1× RTX 3090 24GB | 需开启 QLoRA 4-bit 量化 |
| **推荐配置** | 1× A100 40GB | 1× A100 40GB | 全精度 bfloat16 训练 |
| **训练时间（最低配置）** | 约 2–4 小时 | 约 4–8 小时 | 1.5B 模型，3 个 epoch |


## Step 1：环境搭建

```bash
# 创建项目目录与虚拟环境
mkdir -p agent-rl-training && cd agent-rl-training
python -m venv venv && source venv/bin/activate

# 安装核心依赖（版本经过兼容性验证）
pip install torch>=2.1.0
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install trl>=0.12.0
pip install datasets accelerate bitsandbytes
pip install wandb tensorboard          # 实验追踪（强烈推荐）
```


## Step 2：数据准备

```python
"""
step2_prepare_data.py

将 GSM8K 原始数据转换为 Agent 格式的 SFT 训练数据。

GSM8K 原始格式：
  question: "Natalia sold clips to 48 of her friends..."
  answer:   "Natalia sold 48/2 = <<48/2=24>>24 clips... #### 72"

目标格式（Agent 轨迹）：
  <think>推理过程</think>
  <tool_call>calculator(expression="...")</tool_call>
"""

import re
from datasets import load_dataset, Dataset


def extract_final_answer(solution: str) -> str:
    """从 GSM8K 解答中提取 '#### 数字' 格式的最终答案"""
    match = re.search(r'####\s*(.+)', solution)
    return match.group(1).strip().replace(",", "") if match else ""


def extract_calculations(solution: str) -> list[str]:
    """提取 GSM8K 解答中的计算表达式（格式：<<expr=result>>）"""
    return re.findall(r'<<(.+?)=.+?>>', solution)


def convert_to_agent_format(example: dict) -> dict:
    """
    将 GSM8K 样本转换为 Agent SFT 训练格式
    
    转换策略：
    1. 提取推理步骤作为 <think> 内容
    2. 提取最后一个计算表达式作为 <tool_call> 参数
    3. 构建完整的 ChatML 格式对话
    """
    question = example["question"]
    solution = example["answer"]
    final_answer = extract_final_answer(solution)
    calculations = extract_calculations(solution)

    # 提取推理步骤（去除 #### 行和计算标注）
    steps = [
        re.sub(r'<<.+?>>', '', line).strip()
        for line in solution.split("\n")
        if line.strip() and "####" not in line
    ]
    think_content = "\n".join(steps)

    # 构建 Agent 格式回答
    if calculations:
        # 使用最后一个计算表达式（通常是最终计算步骤）
        final_expr = calculations[-1]
        agent_response = (
            f"<think>\n{think_content}\n"
            f"最终需要计算：{final_expr}\n</think>\n\n"
            f"<tool_call>\ncalculator(expression=\"{final_expr}\")\n</tool_call>"
        )
    else:
        agent_response = (
            f"<think>\n{think_content}\n</think>\n\n"
            f"最终答案是 **{final_answer}**。"
        )

    # 构建 ChatML 格式对话
    conversation = (
        "<|im_start|>system\n"
        "你是一个数学助手。解题时请先在 <think> 标签中写出完整的推理过程，"
        "需要精确计算时使用 calculator 工具。\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{agent_response}\n<|im_end|>"
    )

    return {
        "text": conversation,
        "question": question,
        "answer": final_answer,
    }


# ── 加载并转换数据集 ──────────────────────────────────────────────────────
print("📦 加载 GSM8K 数据集...")
dataset = load_dataset("openai/gsm8k", "main")

print("🔄 转换为 Agent 格式...")
sft_train = dataset["train"].map(convert_to_agent_format, remove_columns=dataset["train"].column_names)
sft_test  = dataset["test"].map(convert_to_agent_format, remove_columns=dataset["test"].column_names)

print(f"✅ 训练集：{len(sft_train)} 条 | 测试集：{len(sft_test)} 条")

# 数据质量验证
valid_train = sft_train.filter(lambda x: "<think>" in x["text"] and x["answer"] != "")
print(f"📊 格式验证通过率：{len(valid_train) / len(sft_train):.1%}")

sft_train.save_to_disk("./data/sft_train")
sft_test.save_to_disk("./data/sft_test")
```


## Step 3：SFT 训练

```python
"""
step3_sft_training.py

SFT 阶段：通过模仿学习让模型习得 Agent 行为格式。
目标：将基座模型的格式符合率从 ~5% 提升至 ~85%+。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk

# ── 数据加载 ──────────────────────────────────────────────────────────────
train_dataset = load_from_disk("./data/sft_train")
eval_dataset  = load_from_disk("./data/sft_test")

# ── 模型加载（QLoRA 配置）────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA 配置 ─────────────────────────────────────────────────────────────
# 1.5B 模型使用 r=16 即可，参数量约 8M（占总参数量 ~0.5%）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# ── 训练配置 ──────────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir="./checkpoints/sft",

    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # 有效 batch size = 16
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",

    bf16=True,
    gradient_checkpointing=True,

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,        # 自动加载验证集最优检查点

    max_seq_length=1024,
    dataset_text_field="text",
    report_to="tensorboard",
)

# ── 训练执行 ──────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset.select(range(200)),
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("🚀 开始 SFT 训练...")
print(f"   模型：{model_name} | LoRA r={lora_config.r} | 训练数据：{len(train_dataset)} 条")
trainer.train()

trainer.save_model("./checkpoints/sft-final")
tokenizer.save_pretrained("./checkpoints/sft-final")
print("✅ SFT 训练完成！")
```


## Step 4：GRPO 强化学习训练

```python
"""
step4_grpo_training.py

GRPO 阶段：通过强化学习信号引导模型探索超越 SFT 数据质量的推理策略。
目标：在 SFT 基础上，将准确率进一步提升 10–20 个百分点。
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk

# ── 加载 SFT 模型（合并 LoRA 权重）──────────────────────────────────────
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "./checkpoints/sft-final")
model = model.merge_and_unload()   # 合并 LoRA 权重，恢复标准模型结构

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ── 准备 GRPO 训练数据（需要 "prompt" 字段）──────────────────────────────
train_dataset = load_from_disk("./data/sft_train")

def prepare_grpo_prompt(example: dict) -> dict:
    """将训练样本转换为 GRPO 所需的 prompt 格式"""
    return {
        "prompt": (
            "<|im_start|>system\n"
            "你是一个数学助手。解题时请先在 <think> 标签中写出完整的推理过程，"
            "需要精确计算时使用 calculator 工具。\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "answer": example["answer"],
    }

grpo_dataset = train_dataset.map(prepare_grpo_prompt)

# ── 奖励函数：数学 Agent 综合评估 ────────────────────────────────────────
def math_agent_reward(completions: list[str], **kwargs) -> list[float]:
    """
    数学 Agent 综合奖励函数
    
    奖励维度及权重：
    - 准确率（0.50）：最终数值是否正确（允许 1% 相对误差）
    - 格式（0.20）：<think>/<tool_call> 标签是否规范
    - 推理质量（0.20）：推理步骤是否充分、包含计算过程
    - 简洁性（0.10）：输出长度是否合理
    """
    rewards = []
    answers = kwargs.get("answer", [""] * len(completions))

    for completion, answer in zip(completions, answers):
        reward = 0.0

        # ── 维度 1：准确率（权重 0.50）────────────────────────────────────
        try:
            numbers = re.findall(r'-?[\d,]+\.?\d*', completion)
            if numbers:
                pred = float(numbers[-1].replace(",", ""))
                true_val = float(str(answer).replace(",", ""))
                if abs(pred - true_val) / (abs(true_val) + 1e-8) < 0.01:
                    reward += 0.50
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # ── 维度 2：格式正确性（权重 0.20）───────────────────────────────
        has_think = "<think>" in completion and "</think>" in completion
        if has_think:
            reward += 0.10
            think = completion.split("<think>")[1].split("</think>")[0].strip()
            if len(think) > 20:
                reward += 0.10   # 有实质性推理内容

        # ── 维度 3：推理质量（权重 0.20）─────────────────────────────────
        if has_think:
            think = completion.split("<think>")[1].split("</think>")[0]
            lines = [l.strip() for l in think.split("\n") if l.strip()]
            if len(lines) >= 2:
                reward += 0.10   # 多步推理
            if re.search(r'[\d+\-*/=]', think):
                reward += 0.10   # 包含数学计算过程

        # ── 维度 4：简洁性（权重 0.10）───────────────────────────────────
        token_count = len(completion.split())
        if token_count <= 300:
            reward += 0.10
        elif token_count > 800:
            reward -= 0.05   # 过长惩罚

        rewards.append(max(0.0, reward))

    return rewards

# ── GRPO 训练配置 ─────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir="./checkpoints/grpo",

    num_generations=8,               # G=8：每题生成 8 个回答用于组内比较
    num_train_epochs=1,              # GRPO 通常 1–2 个 epoch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,              # RL 学习率 ≈ SFT 学习率的 1/40
    warmup_ratio=0.1,
    max_grad_norm=0.5,               # 梯度裁剪，防止 RL 训练中的梯度爆炸

    max_new_tokens=512,
    temperature=0.7,                 # 保证 G 个回答的多样性

    kl_coef=0.05,                    # KL 散度惩罚系数

    bf16=True,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
)

# ── 训练执行 ──────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
    reward_funcs=math_agent_reward,
)

print("🚀 开始 GRPO 训练...")
print(f"   组大小 G={grpo_config.num_generations} | 学习率={grpo_config.learning_rate} | KL 系数={grpo_config.kl_coef}")
trainer.train()
trainer.save_model("./checkpoints/grpo-final")
print("✅ GRPO 训练完成！")
```


## Step 5：系统性评估与对比分析

```python
"""
step5_evaluation.py

对比评估三个阶段的模型性能：
  基座模型（Baseline）→ SFT 模型 → GRPO 模型

评估指标：
  - 准确率（Accuracy）：最终答案正确率
  - 格式符合率（Format Compliance）：<think> 标签使用率
  - 平均输出长度（Avg. Length）：Token 数
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


def evaluate_model(
    model_path: str,
    test_data,
    num_samples: int = 200,
    device: str = "cuda",
) -> dict:
    """
    在 GSM8K 测试集上评估模型性能
    
    Args:
        model_path:  模型路径（HuggingFace 格式）
        test_data:   测试数据集
        num_samples: 评估样本数（完整评估用 1319）
    
    Returns:
        包含各项指标的评估结果字典
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    correct = 0
    format_ok = 0
    total_tokens = 0
    total = 0

    for example in test_data.select(range(num_samples)):
        prompt = (
            "<|im_start|>system\n"
            "你是一个数学助手。解题时请先在 <think> 标签中写出完整的推理过程，"
            "需要精确计算时使用 calculator 工具。\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # 准确率评估
        try:
            true_val = float(example["answer"].replace(",", ""))
            numbers = re.findall(r'-?[\d,]+\.?\d*', response)
            if numbers:
                pred = float(numbers[-1].replace(",", ""))
                if abs(pred - true_val) / (abs(true_val) + 1e-8) < 0.01:
                    correct += 1
        except (ValueError, ZeroDivisionError):
            pass

        # 格式符合率
        if "<think>" in response and "</think>" in response:
            format_ok += 1

        total_tokens += len(response.split())
        total += 1

    del model   # 释放显存，为下一个模型腾出空间

    return {
        "accuracy":          correct / total,
        "format_compliance": format_ok / total,
        "avg_length":        total_tokens / total,
        "total_samples":     total,
    }


# ── 评估三个阶段的模型 ────────────────────────────────────────────────────
test_data = load_from_disk("./data/sft_test")

models_to_eval = [
    ("🔵 基座模型",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("🟡 SFT 模型",  "./checkpoints/sft-merged"),
    ("🟢 GRPO 模型", "./checkpoints/grpo-final"),
]

results = {}
for name, path in models_to_eval:
    print(f"\n{name} 评估中...")
    results[name] = evaluate_model(path, test_data, num_samples=200)

# ── 结果展示 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("📈 Agentic-RL 训练效果对比（GSM8K 测试集，n=200）")
print("=" * 65)
print(f"{'指标':<20} {'基座模型':>12} {'SFT':>12} {'GRPO':>12}")
print("-" * 65)

metrics = [
    ("准确率",     "accuracy",          ".1%"),
    ("格式符合率", "format_compliance", ".1%"),
    ("平均输出长度", "avg_length",      ".0f"),
]

for label, key, fmt in metrics:
    row = f"{label:<20}"
    for name, _ in models_to_eval:
        val = results[name][key]
        row += f" {val:>11{fmt}}"
    print(row)

print("=" * 65)
print("\n📌 预期结果参考（Qwen2.5-1.5B）：")
print("   基座模型：准确率 ~35–45%，格式符合率 ~5%")
print("   SFT 后：  准确率 ~45–55%，格式符合率 ~85%")
print("   GRPO 后： 准确率 ~55–65%，格式符合率 ~90%")
```


## Step 6：模型导出与部署

```python
"""
step6_export.py

将训练好的模型导出为生产可用的格式。
支持 HuggingFace 格式（用于 vLLM/TGI 部署）和 GGUF 格式（用于本地部署）。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 加载最终模型 ──────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/grpo-final",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/grpo-final")

# ── 方式 1：HuggingFace 格式（推荐用于服务端部署）────────────────────────
# 兼容 vLLM、Text Generation Inference (TGI)、Ollama 等推理框架
model.save_pretrained("./export/hf-model", safe_serialization=True)
tokenizer.save_pretrained("./export/hf-model")
print("✅ HuggingFace 格式已导出至 ./export/hf-model")

# ── 方式 2：GGUF 格式（用于 llama.cpp / Ollama 本地部署）────────────────
# 需要安装 llama.cpp 并使用其转换脚本
# python llama.cpp/convert_hf_to_gguf.py ./export/hf-model \
#     --outtype q4_k_m \
#     --outfile ./export/model-q4_k_m.gguf
print("💡 GGUF 格式转换命令：")
print("   python llama.cpp/convert_hf_to_gguf.py ./export/hf-model \\")
print("       --outtype q4_k_m --outfile ./export/model-q4_k_m.gguf")
```


## 完整项目结构

```
agent-rl-training/
├── data/
│   ├── sft_train/              # SFT 训练数据（7,473 条 Agent 格式轨迹）
│   └── sft_test/               # 评估数据（1,319 条）
├── checkpoints/
│   ├── sft/                    # SFT 训练检查点（含 TensorBoard 日志）
│   ├── sft-final/              # SFT 最终 LoRA 适配器权重
│   ├── sft-merged/             # SFT 合并后的完整模型（用于 GRPO 初始化）
│   ├── grpo/                   # GRPO 训练检查点
│   └── grpo-final/             # GRPO 最终模型（用于评估和部署）
├── export/
│   ├── hf-model/               # HuggingFace 格式（服务端部署）
│   └── model-q4_k_m.gguf       # GGUF 格式（本地部署，可选）
├── step2_prepare_data.py
├── step3_sft_training.py
├── step4_grpo_training.py
├── step5_evaluation.py
├── step6_export.py
└── requirements.txt
```

> **📌 工程实践要点**
>
> - **实验追踪**：强烈建议使用 wandb 或 MLflow 记录每次训练的超参数、损失曲线和评估指标，便于复现和对比
> - **数据增强**：可用 GPT-4 对 GSM8K 题目进行改写，生成更多样化的训练数据，通常能带来 2–5% 的准确率提升
> - **课程学习（Curriculum Learning）**：先用简单题（1–2 步推理）训练，再逐步引入复杂题（4–5 步推理），收敛速度通常更快
> - **模型规模效应**：本教程使用 1.5B 模型作为教学演示；实际生产中，7B–14B 模型的 GRPO 提升幅度更显著（通常 15–25%）
> - **成本估算**：A100 40GB 上，1.5B 模型完整训练约需 6–12 小时；7B 模型约需 24–48 小时；14B 模型约需 48–96 小时


## 全文总结

通过以上的系统梳理，我们已掌握了 Agentic-RL 训练的完整知识体系：

| 章节 | 核心知识点 | 关键结论 |
|------|-----------|--------|
| **概述** | MDP 建模、两阶段范式 | RL 训练可涌现出超越训练数据的推理策略 |
| **SFT + LoRA** | 监督微调、参数高效训练 | LoRA 以 <1% 的参数量实现接近全参数微调的效果 |
| **PPO** | 策略梯度、重要性采样、优势函数、Clip 机制 | PPO 是 RLHF 的经典算法，但 Critic 导致显存占用 ≈ 3× |
| **DPO** | 隐式奖励、Bradley-Terry 模型、闭式解 | DPO 将 RL 转化为监督学习，极简但无法在线探索 |
| **GRPO + 奖励设计** | 组内比较、多维度奖励、奖励黑客防御 | GRPO 将显存需求从 3× 降至 1.5×；奖励函数是 RL 训练效果的决定性因素 |
| **实战** | 完整 Pipeline、评估对比 | GSM8K 上：基座 ~40% → SFT ~50% → GRPO ~60% |

Agentic-RL 代表了 LLM 应用的一个重要发展方向：**从"提示工程"到"训练优化"的范式转变**。随着算法的持续演进和计算成本的降低，这一技术将在越来越多的高价值 Agent 场景中发挥关键作用。


# 七、最新研究进展（2025—2026）

> 📖 *"从 DeepSeek-R1 登上 Nature 封面到 DAPO/VAPO 刷新推理基准，Agentic-RL 正以惊人的速度从实验室走向工程实践。下面将带你纵览这一领域最前沿的研究成果。"*

> ⏰ **时效性说明**：本节内容更新至 2026 年 3 月 20 日。由于该领域发展极为迅速，建议读者结合 [Awesome-RL-Reasoning-Recipes](https://github.com/yuezhao-zy/Awesome-RL-Reasoning-Recipes) 等开源项目获取最新动态。


## 7.1 概览：从 RLHF 到推理 RL 的范式转变

过去两年（2025—2026）是大模型强化学习领域爆发式发展的两年。以 **DeepSeek-R1** 登上 Nature 封面为标志，RL 训练 LLM 从"对齐人类偏好"（RLHF）的辅助角色，跃升为**激发模型推理能力**的核心技术。我们可以用一张时间线来概览关键里程碑：

```
2024.09  OpenAI o1 发布，首次展示"推理时间计算扩展"（test-time compute scaling）的潜力
2025.01  DeepSeek-R1 发布，纯 RL 训练激发自主推理能力，使用 GRPO 算法
2025.01  Kimi k1.5 发布，128K 长上下文 RL 训练，Long2Short 蒸馏技术
2025.02  QwQ-32B 发布，展示中等规模模型的推理 RL 训练效果
2025.03  DAPO 开源发布，提出可复现的大规模 RL 训练方案
2025.04  VAPO 发布，基于价值增强的 PPO 框架，AIME 2024 达到 60.4 分
2025.06  OpenAI o3 发布，推理能力进一步跃升
2025.08  Self-Aligned Reward (SAR) 提出，利用困惑度信号解决过度思考
2025.10  PURE 框架发布，最小形式信用分配解决奖励破解问题
2025.12  Co-rewarding (ICLR 2026) 提出自监督 RL 学习方案
2026.01  RLVR 新范式：基于问题拆解的高效强化学习方法
2026.02  DRQA 动态推理配额分配，token 成本降低 31%
2026.03  CoRLHF 提出协同策略-奖励联合优化
```

这些工作可以归纳为以下几个核心研究方向：

| 方向 | 代表工作 | 核心问题 |
|------|---------|---------|
| **推理模型训练** | DeepSeek-R1, Kimi k1.5, QwQ | 如何通过 RL 激发 LLM 的推理能力？ |
| **RL 算法改进** | DAPO, VAPO, GRPO 变体 | 如何让大模型 RL 训练更稳定、更高效？ |
| **奖励设计与反馈** | SAR, Co-rewarding, CoRLHF | 如何设计更好的奖励信号？ |
| **过度思考与效率** | PURE, DRQA, DEER | 如何让模型"恰到好处"地推理？ |
| **Agentic 任务 RL** | AgentPRM, R³L, DeepSWE | 如何将 RL 扩展到工具调用等 Agent 任务？ |

下面我们逐一深入介绍每个方向的重要论文。


## 7.2 推理模型：纯 RL 训练激发自主推理

### 7.2.1 DeepSeek-R1：Nature 封面的突破

**论文**：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (Nature, 2025) [1]

DeepSeek-R1 是该领域最具里程碑意义的工作。它的核心发现是：

> **仅通过 RL 训练（不需要人工标注的推理链），模型可以自主涌现出多步推理、自我反思和动态策略调整等高级认知能力。**


**核心技术要点**


1. **GRPO 算法**：使用组内相对策略优化（详见 GRPO 部分），通过组内回答竞争优化策略，避免昂贵的 Critic 网络，总训练成本约 29.4 万美元。

2. **多阶段训练框架**：
   - **R1-Zero 阶段**：仅使用结果正确性作为奖励（可验证奖励 RLVR），不使用任何 SFT 数据。模型自发涌现出 "Aha moment"——在推理过程中学会自我反思和纠错。
   - **R1 阶段**：在 R1-Zero 基础上，融入少量高质量 SFT 数据和人类偏好对齐，提升综合能力。

3. **可验证奖励（RLVR）**：奖励信号来源于可自动验证的任务（如数学题的最终答案），无需人工标注。


**关键实验结果**


- 在 MMLU、AIME 2024、LiveCodeBench 等 21 个基准上达到 SOTA
- R1-Zero 展示了"从零开始学推理"的可能性——RL 训练过程中推理链长度自发增长
- 蒸馏到 7B/14B 小模型后仍保持强推理能力


**为什么重要？**


DeepSeek-R1 证明了两个关键论点：
1. **RL 可以激发预训练中潜在的推理能力**——这些能力在 SFT 或 prompt engineering 中难以充分释放
2. **推理能力可以在纯 RL 环境中"涌现"**——无需依赖人工标注的推理链作为示范


### 7.2.2 Kimi k1.5：长上下文 RL 的突破

**论文**：*Kimi k1.5: Scaling Reinforcement Learning with LLMs* (2025) [2]

Kimi k1.5 由 Moonshot AI 团队开发，在几个方面做出了独特贡献：


**核心创新**


1. **128K 长上下文 RL 训练**：将 RL 训练的上下文窗口从传统的 4K-8K 扩展到 128K tokens，通过**部分轨迹重用**（Partial Rollout Reuse）提升训练效率。

2. **简化 RL 框架**：摒弃蒙特卡洛树搜索（MCTS）和价值函数，仅通过改进的在线镜像下降（Online Mirror Descent）直接优化模型，大幅降低计算负担。

3. **Long2Short 蒸馏技术**：将长上下文推理能力"压缩"到短上下文模型中。具体做法是：
   - 先在长上下文设置下训练出强推理能力
   - 然后通过知识蒸馏，让短上下文模型学会"精炼"推理


**关键结果**


- 在 LiveCodeBench 等短任务上超越 GPT-4o 达 550%
- Long2Short 技术证明了**长链推理能力可以被压缩而不显著损失**
- 首次展示 128K 上下文窗口的 RL 训练的可行性


### 7.2.3 QwQ-32B：中等规模的推理 RL

**论文**：*QwQ: Reflect and Question to Understand the World* (Alibaba, 2025) [3]

QwQ-32B 是阿里巴巴通义团队发布的中等规模推理模型，其意义在于证明了 **32B 参数量级的模型也能通过 RL 训练获得强大的推理能力**。


**技术特点**


- 基于 Qwen2.5-32B 进行 RL 训练
- 在数学推理任务上接近 DeepSeek-R1 的表现
- 训练成本远低于 670B 级别模型


**为什么重要？**


QwQ 证明了推理 RL 不是"大模型专属"——中等规模模型通过合适的 RL 训练同样能获得显著的推理能力提升，这对资源有限的团队和边缘部署场景具有重大实践价值。


### 7.2.4 OpenAI o1/o3：推理时间计算扩展

**模型**：*OpenAI o1* (2024.09) / *OpenAI o3* (2025.06) [4]

虽然 OpenAI 未公布完整的技术报告，但 o1 和 o3 系列模型在业界产生了深远影响：


**核心理念：Test-Time Compute Scaling**


传统的 Scaling Law 关注**训练时计算扩展**（更大模型 + 更多数据）。o1/o3 系列提出了另一个维度：

> **在推理时投入更多计算（更长的思考链、更多的搜索/验证），也能持续提升模型能力。**

这意味着存在两条互补的扩展路径：
1. **训练时扩展**：增大模型、增加数据
2. **推理时扩展**：增加推理步骤、验证回路


**对领域的影响**


- 催生了"推理模型"这一新品类
- 推动了 GRPO、DAPO、VAPO 等面向推理任务的 RL 算法研发
- 引发了对"推理效率"的关注——过度思考（Overthinking）问题浮出水面


## 7.3 RL 算法改进：让大模型 RL 训练更稳定高效

### 7.3.1 DAPO：大规模可复现的 RL 训练

**论文**：*DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025) [5]

DAPO（Decoupled Clip and Dynamic Sampling PPO）由字节跳动 Seed 团队提出，核心目标是解决大规模 RL 训练的**可复现性**问题。


**核心技术**


1. **解耦裁剪（Decoupled Clipping）**：传统 PPO 使用对称裁剪 $\epsilon$，DAPO 将上下裁剪边界分离：
   - $\epsilon_{\text{high}}$（较大）：鼓励对好回答的探索
   - $\epsilon_{\text{low}}$（较小）：严格抑制坏回答
   
   这种不对称设计让模型在"保守抑制坏行为"的同时"大胆探索好行为"。

2. **动态采样（Dynamic Sampling）**：根据训练进度动态调整每个问题的采样数量：
   - 训练初期：多采样，增加探索
   - 训练后期：少采样，精细优化

3. **Token 级策略约束**：在 token 级别而非序列级别施加 KL 约束，更精细地控制策略偏移。


**开源贡献**


DAPO 完整开源了训练代码和数据集（基于 Qwen2.5-32B），是目前最具可复现性的大规模 RL 训练方案之一。


### 7.3.2 VAPO：基于价值增强的 PPO

**论文**：*VAPO: Efficient and Reliable RL Framework for Advanced Reasoning Tasks* (ByteDance Seed, 2025) [6]

VAPO（Value-based Augmented PPO）是 DAPO 的后续工作，专门针对**长链推理任务**中的难题。


**核心问题**


长链推理（如数学证明、复杂编程）中，RL 训练面临三大挑战：
1. **价值模型偏差**：Critic 网络对长序列的价值估计不准
2. **异构序列长度**：同一批次中回答长度差异极大
3. **稀疏奖励**：只有最终答案才有奖励信号


**核心技术**


1. **价值预训练（Value Pretraining）**：使用蒙特卡洛回报预训练 Critic 网络，减小初始化偏差。

2. **解耦 GAE（Decoupled GAE）**：
   - 对价值网络使用 $\lambda_V = 1.0$（低偏差、高方差）
   - 对策略网络使用 $\lambda_P = 0.95$（平衡偏差与方差）

3. **长度自适应 GAE（Length-Adaptive GAE）**：根据序列长度动态调整 $\lambda$：

$$\lambda = 1 - \frac{1}{0.05 \cdot l}$$

   其中 $l$ 为序列长度。长序列使用更大的 $\lambda$（减少偏差），短序列使用更小的 $\lambda$（减少方差）。

4. **Clip-Higher 探索**：使用不对称裁剪 $\epsilon_{\text{high}} = 0.28$, $\epsilon_{\text{low}} = 0.2$，鼓励多样性采样。


**关键结果**


| 模型 | AIME 2024 | 训练步数 | 稳定性 |
|------|-----------|---------|--------|
| DeepSeek-R1-Zero (671B) | ~50 | 大量 | 偶有崩溃 |
| DAPO (32B) | ~50 | 中等 | 较稳定 |
| **VAPO (32B)** | **60.4** | **~5,000** | **无崩溃** |

VAPO 仅用 Qwen-32B 和 5,000 步训练就超越了 671B 的 DeepSeek-R1-Zero，且训练过程完全无崩溃。


### 7.3.3 GRPO 变体与改进

自 DeepSeek-R1 提出 GRPO 以来，多篇论文对其进行了改进：

| 改进方向 | 代表工作 | 解决的问题 |
|---------|---------|-----------|
| 移除均值归一化 | Dr. GRPO | 原始 GRPO 的组内均值归一化会引入偏差 |
| 自适应组大小 | Adaptive GRPO | 固定组大小不适合所有问题难度 |
| Token 级优势 | Token-level GRPO | 序列级优势对长序列不够精细 |
| 在线/离线混合 | Hybrid GRPO | 纯在线采样效率低 |


## 7.4 奖励设计：如何告诉模型什么是好的推理？

奖励函数是 RL 训练的"灵魂"。2025—2026 年，奖励设计出现了三个重要方向。

### 7.4.1 Self-Aligned Reward (SAR)：利用模型内部信号

**论文**：*Self-Aligned Reward: Towards Effective and Efficient Reasoners* (UIUC & Amazon AWS, 2025) [7]


**核心思想**


SAR 的核心洞察是：**模型内部的困惑度（Perplexity）差异可以作为高质量的奖励信号**。

具体来说，SAR 计算两种条件下的困惑度差异：

$$r_{\text{SAR}}(y|x) = \frac{\text{PPL}(y) - \text{PPL}(y|x)}{\text{PPL}(y)}$$

其中：
- $\text{PPL}(y|x)$：给定问题 $x$ 时生成回答 $y$ 的困惑度
- $\text{PPL}(y)$：将回答 $y$ 视为独立文本时的困惑度

**直觉解释**：
- **高 SAR**：回答高度依赖于问题（是针对性的、简洁的回答）
- **低 SAR**：回答与问题关联弱（可能是冗长、泛泛的内容）


**为什么有效？**


1. **无需外部奖励模型**：利用模型自身的语言建模能力
2. **细粒度评分**：可以区分"正确且简洁"vs"正确但冗长"
3. **跨任务泛化**：在数学数据上训练，在逻辑推理等非数学任务上同样有效


**实验效果**


在 4 个基础模型和 7 个数据集上：
- 准确率平均提升 4%
- 输出长度减少 30%


### 7.4.2 Co-rewarding：自监督 RL 学习

**论文**：*Co-rewarding: Self-Supervised RL for LLM Reasoning* (ICLR 2026) [8]


**核心问题**


Self-rewarding RL（让模型自己给自己打分）容易出现**训练坍塌**——模型学会生成"容易给自己高分"而非"真正好"的回答。


**解决方案**


Co-rewarding 引入**互补监督信号**：
1. 对同一问题生成**改写版本**
2. 使用改写问题的回答作为原问题回答的辅助评估
3. 两个方向的评估互相约束，防止坍塌


**关键结果**


- 在推理任务上性能提升 12.9%（无需真实标签）
- 训练过程显著更稳定


### 7.4.3 CoRLHF：协同策略-奖励联合优化

**论文**：*CoRLHF: Reinforcement Learning from Human Feedback with Cooperative Policy-Reward Optimization* (Expert Systems with Applications, 2026) [9]


**核心创新**


传统 RLHF 分两步：先训练奖励模型，再用奖励模型训练策略。这导致了**分布不匹配**问题——奖励模型训练时看到的数据分布与策略优化时生成的数据分布不一致。

CoRLHF 将策略优化和奖励模型优化**合并为一个迭代过程**：
1. 策略生成新数据
2. 奖励模型在新数据上更新
3. 策略在更新后的奖励上优化
4. 循环迭代

这种方法桥接了 RLHF 和 RLAIF，在减少人工反馈依赖的同时保持了对齐质量。


### 7.4.4 内生奖励：LLM 是自带的奖励模型

**论文**：周志华团队相关工作 (南京大学, 2025) [10]


**颠覆性发现**


这项研究发现：**LLM 的 next-token prediction 能力本身就蕴含了通用奖励函数**（内生奖励，Endogenous Reward）。

也就是说，预训练过程中学到的语言模型分布已经隐式编码了"什么是好的输出"的判断能力，无需额外训练奖励模型。


**实践意义**


- 减少了 RLHF pipeline 中的一个组件（奖励模型）
- 降低了误差累积的风险
- 在多个对齐基准上超越传统奖励模型


## 7.5 过度思考与推理效率

随着推理模型的普及，一个新问题浮出水面：**过度思考（Overthinking）**——模型在简单问题上也生成冗长的推理链，浪费计算资源且可能降低准确率。

### 7.5.1 问题分析：为什么推理模型会"想太多"？

过度思考的根源在于 RLVR（基于可验证奖励的 RL）的奖励结构：

> **只要最终答案正确，不管推理过程多长、多冗余，模型都会获得同样的奖励。**

这导致了两个问题：
1. **奖励膨胀**：标准 RL 的求和形式信用分配使模型偏好生成更多步骤
2. **无差别激励**：无法区分"简洁正确"和"冗长正确"

### 7.5.2 PURE：最小形式信用分配

**论文**：*Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning* (2025) [11]


**核心洞察**


传统 RL 将轨迹价值定义为未来奖励的**总和**：

$$V_{\text{sum}}(s_t) = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

PURE 提出用**最小值**替代总和：

$$V_{\text{min}}(s_t) = \min(r_t, r_{t+1}, \ldots, r_T)$$

**直觉**：推理链的强度取决于**最薄弱的一环**。

| 方式 | 训练信号 | 后果 |
|------|---------|------|
| 求和形式 | "多生成'还行'的步骤来堆分" | 冗长、循环论证 |
| 最小形式 | "每一步都必须正确，一步错满盘输" | 简洁、精确 |


**实现方法**


PURE 通过温度参数 $T$ 将过程奖励转换为新奖励，使标准 RL 算法（PPO/GRPO）的求和公式在数学上等效于取最小值——**无需修改底层算法，只需奖励预处理**。


**实验结果**


- 求和形式训练几乎立即崩溃
- 最小形式训练稳定提升
- 样本效率提升 2-3 倍


### 7.5.3 DRQA：动态推理配额分配

**论文**：*DRQA: Dynamic Reasoning Quota Allocation for Controlling Overthinking in Reasoning Large Language Models* (2026) [12]


**核心观察**


一个有趣的发现：当模型**批量处理**多个问题时（而非逐个处理），总输出长度显著缩短——模型似乎能隐式区分问题难度并"压缩"简单问题的推理。


**方法**


1. 构建偏好数据：
   - 单独生成的推理链（冗长版）
   - 批量生成的推理链（精炼版）
   - 按正确性和简洁性标注偏好

2. 使用 GRPO 训练模型同时优化**逻辑正确性**和**推理简洁性**


**效果**


- 推理 token 成本降低 31%
- 准确率反而提升
- 在简单问题上缩短最多，复杂问题保持充分推理


### 7.5.4 DEER：动态早停推理

**论文**：*Dynamic Early Exit in Reasoning Models (DEER)* (2026) [13]

DEER 是一种**免训练**的推理时优化方法：

1. 在推理过程中实时监控模型置信度
2. 当模型对当前答案高度自信时触发早停
3. 简单问题快速结束，复杂问题继续思考


**效果**


- 推理链长度缩短 19.1%—80.1%
- 准确率提升 0.3%—5.0%
- 无需额外训练，即插即用


### 7.5.5 方案对比

| 方法 | 核心思想 | 是否需要训练 | 效率提升 | 准确率影响 |
|------|---------|-------------|---------|-----------|
| **SAR** | 困惑度差异作为奖励 | 是（RL训练） | 长度-30% | +4% |
| **PURE** | 最小形式信用分配 | 是（奖励预处理） | 2-3x 样本效率 | 显著提升 |
| **DRQA** | 模拟批量推理的配额分配 | 是（GRPO训练） | token-31% | 提升 |
| **DEER** | 置信度触发早停 | 否（推理时） | 长度-19%~80% | +0.3%~5% |
| **简洁 RL** | 二阶段精炼训练 | 是（二阶段RL） | 长度显著缩短 | 不降反升 |


## 7.6 RLVR：基于可验证奖励的强化学习

**RLVR（Reinforcement Learning with Verifiable Rewards）** 是 2025—2026 年最热门的研究方向之一，也是 DeepSeek-R1 成功的关键。

### 7.6.1 什么是 RLVR？

与传统 RLHF 依赖人工标注的偏好数据不同，RLVR 使用**可自动验证**的信号作为奖励：

| 对比维度 | RLHF | RLVR |
|---------|------|------|
| 奖励来源 | 人工标注偏好 | 自动验证（如答案对错） |
| 标注成本 | 高 | 极低 |
| 适用任务 | 开放式（对话、写作） | 有明确正确答案（数学、代码） |
| 扩展性 | 受标注速度限制 | 几乎无限扩展 |

### 7.6.2 RLVR 的问题与改进

**问题拆解框架**（人大 & 字节, 2026）[14]：

传统 RLVR 仅在最终答案处给出奖励（稀疏奖励），导致长链推理中的信用分配困难。该工作提出 **Decomposer-Reasoner 框架**：

1. **Decomposer**：将复杂问题拆解为子问题
2. **Reasoner**：逐步解决子问题
3. **密集奖励**：每个子问题的解决都有可验证的奖励

这种方法将稀疏奖励转化为密集奖励，显著提升了 RL 训练的探索效率。


## 7.7 Agentic 任务的 RL 训练

前面讨论的大多是推理任务（数学、代码）的 RL 训练。一个更前沿的方向是将 RL 应用到真正的 **Agentic 任务**——需要工具调用、环境交互、多步决策的场景。

### 7.7.1 AgentPRM：过程奖励模型用于 Agent 评估

在多轮 Agent 任务（如网页导航、API 调用）中，仅评估最终结果不够——需要评估**每一步决策的质量**。AgentPRM 引入了**过程奖励模型（Process Reward Model）** 来评估 Agent 的中间决策。

### 7.7.2 R³L：反思-重试 RL

**R³L（Reflect-then-Retry RL）** 针对 Agent 任务中的失败恢复：

1. 当 Agent 执行失败时，生成语言反馈诊断错误原因
2. 从失败点重新开始，利用反馈避免重蹈覆辙
3. 大幅减少了 rollout 成本

### 7.7.3 DeepSWE：软件工程 Agent 的 RL 训练

DeepSeek 团队的 DeepSWE 展示了 RL 训练的软件工程 Agent 可以匹配闭源模型的 SWE-bench 表现，证明了 RL 在复杂 Agentic 任务中的潜力。


## 7.8 开放挑战与未来方向

尽管进展迅速，该领域仍面临诸多开放挑战：

### 7.8.1 奖励破解（Reward Hacking）

模型可能找到奖励函数中的漏洞来"作弊"，而非真正提升能力。例如：
- 生成"看起来像推理"但实际是胡说八道的长文本
- 利用格式技巧（如特定关键词）获得高奖励
- 在自我评估中学会"自欺欺人"

### 7.8.2 训练稳定性

大模型 RL 训练仍然不够稳定：
- **KL 散度管理**：策略偏移过大会导致灾难性遗忘
- **奖励规模**：不同奖励维度的尺度不一致
- **数据多样性**：训练数据的多样性直接影响探索质量

### 7.8.3 泛化能力

当前 RL 训练的推理能力主要在数学和代码领域验证，向以下领域的泛化仍需探索：
- 开放域推理（科学推理、常识推理）
- 多模态推理（视觉-语言、视频理解）
- 跨语言推理

### 7.8.4 效率与成本

RL 训练的计算成本仍然很高：
- 大量的 rollout 采样
- 多个模型（Policy、Reference、可能的 Critic）同时在显存中
- 长序列推理的显存和时间开销

### 7.8.5 未来展望

基于当前的研究趋势，我们预期以下方向将成为热点：

| 方向 | 预期进展 |
|------|---------|
| **内部信号挖掘** | 更多利用模型自身信号（如 SAR、内生奖励）替代外部奖励模型 |
| **自我进化训练** | 模型自主生成训练数据和奖励信号的闭环系统 |
| **多模态 RL** | 将推理 RL 扩展到视觉、语音等多模态场景 |
| **Agentic RL 扩展** | 将 RL 从推理任务扩展到工具调用、环境交互等 Agent 场景 |
| **高效训练** | 减少 rollout 成本、提升样本效率的新算法 |
| **理论基础** | 更深入理解 RL 如何激发 LLM 推理能力的理论分析 |


## 7.9 论文列表

以下是本节涉及的主要论文，按主题分类：

### 推理模型

| # | 论文 | 作者/机构 | 年份 | 核心贡献 |
|---|------|---------|------|---------|
| [1] | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek AI | 2025 | 纯 RL 训练激发自主推理，GRPO 算法 |
| [2] | Kimi k1.5: Scaling Reinforcement Learning with LLMs | Moonshot AI | 2025 | 128K 长上下文 RL，Long2Short 蒸馏 |
| [3] | QwQ: Reflect and Question to Understand the World | Alibaba | 2025 | 中等规模推理 RL |
| [4] | OpenAI o1/o3 System Card | OpenAI | 2024/2025 | 推理时间计算扩展 |

### RL 算法

| # | 论文 | 作者/机构 | 年份 | 核心贡献 |
|---|------|---------|------|---------|
| [5] | DAPO: An Open-Source LLM RL System at Scale | ByteDance Seed | 2025 | 解耦裁剪 + 动态采样，开源可复现 |
| [6] | VAPO: Efficient and Reliable RL for Advanced Reasoning | ByteDance Seed | 2025 | 价值预训练 + 长度自适应 GAE，AIME 60.4 |

### 奖励设计

| # | 论文 | 作者/机构 | 年份 | 核心贡献 |
|---|------|---------|------|---------|
| [7] | Self-Aligned Reward (SAR) | UIUC & AWS | 2025 | 困惑度差异作为内在奖励 |
| [8] | Co-rewarding | ICLR 2026 | 2025 | 自监督 RL，互补评估信号 |
| [9] | CoRLHF | Expert Systems with Applications | 2026 | 策略-奖励联合迭代优化 |
| [10] | 内生奖励 | 南京大学（周志华团队） | 2025 | LLM 内含通用奖励函数 |

### 推理效率

| # | 论文 | 作者/机构 | 年份 | 核心贡献 |
|---|------|---------|------|---------|
| [11] | PURE: Min-Form Credit Assignment | — | 2025 | 最小形式替代求和形式信用分配 |
| [12] | DRQA: Dynamic Reasoning Quota Allocation | — | 2026 | 动态推理配额分配，token 降 31% |
| [13] | DEER: Dynamic Early Exit in Reasoning Models | — | 2026 | 免训练动态早停 |
| [14] | RLVR with Adaptive Problem Decomposition | 人大 & 字节 | 2026 | 问题拆解密集奖励 |


## 7.10 推荐阅读路线

如果你是该领域的新入门者，建议按以下顺序阅读：

```
入门路线：
1. DeepSeek-R1 论文（理解 RLVR + GRPO 的核心思想）
   ↓
2. DAPO 论文 + 代码（动手复现大模型 RL 训练）
   ↓
3. VAPO 论文（理解价值函数在长链推理中的作用）
   ↓
4. SAR / PURE 论文（理解奖励设计与过度思考问题）
   ↓
5. Kimi k1.5 / QwQ（了解不同团队的技术路线）
```

如果你对特定主题感兴趣：
- **想做推理模型训练** → 重点读 DeepSeek-R1 + DAPO + VAPO
- **想设计奖励函数** → 重点读 SAR + PURE + Co-rewarding
- **想优化推理效率** → 重点读 DRQA + DEER + PURE
- **想做 Agent RL** → 重点读 DeepSWE + AgentPRM + R³L


## 本节小结

2025—2026 年，Agentic-RL 领域经历了从"对齐辅助工具"到"核心能力激发引擎"的根本转变。几个关键趋势值得关注：

1. **RL 从辅助到核心**：RL 不再仅用于"对齐"，而是用于**激发预训练中潜在的推理能力**
2. **算法从复杂到实用**：从 PPO 的四模型架构到 GRPO 的两模型架构，再到 VAPO 的价值增强方案，训练越来越高效
3. **奖励从外部到内部**：从人工标注到可验证奖励再到模型内部信号，奖励设计越来越自洽
4. **关注从"更强"到"更高效"**：过度思考问题催生了一系列推理效率优化方案

这些进展正在让 **"让模型通过实践自主学习"** 这一愿景逐步成为现实。


## 参考文献

以下是本文涉及的主要参考文献：

**基础理论**
- SUTTON R S, BARTO A G. *Reinforcement Learning: An Introduction*. 2nd ed. Cambridge: MIT Press, 2018.
- WILLIAMS R J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 1992.

**核心算法**
- SCHULMAN J, et al. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.
- SCHULMAN J, et al. High-dimensional continuous control using generalized advantage estimation. ICLR, 2016.
- RAFAILOV R, et al. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 2023.
- SHAO Z, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300, 2024.

**参数高效训练**
- HU E J, et al. LoRA: Low-rank adaptation of large language models. ICLR, 2022.
- DETTMERS T, et al. QLoRA: Efficient finetuning of quantized language models. NeurIPS, 2023.
- AGHAJANYAN A, et al. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. ACL, 2021.

**对齐与微调**
- OUYANG L, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022.
- ZHOU C, et al. LIMA: Less is more for alignment. NeurIPS, 2023.
- BRADLEY R A, TERRY M E. Rank analysis of incomplete block designs. Biometrika, 1952.

**推理模型**
- DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948, 2025.
- DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models. 2025.
- Moonshot AI. Kimi k1.5: Scaling reinforcement learning with LLMs. 2025.
- Alibaba. QwQ: Reflect and question to understand the world. 2025.
- OPENAI. Learning to reason with LLMs. 2024.

**RL 算法改进**
- ByteDance Seed. DAPO: An open-source LLM reinforcement learning system at scale. 2025.
- ByteDance Seed. VAPO: Efficient and reliable RL framework for advanced reasoning tasks. 2025.

**奖励设计与推理效率**
- UIUC & AWS. Self-Aligned Reward (SAR): Towards effective and efficient reasoners. 2025.
- Co-rewarding: Self-supervised RL for LLM reasoning. ICLR 2026.
- CoRLHF: Reinforcement learning from human feedback with cooperative policy-reward optimization. 2026.
- PURE: Stop summation: Min-form credit assignment is all process reward model needs for reasoning. 2025.
- DRQA: Dynamic reasoning quota allocation for controlling overthinking. 2026.
- DEER: Dynamic early exit in reasoning models. 2026.

**Agent 与评估**
- XI Z, et al. The rise and potential of large language model based agents: A survey. arXiv:2309.07864, 2023.
- YANG A, et al. Qwen2.5 technical report. arXiv:2412.15115, 2024.
- TOUVRON H, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288, 2023.
- TUNSTALL L, et al. Zephyr: Direct distillation of LM alignment. arXiv:2310.16944, 2023.
- SKALSE J, et al. Defining and characterizing reward hacking. NeurIPS, 2022.
- COBBE K, et al. Training verifiers to solve math word problems (GSM8K). arXiv:2110.14168, 2021.
