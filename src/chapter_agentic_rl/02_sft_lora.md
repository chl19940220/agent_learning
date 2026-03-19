# 18.2 SFT + LoRA：监督微调与参数高效训练

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

![SFT → RL 两阶段训练流程](../svg/chapter_agentic_rl_02_sft_grpo.svg)

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

LIMA [1] 的研究提供了重要的实证依据：**1,000 条精心筛选的高质量数据，其效果往往优于 10,000 条噪声数据**。对于 Agent SFT，数据质量的评估维度如下：

| 质量维度 | 标准 | 验证方法 |
|---------|------|---------|
| **格式一致性** | 统一的 `<think>`/`<tool_call>` 标签格式 | 正则表达式自动检查 |
| **工具调用正确性** | 参数类型、名称与工具定义完全匹配 | 静态解析验证 |
| **推理连贯性** | `<think>` 内容与最终动作逻辑一致 | 人工抽样审查 |
| **任务覆盖度** | 覆盖所有工具的调用模式和边界情况 | 工具调用分布统计 |
| **难度分布** | 简单/中等/复杂任务比例均衡 | 人工分级标注 |

**推荐数据规模**：500–2,000 条经过人工验证的高质量 Agent 交互轨迹。

---

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

---

## LoRA：低秩适应的理论基础

### 核心假设与数学推导

**LoRA（Low-Rank Adaptation）** [2] 的理论基础来自一个关键的实证发现：

> **预训练模型在微调过程中，权重更新矩阵 $\Delta W$ 具有显著的低秩特性（intrinsic low rank）。**

**为什么微调的更新是低秩的？** 直觉上，预训练模型已经学会了丰富的通用表示，微调只需要在这个表示空间中做小幅度的“方向调整”。这种调整展开在一个低维子空间中，而非需要改变所有 $d \times k$ 个方向。Aghajanyan et al. [4] 通过实验证明，微调的“内在维度”（intrinsic dimensionality）远小于模型参数量。

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

---

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
# QLoRA [3] 的核心思想：将模型权重量化为 4-bit 存储，但计算时反量化为 bfloat16
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

---

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

---

*SFT 阶段让模型习得了 Agent 行为的基本格式与工具调用模式。然而，SFT 的能力上界受限于训练数据的质量——模型无法超越示范数据的水平。下一节介绍的 GRPO 算法将通过强化学习信号，引导模型探索出超越示范数据的最优策略。*

---

## 参考文献

[1] ZHOU C, LIU P, XU P, et al. LIMA: Less is more for alignment[C]//Advances in Neural Information Processing Systems (NeurIPS). 2023.

[2] HU E J, SHEN Y, WALLIS P, et al. LoRA: Low-rank adaptation of large language models[C]//International Conference on Learning Representations (ICLR). 2022.

[3] DETTMERS T, PAGNONI A, HOLTZMAN A, et al. QLoRA: Efficient finetuning of quantized language models[C]//Advances in Neural Information Processing Systems (NeurIPS). 2023.
