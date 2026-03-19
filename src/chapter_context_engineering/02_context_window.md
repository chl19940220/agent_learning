# 上下文窗口管理与注意力预算

> 📖 *"上下文窗口就像一张办公桌——桌面大小固定，你能同时摊开的文件数量有限。关键不是桌子有多大，而是你选择摊开哪些文件。"*

## 理解上下文窗口

### 什么是上下文窗口？

上下文窗口（Context Window）是 LLM 在一次推理中能够"看到"的**最大信息量**，以 token 数量衡量。不同模型的窗口大小差异显著：

| 模型 | 上下文窗口 | 约等于 | 典型场景 |
|------|-----------|--------|---------|
| GPT-3.5 | 16K tokens | ~12,000 汉字 | 简单对话 |
| GPT-4o | 128K tokens | ~96,000 汉字 | 复杂 Agent 任务 |
| Claude 3.5 Sonnet | 200K tokens | ~150,000 汉字 | 长文档分析 |
| Gemini 1.5 Pro | 1M tokens | ~750,000 汉字 | 超长上下文 |

```python
# 上下文窗口的基本约束
def check_context_budget(
    system_prompt_tokens: int,
    conversation_tokens: int,
    tool_result_tokens: int,
    retrieved_docs_tokens: int,
    max_output_tokens: int,
    model_context_window: int = 128000,
) -> dict:
    """
    检查上下文预算
    注意：输出 tokens 也要占用上下文窗口！
    """
    total_input = (
        system_prompt_tokens 
        + conversation_tokens 
        + tool_result_tokens 
        + retrieved_docs_tokens
    )
    
    available_for_output = model_context_window - total_input
    
    return {
        "total_input_tokens": total_input,
        "available_for_output": available_for_output,
        "is_within_budget": available_for_output >= max_output_tokens,
        "utilization": total_input / model_context_window * 100,
    }

# 典型 Agent 场景
budget = check_context_budget(
    system_prompt_tokens=800,
    conversation_tokens=15000,   # 15 轮对话
    tool_result_tokens=20000,    # 多次工具调用结果
    retrieved_docs_tokens=5000,  # RAG 检索结果
    max_output_tokens=4096,
    model_context_window=128000,
)
print(f"输入占用: {budget['total_input_tokens']} tokens")
print(f"利用率: {budget['utilization']:.1f}%")
print(f"剩余输出空间: {budget['available_for_output']} tokens")
```

## 上下文腐蚀（Context Corruption）

### 什么是上下文腐蚀？

**上下文腐蚀**是指随着 Agent 执行任务的推进，上下文中逐渐积累了大量低质量、冗余或过时的信息，导致 LLM 的推理质量持续下降的现象 [1]。

这就像一间不断堆积杂物的房间——最终你在里面什么都找不到了。

```python
# 上下文腐蚀的典型场景

def demonstrate_context_corruption():
    """演示上下文腐蚀是如何发生的"""
    
    context_tokens = 0
    useful_ratio = 1.0  # 有用信息占比
    
    # 模拟 Agent 执行 30 轮交互
    for turn in range(1, 31):
        # 每轮新增内容
        new_user_msg = 100       # 用户消息
        new_thought = 200        # Agent 思考过程
        new_tool_call = 50       # 工具调用请求
        new_tool_result = 500    # 工具返回结果（往往很冗长）
        new_assistant_msg = 300  # Agent 回复
        
        turn_tokens = (new_user_msg + new_thought + 
                       new_tool_call + new_tool_result + 
                       new_assistant_msg)
        context_tokens += turn_tokens
        
        # 随着轮次增加，早期信息的相关性递减
        useful_ratio *= 0.95  # 每轮有效信息比例下降 5%
        
        if turn % 10 == 0:
            print(f"第 {turn} 轮:")
            print(f"  上下文大小: {context_tokens:,} tokens")
            print(f"  有效信息占比: {useful_ratio:.1%}")
            print(f"  噪音信息: {context_tokens * (1 - useful_ratio):,.0f} tokens")
    
    # 第10轮: ~11,500 tokens, 有效 60%
    # 第20轮: ~23,000 tokens, 有效 36%
    # 第30轮: ~34,500 tokens, 有效 21% ← 大量噪音！

demonstrate_context_corruption()
```

### 上下文腐蚀的三种表现

| 症状 | 表现 | 原因 |
|------|------|------|
| **遗忘** | Agent 忘记了之前讨论过的关键信息 | 重要信息被大量新信息"冲刷"到注意力盲区 |
| **重复** | Agent 重复执行已经完成的步骤 | 任务状态在上下文中不够突出 |
| **偏题** | Agent 的行为偏离了原始目标 | 早期的任务目标被后续信息淹没 |

## 注意力预算

### 什么是注意力预算？

虽然现代 LLM 的上下文窗口越来越大（128K、200K 甚至 1M tokens），但研究表明，**LLM 对上下文中不同位置信息的注意力并不均匀** [2]。

这就是所谓的 **Lost in the Middle** 效应：LLM 更关注上下文的**开头**和**结尾**，中间部分的信息容易被忽略。

```python
# 注意力分布示意（简化）

def attention_distribution(context_length: int) -> list[float]:
    """
    模拟 LLM 对不同位置的注意力分布
    U 形曲线：开头和结尾注意力高，中间低
    """
    import math
    
    attentions = []
    for i in range(context_length):
        # 归一化位置 [0, 1]
        pos = i / context_length
        
        # U 形注意力曲线
        # 开头区域（前 10%）：高注意力
        # 中间区域（10%-90%）：低注意力
        # 结尾区域（后 10%）：最高注意力
        if pos < 0.1:
            attention = 0.8 + 0.2 * (1 - pos / 0.1)
        elif pos > 0.9:
            attention = 0.8 + 0.2 * ((pos - 0.9) / 0.1)
        else:
            attention = 0.3 + 0.2 * math.sin(math.pi * pos)
        
        attentions.append(attention)
    
    return attentions

# 实际启示：
# ① System Prompt 放在最前面 → 注意力最稳定
# ② 最新的用户消息和工具结果放在最后 → 注意力最强
# ③ 中间的历史对话 → 最容易被忽略 → 需要精心管理
```

### 注意力预算分配策略

基于 "Lost in the Middle" 效应，我们可以制定以下注意力预算分配策略：

```python
from dataclasses import dataclass

@dataclass
class AttentionBudget:
    """注意力预算分配"""
    
    total_tokens: int = 128000
    
    # 预算分配（根据注意力优先级）
    system_prompt_budget: float = 0.01     # 1% → 系统指令（开头，高注意力）
    task_context_budget: float = 0.05      # 5% → 当前任务上下文（开头区域）
    output_reserve: float = 0.05           # 5% → 预留给输出
    recent_context_budget: float = 0.20    # 20% → 最近几轮对话（结尾，高注意力）
    tool_results_budget: float = 0.30      # 30% → 工具调用结果
    history_budget: float = 0.25           # 25% → 历史对话（中间，低注意力 → 需压缩）
    knowledge_budget: float = 0.14         # 14% → 检索知识
    
    def get_token_allocation(self) -> dict:
        """计算各部分的 token 配额"""
        return {
            "system_prompt": int(self.total_tokens * self.system_prompt_budget),
            "task_context": int(self.total_tokens * self.task_context_budget),
            "output_reserve": int(self.total_tokens * self.output_reserve),
            "recent_context": int(self.total_tokens * self.recent_context_budget),
            "tool_results": int(self.total_tokens * self.tool_results_budget),
            "history": int(self.total_tokens * self.history_budget),
            "knowledge": int(self.total_tokens * self.knowledge_budget),
        }

budget = AttentionBudget()
allocation = budget.get_token_allocation()
for name, tokens in allocation.items():
    print(f"  {name}: {tokens:,} tokens")
```

## 上下文管理的核心技术

### 技术1：滑动窗口（Sliding Window）

最简单的策略——只保留最近 N 轮对话：

```python
def sliding_window(messages: list[dict], max_turns: int = 10) -> list[dict]:
    """
    滑动窗口：只保留最近 N 轮对话
    
    优点：简单高效
    缺点：丢失早期重要信息
    """
    # 始终保留 system message
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    
    # 只保留最近 N 轮（每轮 = user + assistant）
    recent_msgs = other_msgs[-(max_turns * 2):]
    
    return system_msgs + recent_msgs
```

### 技术2：摘要压缩（Summary Compression）

定期将历史对话压缩为摘要：

```python
from openai import OpenAI

client = OpenAI()

def compress_history(
    messages: list[dict], 
    keep_recent: int = 5
) -> list[dict]:
    """
    对话历史压缩：将老的对话总结为摘要
    
    优点：保留关键信息，大幅减少 token
    缺点：会丢失细节
    """
    if len(messages) <= keep_recent * 2:
        return messages  # 不需要压缩
    
    # 分离出需要压缩的老消息和需要保留的新消息
    old_messages = messages[:-(keep_recent * 2)]
    recent_messages = messages[-(keep_recent * 2):]
    
    # 用 LLM 生成摘要
    summary_prompt = f"""请将以下对话历史压缩为一段简洁的摘要，保留：
1. 用户的核心需求和目标
2. 已经完成的关键步骤
3. 重要的中间结果和数据
4. 尚未解决的问题

对话历史：
{format_messages(old_messages)}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 用小模型做摘要
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=500,
    )
    
    summary = response.choices[0].message.content
    
    # 将摘要作为系统消息插入
    summary_message = {
        "role": "system",
        "content": f"[对话历史摘要] {summary}"
    }
    
    return [summary_message] + recent_messages
```

### 技术3：语义过滤（Semantic Filtering）

根据与当前任务的语义相关性来筛选上下文内容：

```python
import numpy as np

def semantic_filter(
    messages: list[dict],
    current_query: str,
    embedding_model,
    top_k: int = 10,
    always_keep_last: int = 3,
) -> list[dict]:
    """
    语义过滤：根据相关性选择保留的历史消息
    
    优点：保留最相关的信息
    缺点：需要额外的 embedding 计算
    """
    # 对当前查询进行 embedding
    query_embedding = embedding_model.encode(current_query)
    
    # 计算每条历史消息与当前查询的相似度
    scored_messages = []
    for i, msg in enumerate(messages):
        msg_embedding = embedding_model.encode(msg["content"])
        similarity = np.dot(query_embedding, msg_embedding)
        scored_messages.append((i, similarity, msg))
    
    # 按相似度排序，选择 top_k
    scored_messages.sort(key=lambda x: x[1], reverse=True)
    selected_indices = set()
    
    # 始终保留最后几条消息
    for i in range(max(0, len(messages) - always_keep_last), len(messages)):
        selected_indices.add(i)
    
    # 添加语义最相关的消息
    for idx, score, msg in scored_messages[:top_k]:
        selected_indices.add(idx)
    
    # 按原始顺序返回
    result = [messages[i] for i in sorted(selected_indices)]
    return result
```

## 本节小结

| 概念 | 说明 |
|------|------|
| **上下文窗口** | LLM 单次推理能看到的最大信息量 |
| **上下文腐蚀** | 低质量信息积累导致推理质量下降 |
| **Lost in the Middle** | LLM 对中间位置信息的注意力较弱 |
| **注意力预算** | 按信息优先级分配上下文空间 |
| **管理技术** | 滑动窗口、摘要压缩、语义过滤 |

## 🤔 思考练习

1. 如果你的 Agent 需要处理一个跨越 100 轮对话的任务，你会如何管理上下文？
2. "Lost in the Middle" 效应对你设计 Agent 的 prompt 结构有什么启示？
3. 摘要压缩和语义过滤各适合什么场景？能否结合使用？

---

## 参考文献

[1] ANTHROPIC. Building effective agents[EB/OL]. 2024. https://www.anthropic.com/engineering/building-effective-agents.

[2] LIU N F, LIN K, HEWITT J, et al. Lost in the middle: How language models use long contexts[J]. Transactions of the ACL, 2024, 12: 157-173.
