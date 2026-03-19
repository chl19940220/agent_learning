# 长时程任务的上下文策略

> 📖 *"短对话靠 prompt 技巧，长任务靠上下文策略——当 Agent 需要工作几小时甚至几天时，上下文管理就是生死线。"*

## 什么是长时程任务？

长时程任务（Long-horizon Tasks）是指需要 Agent 执行**数十到数百轮交互**才能完成的复杂任务。例如：

- 编写一个完整的软件项目（可能需要 100+ 轮工具调用）
- 进行深度研究并撰写报告（需要搜索、阅读、分析、写作多个阶段）
- 执行多步骤数据分析流水线

这类任务面临的核心挑战是：**上下文在任务执行过程中持续膨胀，但上下文窗口是固定的**。

```python
# 长时程任务的上下文膨胀示例

def simulate_long_task():
    """模拟一个需要 50 轮交互的编程任务"""
    context_size = 1000  # 初始上下文（system prompt + 任务描述）
    
    for step in range(50):
        # 每轮交互新增的 tokens
        context_size += 100   # Agent 的思考过程
        context_size += 50    # 工具调用请求
        context_size += 800   # 工具返回结果（如文件内容、搜索结果）
        context_size += 200   # Agent 的回复
        
        if context_size > 128000:
            print(f"⚠️ 第 {step} 轮就超出了 128K 窗口！")
            break
    else:
        print(f"最终上下文大小: {context_size:,} tokens")

simulate_long_task()
# 输出: ⚠️ 第 110 轮... 但实际上更早就会因为注意力稀释而出问题
```

## 三大应对策略

### 策略一：压缩整合（Compaction）

核心思想：**定期将冗长的上下文压缩为精炼的摘要**，释放空间给新信息。

```python
from openai import OpenAI
from dataclasses import dataclass, field

client = OpenAI()

@dataclass
class CompactionStrategy:
    """压缩整合策略"""
    
    messages: list[dict] = field(default_factory=list)
    compaction_threshold: int = 50000  # 超过 50K tokens 触发压缩
    keep_recent_turns: int = 5         # 保留最近 5 轮不压缩
    summaries: list[str] = field(default_factory=list)
    
    def add_turn(self, user_msg: str, assistant_msg: str, 
                 tool_results: list[str] = None):
        """添加一轮对话"""
        self.messages.append({"role": "user", "content": user_msg})
        if tool_results:
            for result in tool_results:
                self.messages.append({"role": "tool", "content": result})
        self.messages.append({"role": "assistant", "content": assistant_msg})
        
        # 检查是否需要压缩
        if self._estimate_tokens() > self.compaction_threshold:
            self._compact()
    
    def _compact(self):
        """执行压缩"""
        # 分离出需要压缩的旧消息
        keep_count = self.keep_recent_turns * 3  # user + tool + assistant
        old_messages = self.messages[:-keep_count]
        recent_messages = self.messages[-keep_count:]
        
        # 生成摘要
        summary = self._generate_summary(old_messages)
        self.summaries.append(summary)
        
        # 用摘要替换旧消息
        self.messages = recent_messages
        print(f"📦 压缩完成: {len(old_messages)} 条消息 → 1 条摘要")
    
    def _generate_summary(self, messages: list[dict]) -> str:
        """用 LLM 生成摘要"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""请将以下 Agent 交互历史压缩为结构化摘要：

{self._format_messages(messages)}

摘要要求：
1. 列出所有已完成的步骤和结果
2. 记录关键的中间数据和发现
3. 标注任何未解决的问题
4. 保留用户的原始需求
格式：使用 Markdown 列表"""
            }],
            max_tokens=800,
        )
        return response.choices[0].message.content
    
    def build_context(self) -> list[dict]:
        """构建最终的上下文"""
        context = []
        
        # 添加历史摘要
        if self.summaries:
            context.append({
                "role": "system",
                "content": "## 任务执行摘要\n\n" + "\n\n---\n\n".join(self.summaries)
            })
        
        # 添加最近的完整对话
        context.extend(self.messages)
        
        return context
    
    def _estimate_tokens(self) -> int:
        total = sum(len(m["content"]) // 4 for m in self.messages)
        return total
    
    def _format_messages(self, messages: list[dict]) -> str:
        return "\n".join(f"[{m['role']}]: {m['content'][:200]}" for m in messages)
```

### 策略二：结构化笔记（Structured Notes）

核心思想：**Agent 在执行过程中主动维护一份结构化的"笔记"**，记录关键信息和任务状态，而不是依赖完整的对话历史。

这个思路来源于 Anthropic 的 Claude Agent 设计——Agent 使用专门的 "NoteTool" 来记录和更新关键信息 [1]。

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentNotepad:
    """
    Agent 的结构化笔记本
    灵感来源：Anthropic Building Effective Agents 中的 NoteTool 概念
    """
    
    # 任务目标（不变）
    objective: str = ""
    
    # 执行计划（可更新）
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    
    # 关键发现（持续追加）
    findings: list[dict] = field(default_factory=list)
    
    # 待解决问题
    open_questions: list[str] = field(default_factory=list)
    
    # 重要数据点
    data_points: dict = field(default_factory=dict)
    
    def update_plan(self, new_plan: list[str]):
        """更新执行计划"""
        self.plan = new_plan
    
    def advance_step(self):
        """推进到下一步"""
        self.current_step += 1
    
    def add_finding(self, finding: str, source: str = ""):
        """记录一个发现"""
        self.findings.append({
            "content": finding,
            "source": source,
            "time": datetime.now().isoformat(),
        })
    
    def add_data_point(self, key: str, value):
        """记录一个重要数据"""
        self.data_points[key] = value
    
    def to_context_string(self) -> str:
        """将笔记序列化为上下文字符串"""
        lines = [
            f"## 任务目标\n{self.objective}",
            f"\n## 执行计划（当前: 步骤 {self.current_step + 1}/{len(self.plan)}）",
        ]
        
        for i, step in enumerate(self.plan):
            status = "✅" if i < self.current_step else ("🔄" if i == self.current_step else "⬜")
            lines.append(f"  {status} {i+1}. {step}")
        
        if self.findings:
            lines.append("\n## 关键发现")
            for f in self.findings[-5:]:  # 只显示最近 5 条
                lines.append(f"  - {f['content']}")
        
        if self.data_points:
            lines.append("\n## 重要数据")
            for k, v in self.data_points.items():
                lines.append(f"  - {k}: {v}")
        
        if self.open_questions:
            lines.append("\n## 待解决问题")
            for q in self.open_questions:
                lines.append(f"  - ❓ {q}")
        
        return "\n".join(lines)


# 使用示例
notepad = AgentNotepad(
    objective="分析 2025 年 Q1 用户留存率下降原因并提出改进方案",
    plan=[
        "查询 Q1 各月用户留存数据",
        "对比 Q4 vs Q1 的留存率变化",
        "分析不同用户群体的留存差异",
        "识别导致下降的关键因素",
        "提出改进方案并预估效果",
    ]
)

# Agent 执行过程中更新笔记
notepad.advance_step()
notepad.add_data_point("Q1 平均 7日留存", "38%")
notepad.add_data_point("Q4 平均 7日留存", "45%")
notepad.add_finding("新用户 7 日留存下降最显著（-12%）", source="SQL 查询")
notepad.advance_step()

print(notepad.to_context_string())
```

### 策略三：子代理架构（Sub-agent Architecture）

核心思想：**将复杂的长任务分解给多个子 Agent 执行，每个子 Agent 有自己独立的上下文窗口**。主 Agent 只需要管理任务进度和子 Agent 的结果摘要。

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class SubAgentResult:
    """子 Agent 的执行结果"""
    agent_name: str
    task: str
    result_summary: str  # 只传摘要，不传完整上下文
    success: bool
    key_data: dict

class OrchestratorAgent:
    """
    编排者 Agent：将长任务分解给子 Agent
    
    关键优势：
    1. 每个子 Agent 都有独立的、干净的上下文
    2. 主 Agent 只需管理摘要级别的信息
    3. 天然支持并行执行
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.sub_agents: dict[str, Callable] = {}
        self.results: list[SubAgentResult] = []
    
    def register_sub_agent(self, name: str, handler: Callable):
        """注册子 Agent"""
        self.sub_agents[name] = handler
    
    def execute_plan(self, task: str, plan: list[dict]):
        """按计划调度子 Agent 执行"""
        for step in plan:
            agent_name = step["agent"]
            sub_task = step["task"]
            
            print(f"📋 分配任务给 [{agent_name}]: {sub_task}")
            
            # 子 Agent 在独立的上下文中执行
            # 只传入必要的信息，不传整个对话历史
            result = self.sub_agents[agent_name](
                task=sub_task,
                context={
                    "overall_objective": task,
                    "previous_results": [
                        r.result_summary for r in self.results
                    ],
                }
            )
            
            self.results.append(result)
            print(f"✅ [{agent_name}] 完成: {result.result_summary[:100]}...")
    
    def get_final_context(self) -> str:
        """获取所有子 Agent 结果的摘要（用于最终总结）"""
        summaries = []
        for r in self.results:
            summaries.append(
                f"### {r.agent_name}: {r.task}\n"
                f"结果: {r.result_summary}\n"
                f"数据: {r.key_data}"
            )
        return "\n\n".join(summaries)


# 使用示例
orchestrator = OrchestratorAgent()

# 一个复杂的研究任务被分解为多个独立子任务
plan = [
    {"agent": "researcher", "task": "搜索2025年Agent领域的最新论文"},
    {"agent": "analyzer", "task": "分析搜索结果中的关键趋势"},
    {"agent": "writer", "task": "根据分析结果撰写研究报告"},
    {"agent": "reviewer", "task": "审核报告的准确性和完整性"},
]
```

## 三大策略对比

| 维度 | 压缩整合 | 结构化笔记 | 子代理架构 |
|------|---------|-----------|-----------|
| **适用场景** | 中等长度任务（20-50 轮） | 持续执行的任务 | 可分解的复杂任务 |
| **实现复杂度** | 低 | 中 | 高 |
| **信息保留** | 摘要级别 | 结构化关键信息 | 每个子任务完整 |
| **额外开销** | 摘要 LLM 调用 | 笔记维护 | 多 Agent 管理 |
| **最大优势** | 简单直接 | 信息精确可控 | 天然隔离，可并行 |
| **典型代表** | ChatGPT 对话压缩 | Claude Agent 的内部笔记 | Devin 的多模块架构 |

## 实践建议：组合使用

在实际项目中，这三种策略往往需要**组合使用**：

```python
class ProductionContextManager:
    """生产级上下文管理器：组合三种策略"""
    
    def __init__(self):
        self.notepad = AgentNotepad()          # 策略2：结构化笔记
        self.compactor = CompactionStrategy()   # 策略1：压缩整合
        self.sub_results: list[str] = []        # 策略3：子 Agent 结果
    
    def build_context(self, system_prompt: str, current_query: str) -> list[dict]:
        """构建最终上下文"""
        messages = []
        
        # 1. System Prompt（始终在最前面）
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. 结构化笔记（紧跟 system prompt，高注意力区域）
        messages.append({
            "role": "system",
            "content": self.notepad.to_context_string()
        })
        
        # 3. 子 Agent 结果摘要
        if self.sub_results:
            messages.append({
                "role": "system",
                "content": "## 子任务结果\n" + "\n".join(self.sub_results)
            })
        
        # 4. 压缩后的对话历史
        messages.extend(self.compactor.build_context())
        
        # 5. 当前用户查询（在最后，最高注意力区域）
        messages.append({"role": "user", "content": current_query})
        
        return messages
```

## 本节小结

| 策略 | 核心思想 | 最适合 |
|------|---------|--------|
| **压缩整合** | 定期摘要，释放空间 | 持续对话场景 |
| **结构化笔记** | 主动记录，精确控制 | 多步骤分析任务 |
| **子代理架构** | 分而治之，独立上下文 | 可并行的复杂任务 |

## 🤔 思考练习

1. 如果你要构建一个能执行"深度研究"的 Agent（需要搜索 50+ 网页并撰写报告），你会如何组合这三种策略？
2. 压缩整合在什么情况下可能造成关键信息丢失？如何缓解？
3. 子代理架构中，主 Agent 和子 Agent 之间应该传递多少信息？传多了和传少了分别有什么问题？

---

## 参考文献

[1] ANTHROPIC. Building effective agents[EB/OL]. 2024. https://www.anthropic.com/engineering/building-effective-agents.
