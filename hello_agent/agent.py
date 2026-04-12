# agent.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import calculator, get_current_time, search_wikipedia, remember_note
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()
client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.environ.get("DASHSCOPE_API_KEY")
)

# 工具注册表：将 Python 函数映射到 OpenAI 工具格式
TOOLS_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_wikipedia": search_wikipedia,
    "remember_note": remember_note,
}

# OpenAI Function Calling 格式的工具定义
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持基本运算和数学函数（sqrt, sin, cos等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区，默认 Asia/Shanghai",
                        "default": "Asia/Shanghai"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "在维基百科搜索信息，适合查询百科知识",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_note",
            "description": "保存重要信息为笔记",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "笔记内容"
                    },
                    "title": {
                        "type": "string",
                        "description": "笔记标题",
                        "default": "未命名"
                    }
                },
                "required": ["content"]
            }
        }
    }
]


class HelloAgent:
    """
    第一个 Agent：Hello Agent！
    具备工具使用、多轮对话、推理能力。
    """
    
    def __init__(self, model: str = "qwen3.5-flash"):
        self.model = model if model else os.environ.get("DEFAULT_MODEL")
        self.messages = [
            {
                "role": "system",
                "content": """你是一个智能助手，可以使用多种工具来帮助用户。

你可以使用以下工具：
- calculator：计算数学问题
- get_current_time：获取当前时间
- search_wikipedia：查询百科知识
- remember_note：保存重要信息

使用工具时，先分析用户需求，选择合适的工具，执行后给出清晰的回答。
如果不需要工具，直接回答即可。请用中文回复。"""
            }
        ]
    
    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行工具调用"""
        tool_func = TOOLS_REGISTRY.get(tool_name)
        if not tool_func:
            return f"错误：未知工具 '{tool_name}'"
        
        try:
            result = tool_func(**tool_args)
            return str(result)
        except Exception as e:
            return f"工具执行失败：{str(e)}"
    
    def chat(self, user_message: str) -> str:
        """
        与 Agent 对话
        实现了完整的 ReAct 循环：Reason → Act → Observe
        """
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_message})
        
        console.print(f"\n[bold blue]用户：[/bold blue]{user_message}")
        
        # Agent 循环（最多10步防止无限循环）
        max_iterations = 10
        for iteration in range(max_iterations):
            
            # 调用 LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS_DEFINITION,
                tool_choice="auto"  # 让模型自己决定是否使用工具
            )
            
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            # 将模型回复加入历史
            self.messages.append(message)
            
            # 如果模型决定直接回答（不使用工具）
            if finish_reason == "stop":
                console.print(f"[bold green]Agent：[/bold green]{message.content}")
                return message.content
            
            # 如果模型决定使用工具
            if finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # 显示工具调用（调试信息）
                    console.print(
                        Panel(
                            f"工具：[yellow]{tool_name}[/yellow]\n"
                            f"参数：{tool_args}",
                            title="🔧 工具调用",
                            border_style="yellow"
                        )
                    )
                    
                    # 执行工具
                    result = self._execute_tool(tool_name, tool_args)
                    
                    console.print(f"[dim]工具结果：{result}[/dim]")
                    
                    # 将工具结果加入历史
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            
        return "抱歉，处理超时，请重试。"
    
    def reset(self):
        """重置对话历史"""
        self.messages = self.messages[:1]  # 只保留 system prompt
        console.print("[dim]对话已重置[/dim]")
