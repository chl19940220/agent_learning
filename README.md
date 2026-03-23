<div align="center">

# 🤖 从零开始学 Agent 

**一本系统、全面、实战导向的 AI Agent 开发教程**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=social)](https://github.com/Haozhe-Xing/agent_learning)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Haozhe-Xing/agent_learning/pulls)
[![mdBook](https://img.shields.io/badge/built%20with-mdBook-blue)](https://rust-lang.github.io/mdBook/)

[📖 在线阅读](https://Haozhe-Xing.github.io/agent_learning) · [🐛 提交问题](https://github.com/Haozhe-Xing/agent_learning/issues) · [💬 参与讨论](https://github.com/Haozhe-Xing/agent_learning/discussions)

</div>

---

## 📌 为什么写这本书？

AI Agent 正在重塑软件开发的边界。从 GitHub Copilot 到 Devin，从 AutoGPT 到 Claude，**会构建 Agent 的工程师正在成为最稀缺的技术人才**。

然而，现有的学习资源要么过于零散，要么停留在理论层面，缺乏一条从入门到生产的完整路径。

这本书的目标只有一个：**让你真正能构建出可用的 AI Agent 系统**。

> 📚 本书已构建为在线电子书，支持全文搜索、暗色模式、KaTeX 数学公式渲染，可直接在浏览器中阅读。

---

## ✨ 本书特色

- 🎯 **循序渐进**：从 LLM 基础到多 Agent 系统，每章都有清晰的知识脉络
- 💻 **代码优先**：每个核心概念都配有可运行的 Python 代码示例
- 🎨 **图文并茂**：120+ 手绘 SVG 架构图 / 流程图 / 时序图，直观理解复杂概念
- 🎬 **交互动画**：内置 5 个交互式 HTML 动画（感知-思考-行动循环、ReAct 推理、Function Calling、RAG 流程、GRPO 采样）
- 🔬 **论文解读**：关键章节附有前沿论文精读（ReAct、Reflexion、MemGPT、GRPO 等），帮你跟上学术最新进展
- 🏗️ **完整项目**：3 个综合实战项目（AI 编程助手、智能数据分析 Agent、多模态 Agent）
- 🛡️ **生产就绪**：涵盖安全、评估、部署等生产环境必备知识
- 🧪 **前沿技术**：涵盖上下文工程、Agentic-RL（GRPO/DPO/PPO）、MCP/A2A/ANP 等 2025—2026 最新进展
- 📐 **公式支持**：使用 KaTeX 渲染数学公式，强化学习章节可清晰阅读策略梯度、KL 散度等公式推导
- 🔄 **持续更新**：跟踪 LangChain、LangGraph、MCP 等框架的最新变化

---

## 📚 内容目录

<details>
<summary><b>第一部分：入门篇（第 1—3 章）</b></summary>

| 章节 | 内容 |
|------|------|
| **第 1 章** | **什么是 Agent？** |
| | 1.1 从聊天机器人到智能体的演进 |
| | 1.2 Agent 的核心概念与定义 |
| | 1.3 Agent 架构：感知-思考-行动循环 |
| | 1.4 Agent 与传统程序的区别 |
| | 1.5 Agent 的应用场景全景图 |
| | 1.6 智能体发展史：从符号主义到大模型驱动 |
| **第 2 章** | **开发环境搭建** |
| | 2.1 Python 环境与依赖管理 |
| | 2.2 关键库安装 |
| | 2.3 API Key 管理与安全最佳实践 |
| | 2.4 第一个 Agent：Hello Agent！ |
| **第 3 章** | **大语言模型基础** |
| | 3.1 LLM 是如何工作的？（直觉理解） |
| | 3.2 Prompt Engineering |
| | 3.3 Few-shot / Zero-shot / Chain-of-Thought 提示策略 |
| | 3.4 模型 API 调用入门 |
| | 3.5 Token、Temperature 与模型参数详解 |

</details>

<details>
<summary><b>第二部分：核心能力篇（第 4—10 章）</b></summary>

| 章节 | 内容 |
|------|------|
| **第 4 章** | **工具调用（Tool Use / Function Calling）** |
| | 4.1 为什么 Agent 需要工具？ |
| | 4.2 Function Calling 机制 |
| | 4.3 自定义工具的设计与实现 |
| | 4.4 工具描述的编写技巧 |
| | 4.5 实战：搜索引擎 + 计算器 Agent |
| | 4.6 论文解读：工具学习前沿进展 |
| **第 5 章** | **记忆系统（Memory）** |
| | 5.1 为什么 Agent 需要记忆？ |
| | 5.2 短期记忆：对话历史管理 |
| | 5.3 长期记忆：向量数据库与检索 |
| | 5.4 工作记忆：Scratchpad 模式 |
| | 5.5 实战：带记忆的个人助理 Agent |
| | 5.6 论文解读：记忆系统前沿进展 |
| **第 6 章** | **规划与推理（Planning & Reasoning）** |
| | 6.1 Agent 如何"思考"？ |
| | 6.2 ReAct：推理 + 行动框架 |
| | 6.3 任务分解：将复杂问题拆解为子任务 |
| | 6.4 反思与自我纠错机制 |
| | 6.5 实战：自动化研究助手 Agent |
| | 6.6 论文解读：规划与推理前沿研究 |
| **第 7 章** | **检索增强生成（RAG）** |
| | 7.1 RAG 的概念与工作原理 |
| | 7.2 文档加载与文本分割 |
| | 7.3 向量嵌入与向量数据库 |
| | 7.4 检索策略与重排序 |
| | 7.5 实战：智能文档问答 Agent |
| | 7.6 论文解读：RAG 前沿进展 |
| **第 8 章** | **上下文工程** |
| | 8.1 从提示工程到上下文工程 |
| | 8.2 上下文窗口管理与注意力预算 |
| | 8.3 长时程任务的上下文策略 |
| | 8.4 实战：构建上下文管理器 |
| **第 9 章** | **Skill System** |
| | 9.1 技能系统概述 |
| | 9.2 技能的定义与封装 |
| | 9.3 技能学习与获取 |
| | 9.4 技能发现与注册 |
| | 9.5 实战：构建可复用的技能系统 |
| | 9.6 论文解读：技能系统前沿研究 |
| **第 10 章** | **Agentic-RL：智能体强化学习训练** |
| | 10.1 什么是 Agentic-RL |
| | 10.2 SFT + LoRA 基础训练 |
| | 10.3 PPO：近端策略优化 |
| | 10.4 DPO：直接偏好优化 |
| | 10.5 GRPO：组内相对策略优化与奖励函数设计 |
| | 10.6 实战：完整 SFT + GRPO 训练 Pipeline |
| | 10.7 最新研究进展（2025—2026） |

</details>

<details>
<summary><b>第三部分：框架实战篇（第 11—13 章）</b></summary>

| 章节 | 内容 |
|------|------|
| **第 11 章** | **LangChain 深入实战** |
| | 11.1 LangChain 架构全景 |
| | 11.2 Chain：构建处理管道 |
| | 11.3 使用 LangChain 构建 Agent |
| | 11.4 LCEL：LangChain 表达式语言 |
| | 11.5 实战：多功能客服 Agent |
| **第 12 章** | **LangGraph：构建有状态的 Agent** |
| | 12.1 为什么需要图结构？ |
| | 12.2 LangGraph 核心概念：节点、边、状态 |
| | 12.3 构建你的第一个 Graph Agent |
| | 12.4 条件路由与循环控制 |
| | 12.5 Human-in-the-Loop：人机协作 |
| | 12.6 实战：工作流自动化 Agent |
| **第 13 章** | **其他主流框架概览** |
| | 13.1 AutoGPT 与 BabyAGI 的启示 |
| | 13.2 CrewAI：角色扮演型多 Agent 框架 |
| | 13.3 AutoGen：多 Agent 对话框架 |
| | 13.4 Dify / Coze 等低代码 Agent 平台 |
| | 13.5 如何选择合适的框架？ |

</details>

<details>
<summary><b>第四部分：多 Agent 系统篇（第 14—15 章）</b></summary>

| 章节 | 内容 |
|------|------|
| **第 14 章** | **多 Agent 协作** |
| | 14.1 单 Agent 的局限性 |
| | 14.2 多 Agent 通信模式 |
| | 14.3 角色分工与任务分配 |
| | 14.4 Supervisor 模式 vs. 去中心化模式 |
| | 14.5 实战：多 Agent 软件开发团队 |
| | 14.6 论文解读：多 Agent 系统前沿研究 |
| **第 15 章** | **Agent 通信协议** |
| | 15.1 MCP（Model Context Protocol）详解 |
| | 15.2 A2A（Agent-to-Agent）协议 |
| | 15.3 ANP（Agent Network Protocol）协议 |
| | 15.4 Agent 间的消息传递与状态共享 |
| | 15.5 实战：基于 MCP 的工具集成 |

</details>

<details>
<summary><b>第五部分：生产化篇（第 16—18 章）</b></summary>

| 章节 | 内容 |
|------|------|
| **第 16 章** | **Agent 的评估与优化** |
| | 16.1 如何评估 Agent 的表现？ |
| | 16.2 基准测试与评估指标（BFCL / GAIA / AgentBench / SWE-bench） |
| | 16.3 Prompt 调优策略 |
| | 16.4 成本控制与性能优化 |
| | 16.5 可观测性：日志、追踪与监控 |
| **第 17 章** | **安全与可靠性** |
| | 17.1 Prompt 注入攻击与防御 |
| | 17.2 幻觉问题与事实性保障 |
| | 17.3 权限控制与沙箱隔离 |
| | 17.4 敏感数据保护 |
| | 17.5 Agent 行为的可控性与对齐 |
| | 17.6 论文解读：安全与可靠性前沿研究 |
| **第 18 章** | **部署与生产化** |
| | 18.1 Agent 应用的部署架构 |
| | 18.2 API 服务化：FastAPI / Flask 封装 |
| | 18.3 容器化与云部署 |
| | 18.4 流式响应与并发处理 |
| | 18.5 实战：部署一个生产级 Agent 服务 |

</details>

<details>
<summary><b>第六部分：综合项目篇（第 19—21 章）</b></summary>

| 章节 | 内容 |
|------|------|
| **第 19 章** | 🔨 **项目实战：AI 编程助手** |
| | 19.1 项目架构设计 |
| | 19.2 代码理解与分析能力 |
| | 19.3 代码生成与修改能力 |
| | 19.4 测试生成与 Bug 修复 |
| | 19.5 完整项目实现 |
| **第 20 章** | 📊 **项目实战：智能数据分析 Agent** |
| | 20.1 需求分析与架构设计 |
| | 20.2 数据连接与查询 |
| | 20.3 自动化分析与可视化 |
| | 20.4 报告生成与导出 |
| | 20.5 完整项目实现 |
| **第 21 章** | 🎨 **项目实战：多模态 Agent** |
| | 21.1 多模态能力概述 |
| | 21.2 图像理解与生成 |
| | 21.3 语音交互集成 |
| | 21.4 实战：多模态个人助理 |

</details>

<details>
<summary><b>附录</b></summary>

| 附录 | 内容 |
|------|------|
| 附录 A | 常用 Prompt 模板大全 |
| 附录 B | Agent 开发常见问题 FAQ |
| 附录 C | 推荐学习资源与社区 |
| 附录 D | 术语表 |
| 附录 E | KL 散度详解 |

</details>

---

## 🗺️ 学习路线图

```
入门篇                 核心能力篇               框架实战篇             多Agent & 生产化         综合项目
─────────            ─────────────           ─────────            ─────────────          ─────────
第1章  Agent概念  →   第4章  工具调用    →   第11章 LangChain  →  第14章 多Agent协作  →  第19章 AI编程助手
第2章  环境搭建   →   第5章  记忆系统    →   第12章 LangGraph  →  第15章 通信协议     →  第20章 数据分析Agent
第3章  LLM基础    →   第6章  规划推理    →   第13章 框架选型   →  第16章 评估优化     →  第21章 多模态Agent
                      第7章  RAG                                 第17章 安全可靠
                      第8章  上下文工程                           第18章 部署生产
                      第9章  技能系统
                      第10章 Agentic-RL
```

---

## 📂 项目结构

```
agent_learning/
├── README.md                   # 项目说明（你正在看的文件）
├── book.toml                   # mdBook 配置文件
├── .github/workflows/
│   └── deploy.yml              # GitHub Pages 自动部署
├── theme/
│   ├── custom.css              # 自定义主题样式
│   └── head.hbs                # 自定义 HTML head 模板
├── src/                        # 📖 书籍源文件目录
│   ├── SUMMARY.md              # 目录结构定义
│   ├── preface.md              # 前言
│   ├── part1~6.md              # 各部分引言页
│   ├── chapter_intro/          # 第1章：什么是 Agent？
│   ├── chapter_setup/          # 第2章：开发环境搭建
│   ├── chapter_llm/            # 第3章：大语言模型基础
│   ├── chapter_tools/          # 第4章：工具调用
│   ├── chapter_memory/         # 第5章：记忆系统
│   ├── chapter_planning/       # 第6章：规划与推理
│   ├── chapter_rag/            # 第7章：检索增强生成 (RAG)
│   ├── chapter_context_engineering/  # 第8章：上下文工程
│   ├── chapter_skill/          # 第9章：Skill System
│   ├── chapter_agentic_rl/     # 第10章：Agentic-RL
│   ├── chapter_langchain/      # 第11章：LangChain 实战
│   ├── chapter_langgraph/      # 第12章：LangGraph
│   ├── chapter_frameworks/     # 第13章：主流框架概览
│   ├── chapter_multi_agent/    # 第14章：多 Agent 协作
│   ├── chapter_protocol/       # 第15章：Agent 通信协议
│   ├── chapter_evaluation/     # 第16章：评估与优化
│   ├── chapter_security/       # 第17章：安全与可靠性
│   ├── chapter_deployment/     # 第18章：部署与生产化
│   ├── chapter_coding_agent/   # 第19章：AI 编程助手
│   ├── chapter_data_agent/     # 第20章：智能数据分析 Agent
│   ├── chapter_multimodal/     # 第21章：多模态 Agent
│   ├── appendix/               # 附录 A~E
│   ├── svg/                    # 120+ SVG 插图资源
│   └── animations/             # 交互式 HTML 动画
└── book/                       # 构建输出目录（自动生成）
```

---

## 🚀 快速开始

### 在线阅读（推荐）

直接访问 👉 **[https://Haozhe-Xing.github.io/agent_learning](https://Haozhe-Xing.github.io/agent_learning)**

### 本地构建

```bash
# 安装 mdBook
cargo install mdbook

# 安装 mdbook-katex 插件（用于数学公式渲染）
cargo install mdbook-katex

# 克隆仓库
git clone https://github.com/Haozhe-Xing/agent_learning.git
cd agent_learning

# 本地预览（自动打开浏览器）
mdbook serve --open
```

### 环境准备（跟随代码实践）

```bash
# Python 3.11+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装核心依赖
pip install langchain langchain-openai langgraph openai anthropic

# 配置 API Key
export OPENAI_API_KEY="your-key-here"
```

---

## 🔥 核心知识点速览

<table>
<tr>
<td width="50%">

**🧠 Agent 核心架构**
- 感知 → 思考 → 行动循环
- ReAct 推理框架
- 任务分解与规划
- 反思与自我纠错

**🛠️ 工具与技能**
- Function Calling 机制
- 自定义工具设计
- 技能系统构建
- 工具描述最佳实践

</td>
<td width="50%">

**💾 记忆、知识与上下文**
- 短期 / 长期 / 工作记忆
- 向量数据库（Chroma / FAISS）
- RAG 检索增强生成
- 上下文工程与注意力预算

**🤝 多 Agent 协作 & 训练**
- MCP / A2A / ANP 协议
- Supervisor 模式
- CrewAI / AutoGen
- Agentic-RL（SFT → PPO → DPO → GRPO）

</td>
</tr>
</table>

---

## 📊 涵盖技术栈

![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic_(Claude)-191919?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Chroma](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat)
![mdBook](https://img.shields.io/badge/mdBook-000000?style=flat&logo=rust&logoColor=white)
![KaTeX](https://img.shields.io/badge/KaTeX-44CC11?style=flat)

---

## 🤝 参与贡献

欢迎任何形式的贡献！

- 🐛 **发现错误**：[提交 Issue](https://github.com/Haozhe-Xing/agent_learning/issues)
- 💡 **内容建议**：[发起 Discussion](https://github.com/Haozhe-Xing/agent_learning/discussions)
- 📝 **改进内容**：Fork → 修改 → 提交 PR
- ⭐ **支持项目**：给本仓库点个 Star！

### 贡献指南

```bash
# Fork 并克隆
git clone https://github.com/YOUR_USERNAME/agent_learning.git  # 替换为你的用户名

# 创建特性分支
git checkout -b feature/improve-chapter-4

# 本地预览
mdbook serve --open

# 提交修改
git commit -m "feat: 改进第4章工具调用示例代码"

# 推送并创建 PR
git push origin feature/improve-chapter-4
```

### 内容组织约定

- 每章内容放在独立目录 `src/chapter_xxx/` 下
- 章节概述放在 `README.md`，各小节按 `01_xxx.md`、`02_xxx.md` 编号
- SVG 插图统一放在 `src/svg/` 目录，命名格式 `chapter_xxx_描述.svg`
- 交互动画放在 `src/animations/` 目录

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐，这是对作者最大的鼓励！

---

<div align="center">

**用 ❤️ 构建，为了让每个开发者都能掌握 AI Agent 开发**

[⬆ 回到顶部](#-从零开始学-agent-开发)

</div>