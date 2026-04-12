import chromadb
from openai import OpenAI
import json
import datetime
import os
from typing import Optional

client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.environ.get("DASHSCOPE_API_KEY")
)
# ============================
# 向量嵌入工具
# ============================

def get_embedding(text: str, model: str = "text-embedding-v4") -> list[float]:
    """获取文本的向量嵌入"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 测试：语义相似的文本有相近的向量
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(v1, v2) -> float:
    """计算余弦相似度"""
    return dot(v1, v2) / (norm(v1) * norm(v2))

# 验证语义相似性
texts = [
    "Python 是一种编程语言",  # 原始句
    "Python 是用于编程的语言",  # 语义相似
    "今天天气很好",            # 语义不相关
]

class LongTermMemory:
    """
    基于 ChromaDB 的长期记忆系统
    支持：添加记忆、语义检索、更新、删除
    """
    
    def __init__(self, user_id: str, persist_dir: str = "./memory_db"):
        self.user_id = user_id
        
        # 创建持久化向量数据库
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # 每个用户有独立的 Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"user_{user_id}_memory",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        print(f"[记忆系统] 加载用户 {user_id} 的记忆库，"
              f"已有 {self.collection.count()} 条记忆")
    
    def _embed(self, text: str) -> list[float]:
        """生成文本嵌入"""
        response = client.embeddings.create(
            input=text,
            model="text-embedding-v4"
        )
        return response.data[0].embedding
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        source: str = "conversation"
    ) -> str:
        """
        添加一条记忆
        
        Args:
            content: 记忆内容
            memory_type: 类型（preference/fact/event/task）
            importance: 重要性（1-10）
            source: 来源
        
        Returns:
            记忆ID
        """
        import uuid
        
        memory_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[{
                "type": memory_type,
                "importance": importance,
                "source": source,
                "created_at": datetime.datetime.now().isoformat(),
                "user_id": self.user_id
            }]
        )
        
        print(f"[记忆] 已保存：{content[:50]}...")
        return memory_id
    
    def search_memories(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: int = 1
    ) -> list[dict]:
        """
        语义搜索记忆
        
        Args:
            query: 查询内容（自然语言）
            n_results: 返回结果数量
            memory_type: 过滤类型
            min_importance: 最低重要性阈值
        
        Returns:
            相关记忆列表，按相似度排序
        """
        query_embedding = self._embed(query)
        
        # 构建过滤条件
        where = {"user_id": self.user_id}
        if memory_type:
            where["type"] = memory_type
        if min_importance > 1:
            where["importance"] = {"$gte": min_importance}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            # 如果没有足够的结果
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                memories.append({
                    "content": doc,
                    "type": meta.get("type"),
                    "importance": meta.get("importance"),
                    "created_at": meta.get("created_at"),
                    "relevance": 1 - dist  # 转换为相似度（0-1）
                })
        
        return memories
    
    def get_all_memories(self, memory_type: Optional[str] = None) -> list[dict]:
        """获取所有记忆"""
        where = {"user_id": self.user_id}
        if memory_type:
            where["type"] = memory_type
        
        if self.collection.count() == 0:
            return []
        
        results = self.collection.get(
            where=where,
            include=["documents", "metadatas"]
        )
        
        memories = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            memories.append({
                "content": doc,
                "type": meta.get("type"),
                "importance": meta.get("importance"),
                "created_at": meta.get("created_at"),
            })
        
        # 按重要性排序
        return sorted(memories, key=lambda x: x.get("importance", 0), reverse=True)
    
    def format_for_prompt(self, memories: list[dict]) -> str:
        """将记忆格式化为 Prompt 可用的文本"""
        if not memories:
            return "无相关记忆"
        
        lines = ["【相关记忆】"]
        for m in memories:
            relevance = m.get("relevance", 0)
            if relevance >= 0.7:
                relevance_label = "高度相关"
            elif relevance >= 0.5:
                relevance_label = "相关"
            else:
                relevance_label = "参考"
            
            lines.append(f"[{m['type']} | {relevance_label}] {m['content']}")
        
        return "\n".join(lines)


class MemoryExtractor:
    """从对话中自动提取值得记忆的信息"""
    
    def __init__(self):
        self.client = client
    
    def extract_memories(self, user_message: str, agent_reply: str) -> list[dict]:
        """
        分析对话，提取值得记忆的信息
        
        Returns:
            记忆列表，每条包含 content、type、importance
        """
        prompt = f"""分析以下对话，提取值得长期记忆的重要信息。

用户说：{user_message}
助手回复：{agent_reply}

提取规则：
- 记录用户的个人信息、偏好、习惯
- 记录重要的决策和结论
- 记录正在进行的项目或任务
- 忽略闲聊、日常问候、重复信息
- 忽略临时性的查询（如"今天几号"）

返回 JSON 格式（若无值得记忆的内容则返回空列表）：
[
  {{
    "content": "记忆内容（简洁陈述句）",
    "type": "preference|fact|event|task|skill",
    "importance": 1-10的整数
  }}
]

只返回 JSON，不要其他内容。"""
        
        response = self.client.chat.completions.create(
            model="qwen3.5-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # 处理可能的嵌套格式
            if isinstance(result, dict):
                memories = result.get("memories", result.get("items", []))
            else:
                memories = result
            return memories if isinstance(memories, list) else []
        except:
            return []


# ============================
# 测试
# ============================

def test_memory_system():
    """测试记忆系统"""
    
    # 初始化
    memory = LongTermMemory(user_id="user_001")
    extractor = MemoryExtractor()
    
    # 添加一些记忆
    memory.add_memory("用户叫张伟，是一名 Python 后端工程师", "fact", importance=8)
    memory.add_memory("用户偏好简洁的代码风格，不喜欢过度注释", "preference", importance=7)
    memory.add_memory("用户正在开发一个基于 FastAPI 的微服务项目", "task", importance=8)
    memory.add_memory("用户擅长 Python 和 Go 语言", "skill", importance=6)
    
    print("\n=== 语义搜索测试 ===")
    
    # 测试1：搜索代码风格相关
    results = memory.search_memories("写代码的时候有什么注意事项？")
    print("\n查询：写代码的时候有什么注意事项？")
    for r in results[:3]:
        print(f"  [{r['relevance']:.2f}] {r['content']}")
    
    # 测试2：搜索项目相关
    results = memory.search_memories("用户在做什么项目？")
    print("\n查询：用户在做什么项目？")
    for r in results[:3]:
        print(f"  [{r['relevance']:.2f}] {r['content']}")
    
    print("\n=== 自动提取记忆测试 ===")
    
    # 测试从对话中提取记忆
    new_memories = extractor.extract_memories(
        user_message="我最近开始学习 Rust，觉得所有权系统很有意思",
        agent_reply="Rust 的所有权系统确实很独特，是保证内存安全的核心机制..."
    )
    
    print("从对话中提取的记忆：")
    for m in new_memories:
        print(f"  [{m['type']} | 重要性:{m['importance']}] {m['content']}")
        memory.add_memory(m['content'], m['type'], m['importance'])
    
    return memory

memory = test_memory_system()
