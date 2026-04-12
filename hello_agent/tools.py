# tools.py
import math
import datetime
import os
import sys
from urllib.parse import quote

import requests
from typing import Annotated, cast

from dotenv import load_dotenv

load_dotenv()


def _requests_proxies() -> dict[str, str] | None:
    """从环境变量读取 HTTP(S) 代理，供 requests 使用。"""
    all_p = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if all_p:
        return {"http": all_p, "https": all_p}
    if https or http:
        p = cast(str, https or http)
        return {"http": p, "https": p}
    return None


def _wikipedia_user_agent() -> str:
    """
    维基媒体 API 要求可识别的 User-Agent 与联系方式，见：
    https://meta.wikimedia.org/wiki/User-Agent_policy
    可在 .env 中设置 WIKIPEDIA_CONTACT，例如：mailto:you@example.com 或 https://你的站点
    """
    contact = (os.environ.get("WIKIPEDIA_CONTACT") or "").strip()
    if not contact:
        contact = "set WIKIPEDIA_CONTACT in .env per https://meta.wikimedia.org/wiki/User-Agent_policy"
    return (
        f"agent-learning/0.1.0 ({contact}) "
        f"Python/{sys.version_info.major}.{sys.version_info.minor} "
        f"requests/{requests.__version__}"
    )


def calculator(expression: Annotated[str, "数学表达式，如 '2 + 3 * 4'"]) -> str:
    """
    计算数学表达式。
    支持基本运算(+,-,*,/)和数学函数(sqrt, pow, sin, cos等)。
    """
    try:
        # 安全地评估数学表达式
        # 只允许数学操作，防止代码注入
        allowed_names = {
            'sqrt': math.sqrt,
            'pow': math.pow,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'pi': math.pi,
            'e': math.e,
            'abs': abs,
            'round': round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def get_current_time(
    timezone: Annotated[str, "时区名称，如 'Asia/Shanghai' 或 'UTC'"] = "Asia/Shanghai"
) -> str:
    """获取当前时间"""
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.datetime.now(tz)
    except (KeyError, Exception):
        # 如果时区无效，使用本地时间
        now = datetime.datetime.now()
    return f"当前时间（{timezone}）：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"


def search_wikipedia(
    query: Annotated[str, "要搜索的关键词"]
) -> str:
    """
    在维基百科中搜索信息。
    适合查询历史事件、人物、地理、科学概念等。
    """
    try:
        # 使用维基百科 API（须带 User-Agent，见 meta.wikimedia.org/wiki/User-Agent_policy）
        url = "https://zh.wikipedia.org/api/rest_v1/page/summary/" + quote(query, safe="")
        proxies = _requests_proxies()
        headers = {"User-Agent": _wikipedia_user_agent(), "Accept": "application/json"}
        response = requests.get(url, timeout=5, proxies=proxies, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return f"维基百科 - {data.get('title', query)}：\n{data.get('extract', '未找到摘要')[:500]}"
        if response.status_code == 404:
            return f"未找到关于 '{query}' 的维基百科页面"
        return f"维基百科请求失败（HTTP {response.status_code}）：{response.text[:300]}"
    except Exception as e:
        return f"搜索失败：{str(e)}"


def remember_note(
    content: Annotated[str, "要记录的笔记内容"],
    title: Annotated[str, "笔记标题"] = "未命名"
) -> str:
    """将信息保存为笔记，以便后续使用"""
    import json
    import os
    
    notes_file = "agent_notes.json"
    
    # 读取已有笔记
    if os.path.exists(notes_file):
        with open(notes_file, 'r', encoding='utf-8') as f:
            notes = json.load(f)
    else:
        notes = []
    
    # 添加新笔记
    note = {
        "title": title,
        "content": content,
        "time": datetime.datetime.now().isoformat()
    }
    notes.append(note)
    
    # 保存
    with open(notes_file, 'w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)
    
    return f"✅ 已保存笔记：《{title}》"
