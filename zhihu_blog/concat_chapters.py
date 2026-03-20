"""
将 chapter_agentic_rl 目录下的所有 markdown 文件原封不动拼接到一起。
唯一的修改：将 SVG 图片引用替换为对应的 PNG 路径。
"""

import os
import re

# 章节文件顺序
chapter_files = [
    "README.md",
    "01_agentic_rl_overview.md",
    "02_sft_lora.md",
    "03_ppo.md",
    "04_dpo.md",
    "05_grpo.md",
    "06_practice_training.md",
    "07_latest_research.md",
]

src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "chapter_agentic_rl")
output_file = os.path.join(os.path.dirname(__file__), "agentic_rl_zhihu.md")

parts = []
for fname in chapter_files:
    fpath = os.path.join(src_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 唯一的修改：将 SVG 图片引用替换为 PNG，并调整路径
    # 原始路径格式: ../svg/chapter_agentic_rl_xxx.svg
    # 替换为:       images/chapter_agentic_rl_xxx.png
    content = re.sub(
        r'\.\./svg/(chapter_agentic_rl_[^)]+)\.svg',
        r'images/\1.png',
        content
    )
    
    parts.append(content)

# 用分隔线拼接
merged = "\n\n---\n\n".join(parts)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(merged)

print(f"✅ 已将 {len(chapter_files)} 个文件拼接到 {output_file}")
print(f"   总字符数: {len(merged):,}")
