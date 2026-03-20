"""
将 agentic_rl_zhihu.md 转换为知乎适配版本
主要处理：
1. 图片链接替换为 GitHub 上可访问的 URL
2. 在图片位置添加清晰的占位提示（方便手动上传）
3. 公式格式适配（知乎兼容）
4. 清理不必要的分隔线
5. 标题层级适配
"""

import re
import os

INPUT_FILE = "agentic_rl_zhihu.md"
OUTPUT_FILE = "agentic_rl_zhihu_ready.md"

# GitHub Pages 上的 SVG URL 基础路径
GITHUB_PAGES_SVG_BASE = "https://haozhe-xing.github.io/agent_learning/svg"

# 图片文件名 -> 图片描述 的映射（方便手动上传时识别）
IMAGE_DESCRIPTIONS = {
    "chapter_agentic_rl_01_overview.png": "图 1：Agentic-RL 训练架构概览（从 Prompt 方式到 SFT+GRPO 两阶段训练）",
    "chapter_agentic_rl_02_sft_grpo.png": "图 2：SFT → RL 两阶段训练流程",
    "chapter_agentic_rl_03_three_algorithms.png": "图 3：PPO / DPO / GRPO 三大策略优化算法架构对比",
    "chapter_agentic_rl_03_ppo_clip.png": "图 4：PPO Clip 机制图解",
    "chapter_agentic_rl_03_ppo_training_flow.png": "图 5：PPO 完整训练迭代流程",
    "chapter_agentic_rl_03_ppo_architecture.png": "图 6：PPO 训练架构（四个模型的协作关系）",
    "chapter_agentic_rl_03_dpo_intuition.png": "图 7：DPO 核心直觉（偏好学习示意）",
    "chapter_agentic_rl_03_dpo_architecture.png": "图 8：DPO 训练架构",
    "chapter_agentic_rl_03_grpo_architecture.png": "图 9：GRPO 训练架构",
    "chapter_agentic_rl_03_grpo_iteration.png": "图 10：GRPO 单次训练迭代流程",
}


def convert_images(content: str) -> str:
    """
    将本地图片引用替换为：
    1. 一个指向 GitHub Pages SVG 的链接（方便读者点击查看高清图）
    2. 清晰的上传提示占位符
    """
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        filename = os.path.basename(img_path)
        svg_filename = filename.replace(".png", ".svg")
        svg_url = f"{GITHUB_PAGES_SVG_BASE}/{svg_filename}"
        
        description = IMAGE_DESCRIPTIONS.get(filename, alt_text)
        
        # 知乎不支持外链图片的 ![alt](url) 语法引用外部图片
        # 所以我们生成一个清晰的占位块 + 在线查看链接
        replacement = (
            f"\n"
            f"> **📷 {description}**\n"
            f"> \n"
            f"> 👉 [点击查看高清大图]({svg_url})\n"
            f"> \n"
            f"> ⚠️ **发布时请在此处手动上传对应图片** `images/{filename}`\n"
            f"\n"
        )
        return replacement
    
    # 匹配 ![alt](images/xxx.png) 格式
    pattern = r'!\[(.*?)\]\((images/[^)]+)\)'
    return re.sub(pattern, replace_image, content)


def adapt_formulas(content: str) -> str:
    """
    适配知乎的公式格式。
    知乎支持 $...$ 行内公式和 $$...$$ 块公式。
    主要处理：
    1. 确保块公式前后有空行（知乎要求）
    2. 表格内的 $ 公式保持不变（知乎表格内公式支持有限，但基本可用）
    """
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 确保 $$ 块公式前后有空行
        if line.strip().startswith('$$') and not line.strip().startswith('$$$'):
            # 如果前一行不是空行，加空行
            if result and result[-1].strip() != '':
                result.append('')
            result.append(line)
            # 如果这行只有 $$（开始标记），找到结束的 $$
            if line.strip() == '$$':
                i += 1
                while i < len(lines):
                    result.append(lines[i])
                    if lines[i].strip() == '$$':
                        break
                    i += 1
            # 确保块公式后有空行
            if i + 1 < len(lines) and lines[i + 1].strip() != '':
                result.append('')
        else:
            result.append(line)
        i += 1
    return '\n'.join(result)


def clean_horizontal_rules(content: str) -> str:
    """移除多余的水平分隔线（知乎中过多分隔线影响阅读体验）"""
    # 连续的分隔线合并为一个
    content = re.sub(r'(\n---\n){2,}', '\n---\n', content)
    return content


def adapt_headers(content: str) -> str:
    """
    知乎文章的标题最多支持到 h3（###）。
    将 h4-h6 转为加粗文本。
    """
    lines = content.split('\n')
    result = []
    for line in lines:
        # #### 标题 -> **标题**
        match = re.match(r'^(#{4,6})\s+(.+)$', line)
        if match:
            title_text = match.group(2)
            result.append(f'\n**{title_text}**\n')
        else:
            result.append(line)
    return '\n'.join(result)


def add_zhihu_header(content: str) -> str:
    """在文章开头添加知乎发布说明"""
    header = """<!--
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

"""
    return header + content


def remove_toc_hr(content: str) -> str:
    """移除目录前后的 --- 分隔线"""
    content = content.replace('\n---\n\n# ', '\n\n# ')
    return content


def main():
    # 读取原始文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"原始文件: {len(content)} 字符, {content.count(chr(10))} 行")
    
    # 1. 替换图片
    content = convert_images(content)
    print("✅ 图片链接已替换为占位符 + 在线查看链接")
    
    # 2. 适配公式格式
    content = adapt_formulas(content)
    print("✅ 公式格式已适配")
    
    # 3. 清理分隔线
    content = clean_horizontal_rules(content)
    content = remove_toc_hr(content)
    print("✅ 多余分隔线已清理")
    
    # 4. 适配标题层级
    content = adapt_headers(content)
    print("✅ 标题层级已适配（h4+ 转为加粗）")
    
    # 5. 添加发布说明
    content = add_zhihu_header(content)
    print("✅ 已添加发布说明")
    
    # 写入输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📝 知乎适配版已生成: {OUTPUT_FILE}")
    print(f"   文件大小: {len(content)} 字符, {content.count(chr(10))} 行")
    
    # 统计转换结果
    img_count = content.count('📷')
    formula_block_count = content.count('$$') // 2
    print(f"   图片占位符: {img_count} 处")
    print(f"   块公式: ~{formula_block_count} 处")


if __name__ == "__main__":
    main()
