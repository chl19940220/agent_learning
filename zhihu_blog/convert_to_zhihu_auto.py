"""
将 agentic_rl_zhihu.md 转换为知乎适配版本（自动图片版）
假设 PNG 图片已推送到 GitHub，使用 raw.githubusercontent.com URL
知乎导入 Markdown 时会自动抓取这些外链图片
"""

import re
import os

INPUT_FILE = "agentic_rl_zhihu.md"
OUTPUT_FILE = "agentic_rl_zhihu_final.md"

# GitHub raw URL 基础路径（推送后生效）
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Haozhe-Xing/agent_learning/main/zhihu_blog/images"


def convert_images(content: str) -> str:
    """将本地图片路径替换为 GitHub raw URL"""
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        filename = os.path.basename(img_path)
        raw_url = f"{GITHUB_RAW_BASE}/{filename}"
        return f"![{alt_text}]({raw_url})"
    
    pattern = r'!\[(.*?)\]\((images/[^)]+)\)'
    return re.sub(pattern, replace_image, content)


def adapt_formulas(content: str) -> str:
    """确保块公式前后有空行"""
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('$$') and not line.strip().startswith('$$$'):
            if result and result[-1].strip() != '':
                result.append('')
            result.append(line)
            if line.strip() == '$$':
                i += 1
                while i < len(lines):
                    result.append(lines[i])
                    if lines[i].strip() == '$$':
                        break
                    i += 1
            if i + 1 < len(lines) and lines[i + 1].strip() != '':
                result.append('')
        else:
            result.append(line)
        i += 1
    return '\n'.join(result)


def clean_horizontal_rules(content: str) -> str:
    content = re.sub(r'(\n---\n){2,}', '\n---\n', content)
    return content


def remove_toc_hr(content: str) -> str:
    content = content.replace('\n---\n\n# ', '\n\n# ')
    return content


def adapt_headers(content: str) -> str:
    """h4-h6 转为加粗文本"""
    lines = content.split('\n')
    result = []
    for line in lines:
        match = re.match(r'^(#{4,6})\s+(.+)$', line)
        if match:
            title_text = match.group(2)
            result.append(f'\n**{title_text}**\n')
        else:
            result.append(line)
    return '\n'.join(result)


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"原始文件: {len(content)} 字符")
    
    content = convert_images(content)
    print("✅ 图片链接已替换为 GitHub raw URL")
    
    content = adapt_formulas(content)
    print("✅ 公式格式已适配")
    
    content = clean_horizontal_rules(content)
    content = remove_toc_hr(content)
    print("✅ 分隔线已清理")
    
    content = adapt_headers(content)
    print("✅ 标题层级已适配")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📝 知乎最终版已生成: {OUTPUT_FILE}")
    print(f"   文件大小: {len(content)} 字符")
    
    # 提示
    print("\n" + "=" * 60)
    print("📋 发布前请先将图片推送到 GitHub：")
    print("   cd /Users/haozhexing/workspace/ft_send/agent_learning")
    print("   git add zhihu_blog/images/")
    print("   git commit -m 'add zhihu blog images'")
    print("   git push")
    print("=" * 60)
    print("\n然后在知乎 → 写文章 → 导入文档 → 选择此文件即可")
    print("知乎会自动抓取 GitHub 上的图片，无需手动上传！")


if __name__ == "__main__":
    main()
