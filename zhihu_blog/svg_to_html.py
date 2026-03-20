#!/usr/bin/env python3
"""将每个 SVG 文件内联嵌入独立 HTML 文件中，方便浏览器截图转 PNG"""
import os, glob

svg_dir = '../src/svg'
html_dir = 'tmp_html'
os.makedirs(html_dir, exist_ok=True)

template = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{
    background: white;
    width: 100%;
    height: 100%;
  }}
  body {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 30px;
  }}
  .svg-wrap {{
    width: 100%;
    max-width: 1340px;
  }}
  .svg-wrap svg {{
    width: 100% !important;
    height: auto !important;
  }}
</style>
</head>
<body>
<div class="svg-wrap">
{svg_content}
</div>
</body>
</html>'''

for svg_path in sorted(glob.glob(os.path.join(svg_dir, 'chapter_agentic_rl_*.svg'))):
    name = os.path.splitext(os.path.basename(svg_path))[0]
    with open(svg_path, 'r') as f:
        svg_content = f.read()
    
    html = template.format(svg_content=svg_content)
    html_path = os.path.join(html_dir, f'{name}.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f'Created: {name}.html')
