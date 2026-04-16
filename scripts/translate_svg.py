#!/usr/bin/env python3
"""
Batch-translate Chinese text in SVG files to English.

Strategy:
  1. Extract all unique Chinese-containing text nodes from the SVG.
  2. Send them as a JSON list to the LLM in one request.
  3. Replace text in the SVG with the translations.
  4. Update the font-family to include common English fonts first.

Usage:
  python3 scripts/translate_svg.py [--test N] [--file filename.svg]
  --test N   : only process first N files (default: all)
  --file     : translate a single file by name
"""

import os
import re
import sys
import json
import time
import argparse
import copy

# ── OpenAI-compatible client pointing at CodeBuddy proxy ──────────────────────
from openai import OpenAI

TOKEN = os.environ.get("CODEBUDDY_TOKEN", "")
client = OpenAI(
    api_key=TOKEN,
    base_url="https://api.codebuddy.cn/v1",
)

SVG_DIR = "/Users/haozhexing/workspace/ft_send/agent_learning/src/en/svg"

# Regex to find Chinese characters
ZH_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df]')

# Font-family replacement: add English-friendly fonts before Chinese ones
FONT_REPLACE = (
    "font-family: 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;",
    "font-family: 'Inter', 'Helvetica Neue', Arial, 'PingFang SC', sans-serif;",
)

def has_chinese(text: str) -> bool:
    return bool(ZH_RE.search(text))

def extract_text_nodes(svg_content: str) -> list[str]:
    """Return all unique text node values that contain Chinese."""
    # Match content between > and < that has Chinese
    pattern = re.compile(r'>([^<>]+)<', re.DOTALL)
    results = []
    seen = set()
    for m in pattern.finditer(svg_content):
        val = m.group(1).strip()
        if val and has_chinese(val) and val not in seen:
            seen.add(val)
            results.append(val)
    return results

def translate_batch(texts: list[str], filename: str) -> dict[str, str]:
    """Send a batch of Chinese strings to the LLM, return {zh: en} mapping."""
    if not texts:
        return {}

    # Build a numbered list for clarity
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    prompt = f"""You are translating UI labels in a technical SVG diagram about AI Agents ({filename}).
Return ONLY a JSON object mapping each original Chinese string to its English translation.
Rules:
- Keep technical terms (API, Token, RAG, LLM, Agent, etc.) as-is.
- Keep emoji as-is.
- Keep punctuation style (bullets •, numbers ①②③, arrows →) as-is.
- Be concise — these are diagram labels, not prose.
- Do NOT translate file paths, code identifiers, or pure-ASCII strings.

Strings to translate:
{numbered}

Respond with ONLY a JSON object like:
{{
  "中文1": "English 1",
  "中文2": "English 2"
}}"""

    resp = client.chat.completions.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content.strip()

    # Extract JSON block (in case the model wraps it in ```json ... ```)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        print(f"  ⚠ Could not parse JSON for {filename}: {raw[:200]}")
        return {}
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error for {filename}: {e}")
        return {}

def translate_svg_file(path: str) -> bool:
    """Translate one SVG file in-place. Returns True on success."""
    filename = os.path.basename(path)
    try:
        content = open(path, encoding="utf-8").read()
    except Exception as e:
        print(f"  ✗ Read error {filename}: {e}")
        return False

    if not has_chinese(content):
        print(f"  ✓ skip (no Chinese): {filename}")
        return True

    texts = extract_text_nodes(content)
    if not texts:
        print(f"  ✓ skip (no text nodes with Chinese): {filename}")
        return True

    print(f"  → translating {len(texts)} strings in {filename}...")

    mapping = translate_batch(texts, filename)
    if not mapping:
        print(f"  ✗ Empty translation for {filename}")
        return False

    # Replace text nodes: replace >zh_text< with >en_text<
    new_content = content
    for zh, en in mapping.items():
        if not en or en == zh:
            continue
        # Replace exact text node occurrences
        # Use a pattern that matches ">...zh...<" boundaries
        escaped_zh = re.escape(zh)
        new_content = re.sub(
            r'(>)' + escaped_zh + r'(<)',
            r'\g<1>' + en + r'\g<2>',
            new_content
        )

    # Update font-family CSS
    new_content = new_content.replace(FONT_REPLACE[0], FONT_REPLACE[1])
    # Also handle variations without leading space
    new_content = new_content.replace(
        "font-family: 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif",
        "font-family: 'Inter', 'Helvetica Neue', Arial, 'PingFang SC', sans-serif"
    )

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        print(f"  ✗ Write error {filename}: {e}")
        return False

    # Verify
    remaining = len(extract_text_nodes(new_content))
    print(f"  ✓ done: {filename} (remaining Chinese text nodes: {remaining})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                        help="Only translate first N files")
    parser.add_argument("--file", type=str, default="",
                        help="Translate a single file")
    parser.add_argument("--resume-from", type=str, default="",
                        help="Skip files alphabetically before this filename")
    args = parser.parse_args()

    if args.file:
        path = os.path.join(SVG_DIR, args.file)
        if not os.path.exists(path):
            path = args.file  # treat as absolute path
        translate_svg_file(path)
        return

    all_files = sorted(f for f in os.listdir(SVG_DIR) if f.endswith(".svg"))

    if args.resume_from:
        all_files = [f for f in all_files if f >= args.resume_from]
        print(f"Resuming from {args.resume_from}, {len(all_files)} files remaining")

    if args.test:
        all_files = all_files[:args.test]
        print(f"Test mode: processing first {args.test} files")

    total = len(all_files)
    ok = 0
    failed = []

    for i, fname in enumerate(all_files):
        print(f"[{i+1}/{total}] {fname}")
        path = os.path.join(SVG_DIR, fname)
        success = translate_svg_file(path)
        if success:
            ok += 1
        else:
            failed.append(fname)
        # Small delay to be nice to the API
        time.sleep(0.3)

    print(f"\n{'='*50}")
    print(f"Done: {ok}/{total} files translated successfully")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
