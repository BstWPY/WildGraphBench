#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
替换已抓取的 Wiki 文章为纯 Markdown 格式

功能:
  1. 扫描指定目录下的所有 Wiki 文章 Markdown 文件
  2. 使用 Jina API (X-Return-Format: markdown) 重新抓取
  3. 保留原有的 reference/ 目录结构不变

使用示例:
  # 单个目录
  python tools/recrawl_wiki.py \
    --wiki-dir ./corpus/culture/Marvel_Cinematic_Universe

  # 批量处理 CSV 中的所有目录
  python tools/recrawl_wiki.py \
    --csv ./Top_references.csv \
    --root-dir ./corpus

  # 批量处理指定分类
  python tools/recrawl_wiki.py \
    --root-dir ./corpus \
    --category culture
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse, unquote

import requests


DEFAULT_API_KEY = os.environ.get('JINA_API_KEY', '')


def normalize_category(category: str) -> str:
    """标准化分类名称 (与 run_ref_scraper.sh 保持一致)"""
    return (
        category
        .replace(' ', '_')
        .replace('&', '_and_')
        .replace(',', '_')
        .lower()
    )


def url_to_slug_title(url: str) -> str:
    """从 Wiki URL 推导标题字符串"""
    path = urlparse(url).path
    last = path.rsplit('/', 1)[-1]
    last = unquote(last)
    title = last.replace('_', ' ')
    return title or 'page'


def slugify(text: str) -> str:
    """简单的 slugify 实现 (与 jina_scraping 保持一致)"""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def fetch_wiki_markdown(url: str, api_key: str, max_retries: int = 3) -> Optional[str]:
    """使用 Jina API 抓取 Wiki 文章的 Markdown 格式
    
    Args:
        url: Wiki 文章 URL
        api_key: Jina API Key
        max_retries: 最大重试次数
        
    Returns:
        Markdown 内容字符串,失败返回 None
    """
    if not api_key.startswith('Bearer '):
        api_key = f'Bearer {api_key}'
    
    headers = {
        'Authorization': api_key,
        'X-Return-Format': 'markdown',  # 关键: 只返回 markdown
    }
    
    jina_url = f'https://r.jina.ai/{url}'
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  🔄 尝试抓取 (第 {attempt}/{max_retries} 次)...")
            
            response = requests.get(
                jina_url,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.text.strip()
                # ★ 在这里先把 "* - Wikipedia" 顶部标题干掉
                content = strip_wikipedia_title_header(content)

                if content and len(content) > 100:  # 基本内容检查
                    print(f"  ✅ 抓取成功: {len(content)} 字符 (清洗后)")
                    return content
                else:
                    print(f"  ⚠️  返回内容过短(清洗后): {len(content)} 字符")
                        
        except requests.exceptions.Timeout:
            print(f"  ⏱️  请求超时")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        
        if attempt < max_retries:
            wait_time = attempt * 5
            print(f"  ⏳ 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    return None


def find_wiki_markdown(wiki_dir: Path) -> Optional[Path]:
    """在目录中查找 Wiki 文章的 Markdown 文件
    
    查找规则:
      1. 文件名不包含 'reference'
      2. 文件扩展名为 .md
      3. 不在 reference/ 子目录下
    
    Returns:
        Wiki Markdown 文件路径,未找到返回 None
    """
    if not wiki_dir.exists() or not wiki_dir.is_dir():
        return None
    
    for md_file in wiki_dir.glob('*.md'):
        # 排除 reference 相关文件
        if 'reference' in md_file.name.lower():
            continue
        
        # 确保不在 reference 子目录
        if 'reference' in str(md_file.relative_to(wiki_dir)).lower():
            continue
        
        return md_file
    
    return None


def extract_original_url(md_content: str) -> Optional[str]:
    """从 Markdown 内容中提取原始 Wiki URL
    
    尝试从以下位置提取:
      1. Markdown metadata 中的 URL 字段
      2. 文件开头的注释
      3. Title 推导 (最后手段)
    """
    # 方法1: 查找 metadata
    url_match = re.search(r'URL Source:\s*(https?://[^\s\)]+)', md_content, re.IGNORECASE)
    if url_match:
        return url_match.group(1).strip()
    
    # 方法2: 查找注释
    comment_match = re.search(r'<!--.*?(https://en\.wikipedia\.org/wiki/[^\s\)]+).*?-->', md_content, re.DOTALL)
    if comment_match:
        return comment_match.group(1).strip()
    
    # 方法3: 从标题推导 (不够准确,但作为备选)
    title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
        # 简单转换为 Wiki URL 格式
        wiki_slug = title.replace(' ', '_')
        return f"https://en.wikipedia.org/wiki/{wiki_slug}"
    
    return None


def replace_wiki_article(
    wiki_dir: Path,
    api_key: str,
    force: bool = False,
    backup: bool = True
) -> bool:
    """替换单个 Wiki 目录中的文章
    
    Args:
        wiki_dir: Wiki 目录路径
        api_key: Jina API Key
        force: 是否强制替换 (忽略备份检查)
        backup: 是否备份原文件
        
    Returns:
        是否成功替换
    """
    print(f"\n📂 处理目录: {wiki_dir}")
    
    # 查找 Wiki Markdown 文件
    md_file = find_wiki_markdown(wiki_dir)
    if not md_file:
        print("  ⚠️  未找到 Wiki Markdown 文件")
        return False
    
    print(f"  📄 找到文件: {md_file.name}")
    
    # 读取原内容提取 URL
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return False
    
    original_url = extract_original_url(original_content)
    if not original_url:
        print("  ⚠️  无法从文件中提取原始 URL,跳过")
        return False
    
    print(f"  🔗 原始 URL: {original_url}")
    
    # 检查是否已有备份
    backup_file = md_file.with_suffix('.md.bak')
    if backup_file.exists() and not force:
        print("  ℹ️  备份文件已存在,跳过 (使用 --force 强制替换)")
        return False
    
    # 抓取新内容
    new_content = fetch_wiki_markdown(original_url, api_key)
    if not new_content:
        print("  ❌ 抓取失败")
        return False
    
    # 备份原文件
    if backup:
        try:
            backup_file.write_text(original_content, encoding='utf-8')
            print(f"  💾 备份原文件: {backup_file.name}")
        except Exception as e:
            print(f"  ⚠️  备份失败: {e}")
    
    # 写入新内容
    try:
        md_file.write_text(new_content, encoding='utf-8')
        print(f"  ✅ 替换成功: {len(new_content)} 字符")
        return True
    except Exception as e:
        print(f"  ❌ 写入失败: {e}")
        # 尝试恢复备份
        if backup and backup_file.exists():
            try:
                md_file.write_text(original_content, encoding='utf-8')
                print("  🔄 已从备份恢复")
            except:
                pass
        return False


def process_csv(
    csv_path: Path,
    root_dir: Path,
    api_key: str,
    start_row: int = 2,
    end_row: Optional[int] = None,
    force: bool = False,
    backup: bool = True
) -> Dict[str, int]:
    """批量处理 CSV 中的 Wiki 目录
    
    Returns:
        统计信息字典 {'total': 总数, 'success': 成功数, 'failed': 失败数, 'skipped': 跳过数}
    """
    stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if end_row is None:
        end_row = len(rows) + 1
    
    print(f"📋 CSV 文件: {csv_path}")
    print(f"📊 处理范围: 第 {start_row} 到 {end_row} 行")
    
    for idx, row in enumerate(rows, start=2):  # CSV 第1行是标题,从第2行开始
        if idx < start_row or idx > end_row:
            continue
        
        stats['total'] += 1
        
        category = row.get('Category', '').strip()
        title = row.get('Title', '').strip()
        url = row.get('URL', '').strip()
        
        if not url:
            print(f"\n[{idx}] ⚠️  URL 为空,跳过")
            stats['skipped'] += 1
            continue
        
        # 构建目录路径
        category_dir = normalize_category(category)
        
        # ★ 修改：直接使用 Title (保留空格)
        # 实际目录名如: "Authoritarian socialism", "Steam service"
        wiki_dir = root_dir / category_dir / title
        
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(rows) + 1}] 处理: {title}")
        print(f"{'='*60}")
        print(f"  🗂️  目录: {wiki_dir}")
        
        if not wiki_dir.exists():
            print(f"  ⚠️  目录不存在")
            stats['skipped'] += 1
            continue
        
        if replace_wiki_article(wiki_dir, api_key, force, backup):
            stats['success'] += 1
        else:
            stats['failed'] += 1
        
        # 避免请求过快
        time.sleep(2)
    
    return stats


def process_category(
    root_dir: Path,
    category: str,
    api_key: str,
    force: bool = False,
    backup: bool = True
) -> Dict[str, int]:
    """批量处理指定分类下的所有 Wiki 目录"""
    stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    category_dir = root_dir / normalize_category(category)
    if not category_dir.exists():
        print(f"❌ 分类目录不存在: {category_dir}")
        return stats
    
    print(f"📂 处理分类: {category}")
    print(f"📁 目录: {category_dir}")
    
    # 遍历所有子目录
    for wiki_dir in sorted(category_dir.iterdir()):
        if not wiki_dir.is_dir():
            continue
        
        # 跳过 reference 相关目录
        if 'reference' in wiki_dir.name.lower():
            continue
        
        stats['total'] += 1
        
        if replace_wiki_article(wiki_dir, api_key, force, backup):
            stats['success'] += 1
        else:
            stats['failed'] += 1
        
        time.sleep(2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='替换已抓取的 Wiki 文章为纯 Markdown 格式'
    )
    
    # 输入模式选择
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--wiki-dir',
        type=Path,
        help='单个 Wiki 目录路径'
    )
    input_group.add_argument(
        '--csv',
        type=Path,
        help='CSV 文件路径 (批量处理)'
    )
    input_group.add_argument(
        '--category',
        type=str,
        help='分类名称 (需配合 --root-dir)'
    )
    
    # 通用参数
    parser.add_argument(
        '--root-dir',
        type=Path,
        default=Path('./corpus'),
        help='根目录 (用于 CSV/分类模式)'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='Jina API Key (or set JINA_API_KEY environment variable)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=2,
        help='CSV 起始行 (1-based, 默认2跳过标题)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='CSV 结束行 (1-based, 包含)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制替换 (忽略已存在的备份)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份原文件'
    )
    
    args = parser.parse_args()
    
    # API Key
    api_key = args.api_key or os.environ.get('JINA_API_KEY') or DEFAULT_API_KEY
    
    print("🚀 Wiki Markdown 替换工具")
    print(f"🔑 API Key: {api_key[:20]}...")
    print(f"💾 备份: {'否' if args.no_backup else '是'}")
    print(f"🔄 强制: {'是' if args.force else '否'}")
    
    # 单目录模式
    if args.wiki_dir:
        success = replace_wiki_article(
            args.wiki_dir,
            api_key,
            force=args.force,
            backup=not args.no_backup
        )
        print(f"\n{'='*60}")
        print(f"✅ 完成" if success else "❌ 失败")
        return
    
    # CSV 批量模式
    if args.csv:
        stats = process_csv(
            args.csv,
            args.root_dir,
            api_key,
            start_row=args.start,
            end_row=args.end,
            force=args.force,
            backup=not args.no_backup
        )
    
    # 分类批量模式
    elif args.category:
        stats = process_category(
            args.root_dir,
            args.category,
            api_key,
            force=args.force,
            backup=not args.no_backup
        )
    
    else:
        parser.error("需要指定 --wiki-dir, --csv 或 --category")
        return
    
    # 打印统计
    print(f"\n{'='*60}")
    print("📊 处理统计")
    print(f"{'='*60}")
    print(f"  总计: {stats['total']}")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  ⏭️  跳过: {stats['skipped']}")


if __name__ == '__main__':
    main()