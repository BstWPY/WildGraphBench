#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convert Markdown to TXT - Batch Processing

"""
批量转换所有分类的 reference Markdown 文件为 TXT

功能:
  1. 遍历 dataset 下所有分类
  2. 将每个主题的 reference_pages/*.md 转换为 .txt
  3. 为每个主题生成 merged.txt（所有 reference 拼接）
"""

import os
from pathlib import Path
from typing import Dict

# 配置 - 使用相对路径，可通过环境变量覆盖
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./corpus"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "./txt_output"))

# 排除的目录
SKIP_DIRS = {"token_statistics_raw.json"}


def convert_topic_references(
    category_name: str,
    topic_name: str,
    ref_pages_dir: Path,
    output_dir: Path
) -> Dict[str, int]:
    """转换单个主题的 reference pages
    
    Returns:
        统计信息 {'converted': 转换数, 'skipped': 跳过数, 'total_chars': 总字符数}
    """
    stats = {'converted': 0, 'skipped': 0, 'total_chars': 0}
    
    # 创建输出目录
    txt_files_dir = output_dir / "txt_files"
    txt_files_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化拼接内容
    merged_content = ""
    
    # 转换所有 .md 文件
    md_files = sorted(ref_pages_dir.glob("*.md"))
    
    if not md_files:
        print(f"  ⚠️  未找到任何 .md 文件")
        return stats
    
    for md_file in md_files:
        try:
            # 读取 .md 文件内容
            content = md_file.read_text(encoding="utf-8")
            
            # 写入到 .txt 文件
            txt_file = txt_files_dir / md_file.name.replace(".md", ".txt")
            txt_file.write_text(content, encoding="utf-8")
            
            # 添加到拼接内容
            merged_content += content + "\n"
            
            stats['converted'] += 1
            stats['total_chars'] += len(content)
            
        except Exception as e:
            print(f"    ❌ 转换失败 {md_file.name}: {e}")
            stats['skipped'] += 1
    
    # 写入拼接后的文件
    merged_file = output_dir / "merged.txt"
    merged_file.write_text(merged_content, encoding="utf-8")
    
    return stats


def main():
    print("="*60)
    print("🚀 批量转换 MD → TXT")
    print("="*60)
    print()
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 全局统计
    global_stats = {
        'total_categories': 0,
        'total_topics': 0,
        'total_files': 0,
        'total_chars': 0,
        'failed_topics': 0,
    }
    
    # 遍历所有分类
    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        
        # 跳过特定目录
        if category_name in SKIP_DIRS:
            print(f"⏭️  跳过: {category_name}")
            continue
        
        global_stats['total_categories'] += 1
        
        print(f"\n{'='*60}")
        print(f"📂 分类: {category_name}")
        print(f"{'='*60}")
        
        # 遍历该分类下的所有主题
        topic_count = 0
        for topic_dir in sorted(category_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            
            topic_name = topic_dir.name
            
            # 检查 reference_pages 目录
            ref_pages_dir = topic_dir / "reference" / "reference_pages"
            if not ref_pages_dir.exists():
                print(f"  ⏭️  跳过 {topic_name}: 无 reference_pages 目录")
                continue
            
            # 创建输出目录
            output_dir = OUTPUT_ROOT / category_name / topic_name
            
            print(f"\n  [{topic_count + 1}] {topic_name}")
            
            # 转换
            try:
                stats = convert_topic_references(
                    category_name,
                    topic_name,
                    ref_pages_dir,
                    output_dir
                )
                
                if stats['converted'] > 0:
                    topic_count += 1
                    global_stats['total_topics'] += 1
                    global_stats['total_files'] += stats['converted']
                    global_stats['total_chars'] += stats['total_chars']
                    
                    # 格式化字符数
                    chars_mb = stats['total_chars'] / 1_000_000
                    
                    print(f"    ✅ 转换 {stats['converted']} 个文件")
                    print(f"    📄 总字符数: {stats['total_chars']:,} ({chars_mb:.2f} MB)")
                    print(f"    📁 输出: {output_dir}")
                else:
                    global_stats['failed_topics'] += 1
                
            except Exception as e:
                print(f"    ❌ 失败: {e}")
                global_stats['failed_topics'] += 1
        
        if topic_count > 0:
            print(f"\n  ✅ {category_name}: 处理 {topic_count} 个主题")
    
    # 打印全局统计
    print("\n" + "="*60)
    print("✅ 转换完成！")
    print("="*60)
    print()
    print("📊 全局统计:")
    print(f"  分类数: {global_stats['total_categories']}")
    print(f"  主题数: {global_stats['total_topics']}")
    print(f"  转换文件数: {global_stats['total_files']}")
    print(f"  总字符数: {global_stats['total_chars']:,} ({global_stats['total_chars'] / 1_000_000:.2f} MB)")
    print(f"  失败主题数: {global_stats['failed_topics']}")
    print()
    print(f"📁 输出目录: {OUTPUT_ROOT}")
    
    # 生成目录树
    print("\n📂 输出目录结构:")
    for category_dir in sorted(OUTPUT_ROOT.iterdir()):
        if category_dir.is_dir():
            topic_dirs = [d for d in category_dir.iterdir() if d.is_dir()]
            print(f"  {category_dir.name}/ ({len(topic_dirs)} 个主题)")
            for topic_dir in sorted(topic_dirs)[:3]:  # 只显示前 3 个
                txt_count = len(list((topic_dir / "txt_files").glob("*.txt")))
                merged_file = topic_dir / "merged.txt"
                merged_size = merged_file.stat().st_size if merged_file.exists() else 0
                print(f"    ├─ {topic_dir.name}/ ({txt_count} 个 TXT, merged: {merged_size:,} bytes)")
            if len(topic_dirs) > 3:
                print(f"    └─ ... 还有 {len(topic_dirs) - 3} 个主题")


if __name__ == "__main__":
    main()