#!/usr/bin/env python3
# Batch Wiki Extractor - Parallel Processing

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import time

# 配置 - 使用相对路径，可通过环境变量覆盖
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./corpus"))
EXTRACTOR_SCRIPT = Path(os.environ.get("EXTRACTOR_SCRIPT", "./tools/wiki_extractor.py"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "./extracted_data"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))  # 并发数


def process_topic(topic_dir: Path, category_name: str) -> Tuple[bool, str, int, int]:
    """处理单个主题
    
    Returns:
        (成功?, 主题名, 有效数, 无效数)
    """
    topic_name = topic_dir.name
    topic_output = OUTPUT_ROOT / category_name / topic_name
    topic_output.mkdir(parents=True, exist_ok=True)
    
    valid_output = topic_output / "valid_triples.jsonl"
    invalid_output = topic_output / "invalid_triples.jsonl"
    log_file = topic_output / "extraction.log"
    
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 开始: {category_name}/{topic_name}")
    
    try:
        # 调用 wiki_extractor.py
        with open(log_file, 'w', encoding='utf-8') as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(EXTRACTOR_SCRIPT),
                    "--raw-dir", str(topic_dir),
                    "--out-valid", str(valid_output),
                    "--out-invalid", str(invalid_output),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        
        if result.returncode == 0:
            valid_count = sum(1 for _ in valid_output.open()) if valid_output.exists() else 0
            invalid_count = sum(1 for _ in invalid_output.open()) if invalid_output.exists() else 0
            
            print(f"[{time.strftime('%H:%M:%S')}] ✅ 完成: {category_name}/{topic_name} "
                  f"(有效: {valid_count}, 无效: {invalid_count})")
            return True, topic_name, valid_count, invalid_count
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ 失败: {category_name}/{topic_name}")
            return False, topic_name, 0, 0
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: {category_name}/{topic_name} - {e}")
        return False, topic_name, 0, 0


def main():
    print("="*60)
    print("🚀 开始批量抽取 Wiki 数据 (并行模式)")
    print(f"⚙️  并发数: {MAX_WORKERS}")
    print("="*60)
    print()
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 收集所有待处理的主题
    tasks = []
    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        
        # 跳过特定目录
        if category_name == "token_statistics_raw.json":
            continue
        
        # 创建分类输出目录
        (OUTPUT_ROOT / category_name).mkdir(parents=True, exist_ok=True)
        
        # 收集该分类下的所有主题
        for topic_dir in sorted(category_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            
            # ★ 新增：跳过 reference 目录
            if topic_dir.name.lower() == 'reference':
                print(f"⏭️  跳过: {category_name}/reference")
                continue
            
            tasks.append((topic_dir, category_name))
    
    total = len(tasks)
    print(f"📊 总共发现 {total} 个主题\n")
    
    # 并行处理
    stats = {
        'success': 0,
        'failed': 0,
        'total_valid': 0,
        'total_invalid': 0,
    }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_topic, topic_dir, category): (topic_dir, category)
            for topic_dir, category in tasks
        }
        
        for future in as_completed(futures):
            success, topic_name, valid_count, invalid_count = future.result()
            
            if success:
                stats['success'] += 1
                stats['total_valid'] += valid_count
                stats['total_invalid'] += invalid_count
            else:
                stats['failed'] += 1
    
    # 打印统计
    print()
    print("="*60)
    print("✅ 批量抽取完成！")
    print("="*60)
    print()
    print("📊 统计信息:")
    print(f"  总主题数: {total}")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  📄 总有效 triples: {stats['total_valid']}")
    print(f"  ⚠️  总无效 triples: {stats['total_invalid']}")
    print()
    print(f"📁 输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()