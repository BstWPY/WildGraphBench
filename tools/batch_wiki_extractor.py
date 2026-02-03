#!/usr/bin/env python3
# Batch Wiki Extractor - Parallel Processing

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import time

# Configuration - use relative paths, can be overridden via environment variables
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./corpus"))
EXTRACTOR_SCRIPT = Path(os.environ.get("EXTRACTOR_SCRIPT", "./tools/wiki_extractor.py"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "./extracted_data"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))  # concurrency


def process_topic(topic_dir: Path, category_name: str) -> Tuple[bool, str, int, int]:
    """Process a single topic
    
    Returns:
        (success?, topic_name, valid_count, invalid_count)
    """
    topic_name = topic_dir.name
    topic_output = OUTPUT_ROOT / category_name / topic_name
    topic_output.mkdir(parents=True, exist_ok=True)
    
    valid_output = topic_output / "valid_triples.jsonl"
    invalid_output = topic_output / "invalid_triples.jsonl"
    log_file = topic_output / "extraction.log"
    
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 Starting: {category_name}/{topic_name}")
    
    try:
        # Call wiki_extractor.py
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
            
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Done: {category_name}/{topic_name} "
                  f"(valid: {valid_count}, invalid: {invalid_count})")
            return True, topic_name, valid_count, invalid_count
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Failed: {category_name}/{topic_name}")
            return False, topic_name, 0, 0
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Error: {category_name}/{topic_name} - {e}")
        return False, topic_name, 0, 0


def main():
    print("="*60)
    print("🚀 Starting batch Wiki data extraction (parallel mode)")
    print(f"⚙️  Concurrency: {MAX_WORKERS}")
    print("="*60)
    print()
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Collect all topics to process
    tasks = []
    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        
        # Skip specific directories
        if category_name == "token_statistics_raw.json":
            continue
        
        # Create category output directory
        (OUTPUT_ROOT / category_name).mkdir(parents=True, exist_ok=True)
        
        # Collect all topics under this category
        for topic_dir in sorted(category_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            
            # Skip reference directory
            if topic_dir.name.lower() == 'reference':
                print(f"⏭️  Skip: {category_name}/reference")
                continue
            
            tasks.append((topic_dir, category_name))
    
    total = len(tasks)
    print(f"📊 Found {total} topics in total\n")
    
    # Parallel processing
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
    
    # Print statistics
    print()
    print("="*60)
    print("✅ Batch extraction completed!")
    print("="*60)
    print()
    print("📊 Statistics:")
    print(f"  Total topics: {total}")
    print(f"  ✅ Success: {stats['success']}")
    print(f"  ❌ Failed: {stats['failed']}")
    print(f"  📄 Total valid triples: {stats['total_valid']}")
    print(f"  ⚠️  Total invalid triples: {stats['total_invalid']}")
    print()
    print(f"📁 Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()