import os
import json
import tiktoken
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text using tiktoken; allows encoding special tokens as normal text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to cl100k_base if the model name is not recognized
        encoding = tiktoken.get_encoding("cl100k_base")

    # Key: disable special token validation
    try:
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback: remove <|...|> markers and encode (rarely used)
        cleaned = re.sub(r"<\|[^|>]+?\|>", "", text)
        return len(encoding.encode(cleaned, disallowed_special=()))

def read_file_content(file_path: str) -> str:
    """Read file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  Failed to read file {file_path}: {e}")
        return ""

def is_content_file(file_path: str) -> bool:
    """Check if the file is a content file (excluding JSON files with only URLs)"""
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        return False
    return file_path.endswith('.md')

def get_default_config(gpu_count: int = 8) -> Dict:
    """Get default configuration"""
    return {
        "gpu_config": f"0-{gpu_count-1}",
        "gpu_ids": list(range(gpu_count)),
        "ports": list(range(30060, 30060 + gpu_count)),
        "service_urls": [f"http://localhost:{port}" for port in range(30060, 30060 + gpu_count)],
        "gpu_count": gpu_count,
        "internal_ip": "localhost"
    }

def analyze_category(category_path: str, use_cleaned: bool = False) -> Dict:
    """Analyze token statistics for a single category
    
    Args:
        category_path: Category path
        use_cleaned: Whether to use cleaned data (from raw_cleaned directory)
    """
    category_name = os.path.basename(category_path)
    print(f"📊 Analyzing category: {category_name}")
    
    wiki_stats = []
    total_wikis = 0
    total_references = 0
    
    # Iterate through all wiki directories under the category
    for wiki_item in os.listdir(category_path):
        wiki_dir = os.path.join(category_path, wiki_item)
        if not os.path.isdir(wiki_dir):
            continue
        
        total_wikis += 1
        wiki_tokens = 0
        ref_tokens = 0
        ref_count = 0
        
        print(f"  📂 Processing wiki: {wiki_item}")
        
        # Process wiki main file
        for file in os.listdir(wiki_dir):
            if file.endswith('.md'):
                wiki_file_path = os.path.join(wiki_dir, file)
                content = read_file_content(wiki_file_path)
                if content:
                    tokens = count_tokens(content)
                    wiki_tokens += tokens
                    print(f"    📄 {file}: {tokens:,} tokens")
        
        # Process reference files under the reference directory
        ref_dir = os.path.join(wiki_dir, "reference")
        if os.path.exists(ref_dir):
            if use_cleaned:
                # Use cleaned files
                ref_pages_dir = os.path.join(ref_dir, "reference_pages_cleaned")
                dir_label = "cleaned references"
            else:
                # Use original reference files
                ref_pages_dir = os.path.join(ref_dir, "reference_pages")
                dir_label = "original references"
            
            if os.path.exists(ref_pages_dir):
                print(f"    📁 Processing {dir_label} directory: {ref_pages_dir}")
                
                # Get all .md files
                ref_files = [f for f in os.listdir(ref_pages_dir) if f.endswith('.md')]
                print(f"    📋 Found {len(ref_files)} reference files")
                
                for ref_file in ref_files:
                    ref_file_path = os.path.join(ref_pages_dir, ref_file)
                    content = read_file_content(ref_file_path)
                    if content:
                        tokens = count_tokens(content)
                        ref_tokens += tokens
                        ref_count += 1
                        total_references += 1
                        # print(f"      📄 {ref_file}: {tokens:,} tokens")  # Optional: show details for each file
                
                if ref_count > 0:
                    print(f"    📊 {dir_label}: {ref_count} files, {ref_tokens:,} tokens")
                else:
                    print(f"    ⚠️  {dir_label} directory is empty or has no valid content")
            else:
                print(f"    ⚠️  {dir_label} directory does not exist: {ref_pages_dir}")
        else:
            print(f"    ⚠️  reference directory does not exist")
        
        # Record statistics for this wiki
        wiki_stats.append({
            'name': wiki_item,
            'wiki_tokens': wiki_tokens,
            'reference_tokens': ref_tokens,
            'reference_count': ref_count,
            'total_tokens': wiki_tokens + ref_tokens
        })
        
        print(f"    📊 Subtotal - Wiki: {wiki_tokens:,}, References: {ref_tokens:,}, Total: {wiki_tokens + ref_tokens:,} tokens")
    
    # Calculate category statistics
    total_wiki_tokens = sum(stat['wiki_tokens'] for stat in wiki_stats)
    total_ref_tokens = sum(stat['reference_tokens'] for stat in wiki_stats)
    total_tokens = total_wiki_tokens + total_ref_tokens
    
    avg_wiki_tokens = total_wiki_tokens / total_wikis if total_wikis > 0 else 0
    avg_ref_tokens = total_ref_tokens / total_references if total_references > 0 else 0
    
    return {
        'category_name': category_name,
        'total_wikis': total_wikis,
        'total_references': total_references,
        'total_wiki_tokens': total_wiki_tokens,
        'total_reference_tokens': total_ref_tokens,
        'total_tokens': total_tokens,
        'avg_wiki_tokens': avg_wiki_tokens,
        'avg_reference_tokens': avg_ref_tokens,
        'wiki_details': wiki_stats
    }

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Calculate tokens for Wiki and reference documents")
    parser.add_argument("--raw-dir", type=str, default="./raw", help="Raw data directory")
    parser.add_argument("--cleaned-dir", type=str, default="./raw_cleaned", help="Cleaned data directory")
    parser.add_argument("--gpu-count", type=int, default=8, help="Number of GPUs/services for load estimation")
    args = parser.parse_args()
    
    print("🔍 Starting Wiki and reference token counting...")
    
    # Configuration paths
    raw_dir = args.raw_dir
    cleaned_dir = args.cleaned_dir
    
    # Use simplified configuration instead of parsing from script
    oss_config = {
        "gpu_config": f"0-{args.gpu_count-1}",
        "gpu_ids": list(range(args.gpu_count)),
        "ports": list(range(30060, 30060 + args.gpu_count)),
        "service_urls": [f"http://localhost:{port}" for port in range(30060, 30060 + args.gpu_count)],
        "gpu_count": args.gpu_count,
        "internal_ip": "localhost"
    }
    print(f"🌐 OSS service configuration:")
    print(f"  GPU config: {oss_config['gpu_config']}")
    print(f"  Using GPUs: {oss_config['gpu_ids']}")
    print(f"  Service ports: {oss_config['ports']}")
    print(f"  Number of services: {oss_config['gpu_count']}")
    print(f"  Internal IP: {oss_config['internal_ip']}")
    print()
    
    # Select data source
    print("📂 Available data sources:")
    print(f"  1. Raw data: {raw_dir}")
    print(f"  2. Cleaned data: {cleaned_dir}")
    
    use_cleaned = False
    data_dir = raw_dir
    
    # Check if cleaned directory exists
    if os.path.exists(cleaned_dir):
        choice = input("\n🤔 Please select data source (1=raw data, 2=cleaned data): ").strip()
        if choice == '2':
            use_cleaned = True
            data_dir = cleaned_dir
            print("✅ Using cleaned data")
        else:
            print("✅ Using raw data")
    else:
        print("⚠️  Cleaned directory does not exist, using raw data")
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return
    
    # Get all category directories
    categories = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            categories.append(item_path)
    
    if not categories:
        print("❌ No category directories found")
        return
    
    print(f"\n📂 Found {len(categories)} categories")
    for i, cat_path in enumerate(sorted(categories), 1):
        print(f"  {i:2d}. {os.path.basename(cat_path)}")
    print()
    
    # Analyze each category
    all_category_stats = []
    for category_path in sorted(categories):
        print(f"\n{'='*60}")
        stats = analyze_category(category_path, use_cleaned)
        all_category_stats.append(stats)
        print(f"✅ Completed category {stats['category_name']}")
        print(f"   📊 {stats['total_wikis']} wikis, {stats['total_references']} references")
        print(f"   🔢 Wiki tokens: {stats['total_wiki_tokens']:,}")
        print(f"   🔢 Reference tokens: {stats['total_reference_tokens']:,}")
        print(f"   🔢 Total tokens: {stats['total_tokens']:,}")
    
    # Calculate overall statistics
    print("\n" + "="*80)
    data_type = "cleaned data" if use_cleaned else "raw data"
    print(f"📈 Overall Statistics ({data_type})")
    print("="*80)
    
    total_wikis = sum(stats['total_wikis'] for stats in all_category_stats)
    total_references = sum(stats['total_references'] for stats in all_category_stats)
    total_wiki_tokens = sum(stats['total_wiki_tokens'] for stats in all_category_stats)
    total_ref_tokens = sum(stats['total_reference_tokens'] for stats in all_category_stats)
    total_all_tokens = total_wiki_tokens + total_ref_tokens
    
    print(f"📂 Total categories: {len(all_category_stats)}")
    print(f"📄 Total wikis: {total_wikis:,}")
    print(f"📋 Total references: {total_references:,}")
    print()
    print(f"🔢 Token Statistics:")
    print(f"  Wiki tokens: {total_wiki_tokens:,}")
    print(f"  Reference tokens: {total_ref_tokens:,}")
    print(f"  Total tokens: {total_all_tokens:,}")
    print()
    print(f"📊 Averages:")
    print(f"  Avg tokens per wiki: {total_wiki_tokens/total_wikis:,.1f}" if total_wikis > 0 else "  Avg tokens per wiki: 0")
    print(f"  Avg tokens per reference: {total_ref_tokens/total_references:,.1f}" if total_references > 0 else "  Avg tokens per reference: 0")
    print(f"  Avg tokens per wiki (incl. refs): {total_all_tokens/total_wikis:,.1f}" if total_wikis > 0 else "  Avg tokens per wiki (incl. refs): 0")
    print()
    print(f"🌐 OSS Service Processing Capacity Estimate:")
    tokens_per_service = total_all_tokens / oss_config['gpu_count']
    print(f"  Avg tokens per OSS service: {tokens_per_service:,.1f}")
    print(f"  Example service URLs:")
    for i, url in enumerate(oss_config['service_urls'][:3], 1):  # Show only first 3
        print(f"    Service {i}: {url}")
    if len(oss_config['service_urls']) > 3:
        print(f"    ... and {len(oss_config['service_urls']) - 3} more services")
    
    # Show detailed statistics by category
    print("\n" + "="*80)
    print("📊 Detailed Statistics by Category")
    print("="*80)
    
    # Sort by total token count
    sorted_stats = sorted(all_category_stats, key=lambda x: x['total_tokens'], reverse=True)
    
    for i, stats in enumerate(sorted_stats, 1):
        print(f"\n{i:2d}. 📁 {stats['category_name']}:")
        print(f"     📄 Wiki count: {stats['total_wikis']}")
        print(f"     📋 Reference count: {stats['total_references']}")
        print(f"     🔢 Wiki tokens: {stats['total_wiki_tokens']:,}")
        print(f"     🔢 Reference tokens: {stats['total_reference_tokens']:,}")
        print(f"     🔢 Total tokens: {stats['total_tokens']:,}")
        print(f"     📊 Avg wiki tokens: {stats['avg_wiki_tokens']:,.1f}")
        if stats['total_references'] > 0:
            print(f"     📊 Avg reference tokens: {stats['avg_reference_tokens']:,.1f}")
        
        # Calculate percentage of total
        percentage = (stats['total_tokens'] / total_all_tokens * 100) if total_all_tokens > 0 else 0
        print(f"     📈 Percentage of total: {percentage:.1f}%")
    
    # Find wikis with most and least tokens
    all_wiki_details = []
    for category_stats in all_category_stats:
        for wiki_detail in category_stats['wiki_details']:
            wiki_detail['category'] = category_stats['category_name']
            all_wiki_details.append(wiki_detail)
    
    if all_wiki_details:
        print("\n" + "="*80)
        print("🏆 Token Count Rankings")
        print("="*80)
        
        # Sort by total token count
        sorted_wikis = sorted(all_wiki_details, key=lambda x: x['total_tokens'], reverse=True)
        
        print("🥇 Top 10 Wikis by Token Count:")
        for i, wiki in enumerate(sorted_wikis[:10], 1):
            print(f"  {i:2d}. [{wiki['category']}] {wiki['name']}")
            print(f"      Total: {wiki['total_tokens']:,} tokens")
            print(f"      (Wiki: {wiki['wiki_tokens']:,}, References: {wiki['reference_tokens']:,}, Ref files: {wiki['reference_count']})")
        
        print("\n📊 Token Distribution Statistics:")
        token_counts = [wiki['total_tokens'] for wiki in all_wiki_details]
        print(f"  Max: {max(token_counts):,} tokens")
        print(f"  Min: {min(token_counts):,} tokens")
        print(f"  Median: {statistics.median(token_counts):,.1f} tokens")
        if len(token_counts) > 1:
            print(f"  Std Dev: {statistics.stdev(token_counts):,.1f} tokens")
        
        # Distribution range statistics
        ranges = [
            (0, 1000, "< 1K"),
            (1000, 5000, "1K-5K"),
            (5000, 10000, "5K-10K"),
            (10000, 50000, "10K-50K"),
            (50000, 100000, "50K-100K"),
            (100000, float('inf'), "> 100K")
        ]
        
        print(f"\n📈 Token Distribution Ranges:")
        for min_val, max_val, label in ranges:
            count = sum(1 for tokens in token_counts if min_val <= tokens < max_val)
            percentage = (count / len(token_counts) * 100) if token_counts else 0
            print(f"  {label:>8}: {count:3d} wikis ({percentage:4.1f}%)")
    
    # Save detailed statistics to JSON file
    output_file = os.path.join(data_dir, f"token_statistics_{'cleaned' if use_cleaned else 'raw'}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "data_source": {
                "type": "cleaned" if use_cleaned else "raw",
                "directory": data_dir,
                "description": data_type
            },
            "oss_config": oss_config,
            "summary": {
                "total_categories": len(all_category_stats),
                "total_wikis": total_wikis,
                "total_references": total_references,
                "total_wiki_tokens": total_wiki_tokens,
                "total_reference_tokens": total_ref_tokens,
                "total_all_tokens": total_all_tokens,
                "avg_wiki_tokens": total_wiki_tokens/total_wikis if total_wikis > 0 else 0,
                "avg_reference_tokens": total_ref_tokens/total_references if total_references > 0 else 0,
                "avg_tokens_per_wiki_with_refs": total_all_tokens/total_wikis if total_wikis > 0 else 0,
                "tokens_per_oss_service": total_all_tokens / oss_config['gpu_count']
            },
            "category_stats": all_category_stats,
            "top_wikis": sorted_wikis[:50] if 'sorted_wikis' in locals() else [],
            "statistics": {
                "max_tokens": max(token_counts) if 'token_counts' in locals() and token_counts else 0,
                "min_tokens": min(token_counts) if 'token_counts' in locals() and token_counts else 0,
                "median_tokens": statistics.median(token_counts) if 'token_counts' in locals() and token_counts else 0,
                "stdev_tokens": statistics.stdev(token_counts) if 'token_counts' in locals() and len(token_counts) > 1 else 0
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed statistics saved to: {output_file}")
    print(f"\n🎯 Data Processing Summary:")
    print(f"  Data type: {data_type}")
    print(f"  Data directory: {data_dir}")
    print(f"  Reference directory: {'reference_pages_cleaned' if use_cleaned else 'reference_pages'}")
    print(f"  OSS service count: {oss_config['gpu_count']}")
    print(f"  Avg load per service: {tokens_per_service:,.1f} tokens")
    print("\n🎉 Statistics completed!")

if __name__ == "__main__":
    main()