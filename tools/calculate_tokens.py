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
    使用tiktoken计算文本的token数量；允许把特殊token当普通文本编码
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # 某些模型名不被识别时，回退到 cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")

    # 关键：关闭特殊token校验
    try:
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # 兜底：去掉 <|...|> 这种标记再编码（极少用到）
        cleaned = re.sub(r"<\|[^|>]+?\|>", "", text)
        return len(encoding.encode(cleaned, disallowed_special=()))

def read_file_content(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  读取文件失败 {file_path}: {e}")
        return ""

def is_content_file(file_path: str) -> bool:
    """判断是否是内容文件（排除只有URL的json文件）"""
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        return False
    return file_path.endswith('.md')

def get_default_config(gpu_count: int = 8) -> Dict:
    """获取默认配置信息"""
    return {
        "gpu_config": f"0-{gpu_count-1}",
        "gpu_ids": list(range(gpu_count)),
        "ports": list(range(30060, 30060 + gpu_count)),
        "service_urls": [f"http://localhost:{port}" for port in range(30060, 30060 + gpu_count)],
        "gpu_count": gpu_count,
        "internal_ip": "localhost"
    }

def analyze_category(category_path: str, use_cleaned: bool = False) -> Dict:
    """分析单个类别的token统计
    
    Args:
        category_path: 类别路径
        use_cleaned: 是否使用清洗后的数据（从raw_cleaned目录）
    """
    category_name = os.path.basename(category_path)
    print(f"📊 分析类别: {category_name}")
    
    wiki_stats = []
    total_wikis = 0
    total_references = 0
    
    # 遍历类别下的所有wiki目录
    for wiki_item in os.listdir(category_path):
        wiki_dir = os.path.join(category_path, wiki_item)
        if not os.path.isdir(wiki_dir):
            continue
        
        total_wikis += 1
        wiki_tokens = 0
        ref_tokens = 0
        ref_count = 0
        
        print(f"  📂 处理wiki: {wiki_item}")
        
        # 处理wiki主文件
        for file in os.listdir(wiki_dir):
            if file.endswith('.md'):
                wiki_file_path = os.path.join(wiki_dir, file)
                content = read_file_content(wiki_file_path)
                if content:
                    tokens = count_tokens(content)
                    wiki_tokens += tokens
                    print(f"    📄 {file}: {tokens:,} tokens")
        
        # 处理reference目录下的参考文件
        ref_dir = os.path.join(wiki_dir, "reference")
        if os.path.exists(ref_dir):
            if use_cleaned:
                # 使用清洗后的文件
                ref_pages_dir = os.path.join(ref_dir, "reference_pages_cleaned")
                dir_label = "清洗后的参考文献"
            else:
                # 使用原始的参考文件
                ref_pages_dir = os.path.join(ref_dir, "reference_pages")
                dir_label = "原始参考文献"
            
            if os.path.exists(ref_pages_dir):
                print(f"    📁 处理{dir_label}目录: {ref_pages_dir}")
                
                # 获取所有.md文件
                ref_files = [f for f in os.listdir(ref_pages_dir) if f.endswith('.md')]
                print(f"    📋 找到 {len(ref_files)} 个参考文献文件")
                
                for ref_file in ref_files:
                    ref_file_path = os.path.join(ref_pages_dir, ref_file)
                    content = read_file_content(ref_file_path)
                    if content:
                        tokens = count_tokens(content)
                        ref_tokens += tokens
                        ref_count += 1
                        total_references += 1
                        # print(f"      📄 {ref_file}: {tokens:,} tokens")  # 可选：显示每个文件的详情
                
                if ref_count > 0:
                    print(f"    📊 {dir_label}: {ref_count} 文件, {ref_tokens:,} tokens")
                else:
                    print(f"    ⚠️  {dir_label}目录为空或无有效内容")
            else:
                print(f"    ⚠️  {dir_label}目录不存在: {ref_pages_dir}")
        else:
            print(f"    ⚠️  reference目录不存在")
        
        # 记录该wiki的统计
        wiki_stats.append({
            'name': wiki_item,
            'wiki_tokens': wiki_tokens,
            'reference_tokens': ref_tokens,
            'reference_count': ref_count,
            'total_tokens': wiki_tokens + ref_tokens
        })
        
        print(f"    📊 小计 - Wiki: {wiki_tokens:,}, 参考: {ref_tokens:,}, 总计: {wiki_tokens + ref_tokens:,} tokens")
    
    # 计算类别统计
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
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="Calculate tokens for Wiki and reference documents")
    parser.add_argument("--raw-dir", type=str, default="./raw", help="Raw data directory")
    parser.add_argument("--cleaned-dir", type=str, default="./raw_cleaned", help="Cleaned data directory")
    parser.add_argument("--gpu-count", type=int, default=8, help="Number of GPUs/services for load estimation")
    args = parser.parse_args()
    
    print("🔍 开始统计Wiki和参考文献Token数量...")
    
    # 配置路径
    raw_dir = args.raw_dir
    cleaned_dir = args.cleaned_dir
    
    # 使用简化的配置而非从脚本解析
    oss_config = {
        "gpu_config": f"0-{args.gpu_count-1}",
        "gpu_ids": list(range(args.gpu_count)),
        "ports": list(range(30060, 30060 + args.gpu_count)),
        "service_urls": [f"http://localhost:{port}" for port in range(30060, 30060 + args.gpu_count)],
        "gpu_count": args.gpu_count,
        "internal_ip": "localhost"
    }
    print(f"🌐 OSS服务配置:")
    print(f"  GPU配置: {oss_config['gpu_config']}")
    print(f"  使用GPU: {oss_config['gpu_ids']}")
    print(f"  服务端口: {oss_config['ports']}")
    print(f"  服务数量: {oss_config['gpu_count']}")
    print(f"  内网IP: {oss_config['internal_ip']}")
    print()
    
    # 选择数据源
    print("📂 可用的数据源:")
    print(f"  1. 原始数据: {raw_dir}")
    print(f"  2. 清洗后数据: {cleaned_dir}")
    
    use_cleaned = False
    data_dir = raw_dir
    
    # 检查清洗后目录是否存在
    if os.path.exists(cleaned_dir):
        choice = input("\n🤔 请选择数据源 (1=原始数据, 2=清洗后数据): ").strip()
        if choice == '2':
            use_cleaned = True
            data_dir = cleaned_dir
            print("✅ 使用清洗后数据")
        else:
            print("✅ 使用原始数据")
    else:
        print("⚠️  清洗后目录不存在，使用原始数据")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 获取所有类别目录
    categories = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            categories.append(item_path)
    
    if not categories:
        print("❌ 没有找到类别目录")
        return
    
    print(f"\n📂 找到 {len(categories)} 个类别")
    for i, cat_path in enumerate(sorted(categories), 1):
        print(f"  {i:2d}. {os.path.basename(cat_path)}")
    print()
    
    # 分析每个类别
    all_category_stats = []
    for category_path in sorted(categories):
        print(f"\n{'='*60}")
        stats = analyze_category(category_path, use_cleaned)
        all_category_stats.append(stats)
        print(f"✅ 完成类别 {stats['category_name']}")
        print(f"   📊 {stats['total_wikis']} wikis, {stats['total_references']} references")
        print(f"   🔢 Wiki tokens: {stats['total_wiki_tokens']:,}")
        print(f"   🔢 参考tokens: {stats['total_reference_tokens']:,}")
        print(f"   🔢 总tokens: {stats['total_tokens']:,}")
    
    # 计算总体统计
    print("\n" + "="*80)
    data_type = "清洗后数据" if use_cleaned else "原始数据"
    print(f"📈 总体统计 ({data_type})")
    print("="*80)
    
    total_wikis = sum(stats['total_wikis'] for stats in all_category_stats)
    total_references = sum(stats['total_references'] for stats in all_category_stats)
    total_wiki_tokens = sum(stats['total_wiki_tokens'] for stats in all_category_stats)
    total_ref_tokens = sum(stats['total_reference_tokens'] for stats in all_category_stats)
    total_all_tokens = total_wiki_tokens + total_ref_tokens
    
    print(f"📂 总类别数: {len(all_category_stats)}")
    print(f"📄 总Wiki数: {total_wikis:,}")
    print(f"📋 总参考文献数: {total_references:,}")
    print()
    print(f"🔢 Token统计:")
    print(f"  Wiki tokens: {total_wiki_tokens:,}")
    print(f"  参考文献tokens: {total_ref_tokens:,}")
    print(f"  总tokens: {total_all_tokens:,}")
    print()
    print(f"📊 平均值:")
    print(f"  每个Wiki平均tokens: {total_wiki_tokens/total_wikis:,.1f}" if total_wikis > 0 else "  每个Wiki平均tokens: 0")
    print(f"  每个参考文献平均tokens: {total_ref_tokens/total_references:,.1f}" if total_references > 0 else "  每个参考文献平均tokens: 0")
    print(f"  每个Wiki(含参考文献)平均tokens: {total_all_tokens/total_wikis:,.1f}" if total_wikis > 0 else "  每个Wiki(含参考文献)平均tokens: 0")
    print()
    print(f"🌐 OSS服务处理能力评估:")
    tokens_per_service = total_all_tokens / oss_config['gpu_count']
    print(f"  每个OSS服务平均处理tokens: {tokens_per_service:,.1f}")
    print(f"  服务URL示例:")
    for i, url in enumerate(oss_config['service_urls'][:3], 1):  # 只显示前3个
        print(f"    服务{i}: {url}")
    if len(oss_config['service_urls']) > 3:
        print(f"    ... 以及其他 {len(oss_config['service_urls']) - 3} 个服务")
    
    # 按类别显示详细统计
    print("\n" + "="*80)
    print("📊 各类别详细统计")
    print("="*80)
    
    # 按总token数排序
    sorted_stats = sorted(all_category_stats, key=lambda x: x['total_tokens'], reverse=True)
    
    for i, stats in enumerate(sorted_stats, 1):
        print(f"\n{i:2d}. 📁 {stats['category_name']}:")
        print(f"     📄 Wiki数: {stats['total_wikis']}")
        print(f"     📋 参考文献数: {stats['total_references']}")
        print(f"     🔢 Wiki tokens: {stats['total_wiki_tokens']:,}")
        print(f"     🔢 参考文献tokens: {stats['total_reference_tokens']:,}")
        print(f"     🔢 总tokens: {stats['total_tokens']:,}")
        print(f"     📊 平均Wiki tokens: {stats['avg_wiki_tokens']:,.1f}")
        if stats['total_references'] > 0:
            print(f"     📊 平均参考文献tokens: {stats['avg_reference_tokens']:,.1f}")
        
        # 计算该类别占总体的比例
        percentage = (stats['total_tokens'] / total_all_tokens * 100) if total_all_tokens > 0 else 0
        print(f"     📈 占总体比例: {percentage:.1f}%")
    
    # 找出token数最多和最少的wiki
    all_wiki_details = []
    for category_stats in all_category_stats:
        for wiki_detail in category_stats['wiki_details']:
            wiki_detail['category'] = category_stats['category_name']
            all_wiki_details.append(wiki_detail)
    
    if all_wiki_details:
        print("\n" + "="*80)
        print("🏆 Token数量排行")
        print("="*80)
        
        # 按总token数排序
        sorted_wikis = sorted(all_wiki_details, key=lambda x: x['total_tokens'], reverse=True)
        
        print("🥇 Token数最多的前10个Wiki:")
        for i, wiki in enumerate(sorted_wikis[:10], 1):
            print(f"  {i:2d}. [{wiki['category']}] {wiki['name']}")
            print(f"      总计: {wiki['total_tokens']:,} tokens")
            print(f"      (Wiki: {wiki['wiki_tokens']:,}, 参考: {wiki['reference_tokens']:,}, 参考文件: {wiki['reference_count']})")
        
        print("\n📊 Token数分布统计:")
        token_counts = [wiki['total_tokens'] for wiki in all_wiki_details]
        print(f"  最大值: {max(token_counts):,} tokens")
        print(f"  最小值: {min(token_counts):,} tokens")
        print(f"  中位数: {statistics.median(token_counts):,.1f} tokens")
        if len(token_counts) > 1:
            print(f"  标准差: {statistics.stdev(token_counts):,.1f} tokens")
        
        # 分布区间统计
        ranges = [
            (0, 1000, "< 1K"),
            (1000, 5000, "1K-5K"),
            (5000, 10000, "5K-10K"),
            (10000, 50000, "10K-50K"),
            (50000, 100000, "50K-100K"),
            (100000, float('inf'), "> 100K")
        ]
        
        print(f"\n📈 Token数分布区间:")
        for min_val, max_val, label in ranges:
            count = sum(1 for tokens in token_counts if min_val <= tokens < max_val)
            percentage = (count / len(token_counts) * 100) if token_counts else 0
            print(f"  {label:>8}: {count:3d} wikis ({percentage:4.1f}%)")
    
    # 保存详细统计到JSON文件
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
    
    print(f"\n💾 详细统计已保存到: {output_file}")
    print(f"\n🎯 数据处理摘要:")
    print(f"  数据类型: {data_type}")
    print(f"  数据目录: {data_dir}")
    print(f"  参考文献目录: {'reference_pages_cleaned' if use_cleaned else 'reference_pages'}")
    print(f"  OSS服务数量: {oss_config['gpu_count']}")
    print(f"  每服务平均负载: {tokens_per_service:,.1f} tokens")
    print("\n🎉 统计完成！")

if __name__ == "__main__":
    main()