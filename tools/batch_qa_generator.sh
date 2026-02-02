#!/bin/bash
# Batch QA Generator Script for WildGraphBench
# Usage: ./batch_qa_generator.sh [EXTRACTED_DATA_ROOT] [QA_OUTPUT_ROOT]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration paths - can be overridden by command line arguments or environment variables
EXTRACTED_DATA_ROOT="${1:-${EXTRACTED_DATA_ROOT:-./extracted_data}}"
QA_GENERATOR_SCRIPT="${SCRIPT_DIR}/qa_generator.py"
QA_OUTPUT_ROOT="${2:-${QA_OUTPUT_ROOT:-./qa_output}}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建输出根目录
mkdir -p "$QA_OUTPUT_ROOT"

# 统计变量
total_topics=0
success_count=0
failed_count=0
total_qa_count=0

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}开始批量生成 QA 数据集${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 遍历 extracted_data 下的所有一级分类目录
for category_dir in "$EXTRACTED_DATA_ROOT"/*/ ; do
    # 跳过非目录
    if [ ! -d "$category_dir" ]; then
        continue
    fi
    
    category_name=$(basename "$category_dir")
    
    echo -e "${YELLOW}================================================${NC}"
    echo -e "${YELLOW}处理分类: $category_name${NC}"
    echo -e "${YELLOW}================================================${NC}"
    
    # 为该分类创建输出目录
    category_output="$QA_OUTPUT_ROOT/$category_name"
    mkdir -p "$category_output"
    
    # 遍历该分类下的所有主题目录
    for topic_dir in "$category_dir"*/ ; do
        # 跳过非目录
        if [ ! -d "$topic_dir" ]; then
            continue
        fi
        
        topic_name=$(basename "$topic_dir")
        total_topics=$((total_topics + 1))
        
        echo ""
        echo -e "${GREEN}[$total_topics] 正在处理主题: $topic_name${NC}"
        echo "  路径: $topic_dir"
        
        # 检查 valid_triples.jsonl 或 valid.jsonl 是否存在
        valid_triples=""
        if [ -f "$topic_dir/valid_triples.jsonl" ]; then
            valid_triples="$topic_dir/valid_triples.jsonl"
        elif [ -f "$topic_dir/valid.jsonl" ]; then
            valid_triples="$topic_dir/valid.jsonl"
        else
            echo -e "  ${RED}✗ 未找到 valid_triples.jsonl 或 valid.jsonl，跳过${NC}"
            failed_count=$((failed_count + 1))
            continue
        fi
        
        echo "  使用输入文件: $(basename "$valid_triples")"
        
        # 统计 triple 数量
        triple_count=$(wc -l < "$valid_triples")
        echo "  有效 triples 数量: $triple_count"
        
        if [ "$triple_count" -eq 0 ]; then
            echo -e "  ${YELLOW}⚠ 没有有效 triples，跳过${NC}"
            failed_count=$((failed_count + 1))
            continue
        fi
        
        # 为该主题创建输出目录
        topic_output="$category_output/$topic_name"
        mkdir -p "$topic_output"
        
        qa_output="$topic_output/qa.jsonl"
        
        # 调用 qa_generator.py
        echo "  开始生成 QA..."
        
        if python3 "$QA_GENERATOR_SCRIPT" \
            --triples-valid "$valid_triples" \
            --out "$qa_output" \
            --num-type1 0 \
            --num-type2 0 \
            --num-type3 100 \
            --val-max-refs 6 \
            --seed 2025 \
            2>&1 | tee "$topic_output/qa_generation.log"; then
            
            echo -e "  ${GREEN}✓ QA 生成成功${NC}"
            success_count=$((success_count + 1))
            
            # 显示统计信息
            if [ -f "$qa_output" ]; then
                qa_lines=$(wc -l < "$qa_output")
                total_qa_count=$((total_qa_count + qa_lines))
                echo "  - 生成 QA 数量: $qa_lines 条"
                
                # 统计各类型 QA
                type1_count=$(grep -o '"question_type": \["single-fact"\]' "$qa_output" | wc -l)
                type2_count=$(grep -o '"question_type": \["multi_fact"\]' "$qa_output" | wc -l)
                type3_count=$(grep -o '"question_type": \["summary"\]' "$qa_output" | wc -l)
                
                echo "    • Type1 (single-fact): $type1_count"
                echo "    • Type2 (multi-fact): $type2_count"
                echo "    • Type3 (summary): $type3_count"
            fi
        else
            echo -e "  ${RED}✗ QA 生成失败${NC}"
            failed_count=$((failed_count + 1))
        fi
        
        echo "  日志已保存到: $topic_output/qa_generation.log"
    done
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}批量 QA 生成完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "统计信息:"
echo "  总主题数: $total_topics"
echo -e "  成功: ${GREEN}$success_count${NC}"
echo -e "  失败: ${RED}$failed_count${NC}"
echo "  总 QA 数量: $total_qa_count"
echo ""
echo "输出目录: $QA_OUTPUT_ROOT"
echo ""
echo -e "${BLUE}提示：可以使用以下命令查看所有生成的 QA：${NC}"
echo "  find $QA_OUTPUT_ROOT -name 'qa.jsonl' -exec wc -l {} +"