#!/bin/bash
# Batch QA Generator Script for WildGraphBench
# Usage: ./batch_qa_generator.sh [EXTRACTED_DATA_ROOT] [QA_OUTPUT_ROOT]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration paths - can be overridden by command line arguments or environment variables
EXTRACTED_DATA_ROOT="${1:-${EXTRACTED_DATA_ROOT:-./extracted_data}}"
QA_GENERATOR_SCRIPT="${SCRIPT_DIR}/qa_generator.py"
QA_OUTPUT_ROOT="${2:-${QA_OUTPUT_ROOT:-./qa_output}}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output root directory
mkdir -p "$QA_OUTPUT_ROOT"

# Statistics variables
total_topics=0
success_count=0
failed_count=0
total_qa_count=0

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting batch QA dataset generation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Iterate through all first-level category directories under extracted_data
for category_dir in "$EXTRACTED_DATA_ROOT"/*/ ; do
    # Skip non-directories
    if [ ! -d "$category_dir" ]; then
        continue
    fi
    
    category_name=$(basename "$category_dir")
    
    echo -e "${YELLOW}================================================${NC}"
    echo -e "${YELLOW}Processing category: $category_name${NC}"
    echo -e "${YELLOW}================================================${NC}"
    
    # Create output directory for this category
    category_output="$QA_OUTPUT_ROOT/$category_name"
    mkdir -p "$category_output"
    
    # Iterate through all topic directories under this category
    for topic_dir in "$category_dir"*/ ; do
        # Skip non-directories
        if [ ! -d "$topic_dir" ]; then
            continue
        fi
        
        topic_name=$(basename "$topic_dir")
        total_topics=$((total_topics + 1))
        
        echo ""
        echo -e "${GREEN}[$total_topics] Processing topic: $topic_name${NC}"
        echo "  Path: $topic_dir"
        
        # Check if valid_triples.jsonl or valid.jsonl exists
        valid_triples=""
        if [ -f "$topic_dir/valid_triples.jsonl" ]; then
            valid_triples="$topic_dir/valid_triples.jsonl"
        elif [ -f "$topic_dir/valid.jsonl" ]; then
            valid_triples="$topic_dir/valid.jsonl"
        else
            echo -e "  ${RED}✗ valid_triples.jsonl or valid.jsonl not found, skipping${NC}"
            failed_count=$((failed_count + 1))
            continue
        fi
        
        echo "  Using input file: $(basename "$valid_triples")"
        
        # Count triples
        triple_count=$(wc -l < "$valid_triples")
        echo "  Valid triples count: $triple_count"
        
        if [ "$triple_count" -eq 0 ]; then
            echo -e "  ${YELLOW}⚠ No valid triples, skipping${NC}"
            failed_count=$((failed_count + 1))
            continue
        fi
        
        # Create output directory for this topic
        topic_output="$category_output/$topic_name"
        mkdir -p "$topic_output"
        
        qa_output="$topic_output/qa.jsonl"
        
        # Call qa_generator.py
        echo "  Starting QA generation..."
        
        if python3 "$QA_GENERATOR_SCRIPT" \
            --triples-valid "$valid_triples" \
            --out "$qa_output" \
            --num-type1 0 \
            --num-type2 0 \
            --num-type3 100 \
            --val-max-refs 6 \
            --seed 2025 \
            2>&1 | tee "$topic_output/qa_generation.log"; then
            
            echo -e "  ${GREEN}✓ QA generation successful${NC}"
            success_count=$((success_count + 1))
            
            # Display statistics
            if [ -f "$qa_output" ]; then
                qa_lines=$(wc -l < "$qa_output")
                total_qa_count=$((total_qa_count + qa_lines))
                echo "  - Generated QA count: $qa_lines"
                
                # Count each type of QA
                type1_count=$(grep -o '"question_type": \["single-fact"\]' "$qa_output" | wc -l)
                type2_count=$(grep -o '"question_type": \["multi_fact"\]' "$qa_output" | wc -l)
                type3_count=$(grep -o '"question_type": \["summary"\]' "$qa_output" | wc -l)
                
                echo "    • Type1 (single-fact): $type1_count"
                echo "    • Type2 (multi-fact): $type2_count"
                echo "    • Type3 (summary): $type3_count"
            fi
        else
            echo -e "  ${RED}✗ QA generation failed${NC}"
            failed_count=$((failed_count + 1))
        fi
        
        echo "  Log saved to: $topic_output/qa_generation.log"
    done
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Batch QA generation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Statistics:"
echo "  Total topics: $total_topics"
echo -e "  Success: ${GREEN}$success_count${NC}"
echo -e "  Failed: ${RED}$failed_count${NC}"
echo "  Total QA count: $total_qa_count"
echo ""
echo "Output directory: $QA_OUTPUT_ROOT"
echo ""
echo -e "${BLUE}Tip: Use the following command to view all generated QA:${NC}"
echo "  find $QA_OUTPUT_ROOT -name 'qa.jsonl' -exec wc -l {} +"