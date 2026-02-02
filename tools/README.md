# WildGraphBench Tools

This directory contains tools for data construction and evaluation of WildGraphBench.

## 📋 Prerequisites

```bash
pip install requests tiktoken tqdm aiohttp python-dotenv
```

## 🔑 Configuration

Before using these tools, you need to configure your API credentials. Create a `.env` file in your working directory or set environment variables:

```bash
# LLM API Configuration (OpenAI-compatible API)
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1/"  # or your compatible API endpoint
export OPENAI_MODEL="gpt-4o"  # or your preferred model

# Jina Reader API (for web scraping)
# Get your API key from https://jina.ai/reader
export JINA_API_KEY="your-jina-api-key"

# Spider API (optional, alternative web scraping)
export SPIDER_API_URL="your-spider-api-url"

# Evaluation API (defaults to OPENAI settings if not set)
export EVAL_API_KEY="your-eval-api-key"
export EVAL_BASE_URL="https://api.openai.com/v1"
export EVAL_MODEL="gpt-4o-mini"
```

## 📁 Tool Descriptions

### Data Construction Pipeline

| Tool | Description |
|------|-------------|
| `build_wiki_url_manifest.py` | Generate a CSV manifest of Wikipedia article URLs from seed categories |
| `jina_scraping.py` | Scrape web pages using Jina Reader API and save as Markdown |
| `goliath.py` | Alternative web scraping tool using Spider API |
| `extract_references.py` | Extract reference links from Wikipedia Markdown files |
| `fetch_reference_pages.py` | Batch fetch reference source pages |
| `wiki_extractor.py` | Extract (sentence, statement, refs) triples from Wikipedia articles |
| `qa_generator.py` | Generate QA pairs (single-fact, multi-fact, summary) from extracted triples |
| `batch_qa_generator.sh` | Batch process multiple topics for QA generation |

### Evaluation

| Tool | Description |
|------|-------------|
| `eval.py` | Evaluate model predictions against gold answers |
| `calculate_tokens.py` | Calculate token statistics for corpus analysis |

### Utilities

| Tool | Description |
|------|-------------|
| `conver_md_to_txt.py` | Convert Markdown files to plain text |
| `post_filter_ref_pages.py` | Post-filter reference pages for quality |
| `recrawl_wiki.py` | Re-crawl Wikipedia pages |
| `enrich_human_performance.py` | Enrich dataset with human performance data |

## 🚀 Usage Examples

### 1. Build Wikipedia URL Manifest

```bash
python build_wiki_url_manifest.py \
    --topics "Computer science,Biology,History" \
    --per-topic 800 \
    --depth 2 \
    --min-refs 8 \
    --out urls.csv
```

### 2. Scrape Wikipedia Pages

```bash
python jina_scraping.py \
    --url "https://en.wikipedia.org/wiki/Machine_learning" \
    --output_dir ./output
```

### 3. Extract Triples from Wikipedia

```bash
python wiki_extractor.py \
    --input ./wiki_data/Machine_learning \
    --out-valid ./output/valid_triples.jsonl \
    --out-invalid ./output/invalid_triples.jsonl
```

### 4. Generate QA Dataset

```bash
python qa_generator.py \
    --triples-valid ./output/valid_triples.jsonl \
    --out ./output/qa.jsonl \
    --num-type1 100 \
    --num-type2 50 \
    --num-type3 50
```

### 5. Evaluate Predictions

```bash
python eval.py \
    --gold ./qa.jsonl \
    --pred ./predictions.jsonl \
    --outdir ./eval_results \
    --max-concurrent 10
```

### 6. Batch QA Generation

```bash
# Set paths via environment variables or command line arguments
./batch_qa_generator.sh ./extracted_data ./qa_output
```

## 📊 Output Formats

### QA Dataset (qa.jsonl)

```json
{
  "question": "What is the primary architecture used in GPT models?",
  "answer": "GPT models use the Transformer architecture...",
  "question_type": ["single-fact"],
  "source": [{
    "wiki_title": "GPT-4",
    "section_path": ["Architecture"],
    "wiki_sentences": ["GPT-4 is based on the Transformer architecture..."],
    "ref_urls": ["https://arxiv.org/..."]
  }]
}
```

### Evaluation Report (report.json)

```json
{
  "total_items": 1197,
  "single_fact": {"num": 667, "correct": 450, "accuracy": 0.6747},
  "multi_fact": {"num": 191, "correct": 89, "accuracy": 0.4660},
  "summary": {"num": 339, "correct": 120, "accuracy": 0.3540}
}
```

## ⚠️ Notes

- These tools require API access to LLM services (OpenAI or compatible)
- Web scraping respects rate limits; be patient with large-scale operations
- For production use, consider setting up proper error handling and logging
- Some tools may require additional dependencies; check individual file headers

## 📄 License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

