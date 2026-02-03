<div align="center">

# 🌲 WildGraphBench

**Benchmarking GraphRAG with Wild-Source Corpora**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](YOUR_ARXIV_LINK_HERE)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-WildGraphBench-yellow)](https://huggingface.co/datasets/YOUR_HF_LINK_HERE)

</div>

## 📖 Overview

**WildGraphBench** is a benchmark designed to evaluate Graph-based Retrieval-Augmented Generation (GraphRAG) systems in realistic, challenging scenarios. Unlike existing benchmarks that rely on short, curated passages, WildGraphBench uses Wikipedia's unique structure—where concise summaries are grounded in long, heterogeneous external reference documents—to create a truly "wild" evaluation setting.

![WildGraphBench Overview](assets/wild.png)

### 🔑 Key Features

- **Wild Evidence**: External reference pages from Wikipedia, including news sites, blogs, PDFs, and public reports
- **12 Diverse Topics**: Culture, Geography, Health, History, Human Activities, Mathematics, Nature, People, Philosophy, Religion, Society, and Technology
- **1,197 Questions** across three complexity levels:
  - 🔹 **Single-Fact QA** (667 questions): Lookup-style questions grounded by a single reference
  - 🔹 **Multi-Fact QA** (191 questions): Questions requiring evidence aggregation across multiple references
  - 🔹 **Summary** (339 questions): Section-level summarization tasks evaluated at the statement level

### 📋 Task Examples

![Task Examples](assets/cases.png)

### 🔧 Benchmark Pipeline

![Benchmark Pipeline](assets/pipeline.png)

---

## 📊 Main Results

We evaluate representative flat-RAG and GraphRAG baselines on WildGraphBench. All methods use `gpt-4o-mini` for graph construction and answering.

### Overall Performance

| Method | Avg. Acc. | Single-fact Acc. | Multi-fact Acc. | Recall | Precision | F1 |
|:-------|:---------:|:----------------:|:---------------:|:------:|:---------:|:--:|
| **NaiveRAG** | 59.79 | 66.87 | 35.08 | **13.54** | 19.07 | **15.84** |
| BM25 | 36.83 | 41.38 | 20.94 | 9.38 | 19.46 | 12.66 |
| Fast-GraphRAG | 33.56 | 35.83 | 25.65 | 6.81 | 23.48 | 10.56 |
| **HippoRAG2** | **64.33** | **71.51** | 39.27 | 11.15 | 16.76 | 13.39 |
| MS GraphRAG (local) | 38.23 | 39.43 | 34.03 | 9.82 | 12.64 | 11.05 |
| MS GraphRAG (global) | 54.54 | 56.52 | **47.64** | 12.66 | 15.13 | 13.78 |
| LightRAG (hybrid) | 56.76 | 61.32 | 40.84 | 12.44 | 17.70 | 14.61 |
| LinearRAG | 44.87 | 47.53 | 35.60 | 5.81 | **29.20** | 9.69 |

### People Subset (with Human Performance)

| Method | Avg. Acc. | Single-fact Acc. | Multi-fact Acc. | Recall | Precision | F1 |
|:-------|:---------:|:----------------:|:---------------:|:------:|:---------:|:--:|
| NaiveRAG | 65.82 | 76.62 | 28.12 | **10.48** | 15.29 | **8.03** |
| BM25 | 65.20 | 74.03 | 34.38 | 5.74 | 16.98 | 5.03 |
| Fast-GraphRAG | 30.43 | 33.77 | 18.75 | 1.48 | **22.83** | 1.62 |
| HippoRAG2 | 64.89 | 72.73 | 37.50 | 7.63 | 15.69 | 6.14 |
| MS GraphRAG (local) | 35.16 | 38.96 | 21.88 | 4.59 | 9.17 | 2.98 |
| MS GraphRAG (global) | 56.81 | 62.34 | 37.50 | 5.52 | 14.13 | 5.41 |
| **LightRAG (hybrid)** | **74.42** | **80.52** | **53.12** | 5.56 | 15.69 | 4.73 |
| LinearRAG | 45.26 | 51.95 | 21.88 | 1.52 | 22.51 | 1.69 |
| 👤 **Human** | **85.66** | **89.61** | **71.88** | 38.59 | 12.62 | 15.30 |

### 💡 Key Findings

1. **Single-Fact QA**: Flat retrieval baselines (NaiveRAG) remain competitive; graph structure doesn't automatically translate into gains for simple lookups
2. **Multi-Fact QA**: GraphRAG methods (especially MS GraphRAG global) show clear advantages when evidence must be aggregated from multiple documents
3. **Summary Tasks**: All methods struggle with low statement-level scores; NaiveRAG achieves highest recall due to broader context coverage, while GraphRAG bottlenecks may limit evidence gathering

---

## 📁 Dataset Statistics

### Question Distribution by Domain

| Domain | Single-Fact | Multi-Fact | Summary | Total |
|:-------|:-----------:|:----------:|:-------:|:-----:|
| Culture | 86 | 37 | 32 | 155 |
| Geography | 41 | 24 | 33 | 98 |
| Health | 76 | 19 | 55 | 150 |
| History | 25 | 1 | 10 | 36 |
| Human Activities | 83 | 13 | 44 | 140 |
| Mathematics | 21 | 1 | 11 | 33 |
| Nature | 18 | 0 | 10 | 28 |
| People | 77 | 32 | 45 | 154 |
| Philosophy | 46 | 6 | 18 | 70 |
| Religion | 72 | 4 | 30 | 106 |
| Society | 66 | 21 | 27 | 114 |
| Technology | 56 | 33 | 24 | 113 |
| **Total** | **667** | **191** | **339** | **1,197** |

---

## 📂 Repository Structure

```
WildGraphBench/
├── corpus/                     # Corpus for graph construction
│   └── {domain}/{topic}/
│       ├── {topic}.txt              # Wikipedia article (for reference only)
│       ├── reference_pages/         # 📌 Use these for graph construction!
│       └── references.jsonl         # Reference metadata
├── QA/                         # Questions for evaluation
│   └── {domain}/
│       └── questions.jsonl
├── statements/                 # Gold statements (for Summary tasks)
│   └── {domain}/{topic}/
│       └── statements.jsonl
└── LICENSE
```

---

## 🚀 Quick Start

WildGraphBench evaluates GraphRAG systems in two steps:

| Step | What to do | Data to use |
|:-----|:-----------|:------------|
| **1. Build Graph** | Construct your knowledge graph | `corpus/{domain}/{topic}/reference_pages/` |
| **2. Answer Questions** | Run your GraphRAG to answer | `QA/{domain}/questions.jsonl` |

### Step 1: Load Corpus for Graph Construction

> ⚠️ **Important**: Build your graph from **reference pages**, not the Wikipedia article!

```python
from pathlib import Path

# Choose a domain and topic
domain = "people"
topic = "Donald Trump"

# Load all reference documents
ref_dir = Path(f"corpus/{domain}/{topic}/reference_pages")
documents = [f.read_text() for f in ref_dir.glob("*.txt")]

print(f"Loaded {len(documents)} reference documents")
```

```python
# Now build your graph with your own GraphRAG system
# Example: your_graph = YourGraphRAG.build(documents)
```

### Step 2: Load Questions & Evaluate

```python
import json

# Load questions for this domain
domain = "people"
with open(f"QA/{domain}/questions.jsonl") as f:
    questions = [json.loads(line) for line in f]

print(f"Loaded {len(questions)} questions")
```

```python
# Answer each question with your GraphRAG
for q in questions:
    question = q["question"]
    q_type = q["type"]  # "single-fact", "multi-fact", or "summary"
    answer = q["answer"]
    
    # your_answer = your_graph.query(question)
    # evaluate(your_answer, answer)
```

### Available Domains

| Domain | Topic | # References | # Questions |
|:-------|:------|:------------:|:-----------:|
| culture | Marvel Cinematic Universe | 452 | 155 |
| geography | United States | 470 | 98 |
| health | COVID-19 pandemic | 510 | 150 |
| history | World War II | 74 | 36 |
| human_activities | 2022 FIFA World Cup | 367 | 140 |
| mathematics | Prime number | 50 | 33 |
| nature | 2012 Pacific typhoon season | 72 | 28 |
| people | Donald Trump | 547 | 154 |
| philosophy | Authoritarian socialism | 257 | 70 |
| religion | Persecution of Muslims | 346 | 106 |
| society | Human | 319 | 114 |
| technology | Steam (service) | 442 | 113 |

---

## 📝 Citation

If you find WildGraphBench useful in your research, please cite our paper:

```bibtex
@article{wildgraphbench2025,
  title={WildGraphBench: Benchmarking GraphRAG with Wild-Source Corpora},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

<a href="https://star-history.com/#BstWPY/WildGraphBench&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=BstWPY/WildGraphBench&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=BstWPY/WildGraphBench&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=BstWPY/WildGraphBench&type=Date" />
 </picture>
</a>

---

<div align="center">

**⭐ Star us on GitHub if you find this benchmark useful! ⭐**

</div>
