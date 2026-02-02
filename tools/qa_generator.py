#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

阶段二：基于抽取好的 triple 出题，生成 Type1 / Type2 / Type3 QA。

输入：
  - --triples-valid: wiki_extractor.py 输出的有效 triple JSONL

输出：
  - --out: QA 数据集（与你原来 qa.jsonl 结构类似）
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
import unicodedata
import difflib

load_dotenv()

# Configure your LLM API credentials via environment variables or .env file
# Supports OpenAI-compatible APIs (OpenAI, Azure, etc.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_JSON_CHAT_CACHE: Dict[Tuple[str, str, int, float], dict] = {}
JSON_BOOL_SUPPORT = '{"supported": true/false, "reason": "brief"}'
JSON_ALL_NEEDED = '{"all_needed": true/false, "reason": "brief"}'
JSON_FILTER_STATEMENTS = '{"items":[{"idx":1,"keep":true/false,"reason":"brief"}], "summary":"brief"}'
JSON_FILTER_REF_SUPPORT = '{"items":[{"idx":1,"keep":true/false,"reason":"brief"}], "summary":"brief"}'

class ReferenceResolver:
    """
    QA 阶段用的轻量版引用解析器：
    - 从 references.jsonl 读取每个 URL 对应的标题
    - 扫描 reference_pages/*.md
    - 用标题匹配（精确 + 前缀 + 相似度）找到对应 MD 文件
    - url_meta[url]["file"] 指向 md 路径
    """
    def __init__(self, topic_dir: Path):
        self.topic_dir = topic_dir
        self.ref_jsonl_path = topic_dir / "reference" / "references.jsonl"
        self.ref_pages_dir = topic_dir / "reference" / "reference_pages"
        self.url_to_title: Dict[str, str] = {}
        self.title_to_file: Dict[str, Path] = {}
        self.url_meta: Dict[str, dict] = {}
        self._load_references()

    def _normalize_for_matching(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace("_", " ")
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _load_references(self):
        if not self.ref_jsonl_path.exists():
            print(f"  [参考解析器] 未找到 references.jsonl: {self.ref_jsonl_path}")
            return

        self.url_to_title = {}
        self.title_to_file = {}
        self.url_meta = {}

        # 1) 读取 references.jsonl → url_to_title / url_meta
        with self.ref_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    ref = json.loads(line.strip())
                    url = ref.get("url", "").split("#")[0]
                    title = (ref.get("title", "") or "").strip()
                    if not url or not title:
                        continue
                    self.url_to_title[url] = title
                    self.url_meta[url] = {
                        "title": title,
                        "scraped": bool(ref.get("scraped", False)),
                        "is_external": bool(ref.get("is_external", True)),
                        "file": None,
                    }
                except Exception:
                    continue

        # 2) 扫描 reference_pages 下所有 *.md
        actual_files: Dict[str, Path] = {}
        actual_files_original: Dict[str, str] = {}
        if self.ref_pages_dir.exists():
            for md_file in self.ref_pages_dir.glob("*.md"):
                norm = self._normalize_for_matching(md_file.stem)
                actual_files[norm] = md_file
                actual_files_original[norm] = md_file.stem

        print(f"  [参考解析器] 找到 {len(actual_files)} 个实际 MD 文件")

        # 3) 第一轮：精确匹配
        matched_count = 0
        matched_normalized_files = set()
        unmatched_titles_info: List[Tuple[str, str, str]] = []

        for url, meta in self.url_meta.items():
            title = meta["title"]
            norm_title = self._normalize_for_matching(title)
            if norm_title in actual_files:
                self.title_to_file[title] = actual_files[norm_title]
                meta["file"] = actual_files[norm_title]
                matched_count += 1
                matched_normalized_files.add(norm_title)
            else:
                unmatched_titles_info.append((url, title, norm_title))
        print(f"  [参考解析器] 精确匹配: {matched_count}/{len(self.url_to_title)} 个 URL")

        # 4) 第二轮：前缀模糊匹配
        if unmatched_titles_info:
            print(f"  [参考解析器] 尝试对 {len(unmatched_titles_info)} 个未匹配标题进行前缀模糊匹配...")
            fuzzy_matched = 0
            still_unmatched: List[Tuple[str, str, str]] = []

            unmatched_files = {
                norm: (actual_files[norm], actual_files_original[norm])
                for norm in actual_files.keys()
                if norm not in matched_normalized_files
            }

            for url, title, norm_title in unmatched_titles_info:
                matched = False

                for file_norm, (file_path, file_orig) in list(unmatched_files.items()):
                    min_len = min(len(norm_title), len(file_norm))
                    if min_len < 20:
                        continue

                    prefix_len = min(200, min_len)
                    title_prefix = norm_title[:prefix_len]
                    file_prefix = file_norm[:prefix_len]

                    if title_prefix == file_prefix:
                        matched = True
                    elif norm_title.startswith(file_norm) and len(file_norm) >= 50:
                        matched = True
                    elif file_norm.startswith(norm_title) and len(norm_title) >= 50:
                        matched = True

                    if matched:
                        self.title_to_file[title] = file_path
                        self.url_meta[url]["file"] = file_path
                        matched_normalized_files.add(file_norm)
                        fuzzy_matched += 1
                        unmatched_files.pop(file_norm, None)
                        print(f"    [模糊1/2] '{title[:60]}...' → '{file_orig[:60]}...'")
                        break

                if not matched:
                    still_unmatched.append((url, title, norm_title))

            matched_count += fuzzy_matched
            unmatched_titles_info = still_unmatched
            print(f"  [参考解析器] 前缀模糊匹配: {fuzzy_matched} 个额外匹配")

        # 5) 第三轮：相似度匹配
        if unmatched_titles_info:
            print(f"  [参考解析器] 尝试对剩余 {len(unmatched_titles_info)} 个标题进行相似度匹配...")
            sim_matched = 0
            still_unmatched2: List[Tuple[str, str, str]] = []

            unmatched_files = {
                norm: (actual_files[norm], actual_files_original[norm])
                for norm in actual_files.keys()
                if norm not in matched_normalized_files
            }

            for url, title, norm_title in unmatched_titles_info:
                best_norm = None
                best_score = 0.0

                for file_norm, (file_path, file_orig) in unmatched_files.items():
                    score = difflib.SequenceMatcher(None, norm_title, file_norm).ratio()
                    if score > best_score:
                        best_score = score
                        best_norm = file_norm

                if best_norm is not None and best_score >= 0.70:
                    file_path, file_orig = unmatched_files.pop(best_norm)
                    self.title_to_file[title] = file_path
                    self.url_meta[url]["file"] = file_path
                    matched_normalized_files.add(best_norm)
                    sim_matched += 1
                    matched_count += 1
                    print(f"    [模糊3] 相似度 {best_score:.3f}: '{title[:60]}...' → '{file_orig[:60]}...'")
                else:
                    still_unmatched2.append((url, title, norm_title))

            unmatched_titles_info = still_unmatched2
            print(f"  [参考解析器] 相似度匹配: {sim_matched} 个额外匹配")

        # 6) 报告未匹配 md
        unmatched_md_files = []
        for norm_file, orig_file in actual_files_original.items():
            if norm_file not in matched_normalized_files:
                unmatched_md_files.append({
                    "original_name": orig_file,
                    "normalized_name": norm_file,
                    "file_path": actual_files[norm_file],
                })

        if unmatched_md_files:
            print(f"\n  [参考解析器] 有 {len(unmatched_md_files)} 个 MD 文件未匹配到任何引用标题")
            for i, info in enumerate(unmatched_md_files[:10], 1):
                print(f"    {i}. 文件: '{info['original_name'][:80]}...' "
                      f"标准化: '{info['normalized_name'][:80]}'")

        print(f"[参考解析器] 总共匹配 URL: {matched_count}/{len(self.url_to_title)}")
def _resolve_ref_urls_to_docs(resolver: ReferenceResolver, ref_urls: List[str]) -> List[Dict]:
    """
    根据 triple 里保存的 ref_urls，加载对应的 MD 内容：
    返回形如 [{"url":..., "title":..., "content":...}, ...]
    """
    refs = []
    seen_keys = set()
    for url in ref_urls or []:
        clean_url = url.split("#")[0]
        meta = resolver.url_meta.get(clean_url)
        if not meta:
            continue
        file_path = meta.get("file")
        if not file_path:
            continue
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        title = (meta.get("title") or "").strip()
        key = title or clean_url
        if key in seen_keys:
            continue
        seen_keys.add(key)
        refs.append({
            "url": clean_url,
            "title": title or clean_url,
            "content": content,
        })
    return refs
def llm_filter_statement_items_by_ref_support(
    model: str,
    wiki_title: str,
    section_path: List[str],
    items: List[Dict],
    resolver: ReferenceResolver,
    max_items: int = 20,
    max_refs_per_item: int = 3,
    ref_excerpt_chars: int = 1200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    对每条 statement，用其 ref_urls 对应的 reference_pages 内容做“是否支持 statement”的后验。
    返回：
      kept_items:  [{"statement":..., "ref_urls":[...], ...}, ...]
      dropped:     [{"idx":..., "statement":..., "reason":...}, ...]
    """
    if not items:
        return [], []

    use_items = items[:max_items]
    path_str = " > ".join(section_path)
    leaf_topic = section_path[-1] if section_path else ""

    blocks = []
    # 先把每条 statement + 其 refs 内容打包进 prompt
    for i, it in enumerate(use_items, 1):
        stmt = (it.get("statement") or "").strip()
        ref_urls = it.get("ref_urls") or []
        refs = _resolve_ref_urls_to_docs(resolver, ref_urls)[:max_refs_per_item]

        if not refs:
            blocks.append(f"""
### ITEM {i}
STATEMENT:
{stmt}

REFERENCES:
<NO REFERENCE CONTENT LOADED>
""".strip())
            continue

        ref_parts = []
        for j, r in enumerate(refs, 1):
            title = r.get("title") or f"Ref {j}"
            content = (r.get("content") or "")[:ref_excerpt_chars]
            ref_parts.append(f"#### REF {i}.{j} {title}\n{content}\n{'-'*20}")

        blocks.append(f"""
### ITEM {i}
STATEMENT:
{stmt}

REFERENCES:
{chr(10).join(ref_parts)}
""".strip())

    bundle = "\n\n".join(blocks)

    prompt = f"""
You are doing a POST-HOC VERIFICATION for a citation-based summary dataset.

ARTICLE TITLE:
{wiki_title}

LEAF SECTION TOPIC PATH:
{path_str}

Leaf topic (most specific):
{leaf_topic}

You will be given multiple ITEMS. Each item has:
- a STATEMENT (candidate gold statement)
- several REFERENCES (content excerpts)

Task:
For EACH item:
- keep=true only if the REFERENCES (collectively) contain enough information to support ALL key factual claims in the STATEMENT.
- keep=false if key facts are missing, contradicted, or the references are irrelevant/noisy.

Rules:
- Use ONLY the given references; ignore outside knowledge.
- Be fairly strict: if unsure due to missing evidence, set keep=false.

Return JSON ONLY:
{JSON_FILTER_REF_SUPPORT}
""".strip()

    data = _json_chat(model=model, prompt=prompt, max_tokens=8000, temperature=0.0)
    if not isinstance(data, dict):
        # 失败时：保守策略（不误杀）
        return use_items, []

    out_items = data.get("items", [])
    if not isinstance(out_items, list):
        return use_items, []

    keep_flags = [False] * len(use_items)  # 这里建议默认 False（严格一些），与“证据不足就丢”一致
    reasons = [""] * len(use_items)

    for it in out_items:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("idx"))
        except Exception:
            continue
        if 1 <= idx <= len(use_items):
            keep_flags[idx - 1] = bool(it.get("keep", False))
            reasons[idx - 1] = str(it.get("reason", "") or "").strip()

    kept, dropped = [], []
    for i, it in enumerate(use_items):
        if keep_flags[i]:
            kept.append(it)
        else:
            dropped.append({"idx": i + 1, "statement": it.get("statement", ""), "reason": reasons[i]})

    # max_items 之外的尾部：你可以直接丢弃（更严格）或保留（更保守）
    # 我这里建议：尾部直接丢弃，避免“未验证的 statement 混进来”
    return kept, dropped

def _validate_support_with_refs(
    model: str,
    question: str,
    answer: str,
    refs: List[Dict],
    max_refs: int = 6,
) -> bool:
    """
    检查：这些参考文献“合起来”是否足以支持这个 Q/A。
    （对应你大脚本里的 _validate_support_with_refs）
    """
    if not refs:
        return False

    print(f"  [后验检查] 正在检查 {len(refs)} 个参考文献是否支持该问答...")

    bundle_parts = []
    for i, r in enumerate(refs[:max_refs], 1):
        title = r.get("title") or f"Reference {i}"
        content = r.get("content") or ""
        if not content.strip():
            continue
        bundle_parts.append(f"### [{i}] {title}\n{content}\n{'-'*40}")

    if not bundle_parts:
        print("  [后验检查] 所有参考文献内容为空，判定为不支持")
        return False

    bundle = "\n".join(bundle_parts)
    print(f"  [后验检查] 发送 {len(bundle)} 字符的参考文献给 LLM 检查支持度...")

    prompt = f"""
You are checking whether the provided REFERENCES collectively support a Q&A.

Q: {question}
A: {answer}

REFERENCES (may include some noise, read holistically):
{bundle}

Rules:
- If the references together contain the key facts to justify the answer, return supported=true.
- If key facts are missing or contradicted, return supported=false.

Return JSON ONLY (do NOT explain your reasoning process, be concise):
{JSON_BOOL_SUPPORT}
""".strip()

    data = _json_chat(
        model=model,
        prompt=prompt,
        max_tokens=50000,
        temperature=0.0,
    )
    if not isinstance(data, dict):
        print("  [后验检查] LLM 返回格式异常，默认判定为不支持")
        return False

    supported = bool(data.get("supported", False))
    reason = str(data.get("reason", "") or "")
    if supported:
        print(f"  [后验检查] 支持度通过：{reason[:120]}")
    else:
        print(f"  [后验检查] 支持度不通过：{reason[:120]}")
    return supported


def _validate_multi_ref_necessity_for_statement(
    model: str,
    question: str,
    statement: str,
    refs: List[Dict],
    max_refs: int = 4,
) -> bool:
    """
    多引用“缺一不可”检查（专门针对 Type2）：

    语义：
      - 如果存在某一个参考文献单独就能支撑 STATEMENT 里的所有关键事实，
        那么 all_needed=false → 这一问答不算真正的“多引用”题，应该丢弃。
      - 只有在“每一篇 ref 单独都不够，必须综合 >=2 篇 ref 才能覆盖整个 statement”
        的情况下，才 all_needed=true。

    这里同时把 question 提供给模型，方便它理解哪些事实是跟问题相关的。
    """
    if not refs or len(refs) < 2:
        # 本身就不是多引用
        return False

    use_refs = refs[:max_refs]

    bundle_parts = []
    for i, r in enumerate(use_refs, 1):
        title = r.get("title") or f"Reference {i}"
        content = r.get("content") or ""
        if not content.strip():
            continue
        bundle_parts.append(f"### [REF {i}] {title}\n{content}\n{'-'*40}")

    if not bundle_parts:
        print("  [多引用检查] 所选引用内容全部为空，默认判定为不通过")
        return False

    bundle = "\n".join(bundle_parts)
    print(f"  [多引用检查] 发送 {len(use_refs)} 个引用给 LLM，判断是否“缺一不可”...")

    prompt = f"""
You are given a factual STATEMENT (used as the reference answer for a QA pair),
together with the QUESTION and several reference documents cited from Wikipedia.

QUESTION:
{question}

REFERENCE ANSWER (STATEMENT):
{statement}

REFERENCES:
{bundle}

Your task is to judge whether these references are **jointly necessary**
to support the FULL factual content of the STATEMENT.

Rules:

1. Consider ONLY the information contained in the given references. Ignore any outside world knowledge.

2. For EACH reference individually, imagine you only had that single reference:
   - If that single reference ALONE already contains enough information to support
     ALL key factual claims in the STATEMENT (numbers, named entities, relationships,
     important conditions), then that reference is "individually sufficient"
     to justify the STATEMENT.

3. If **ANY** single reference is individually sufficient, then the multi-reference pattern is
   NOT truly necessary.
   → In this case, set all_needed = false.

4. Only if **NO** single reference is individually sufficient (each one misses some essential facts),
   and you really need to COMBINE at least two references to cover the full STATEMENT,
   set all_needed = true.

"Key factual claims" means the main facts expressed by the STATEMENT,
not minor stylistic details.

Return JSON ONLY in the following format:
{JSON_ALL_NEEDED}
""".strip()

    data = _json_chat(
        model=model,
        prompt=prompt,
        max_tokens=8000,
        temperature=0.0,
    )

    if not isinstance(data, dict):
        print("  [多引用检查] LLM 返回格式异常，默认判定为不通过")
        return False

    all_needed = bool(data.get("all_needed", False))
    reason = str(data.get("reason", "") or "")
    if all_needed:
        print(f"  [多引用检查通过] 判定为“缺一不可”：{reason[:120]}")
        return True
    else:
        print(f"  [多引用检查失败] 判定为“可由单个引用覆盖”：{reason[:120]}")
        return False

def _json_chat(model: str, prompt: str, max_tokens: int = 4000, temperature: float = 0.3) -> dict:
    # 和前面 extract_triples.py 里的实现相同，可以直接 copy
    import requests
    global _JSON_CHAT_CACHE

    full_prompt = f"You are a careful data-wrangler. Return ONLY valid JSON.\n\n{prompt}"
    cache_key = (model, full_prompt, int(max_tokens), float(temperature))
    if cache_key in _JSON_CHAT_CACHE:
        return _JSON_CHAT_CACHE[cache_key]

    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "contents": [
            {"role": "user", "parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    try:
        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=600,
            proxies={"http": None, "https": None}
        )
        if resp.status_code != 200:
            print(f"[LLM Err] {resp.status_code}: {resp.text[:300]}")
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}
        data = resp.json()
        content_text = None
        if "candidates" in data and data["candidates"]:
            cand = data["candidates"][0]
            if "content" in cand and "parts" in cand["content"]:
                parts = cand["content"]["parts"]
                if parts and "text" in parts[0]:
                    content_text = parts[0]["text"]
        elif "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            content_text = msg.get("content", "")
        elif "data" in data and data["data"]:
            content_text = data["data"][0].get("text") or ""

        if not content_text:
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}

        content_text = re.sub(
            r'^```(?:json)?\s*|\s*```$',
            '',
            content_text,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        try:
            result = json.loads(content_text)
            if isinstance(result, dict):
                _JSON_CHAT_CACHE[cache_key] = result
            else:
                _JSON_CHAT_CACHE[cache_key] = {}
            return result
        except Exception:
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}
    except Exception:
        _JSON_CHAT_CACHE[cache_key] = {}
        return {}

# ============= 小工具 =============

def clean_body_to_answer(body: str) -> str:
    """
    Type3 用：从 section_body 生成 answer_clean
    - 去掉 [[N]](...) / [[N]]
    - 去掉裸 URL
    """
    if not body:
        return ""
    # 去 [[N]](url)
    body = re.sub(r'\[\[\d+\]\]\([^)]+\)', '', body)
    # 去 [[N]]
    body = re.sub(r'\[\[\d+\]\]', '', body)
    # 去裸 URL
    body = re.sub(r'https?://[^\s)]+', '', body)
    # 压空格
    body = re.sub(r'[ \t]+', ' ', body)
    body = re.sub(r'\n{3,}', '\n\n', body)
    return body.strip()

def dedup_list_keep_order(xs: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# ============= LLM 出题：Type1 / Type2（只出 question） =============

def llm_make_question_for_statement(
    model: str,
    wiki_title: str,
    section_path: List[str],
    sentence: str,
    statement: str,
    ref_urls: List[str],
    style: str,
) -> str:
    """
    style: "single-fact" or "multi_fact"
    只返回 question 字符串；answer 直接用 statement
    """
    path_str = " > ".join(section_path)
    refs_str = "\n".join(f"- {u}" for u in ref_urls[:5])

    example_json = {"question": "..."}

    style_desc = "SINGLE-FACT (supported by one citation)" if style == "single-fact" \
                 else "MULTI-FACT (requires several citations together)"

    prompt = f"""
You are constructing a question for a {style_desc} citation-based QA dataset.

ARTICLE TITLE:
{wiki_title}

SECTION PATH:
{path_str}

WIKI SENTENCE (with inline citations):
{sentence}

CLEAN FACTUAL STATEMENT (this will be used as the reference answer):
{statement}

REFERENCE URLS (for context, do NOT quote them explicitly):
{refs_str}

Your task:
- Write ONE natural-language QUESTION in English or Chinese (depending on the style of the article),
  such that:
  - The gold answer should be exactly the given STATEMENT (possibly with tiny paraphrasing).
  - The question should contain **multiple constraints** (e.g. entity + time, quantity + condition, entity + location).
  - If any of these constraints was removed, the question would become under-specified or wrong.
  - The question must be answerable solely from the given statement and sentence.
- The question should feel natural and non-trivial:
  - Do NOT copy any span of 4 or more consecutive words from the sentence or the statement.
  - Avoid generic patterns like "What is X?", "Who is Y?", "When did X happen?".

Return JSON ONLY:
{json.dumps(example_json, ensure_ascii=False)}
""".strip()

    data = _json_chat(model=model, prompt=prompt, max_tokens=50000, temperature=0.6)
    if not isinstance(data, dict):
        return ""
    q = str(data.get("question", "")).strip()
    return q

def llm_make_question_for_section_guided(
    model: str,
    wiki_title: str,
    section_path: List[str],
    gold_statements: List[str],
    section_body: str = "",
    max_statements: int = 8,
) -> str:
    path_str = " > ".join(section_path)
    example_json = {"question": "..."}

    # 控制长度：只给前 max_statements 条，并截断每条
    gs = []
    for s in (gold_statements or [])[:max_statements]:
        s = (s or "").strip()
        if not s:
            continue
        gs.append(s[:300])
    gs_text = "\n".join([f"- {s}" for s in gs])

    body_excerpt = clean_body_to_answer(section_body)[:600] if section_body else ""

    prompt = f"""
You are constructing a TOPIC-CENTERED SUMMARY QUESTION for a topic.

TOPIC PATH (broad -> specific):
{path_str}

OPTIONAL BODY EXCERPT (for natural phrasing only):
{body_excerpt}

GOLD STATEMENTS (facts that a good answer SHOULD cover; do NOT quote them):
{gs_text}

Your task:
- Write ONE natural-language question that asks for a concise, encyclopedic-style overview of the MOST SPECIFIC topic
  (typically the LAST 1–2 elements of the path).
- Use the GOLD STATEMENTS only as soft guidance to choose what aspects to emphasize,
  so that the answer naturally tends to cover those facts.
- The question must remain strongly anchored to the leaf topic in the path.

STRICT constraints:
- DO NOT mention Wikipedia/article/section/heading or similar meta words.
- DO NOT copy any span of 4+ consecutive words from any gold statement.
- Avoid leaking specific factual details from the gold statements in the question
  (especially exact numbers, exact dates, long proper names, or verbatim event descriptions).
  You may mention high-level aspects (e.g., "history", "structure", "major components", "development", "reception")
  if they align with the leaf topic and the gold statements.
- 20–200 characters.

Return JSON ONLY:
{json.dumps(example_json, ensure_ascii=False)}
""".strip()

    data = _json_chat(model=model, prompt=prompt, max_tokens=4000, temperature=0.5)
    if not isinstance(data, dict):
        return ""
    return str(data.get("question", "")).strip()


# ============= 主流程：读 triples_valid.jsonl 出题 =============

def main():
    ap = argparse.ArgumentParser(description="阶段二：基于 triple 出题生成 QA")
    ap.add_argument("--triples-valid", type=str, required=True,
                    help="wiki_extraction.py 生成的 valid_triples.jsonl 文件路径")
    ap.add_argument("--out", type=str, required=True,
                    help="输出 QA JSONL 路径")
    ap.add_argument("--num-type1", type=int, default=0,
                    help="Type1 single-fact 目标数量（全局）")
    ap.add_argument("--num-type2", type=int, default=0,
                    help="Type2 multi-fact 目标数量（全局）")
    ap.add_argument("--num-type3", type=int, default=100,
                    help="Type3 summary 目标数量（全局）")
    ap.add_argument("--val-max-refs", type=int, default=6,
                    help="Type1/Type2 后验检查时最多使用多少篇参考文献")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    random.seed(args.seed)

    triples_path = Path(args.triples_valid)
    if not triples_path.exists():
        print(f"[错误] triples-valid 不存在: {triples_path}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取所有 leaf section 记录
    leaf_records = []
    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            leaf_records.append(json.loads(line))

    print(f"[加载] 共 {len(leaf_records)} 个 leaf section 记录")

    # 拆成 single / multi 候选 + summary 候选
    single_triples = []  # (meta, triple_dict)
    multi_triples = []
    summary_groups = {}  # key: (wiki_title, topic_dir, tuple(path)) -> {"body":..., "statements":[], "ref_urls":[]}

    for rec in leaf_records:
        wiki_title = rec["wiki_title"]
        topic_dir = rec["topic_dir"]
        section_path = rec["section_path"]
        section_body = rec["section_body"]
        triples = rec.get("triples", [])

        key = (wiki_title, topic_dir, tuple(section_path))
        g = summary_groups.setdefault(key, {"body": section_body, "items": [], "ref_urls": []})


        for t in triples:
            stmt = t.get("statement")
            if isinstance(stmt, str):
                stmt = stmt.strip()
            if not stmt or stmt.lower() == "none":
                continue
            ref_urls = t.get("ref_urls", []) or []
            ref_count = int(t.get("ref_count", len(ref_urls)))
            meta = {
                "wiki_title": wiki_title,
                "topic_dir": topic_dir,
                "section_path": section_path,
                "sentence": t["sentence"],
                "statement": stmt,
                "ref_urls": ref_urls,
            }
            if ref_count == 1:
                single_triples.append(meta)
            elif ref_count >= 2:
                multi_triples.append(meta)

            # summary 聚合
            g["items"].append({
            "statement": stmt,
            "ref_urls": ref_urls,
            "sentence": t.get("sentence", ""),
            })
            g["ref_urls"].extend(ref_urls)

    print(f"[候选统计]")
    print(f"  single-fact 候选 triple: {len(single_triples)}")
    print(f"  multi-fact 候选 triple:  {len(multi_triples)}")
    print(f"  summary 候选 section:   {len(summary_groups)}")

    # 打乱顺序再按需求数量截断
    random.shuffle(single_triples)
    random.shuffle(multi_triples)
    summary_keys = list(summary_groups.keys())
    random.shuffle(summary_keys)

    target_t1 = min(args.num_type1, len(single_triples)) if args.num_type1 > 0 else 0
    target_t2 = min(args.num_type2, len(multi_triples)) if args.num_type2 > 0 else 0
    target_t3 = min(args.num_type3, len(summary_keys)) if args.num_type3 > 0 else 0
    # 为后验检查准备：按 topic_dir 缓存 ReferenceResolver
    resolver_cache: Dict[str, ReferenceResolver] = {}

    def get_resolver(topic_dir: str) -> ReferenceResolver:
        if topic_dir not in resolver_cache:
            print(f"[后验检查] 初始化 ReferenceResolver: {topic_dir}")
            resolver_cache[topic_dir] = ReferenceResolver(Path(topic_dir))
        return resolver_cache[topic_dir]

    print(f"[采样目标]")
    print(f"  Type1: {target_t1}")
    print(f"  Type2: {target_t2}")
    print(f"  Type3: {target_t3}")

    total_qa = 0
    type_counts = {"single-fact": 0, "multi_fact": 0, "summary": 0}

    with out_path.open("w", encoding="utf-8") as fout:

        # ---- Type1 ----
        for meta in single_triples[:target_t1]:
            q = llm_make_question_for_statement(
                model=OPENAI_MODEL,
                wiki_title=meta["wiki_title"],
                section_path=meta["section_path"],
                sentence=meta["sentence"],
                statement=meta["statement"],
                ref_urls=meta["ref_urls"],
                style="single-fact",
            )
            if not q:
                continue

            # ★ 后验 1：加载引用文档
            resolver = get_resolver(meta["topic_dir"])
            refs = _resolve_ref_urls_to_docs(resolver, meta["ref_urls"])
            if not refs:
                print("[Type1 后验] 无法加载任何参考文档内容，丢弃该问答")
                continue

            # ★ 后验 2：检查“所有引用合起来是否支持 Q/A”
            ok_support = _validate_support_with_refs(
                model=OPENAI_MODEL,
                question=q,
                answer=meta["statement"],
                refs=refs,
                max_refs=args.val_max_refs,
            )
            if not ok_support:
                print("[Type1 后验] 参考文献整体不足以支持该 Q/A，丢弃")
                continue

            qa = {
                "question": q,
                "answer": meta["statement"],
                "question_type": ["single-fact"],
                "source": [{
                    "wiki_title": meta["wiki_title"],
                    "section_path": meta["section_path"],
                    "wiki_sentences": [meta["sentence"]],
                    # 用实际成功加载到内容的 URL，更稳妥
                    "ref_urls": [r["url"] for r in refs],
                }]
            }
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            total_qa += 1
            type_counts["single-fact"] += 1

        # ---- Type2 ----
        for meta in multi_triples[:target_t2]:
            q = llm_make_question_for_statement(
                model=OPENAI_MODEL,
                wiki_title=meta["wiki_title"],
                section_path=meta["section_path"],
                sentence=meta["sentence"],
                statement=meta["statement"],
                ref_urls=meta["ref_urls"],
                style="multi_fact",
            )
            if not q:
                continue

            # ★ 后验 1：加载引用文档
            resolver = get_resolver(meta["topic_dir"])
            refs = _resolve_ref_urls_to_docs(resolver, meta["ref_urls"])

            if len(refs) < 2:
                print("[Type2 后验] 可用参考文献少于 2 篇，无法构成多引用问答，丢弃")
                continue

            # ★ 现在只做“缺一不可”检查（多引用必要性）
            ok_multi = _validate_multi_ref_necessity_for_statement(
                model=OPENAI_MODEL,
                question=q,
                statement=meta["statement"],
                refs=refs,
                max_refs=min(args.val_max_refs, len(refs)),
            )
            if not ok_multi:
                print("[Type2 后验] 缺一不可条件不满足（存在单个引用就能覆盖，或整体信息不足），丢弃该问答")
                continue

            qa = {
                "question": q,
                "answer": meta["statement"],
                "question_type": ["multi_fact"],
                "source": [{
                    "wiki_title": meta["wiki_title"],
                    "section_path": meta["section_path"],
                    "wiki_sentences": [meta["sentence"]],
                    "ref_urls": [r["url"] for r in refs],
                }]
            }
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            total_qa += 1
            type_counts["multi_fact"] += 1

        # ---- Type3 ----
        for key in summary_keys[:target_t3]:
            wiki_title, topic_dir, path_tpl = key
            group = summary_groups[key]
            section_path = list(path_tpl)
            body = group["body"]

            # 1) 原始 items（statement ↔ ref_urls 对应）
            raw_items = group.get("items", [])
            # 去掉空 statement / none
            raw_items = [
                it for it in raw_items
                if (it.get("statement") or "").strip()
                and (it.get("statement") or "").strip().lower() != "none"
            ]

            # 2) 先做“按 statement 文本去重，并合并 ref_urls”
            merged = {}
            for it in raw_items:
                s = it["statement"].strip()
                if s not in merged:
                    merged[s] = {"statement": s, "ref_urls": [], "sentences": []}
                merged[s]["ref_urls"].extend(it.get("ref_urls") or [])
                if it.get("sentence"):
                    merged[s]["sentences"].append(it["sentence"])

            items_dedup = []
            for s, it in merged.items():
                it["ref_urls"] = dedup_list_keep_order(it["ref_urls"])
                items_dedup.append(it)

            # 3) 后验：逐条 statement 用对应 refs 验证是否“被支持”
            resolver = get_resolver(topic_dir)
            kept_items, dropped = llm_filter_statement_items_by_ref_support(
                model=OPENAI_MODEL,
                wiki_title=wiki_title,
                section_path=section_path,
                items=items_dedup,
                resolver=resolver,
                max_items=20,
                max_refs_per_item=3,
                ref_excerpt_chars=1200,
            )

            # 4) 数量门槛：过滤后 <2 直接跳过
            if len(kept_items) < 2:
                print(f"[Type3 跳过] refs-support 后仅剩 {len(kept_items)} 条 statement（raw={len(items_dedup)}），跳过该 section")
                continue

            # 5) answer_clean 过滤
            answer_clean = clean_body_to_answer(body)
            if len(answer_clean) < 40:
                continue

            # 6) 生成问题：把过滤后的 gold_statements 喂给模型做轻引导
            gold_stmts = [it["statement"] for it in kept_items]
            q = llm_make_question_for_section_guided(
                model=OPENAI_MODEL,
                wiki_title=wiki_title,
                section_path=section_path,
                gold_statements=gold_stmts,
                section_body=body,
            )
            if not q:
                continue

            # 7) refs：只保留“通过后验的 statements”的 refs（更干净）
            ref_urls = dedup_list_keep_order([u for it in kept_items for u in (it.get("ref_urls") or [])])

            qa = {
                "question": q,
                "question_type": ["summary"],
                "gold_statements": gold_stmts,
                "answer": answer_clean,
                "source": [{
                    "wiki_title": wiki_title,
                    "section_path": section_path,
                    "wiki_snippet": body,
                    "ref_urls": ref_urls,
                }],
                # 可选：调试信息（不想改数据格式就删掉这一段）
                "posthoc": {
                    "dropped_count": len(dropped),
                    "dropped_examples": dropped[:5],
                }
            }
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            total_qa += 1
            type_counts["summary"] += 1

    print("\n[完成] QA 生成结束")
    print(f"  总 QA 数: {total_qa}")
    print(f"  single-fact: {type_counts['single-fact']}")
    print(f"  multi_fact:  {type_counts['multi_fact']}")
    print(f"  summary:     {type_counts['summary']}")
    print(f"  输出文件: {out_path}")

if __name__ == "__main__":
    main()