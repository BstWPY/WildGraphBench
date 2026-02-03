#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Phase 2: Generate Type1 / Type2 / Type3 QA based on extracted triples.

Input:
  - --triples-valid: valid triple JSONL output from wiki_extractor.py

Output:
  - --out: QA dataset (similar to your original qa.jsonl structure)
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
    Lightweight reference resolver for QA phase:
    - Read title for each URL from references.jsonl
    - Scan reference_pages/*.md
    - Use title matching (exact + prefix + similarity) to find corresponding MD file
    - url_meta[url]["file"] points to md path
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
            print(f"  [ReferenceResolver] references.jsonl not found: {self.ref_jsonl_path}")
            return

        self.url_to_title = {}
        self.title_to_file = {}
        self.url_meta = {}

        # 1) Read references.jsonl → url_to_title / url_meta
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

        # 2) Scan all *.md files under reference_pages
        actual_files: Dict[str, Path] = {}
        actual_files_original: Dict[str, str] = {}
        if self.ref_pages_dir.exists():
            for md_file in self.ref_pages_dir.glob("*.md"):
                norm = self._normalize_for_matching(md_file.stem)
                actual_files[norm] = md_file
                actual_files_original[norm] = md_file.stem

        print(f"  [ReferenceResolver] Found {len(actual_files)} actual MD files")

        # 3) Round 1: Exact match
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
        print(f"  [ReferenceResolver] Exact match: {matched_count}/{len(self.url_to_title)} URLs")

        # 4) Round 2: Prefix fuzzy match
        if unmatched_titles_info:
            print(f"  [ReferenceResolver] Attempting prefix fuzzy match for {len(unmatched_titles_info)} unmatched titles...")
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
                        print(f"    [Fuzzy1/2] '{title[:60]}...' → '{file_orig[:60]}...'")
                        break

                if not matched:
                    still_unmatched.append((url, title, norm_title))

            matched_count += fuzzy_matched
            unmatched_titles_info = still_unmatched
            print(f"  [ReferenceResolver] Prefix fuzzy match: {fuzzy_matched} additional matches")

        # 5) Round 3: Similarity match
        if unmatched_titles_info:
            print(f"  [ReferenceResolver] Attempting similarity match for remaining {len(unmatched_titles_info)} titles...")
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
                    print(f"    [Fuzzy3] Similarity {best_score:.3f}: '{title[:60]}...' → '{file_orig[:60]}...'")
                else:
                    still_unmatched2.append((url, title, norm_title))

            unmatched_titles_info = still_unmatched2
            print(f"  [ReferenceResolver] Similarity match: {sim_matched} additional matches")

        # 6) Report unmatched MD files
        unmatched_md_files = []
        for norm_file, orig_file in actual_files_original.items():
            if norm_file not in matched_normalized_files:
                unmatched_md_files.append({
                    "original_name": orig_file,
                    "normalized_name": norm_file,
                    "file_path": actual_files[norm_file],
                })

        if unmatched_md_files:
            print(f"\n  [ReferenceResolver] {len(unmatched_md_files)} MD files not matched to any reference title")
            for i, info in enumerate(unmatched_md_files[:10], 1):
                print(f"    {i}. File: '{info['original_name'][:80]}...' "
                      f"Normalized: '{info['normalized_name'][:80]}'")

        print(f"[ReferenceResolver] Total URLs matched: {matched_count}/{len(self.url_to_title)}")
def _resolve_ref_urls_to_docs(resolver: ReferenceResolver, ref_urls: List[str]) -> List[Dict]:
    """
    Load corresponding MD content based on ref_urls saved in triple:
    Returns list like [{"url":..., "title":..., "content":...}, ...]
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
    For each statement, use the reference_pages content corresponding to its ref_urls to do post-hoc verification.
    Returns:
      kept_items:  [{"statement":..., "ref_urls":[...], ...}, ...]
      dropped:     [{"idx":..., "statement":..., "reason":...}, ...]
    """
    if not items:
        return [], []

    use_items = items[:max_items]
    path_str = " > ".join(section_path)
    leaf_topic = section_path[-1] if section_path else ""

    blocks = []
    # First pack each statement + its refs content into prompt
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
        # On failure: conservative strategy (avoid false negatives)
        return use_items, []

    out_items = data.get("items", [])
    if not isinstance(out_items, list):
        return use_items, []

    keep_flags = [False] * len(use_items)  # Default to False here (stricter), consistent with "discard if insufficient evidence"
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

    # Items beyond max_items: you can discard directly (stricter) or keep (more conservative)
    # Recommended: discard tail items directly to avoid "unverified statements mixing in"
    return kept, dropped

def _validate_support_with_refs(
    model: str,
    question: str,
    answer: str,
    refs: List[Dict],
    max_refs: int = 6,
) -> bool:
    """
    Check if these references "together" are sufficient to support this Q/A.
    (Corresponds to _validate_support_with_refs in the larger script)
    """
    if not refs:
        return False

    print(f"  [Validation Check] Checking if {len(refs)} references support this Q/A...")

    bundle_parts = []
    for i, r in enumerate(refs[:max_refs], 1):
        title = r.get("title") or f"Reference {i}"
        content = r.get("content") or ""
        if not content.strip():
            continue
        bundle_parts.append(f"### [{i}] {title}\n{content}\n{'-'*40}")

    if not bundle_parts:
        print("  [Validation Check] All reference contents are empty, determined as not supported")
        return False

    bundle = "\n".join(bundle_parts)
    print(f"  [Validation Check] Sending {len(bundle)} chars of references to LLM to check support...")

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
        print("  [Validation Check] LLM returned abnormal format, defaulting to not supported")
        return False

    supported = bool(data.get("supported", False))
    reason = str(data.get("reason", "") or "")
    if supported:
        print(f"  [Validation Check] Support check passed: {reason[:120]}")
    else:
        print(f"  [Validation Check] Support check failed: {reason[:120]}")
    return supported


def _validate_multi_ref_necessity_for_statement(
    model: str,
    question: str,
    statement: str,
    refs: List[Dict],
    max_refs: int = 4,
) -> bool:
    """
    Multi-reference "all needed" check (specifically for Type2):

    Semantics:
      - If any single reference alone can support ALL key facts in the STATEMENT,
        then all_needed=false -> this QA is not truly a "multi-reference" question and should be discarded.
      - Only when "each individual ref is insufficient and >=2 refs must be combined to cover the entire statement"
        do we return all_needed=true.

    The question is also provided to help the model understand which facts are relevant to the question.
    """
    if not refs or len(refs) < 2:
        # Not multi-reference
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
        print("  [MultiRefCheck] All selected reference contents are empty, defaulting to not passed")
        return False

    bundle = "\n".join(bundle_parts)
    print(f"  [MultiRefCheck] Sending {len(use_refs)} references to LLM to check if \"all needed\"...")

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
        print("  [MultiRefCheck] LLM returned abnormal format, defaulting to not passed")
        return False

    all_needed = bool(data.get("all_needed", False))
    reason = str(data.get("reason", "") or "")
    if all_needed:
        print(f"  [MultiRefCheck PASSED] Determined as \"all needed\": {reason[:120]}")
        return True
    else:
        print(f"  [MultiRefCheck FAILED] Determined as \"can be covered by single ref\": {reason[:120]}")
        return False

def _json_chat(model: str, prompt: str, max_tokens: int = 4000, temperature: float = 0.3) -> dict:
    # Same implementation as in extract_triples.py, can be directly copied
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

# ============= Utility Functions =============

def clean_body_to_answer(body: str) -> str:
    """
    For Type3: generate answer_clean from section_body
    - Remove [[N]](...) / [[N]]
    - Remove bare URLs
    """
    if not body:
        return ""
    # Remove [[N]](url)
    body = re.sub(r'\[\[\d+\]\]\([^)]+\)', '', body)
    # Remove [[N]]
    body = re.sub(r'\[\[\d+\]\]', '', body)
    # Remove bare URLs
    body = re.sub(r'https?://[^\s)]+', '', body)
    # Compress whitespace
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

# ============= LLM Question Generation: Type1 / Type2 (question only) =============

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
    Returns only the question string; answer uses the statement directly
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

    # Control length: only use first max_statements items, and truncate each
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


# ============= Main Flow: Read triples_valid.jsonl and generate QA =============

def main():
    ap = argparse.ArgumentParser(description="Phase 2: Generate QA based on triples")
    ap.add_argument("--triples-valid", type=str, required=True,
                    help="Path to valid_triples.jsonl generated by wiki_extraction.py")
    ap.add_argument("--out", type=str, required=True,
                    help="Output QA JSONL path")
    ap.add_argument("--num-type1", type=int, default=0,
                    help="Type1 single-fact target count (global)")
    ap.add_argument("--num-type2", type=int, default=0,
                    help="Type2 multi-fact target count (global)")
    ap.add_argument("--num-type3", type=int, default=100,
                    help="Type3 summary target count (global)")
    ap.add_argument("--val-max-refs", type=int, default=6,
                    help="Max number of references to use for Type1/Type2 post-hoc validation to use for Type1/Type2 post-hoc validation")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    random.seed(args.seed)

    triples_path = Path(args.triples_valid)
    if not triples_path.exists():
        print(f"[Error] triples-valid does not exist: {triples_path}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all leaf section recordsn records
    leaf_records = []
    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            leaf_records.append(json.loads(line))

    print(f"[Loaded] Total {len(leaf_records)} leaf section records")

    # Split into single / multi candidates + summary candidates
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

            # Summary aggregation
            g["items"].append({
            "statement": stmt,
            "ref_urls": ref_urls,
            "sentence": t.get("sentence", ""),
            })
            g["ref_urls"].extend(ref_urls)

    print(f"[Candidate Statistics]")
    print(f"  single-fact candidate triples: {len(single_triples)}")
    print(f"  multi-fact candidate triples:  {len(multi_triples)}")
    print(f"  summary candidate sections:    {len(summary_groups)}")

    # Shuffle order and truncate by target count
    random.shuffle(single_triples)
    random.shuffle(multi_triples)
    summary_keys = list(summary_groups.keys())
    random.shuffle(summary_keys)

    target_t1 = min(args.num_type1, len(single_triples)) if args.num_type1 > 0 else 0
    target_t2 = min(args.num_type2, len(multi_triples)) if args.num_type2 > 0 else 0
    target_t3 = min(args.num_type3, len(summary_keys)) if args.num_type3 > 0 else 0
    # Prepare for post-hoc validation: cache ReferenceResolver by topic_dir
    resolver_cache: Dict[str, ReferenceResolver] = {}

    def get_resolver(topic_dir: str) -> ReferenceResolver:
        if topic_dir not in resolver_cache:
            print(f"[Post-hoc Validation] Initializing ReferenceResolver: {topic_dir}")
            resolver_cache[topic_dir] = ReferenceResolver(Path(topic_dir))
        return resolver_cache[topic_dir]

    print(f"[Sampling Targets]")
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

            # ★ Post-hoc 1: Load reference documents
            resolver = get_resolver(meta["topic_dir"])
            refs = _resolve_ref_urls_to_docs(resolver, meta["ref_urls"])
            if not refs:
                print("[Type1 Validation] Cannot load any reference document content, discarding this QA")
                continue

            # ★ Validation 2: Check if "all references together support Q/A"
            ok_support = _validate_support_with_refs(
                model=OPENAI_MODEL,
                question=q,
                answer=meta["statement"],
                refs=refs,
                max_refs=args.val_max_refs,
            )
            if not ok_support:
                print("[Type1 Validation] References collectively insufficient to support this Q/A, discarding")
                continue

            qa = {
                "question": q,
                "answer": meta["statement"],
                "question_type": ["single-fact"],
                "source": [{
                    "wiki_title": meta["wiki_title"],
                    "section_path": meta["section_path"],
                    "wiki_sentences": [meta["sentence"]],
                    # Use URLs that actually loaded content successfully for reliability
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

            # ★ Validation 1: Load reference documents
            resolver = get_resolver(meta["topic_dir"])
            refs = _resolve_ref_urls_to_docs(resolver, meta["ref_urls"])

            if len(refs) < 2:
                print("[Type2 Validation] Less than 2 available references, cannot form multi-ref QA, discarding")
                continue

            # ★ Now only do "all needed" check (multi-reference necessity)
            ok_multi = _validate_multi_ref_necessity_for_statement(
                model=OPENAI_MODEL,
                question=q,
                statement=meta["statement"],
                refs=refs,
                max_refs=min(args.val_max_refs, len(refs)),
            )
            if not ok_multi:
                print("[Type2 Validation] All-needed condition not met (single ref can cover, or overall info insufficient), discarding")
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

            # 1) Original items (statement <-> ref_urls correspondence)
            raw_items = group.get("items", [])
            # Remove empty statement / none
            raw_items = [
                it for it in raw_items
                if (it.get("statement") or "").strip()
                and (it.get("statement") or "").strip().lower() != "none"
            ]

            # 2) First deduplicate by statement text and merge ref_urls
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

            # 3) Validation: verify each statement is "supported" by corresponding refs
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

            # 4) Count threshold: skip if <2 after filtering
            if len(kept_items) < 2:
                print(f"[Type3 Skip] Only {len(kept_items)} statements left after refs-support (raw={len(items_dedup)}), skipping this section")
                continue

            # 5) answer_clean filter
            answer_clean = clean_body_to_answer(body)
            if len(answer_clean) < 40:
                continue

            # 6) Generate question: feed validated gold_statements to model for guidance
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

            # 7) refs: only keep refs from validated statements (cleaner)
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
                # Optional: debug info (remove if you don't want to change data format)
                "posthoc": {
                    "dropped_count": len(dropped),
                    "dropped_examples": dropped[:5],
                }
            }
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            total_qa += 1
            type_counts["summary"] += 1

    print("\n[Complete] QA generation finished")
    print(f"  Total QA count: {total_qa}")
    print(f"  single-fact: {type_counts['single-fact']}")
    print(f"  multi_fact:  {type_counts['multi_fact']}")
    print(f"  summary:     {type_counts['summary']}")
    print(f"  Output file: {out_path}")

if __name__ == "__main__":
    main()