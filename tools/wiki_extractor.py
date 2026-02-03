
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_triples.py

Phase 1: Extract (sentence, statement, refs) triples

Outputs two JSONL files:
  - --out-valid:  each line contains a list of valid triples for a leaf section
  - --out-invalid: each line contains a list of invalid triples for a leaf section
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import unicodedata
import difflib
from dotenv import load_dotenv

load_dotenv()

# ====================== LLM Configuration ======================
# Configure your LLM API credentials via environment variables or .env file
# Supports OpenAI-compatible APIs (OpenAI, Azure, Google, etc.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_JSON_CHAT_CACHE: Dict[Tuple[str, str, int, float], dict] = {}
def strip_wikipedia_title_header(md: str) -> str:
    """
    Remove headers like:
      Marvel Cinematic Universe - Wikipedia
      ===============
    So the real article starts from
      Marvel Cinematic Universe
      ===============
    """
    if not md:
        return md

    lines = md.splitlines()

    # Remove leading empty lines / BOM
    while lines and not lines[0].strip():
        lines.pop(0)

    if len(lines) < 2:
        return md

    first = lines[0].strip()
    second = lines[1].strip()

    # Case 1: Setext format
    # Marvel Cinematic Universe - Wikipedia
    # ===============
    if first.endswith(" - Wikipedia") and second and all(ch == "=" for ch in second):
        # Remove these two lines
        lines = lines[2:]
        # Also remove following empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join(lines)

    # Case 2: ATX format (in case Jina uses this)
    # # Marvel Cinematic Universe - Wikipedia
    if first.startswith("#") and first.rstrip().endswith(" - Wikipedia"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join(lines)

    return md

def _should_skip_section(text: str) -> bool:
    """
    Determine if this section should be skipped
    
    Skip conditions:
    1. In navigation section title list (See also, References, etc.)
    2. Title format is "XXX - Wikipedia"
    """
    norm_text = _norm_heading_title(text)
    
    # Check if in skip list
    if norm_text in SKIP_SECTION_TITLES:
        return True
    
    # Check if ends with " - wikipedia" (case insensitive)
    if text.lower().endswith(" - wikipedia"):
        return True
    
    return False

def _json_chat(model: str, prompt: str, max_tokens: int = 50000, temperature: float = 0.1) -> dict:
    """
    Call LLM and return JSON response.
    Reuses the implementation from QA script without the client parameter.
    """
    import requests
    global _JSON_CHAT_CACHE

    full_prompt = f"You are a careful data-wrangler. Return ONLY valid JSON.\n\n{prompt}"
    cache_key = (model, full_prompt, int(max_tokens), float(temperature))
    if cache_key in _JSON_CHAT_CACHE:
        print(f"[Cache Hit] Reusing existing LLM response (model={model})")
        return _JSON_CHAT_CACHE[cache_key]

    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": full_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    try:
        print(f"[LLM Call] Model: {model}")
        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=1800,
            proxies={"http": None, "https": None}
        )
        if resp.status_code != 200:
            print(f"[LLM Error] HTTP {resp.status_code}: {resp.text[:500]}")
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}
        data = resp.json()
        content_text = None

        # Gemini-style
        if isinstance(data, dict) and "candidates" in data and data["candidates"]:
            cand = data["candidates"][0]
            if "content" in cand and "parts" in cand["content"]:
                parts = cand["content"]["parts"]
                if parts and "text" in parts[0]:
                    content_text = parts[0]["text"]
        # OpenAI-style
        elif isinstance(data, dict) and "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            content_text = msg.get("content", "")
        # Simplified {"data":[{"text":"..."}]}
        elif isinstance(data, dict) and "data" in data and data["data"]:
            content_text = data["data"][0].get("text") or ""

        if not content_text:
            print(f"[Response Format Error] Unable to parse: {json.dumps(data)[:300]}")
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}

        # Remove ```json code blocks
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
        except json.JSONDecodeError as e:
            print(f"[JSON Parse Failed] {e}")
            print(f"[Raw Content] {content_text[:400]}")
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}
    except Exception as e:
        print(f"[LLM Call Exception] {type(e).__name__}: {e}")
        _JSON_CHAT_CACHE[cache_key] = {}
        return {}

# ====================== Citation Tools ======================

# # [[12]](https://...#cite_note-:2-1)
# WIKI_CITE_PATTERN = re.compile(r'\[\[(\d+)\]\]\(https://[^\)]*?#cite_note-([^)]+)\)')
# More general pattern: [[N]] or [[N]](any_url)
CITE_PATTERN = re.compile(r'\[\[(\d+)\]\](?:\(([^)]*)\))?')

def _extract_all_citations(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (display_num, cite_note_id), cite_note_id may be "".
    Supports:
      - [[N]]
      - [[N]](https://...#cite_note-xxx)
      - [[N]](https://...  any other format)
    """
    if not text:
        return []

    pairs = []
    for m in CITE_PATTERN.finditer(text):
        display_num = m.group(1)
        url_part = m.group(2) or ""

        cite_note_id = ""
        # Try to extract '#cite_note-xxx' from url
        m_id = re.search(r'#cite_note-([^&)\s]+)', url_part)
        if m_id:
            cite_note_id = m_id.group(1)

        pairs.append((display_num, cite_note_id))

    # Deduplicate
    seen = set()
    out: List[Tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

class ReferenceResolver:
    """
    Copied directly from original script: responsible for mapping [[N]]#cite_note-XX to md files / URLs under reference_pages
    Comments omitted, only key logic retained.
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
            print(f"[Warning] references.jsonl not found: {self.ref_jsonl_path}")
            return
        self.url_to_title = {}
        self.title_to_file = {}
        self.url_meta = {}

        # Read url + title from references.jsonl
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

        # Scan md files under reference_pages
        actual_files: Dict[str, Path] = {}
        actual_files_original: Dict[str, str] = {}
        if self.ref_pages_dir.exists():
            for md_file in self.ref_pages_dir.glob("*.md"):
                norm = self._normalize_for_matching(md_file.stem)
                actual_files[norm] = md_file
                actual_files_original[norm] = md_file.stem

        # Round 1: Exact match
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

        # 4) Round 2: Prefix fuzzy match (handle truncation/identical prefixes)
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

                    # Prefix exactly the same
                    if title_prefix == file_prefix:
                        matched = True
                    # Filename is complete prefix of title (title is longer)
                    elif norm_title.startswith(file_norm) and len(file_norm) >= 50:
                        matched = True
                    # Title is complete prefix of filename (filename is longer)
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

        # 5) Round 3: Similarity match (true fuzzy, using difflib)
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

                # Threshold can be adjusted based on actual data, using 0.90 here first
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

        print(f"  [ReferenceResolver] Total matched: {matched_count}/{len(self.url_to_title)} URLs")

        # 6) Find unmatched MD files
        unmatched_md_files = []
        for norm_file, orig_file in actual_files_original.items():
            if norm_file not in matched_normalized_files:
                unmatched_md_files.append({
                    "original_name": orig_file,
                    "normalized_name": norm_file,
                    "file_path": actual_files[norm_file],
                })

        # 7) Print unmatched titles
        if unmatched_titles_info:
            print(f"  [Warning] {len(unmatched_titles_info)} titles still unmatched:")
            for url, title, norm_title in unmatched_titles_info[:5]:
                print(f"    - '{title[:80]}...'")
                print(f"      Normalized: '{norm_title[:80]}...'")
            if len(unmatched_titles_info) > 5:
                print(f"    ... and another {len(unmatched_titles_info) - 5}")

        # 8) Print & save unmatched MD files
        if unmatched_md_files:
            print(f"\n  [Warning] {len(unmatched_md_files)} MD files exist but not matched by any title:")
            for i, info in enumerate(unmatched_md_files[:10], 1):
                print(f"    {i}. File: '{info['original_name'][:80]}...'")
                print(f"       Normalized: '{info['normalized_name'][:80]}...'")

            if len(unmatched_md_files) > 10:
                print(f"    ... and another {len(unmatched_md_files) - 10}")

        print(f"[ReferenceResolver] Total URLs matched: {matched_count}/{len(self.url_to_title)}")

    # ---- The parsing functions below are directly adapted from the original implementation ----
    def _extract_reference_section(self, wiki_text: str) -> str:
        patterns = [r'##\s*References\s*\n(.*)', r'References\s*\n-{3,}\n(.*)']
        for pattern in patterns:
            m = re.search(pattern, wiki_text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1)
        return ""

    def _note_to_ref_base(self, cite_note_id: str) -> str:
        return re.sub(r'-(\d+)$', r'_\1', cite_note_id)

    def _split_reference_items(self, ref_section: str) -> Dict[int, Tuple[int, int, str]]:
        items = {}
        boundaries = []
        for m in re.finditer(r'(?m)^\s*(\d+)\.\s', ref_section):
            boundaries.append((int(m.group(1)), m.start()))
        boundaries.sort(key=lambda x: x[1])
        for i, (num, start) in enumerate(boundaries):
            end = boundaries[i+1][1] if i+1 < len(boundaries) else len(ref_section)
            items[num] = (start, end, ref_section[start:end].strip())
        return items

    def _extract_ref_item(self, ref_section: str, display_num: str, cite_note_id: Optional[str]) -> str:
        items = self._split_reference_items(ref_section)
        try:
            num = int(display_num)
        except Exception:
            num = None

        candidate = items.get(num, (None, None, ""))[2] if num in items else ""
        if cite_note_id:
            base = self._note_to_ref_base(cite_note_id)
            anchor_pat = re.compile(rf'cite_ref-{re.escape(base)}(?:-\d+)?')
            if candidate and anchor_pat.search(candidate):
                return candidate
            for _, (_, _, text) in items.items():
                if anchor_pat.search(text):
                    return text
        return candidate

    def resolve_cite_note(self, wiki_text: str, display_num: str, cite_note_id: str) -> Optional[Dict]:
        ref_section = self._extract_reference_section(wiki_text)
        if not ref_section:
            return None
        ref_item_text = self._extract_ref_item(ref_section, display_num, cite_note_id)
        if not ref_item_text:
            return None

        urls = re.findall(r'https?://[^\s\)\]">]+', ref_item_text)
        urls = [u for u in urls if urlparse(u).netloc and "wikipedia.org" not in urlparse(u).netloc]

        matched_refs = []
        for url in urls:
            clean_url = url.split("#")[0]
            meta = self.url_meta.get(clean_url)
            if not meta or not meta.get("file"):
                continue
            matched_refs.append({
                "url": clean_url,
                "title": meta["title"],
                "file": str(meta["file"]),
            })

        if not matched_refs:
            return None
        return {
            "display_num": display_num,
            "cite_note_id": cite_note_id,
            "urls": urls,
            "matched_refs": matched_refs,
        }

def _require_all_refs_md(resolver: ReferenceResolver, wiki_text: str, sentence: str) -> Tuple[bool, List[str], List[str]]:
    """
    Check if all footnotes in a sentence can be resolved to reference md files.
    
    Enhancement: For citations that fail exact matching, attempt fuzzy matching (prefix match + similarity match)

    Returns:
      all_ok: Whether all citations are resolved
      missing_keys: List of unresolved citation keys
      ref_urls: Deduplicated list of successfully resolved URLs
    """
    citations = set(_extract_all_citations(sentence))
    missing = []
    ref_urls = set()
    unmatched_citations = []  # Store citations that failed exact matching
    
    # Round 1: Exact matching
    for dn, nid in sorted(citations, key=lambda x: (int(x[0]), x[1])):
        info = resolver.resolve_cite_note(wiki_text, dn, nid)
        if not info or not info.get("matched_refs"):
            # Exact match failed, store for later
            unmatched_citations.append((dn, nid, info))
        else:
            # Success, collect URL
            for rr in info["matched_refs"]:
                ref_urls.add(rr["url"])
    
    # Round 2: Fuzzy matching for unmatched citations
    if unmatched_citations:
        print(f"  [Fuzzy Match] Attempting fuzzy matching for {len(unmatched_citations)} unmatched citations...")
        
        # Collect all matched MD files
        matched_files = set()
        for meta in resolver.url_meta.values():
            if meta.get("file"):
                matched_files.add(resolver._normalize_for_matching(meta["file"].stem))
        
        # Scan all MD files
        all_md_files = {}
        if resolver.ref_pages_dir.exists():
            for md_file in resolver.ref_pages_dir.glob("*.md"):
                norm = resolver._normalize_for_matching(md_file.stem)
                all_md_files[norm] = md_file
        
        # Find unmatched files
        unmatched_files = {
            norm: path 
            for norm, path in all_md_files.items() 
            if norm not in matched_files
        }
        
        for dn, nid, info in unmatched_citations:
            if not info:
                key = f"[[{dn}]]#{nid}" if nid else f"[[{dn}]]"
                missing.append(key)
                continue
            
            # Get URL from info (may have URL even if MD not matched)
            urls = info.get("urls", [])
            if not urls:
                key = f"[[{dn}]]#{nid}" if nid else f"[[{dn}]]"
                missing.append(key)
                continue
            
            # Try to find corresponding MD file for each URL
            matched = False
            for url in urls:
                clean_url = url.split("#")[0]
                
                # Try to get title from url_meta
                meta = resolver.url_meta.get(clean_url)
                if not meta:
                    continue
                
                title = meta.get("title", "")
                if not title:
                    continue
                
                norm_title = resolver._normalize_for_matching(title)
                
                # 2.1 Prefix fuzzy matching
                for file_norm, file_path in list(unmatched_files.items()):
                    min_len = min(len(norm_title), len(file_norm))
                    if min_len < 20:
                        continue
                    
                    prefix_len = min(200, min_len)
                    title_prefix = norm_title[:prefix_len]
                    file_prefix = file_norm[:prefix_len]
                    
                    prefix_matched = False
                    # Exact prefix match
                    if title_prefix == file_prefix:
                        prefix_matched = True
                    # File name is complete prefix of title
                    elif norm_title.startswith(file_norm) and len(file_norm) >= 50:
                        prefix_matched = True
                    # Title is complete prefix of file name
                    elif file_norm.startswith(norm_title) and len(norm_title) >= 50:
                        prefix_matched = True
                    
                    if prefix_matched:
                        print(f"    [Prefix Match] [[{dn}]] '{title[:60]}...' → '{file_path.stem[:60]}...'")
                        # Update resolver mapping table
                        resolver.title_to_file[title] = file_path
                        resolver.url_meta[clean_url]["file"] = file_path
                        matched_files.add(file_norm)
                        
                        # Collect URL
                        ref_urls.add(clean_url)
                        matched = True
                        
                        # Remove from unmatched list
                        unmatched_files.pop(file_norm, None)
                        break
                
                if matched:
                    break
                
                # 2.2 Similarity fuzzy matching
                if not matched:
                    best_norm = None
                    best_score = 0.0
                    best_path = None
                    
                    for file_norm, file_path in unmatched_files.items():
                        score = difflib.SequenceMatcher(None, norm_title, file_norm).ratio()
                        if score > best_score:
                            best_score = score
                            best_norm = file_norm
                            best_path = file_path
                    
                    # Threshold set to 0.70
                    if best_norm is not None and best_score >= 0.70:
                        print(f"    [Similarity Match {best_score:.3f}] [[{dn}]] '{title[:60]}...' → '{best_path.stem[:60]}...'")
                        # Update resolver mapping table
                        resolver.title_to_file[title] = best_path
                        resolver.url_meta[clean_url]["file"] = best_path
                        matched_files.add(best_norm)
                        
                        # Collect URL
                        ref_urls.add(clean_url)
                        matched = True
                        
                        # Remove from unmatched list
                        unmatched_files.pop(best_norm, None)
                        break
            
            # If still unmatched, add to missing list
            if not matched:
                key = f"[[{dn}]]#{nid}" if nid else f"[[{dn}]]"
                missing.append(key)
    
    return len(missing) == 0, missing, sorted(ref_urls)

# ====================== Markdown Parsing: Leaf Sections ======================

class LeafSection:
    def __init__(self, path: List[str], body: str):
        self.path = path          # ["Title", "Section", "Subsection", ...]
        self.body = body          # Original text (including citations)
# Place before parse_leaf_sections
SKIP_SECTION_TITLES = {
    s.lower()
    for s in [
        "See also",
        "References",
        "Cited sources",
        "External links",
        "Further reading",
        "Notes",
        "Contents",
    ]
}

def _norm_heading_title(s: str) -> str:
    # Normalize case, strip leading/trailing whitespace, reduce multiple spaces
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

# def parse_leaf_sections(wiki_text: str, wiki_title: str) -> List[LeafSection]:
#     """
#     Parse markdown to find all leaf sections:

#     - "# title" is treated as level 1
#     - setext "Section\n-------" is treated as level 2
#     - "### ..." level = hash count - 1 (adjustable)
#     """
#     lines = wiki_text.splitlines()
#     n = len(lines)

#     # Collect all headings
#     headings = []  # (idx, level, text, is_setext)
#     i = 0
#     while i < n:
#         line = lines[i]

#         # ATX heading: ### xxx
#         m = re.match(r'^(#+)\s*(.+?)\s*$', line)
#         if m:
#             level = len(m.group(1))   # 1,2,3...
#             text = m.group(2).strip()
#             headings.append((i, level, text, False))
#             i += 1
#             continue

#         # setext: "Text" + "-----"
#         if i + 1 < n and re.match(r'^[^\s].*$', line) and re.match(r'^\s*-{3,}\s*$', lines[i+1]):
#             text = line.strip()
#             level = 2
#             headings.append((i, level, text, True))
#             i += 2
#             continue

#         i += 1

#     if not headings:
#         return []

#     # Determine title (level 1)
#     # Use "# title" if it's the first line; otherwise use wiki_title
#     title = wiki_title
#     if headings[0][1] == 1:
#         title = headings[0][2]

#     # Divide body sections based on headings
#     sections = []  # (start_idx, end_idx, level, text, is_setext)
#     for idx, (h_i, level, text, is_setext) in enumerate(headings):
#         body_start = h_i + (2 if is_setext else 1)
#         if idx + 1 < len(headings):
#             next_h_i, _, _, _ = headings[idx+1]
#             body_end = next_h_i
#         else:
#             body_end = n
#         sections.append((h_i, body_start, body_end, level, text, is_setext))

#     # Use a stack to maintain heading levels and compute path
#     leaf_sections: List[LeafSection] = []
#     stack: List[Tuple[int, str]] = []  # (level, text)

#     for (h_i, body_start, body_end, level, text, is_setext) in sections:
#         # Update stack: pop headings >= current level
#         while stack and stack[-1][0] >= level:
#             stack.pop()
#         stack.append((level, text))

#         # Check if leaf: whether there are deeper level headings in (h_i, body_end) range
#         has_child = False
#         for (other_h_i, other_level, _, _) in headings:
#             if other_h_i <= h_i:
#                 continue
#             if other_h_i >= body_end:
#                 break
#             if other_level > level:
#                 has_child = True
#                 break
#         if has_child:
#             continue
#         # NEW: Filter out "See also / References / Cited sources / External links" type leaf sections
#         if _norm_heading_title(text) in SKIP_SECTION_TITLES:
#             print(f"[Section Parsing] Skipping navigation section: {text!r}")
#             continue
#         # Build path: use title as root, followed by stack text (remove duplicate title from level1)
#         path = [title]
#         for lv, tx in stack:
#             if lv == 1:
#                 continue
#             path.append(tx)

#         body_lines = lines[body_start:body_end]
#         body = "\n".join(body_lines).strip()
#         if body:
#             leaf_sections.append(LeafSection(path=path, body=body))

#     print(f"[Section Parsing] Found leaf sections: {len(leaf_sections)}")
#     return leaf_sections
def parse_leaf_sections(wiki_text: str, wiki_title: str) -> List[LeafSection]:
    """
    Parse markdown to find all leaf sections
    
    Format rules:
    - Title\n===== is treated as level 1
    - Section\n----- is treated as level 2
    - ### xxx is treated as level 3
    - #### xxx is treated as level 4
    
    Leaf section criteria:
    - A section is a leaf if it has no deeper-level subsections after it
    - Content between wiki_title and the first section is also considered a leaf section
    """
    lines = wiki_text.splitlines()
    n = len(lines)

    # Collect all headings: (line_number, level, title_text, is_setext)
    headings = []
    i = 0
    
    while i < n:
        line = lines[i]
        
        # ===== style (Title, level 1)
        if i + 1 < n and re.match(r'^[^\s].*$', line) and re.match(r'^\s*={3,}\s*$', lines[i+1]):
            text = line.strip()
            headings.append((i, 1, text, True))
            i += 2
            continue
        
        # ----- style (Section, level 2)
        if i + 1 < n and re.match(r'^[^\s].*$', line) and re.match(r'^\s*-{3,}\s*$', lines[i+1]):
            text = line.strip()
            headings.append((i, 2, text, True))
            i += 2
            continue
        
        # ATX style: ### xxx (level = number of #)
        m = re.match(r'^(#+)\s*(.+?)\s*$', line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            headings.append((i, level, text, False))
            i += 1
            continue
        
        i += 1

    if not headings:
        return []

    # NEW: Process content from wiki_title to first section as first leaf section
    sections = []
    
    # If first heading is title (level 1)
    if headings and headings[0][1] == 1:
        title = headings[0][2]
        title_line_idx = headings[0][0]
        
        # Check if title should be skipped
        if _should_skip_section(title):
            print(f"[Section Parsing] Skipping title section: {title!r}")
            # Start processing from second heading
            start_idx = 1
        else:
            # Title content: from line after title to first section (or end of file)
            if len(headings) > 1:
                body_start = title_line_idx + 2  # Skip title and ===== line
                body_end = headings[1][0]  # Start of second heading
            else:
                body_start = title_line_idx + 2
                body_end = n
            
            # Collect title section
            sections.append((title_line_idx, body_start, body_end, 1, title, True))
            
            # Start processing from second heading
            start_idx = 1
    else:
        # If no title, use the passed wiki_title
        title = wiki_title
        start_idx = 0

    # Process remaining sections
    for idx in range(start_idx, len(headings)):
        h_i, level, text, is_setext = headings[idx]
        
        # Calculate body range
        if is_setext:
            body_start = h_i + 2  # Skip title line and underline
        else:
            body_start = h_i + 1  # Skip ### line
        
        # body_end: to next heading or end of file
        if idx + 1 < len(headings):
            body_end = headings[idx + 1][0]
        else:
            body_end = n
        
        sections.append((h_i, body_start, body_end, level, text, is_setext))

    # Determine leaf sections + build path
    leaf_sections: List[LeafSection] = []
    stack: List[Tuple[int, str]] = []  # (level, text)
    
    for (h_i, body_start, body_end, level, text, is_setext) in sections:
        # Update stack: pop headings >= current level
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        
        # Check if leaf: whether there are deeper level headings in body range
        has_child = False
        for (other_h_i, other_level, _, _) in headings:
            if other_h_i <= h_i:
                continue
            if other_h_i >= body_end:
                break
            if other_level > level:
                has_child = True
                break
        
        if has_child:
            continue  # Not a leaf, skip
        
        # Filter out navigation sections and "XXX - Wikipedia" format sections
        if _should_skip_section(text):
            print(f"[Section Parsing] Skipping section: {text!r}")
            continue
        
        # Build path: [title, section1, subsection1, ...]
        path = [title]
        for lv, tx in stack:
            if lv == 1:  # Skip level 1 (title already added)
                continue
            path.append(tx)
        
        # Extract body content
        body_lines = lines[body_start:body_end]
        body = "\n".join(body_lines).strip()
        
        if body:
            leaf_sections.append(LeafSection(path=path, body=body))
    
    print(f"[Section Parsing] Found {len(leaf_sections)} leaf sections")
    return leaf_sections

# ====================== LLM: Extract sentence + statement from leaf section ======================

def llm_extract_triples_from_leaf(
    model: str,
    wiki_title: str,
    section_path: List[str],
    section_body: str,
) -> List[Dict]:
    path_str = " > ".join(section_path)
    example_json = {
        "triples": [
            {
                "sentence": "As of 2021, small farms ... [[1]](https://en.wikipedia.org/...#cite_note-:2-1)",
                "statement": "As of 2021, small farms produce about one-third of the world's food.",
                "citation_numbers": ["1"]
            }
        ]
    }

    prompt = f"""
You are given the BODY text of a leaf subsection from a Wikipedia article.

ARTICLE TITLE: {wiki_title}

SECTION PATH (from root to this leaf):
{path_str}

BODY (Markdown-style, may contain inline citations and line breaks):
{section_body}

Your tasks:

1. Split BODY into SENTENCES.
   - A sentence should be a contiguous span of text that would still be grammatical if read alone.
   - Sentences MUST be copied VERBATIM from BODY (exact substring).
   - KEEP all inline numeric citation markers such as:
       [[12]](https://...#cite_note-...)
     or bare [[12]].
   - Do NOT reorder sentences; preserve original order.
   - Markdown tables or other table-like blocks (e.g., rows with '|' separators)
     SHOULD be treated as a SINGLE sentence:
       * include the entire table block as ONE continuous substring from BODY;
       * keep line breaks exactly as they appear;
       * do NOT split the table into multiple sentences.

2. For each sentence, decide whether it contains a meaningful factual statement.
   - If yes, write a short, cleaned "statement" for it:
       - Declarative sentence.
       - Remove citation markers and URLs.
       - Summarize the key fact in your own words, but do NOT add new information.
   - If the sentence contains no useful factual content (e.g., purely structural, list headings, etc.),
     set "statement" to null.

3. Extract citation_numbers:
   - For each sentence, collect the UNIQUE numeric citation ids appearing in that sentence.
   - If a sentence has [[1]](https://...#cite_note-xyz) or bare [[1]], then "1" is a citation number.
   - Return them as strings, e.g., ["1","2"]. If no citations, use [].

**CRITICAL VERIFICATION STEP - YOU MUST DO THIS BEFORE RETURNING:**

4. After extracting all triples, VERIFY your work:
   a) Search the BODY text for ALL citation patterns:
      - [[N]](https://...#cite_note-...)
      - [[N]]
   b) For EACH citation found, verify that the sentence containing it is included in your "triples" array.
   c) If you find ANY sentence with citations that you MISSED:
      - GO BACK and add it to your triples list
      - DO NOT skip any sentence with citations
   d) Double-check: Count the total number of citation markers in BODY, 
      then count citations in your extracted sentences - they MUST match.

**IMPORTANT:** Your extraction is INCOMPLETE if any cited sentence is missing from the output.
You MUST extract ALL sentences that contain citation markers [[N]] or [[N]](url).

Return JSON ONLY in this format:

{json.dumps(example_json, ensure_ascii=False, indent=2)}

Remember: VERIFY that you captured ALL cited sentences before returning your answer.
""".strip()

    data = _json_chat(model=model, prompt=prompt, max_tokens=50000, temperature=0.0)
    triples = data.get("triples", []) if isinstance(data, dict) else []
    out: List[Dict] = []
    if not isinstance(triples, list):
        return out

    for t in triples:
        sent = str(t.get("sentence", "")).strip()
        if not sent:
            continue
        # # Soft validation: sentence must appear in body
        # if sent not in section_body:
        #     print(f"  [Warning] sentence not in body, discarding: {sent[:60]}...")
        #     continue
        stmt = t.get("statement", None)
        if isinstance(stmt, str):
            stmt = stmt.strip()
            if not stmt:
                stmt = None
        cites = t.get("citation_numbers", [])
        cites = [str(c).strip() for c in (cites or []) if str(c).strip()]
        out.append({
            "sentence": sent,
            "statement": stmt,
            "citation_numbers": cites,
        })
    return out

# ====================== Main Process: Extract triples by topic ======================

def process_topic(
    raw_root: Path,
    topic_dir: Path,
    out_valid_f,
    out_invalid_f,
):
    wiki_md = max(
        (f for f in topic_dir.glob("*.md") if f.name != "README.md"),
        key=lambda p: p.stat().st_size,
        default=None
    )
    if not wiki_md:
        print(f"[Skip] {topic_dir.name}: Wiki Markdown file not found")
        return

    wiki_text = wiki_md.read_text(encoding="utf-8", errors="ignore")
    wiki_text = strip_wikipedia_title_header(wiki_text)  # Clean up “*- Wikipedia” header block
    wiki_title = wiki_md.stem.replace("_", " ")
    # Count [[N]] occurrences in the entire article
    total_cites = len(re.findall(r"\[\[(\d+)\]\]", wiki_text))
    print(f"[DEBUG] Total citation markers in article: {total_cites}")

    resolver = ReferenceResolver(topic_dir)
    leaf_sections = parse_leaf_sections(wiki_text, wiki_title)

    # Count [[N]] occurrences in all leaf.body
    cites_in_leaves = sum(
        len(re.findall(r"\[\[(\d+)\]\]", leaf.body)) for leaf in leaf_sections
    )
    print(f"[DEBUG] Total citation markers in all leaf sections: {cites_in_leaves}")
    print(f"\n{'#'*80}")
    print(f"# Topic: {wiki_title} ({topic_dir.name})")
    print(f"# File: {wiki_md.name} ({len(wiki_text)} chars)")
    print(f"{'#'*80}")

    resolver = ReferenceResolver(topic_dir)
    leaf_sections = parse_leaf_sections(wiki_text, wiki_title)
    total_valid = 0
    total_invalid = 0
    for leaf in leaf_sections:
        print(f"\n[Leaf] PATH = {' > '.join(leaf.path)}")
        triples_llm = llm_extract_triples_from_leaf(
            model=OPENAI_MODEL,
            wiki_title=wiki_title,
            section_path=leaf.path,
            section_body=leaf.body,
        )
        print(f"  [Leaf DEBUG] LLM returned triples count: {len(triples_llm)}")

        # Quick look at first few sentences
        for t in triples_llm[:5]:
            s = t.get("sentence", "")
            print("    [Sample] ", s[:200].replace("\n", " "))
            print("    [Sample-original [[N]] count] ", len(re.findall(r"\[\[(\d+)\]\]", s)))
            print("    [Sample-[N] count] ", len(re.findall(r"\[(\d+)\]", s)))
        if not triples_llm:
            print("  [Leaf] LLM returned no triples, skipping")
            continue

        valid_triples = []
        invalid_triples = []

        for t in triples_llm:
            sentence = t["sentence"]
            statement = t["statement"]
            # Only keep sentences with citations (otherwise useless for us)
            if not _extract_all_citations(sentence):
                continue

            all_ok, missing, ref_urls = _require_all_refs_md(
                resolver=resolver,
                wiki_text=wiki_text,
                sentence=sentence
            )
            triple_obj = {
                "sentence": sentence,
                "statement": statement,
                "citation_numbers": t["citation_numbers"],
                "citation_keys": missing if not all_ok else [],  # storing missing here is not really necessary
                "ref_urls": ref_urls,
                "ref_count": len(ref_urls),
            }
            if all_ok and ref_urls:
                valid_triples.append(triple_obj)
                total_valid += 1
            else:
                triple_obj["citation_keys"] = missing
                invalid_triples.append(triple_obj)
                total_invalid += 1
        print(f"[Topic Summary] {wiki_title}: valid_triples={total_valid}, invalid_triples={total_invalid}")

        if valid_triples:
            rec = {
                "wiki_title": wiki_title,
                "topic_dir": str(topic_dir),
                "section_path": leaf.path,
                "section_body": leaf.body,
                "triples": valid_triples,
            }
            out_valid_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if invalid_triples:
            rec = {
                "wiki_title": wiki_title,
                "topic_dir": str(topic_dir),
                "section_path": leaf.path,
                "section_body": leaf.body,
                "triples": invalid_triples,
            }
            out_invalid_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Phase 1: Extract (sentence, statement, refs) triples from wiki md")
    ap.add_argument("--raw-dir", type=str, required=True,
                    help="Root directory of raw wiki topics (one topic per subdirectory)")
    ap.add_argument("--out-valid", type=str, required=True,
                    help="JSONL output path for valid triples (all citations resolved successfully)")
    ap.add_argument("--out-invalid", type=str, required=True,
                    help="JSONL output path for invalid triples")
    args = ap.parse_args()

    raw_root = Path(args.raw_dir)
    if not raw_root.exists():
        print(f"[Error] raw-dir does not exist: {raw_root}")
        return

    out_valid_path = Path(args.out_valid)
    out_valid_path.parent.mkdir(parents=True, exist_ok=True)
    out_invalid_path = Path(args.out_invalid)
    out_invalid_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if raw_root is a category or a single topic
    has_article_md = any(
        f for f in raw_root.glob("*.md")
        if f.name != "README.md"
    )

    if has_article_md:
        # Single topic mode: raw_root itself is a topic directory
        topic_dirs = [raw_root]
        print(f"[Mode] Single topic directory: {raw_root}")
    else:
        # Category mode: enumerate subdirectories
        topic_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
        print(f"[Mode] Category directory: {raw_root}，containing {len(topic_dirs)} topics")

    print(f"[Start] Total {len(topic_dirs)} topics")

    with out_valid_path.open("w", encoding="utf-8") as fv, \
        out_invalid_path.open("w", encoding="utf-8") as fi:
        for idx, topic_dir in enumerate(topic_dirs, 1):
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(topic_dirs)}] {topic_dir.name}")
            print(f"{'='*80}")
            process_topic(raw_root, topic_dir, fv, fi)
            fv.flush()
            fi.flush()
    print("\n[Complete] Triple extraction finished")
    print(f"  Valid triples file: {out_valid_path}")
    print(f"  Invalid triples file: {out_invalid_path}")

if __name__ == "__main__":
    main()
