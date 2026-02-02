
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_triples.py

阶段一：抽取 (sentence, statement, refs) triple

输出两个 JSONL：
  - --out-valid:  每行一个 leaf section 的有效 triple 列表
  - --out-invalid: 每行一个 leaf section 的无效 triple 列表
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
    去掉形如:
      Marvel Cinematic Universe - Wikipedia
      ===============
    这种顶部 header，让真正的文章从
      Marvel Cinematic Universe
      ===============
    开始。
    """
    if not md:
        return md

    lines = md.splitlines()

    # 去掉前面的空行 / BOM
    while lines and not lines[0].strip():
        lines.pop(0)

    if len(lines) < 2:
        return md

    first = lines[0].strip()
    second = lines[1].strip()

    # 情形1：Setext 格式
    # Marvel Cinematic Universe - Wikipedia
    # ===============
    if first.endswith(" - Wikipedia") and second and all(ch == "=" for ch in second):
        # 删除这两行
        lines = lines[2:]
        # 再把后面紧跟的空行去掉
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join(lines)

    # 情形2：ATX 格式（万一 Jina 有这种）
    # # Marvel Cinematic Universe - Wikipedia
    if first.startswith("#") and first.rstrip().endswith(" - Wikipedia"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join(lines)

    return md

def _should_skip_section(text: str) -> bool:
    """
    判断是否应该跳过该 section
    
    跳过条件:
    1. 在导航类标题列表中 (See also, References 等)
    2. 格式为 "XXX - Wikipedia" 的标题
    """
    norm_text = _norm_heading_title(text)
    
    # 检查是否在跳过列表中
    if norm_text in SKIP_SECTION_TITLES:
        return True
    
    # 检查是否以 " - wikipedia" 结尾（统一转小写）
    if text.lower().endswith(" - wikipedia"):
        return True
    
    return False

def _json_chat(model: str, prompt: str, max_tokens: int = 50000, temperature: float = 0.1) -> dict:
    """
    直接复用你 QA 脚本里的实现，只是去掉 client 形参。
    """
    import requests
    global _JSON_CHAT_CACHE

    full_prompt = f"You are a careful data-wrangler. Return ONLY valid JSON.\n\n{prompt}"
    cache_key = (model, full_prompt, int(max_tokens), float(temperature))
    if cache_key in _JSON_CHAT_CACHE:
        print(f"[缓存命中] 复用已有 LLM 响应 (model={model})")
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
        print(f"[调用 LLM] 模型: {model}")
        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=1800,
            proxies={"http": None, "https": None}
        )
        if resp.status_code != 200:
            print(f"[LLM 错误] HTTP {resp.status_code}: {resp.text[:500]}")
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
            print(f"[响应格式错误] 无法解析: {json.dumps(data)[:300]}")
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}

        # 去掉 ```json 代码块
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
            print(f"[JSON 解析失败] {e}")
            print(f"[原始内容] {content_text[:400]}")
            _JSON_CHAT_CACHE[cache_key] = {}
            return {}
    except Exception as e:
        print(f"[LLM 调用异常] {type(e).__name__}: {e}")
        _JSON_CHAT_CACHE[cache_key] = {}
        return {}

# ====================== 引用相关工具 ======================

# # [[12]](https://...#cite_note-:2-1)
# WIKI_CITE_PATTERN = re.compile(r'\[\[(\d+)\]\]\(https://[^\)]*?#cite_note-([^)]+)\)')
# 新的更通用的模式：[[N]] 或 [[N]](任意url)
CITE_PATTERN = re.compile(r'\[\[(\d+)\]\](?:\(([^)]*)\))?')

def _extract_all_citations(text: str) -> List[Tuple[str, str]]:
    """
    返回 (display_num, cite_note_id) 列表，cite_note_id 可能为 "".
    支持：
      - [[N]]
      - [[N]](https://...#cite_note-xxx)
      - [[N]](https://...  任意其它形式)
    """
    if not text:
        return []

    pairs = []
    for m in CITE_PATTERN.finditer(text):
        display_num = m.group(1)
        url_part = m.group(2) or ""

        cite_note_id = ""
        # 尝试从 url 中提取 '#cite_note-xxx'
        m_id = re.search(r'#cite_note-([^&)\s]+)', url_part)
        if m_id:
            cite_note_id = m_id.group(1)

        pairs.append((display_num, cite_note_id))

    # 去重
    seen = set()
    out: List[Tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

class ReferenceResolver:
    """
    直接从你原脚本 copy：负责把 [[N]]#cite_note-XX 映射到 reference_pages 下的 md 文件 / URL
    这里略掉注释，只保留关键逻辑。
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
            print(f"[警告] 未找到 references.jsonl: {self.ref_jsonl_path}")
            return
        self.url_to_title = {}
        self.title_to_file = {}
        self.url_meta = {}

        # 从 references.jsonl 读 url + title
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

        # 扫描 reference_pages 下的 md 文件
        actual_files: Dict[str, Path] = {}
        actual_files_original: Dict[str, str] = {}
        if self.ref_pages_dir.exists():
            for md_file in self.ref_pages_dir.glob("*.md"):
                norm = self._normalize_for_matching(md_file.stem)
                actual_files[norm] = md_file
                actual_files_original[norm] = md_file.stem

        # 第一轮：精确匹配
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

        # 4) 第二轮：前缀模糊匹配（处理截断/前缀完全相同）
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

                    # 前缀完全一样
                    if title_prefix == file_prefix:
                        matched = True
                    # 文件名是标题的完整前缀（标题更长）
                    elif norm_title.startswith(file_norm) and len(file_norm) >= 50:
                        matched = True
                    # 标题是文件名的完整前缀（文件更长）
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

        # 5) 第三轮：相似度匹配（真正 fuzzy，用 difflib）
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

                # 阈值可以根据实际数据调，这里先用 0.90
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

        print(f"  [参考解析器] 总共匹配: {matched_count}/{len(self.url_to_title)} 个 URL")

        # 6) 找出未匹配的 MD 文件
        unmatched_md_files = []
        for norm_file, orig_file in actual_files_original.items():
            if norm_file not in matched_normalized_files:
                unmatched_md_files.append({
                    "original_name": orig_file,
                    "normalized_name": norm_file,
                    "file_path": actual_files[norm_file],
                })

        # 7) 打印未匹配的标题
        if unmatched_titles_info:
            print(f"  [警告] {len(unmatched_titles_info)} 个标题仍未匹配:")
            for url, title, norm_title in unmatched_titles_info[:5]:
                print(f"    - '{title[:80]}...'")
                print(f"      标准化后: '{norm_title[:80]}...'")
            if len(unmatched_titles_info) > 5:
                print(f"    ... 以及另外 {len(unmatched_titles_info) - 5} 个")

        # 8) 打印 & 保存未匹配的 MD 文件
        if unmatched_md_files:
            print(f"\n  [警告] {len(unmatched_md_files)} 个 MD 文件存在但未被任何标题匹配:")
            for i, info in enumerate(unmatched_md_files[:10], 1):
                print(f"    {i}. 文件: '{info['original_name'][:80]}...'")
                print(f"       标准化后: '{info['normalized_name'][:80]}...'")

            if len(unmatched_md_files) > 10:
                print(f"    ... 以及另外 {len(unmatched_md_files) - 10} 个")

        print(f"[参考解析器] 总匹配 URL: {matched_count}/{len(self.url_to_title)}")

    # ---- 下面这些解析函数也直接照搬你原来的 ----
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
    检查一个 sentence 中出现的脚注能否全部解析到 reference md。
    
    新增：对于精确匹配失败的引用，尝试模糊匹配（前缀匹配 + 相似度匹配）

    返回:
      all_ok: 是否全部命中
      missing_keys: 无法命中的键列表
      ref_urls: 成功解析到的去重 URL 列表
    """
    citations = set(_extract_all_citations(sentence))
    missing = []
    ref_urls = set()
    unmatched_citations = []  # 存储精确匹配失败的引用
    
    # 第一轮：精确匹配
    for dn, nid in sorted(citations, key=lambda x: (int(x[0]), x[1])):
        info = resolver.resolve_cite_note(wiki_text, dn, nid)
        if not info or not info.get("matched_refs"):
            # 精确匹配失败，暂存
            unmatched_citations.append((dn, nid, info))
        else:
            # 成功，收集 URL
            for rr in info["matched_refs"]:
                ref_urls.add(rr["url"])
    
    # 第二轮：对未匹配的引用进行模糊匹配
    if unmatched_citations:
        print(f"  [模糊匹配] 尝试对 {len(unmatched_citations)} 个未匹配的引用进行模糊匹配...")
        
        # 收集所有未匹配的 MD 文件
        matched_files = set()
        for meta in resolver.url_meta.values():
            if meta.get("file"):
                matched_files.add(resolver._normalize_for_matching(meta["file"].stem))
        
        # 扫描所有 MD 文件
        all_md_files = {}
        if resolver.ref_pages_dir.exists():
            for md_file in resolver.ref_pages_dir.glob("*.md"):
                norm = resolver._normalize_for_matching(md_file.stem)
                all_md_files[norm] = md_file
        
        # 找出未被匹配的文件
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
            
            # 从 info 中获取 URL（虽然没匹配到 MD，但可能有 URL）
            urls = info.get("urls", [])
            if not urls:
                key = f"[[{dn}]]#{nid}" if nid else f"[[{dn}]]"
                missing.append(key)
                continue
            
            # 尝试为每个 URL 找到对应的 MD 文件
            matched = False
            for url in urls:
                clean_url = url.split("#")[0]
                
                # 尝试从 url_meta 中获取标题
                meta = resolver.url_meta.get(clean_url)
                if not meta:
                    continue
                
                title = meta.get("title", "")
                if not title:
                    continue
                
                norm_title = resolver._normalize_for_matching(title)
                
                # 2.1 前缀模糊匹配
                for file_norm, file_path in list(unmatched_files.items()):
                    min_len = min(len(norm_title), len(file_norm))
                    if min_len < 20:
                        continue
                    
                    prefix_len = min(200, min_len)
                    title_prefix = norm_title[:prefix_len]
                    file_prefix = file_norm[:prefix_len]
                    
                    prefix_matched = False
                    # 前缀完全一致
                    if title_prefix == file_prefix:
                        prefix_matched = True
                    # 文件名是标题的完整前缀
                    elif norm_title.startswith(file_norm) and len(file_norm) >= 50:
                        prefix_matched = True
                    # 标题是文件名的完整前缀
                    elif file_norm.startswith(norm_title) and len(norm_title) >= 50:
                        prefix_matched = True
                    
                    if prefix_matched:
                        print(f"    [前缀匹配] [[{dn}]] '{title[:60]}...' → '{file_path.stem[:60]}...'")
                        # 更新到 resolver 的映射表
                        resolver.title_to_file[title] = file_path
                        resolver.url_meta[clean_url]["file"] = file_path
                        matched_files.add(file_norm)
                        
                        # 收集 URL
                        ref_urls.add(clean_url)
                        matched = True
                        
                        # 从未匹配列表中移除
                        unmatched_files.pop(file_norm, None)
                        break
                
                if matched:
                    break
                
                # 2.2 相似度模糊匹配
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
                    
                    # 阈值设为 0.70
                    if best_norm is not None and best_score >= 0.70:
                        print(f"    [相似度匹配 {best_score:.3f}] [[{dn}]] '{title[:60]}...' → '{best_path.stem[:60]}...'")
                        # 更新到 resolver 的映射表
                        resolver.title_to_file[title] = best_path
                        resolver.url_meta[clean_url]["file"] = best_path
                        matched_files.add(best_norm)
                        
                        # 收集 URL
                        ref_urls.add(clean_url)
                        matched = True
                        
                        # 从未匹配列表中移除
                        unmatched_files.pop(best_norm, None)
                        break
            
            # 如果仍未匹配，加入 missing 列表
            if not matched:
                key = f"[[{dn}]]#{nid}" if nid else f"[[{dn}]]"
                missing.append(key)
    
    return len(missing) == 0, missing, sorted(ref_urls)

# ====================== markdown 解析：叶子 section ======================

class LeafSection:
    def __init__(self, path: List[str], body: str):
        self.path = path          # ["Title", "Section", "Subsection", ...]
        self.body = body          # 原文（含 cite）
# 放在 parse_leaf_sections 之前
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
    # 统一大小写、去掉两端空白、多余空格
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

# def parse_leaf_sections(wiki_text: str, wiki_title: str) -> List[LeafSection]:
#     """
#     解析 markdown，找出所有叶子 section：

#     - "# title" 视作 level 1
#     - setext "Section\n-------" 视作 level 2
#     - "### ..." level = 个数 - 1 (你可以按自己习惯调)
#     """
#     lines = wiki_text.splitlines()
#     n = len(lines)

#     # 收集所有 heading
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

#     # 确定 title（level 1）
#     # 若第一行就是 "# title" 就用它；否则用 wiki_title
#     title = wiki_title
#     if headings[0][1] == 1:
#         title = headings[0][2]

#     # 根据 heading 划分 body 区间
#     sections = []  # (start_idx, end_idx, level, text, is_setext)
#     for idx, (h_i, level, text, is_setext) in enumerate(headings):
#         body_start = h_i + (2 if is_setext else 1)
#         if idx + 1 < len(headings):
#             next_h_i, _, _, _ = headings[idx+1]
#             body_end = next_h_i
#         else:
#             body_end = n
#         sections.append((h_i, body_start, body_end, level, text, is_setext))

#     # 用一个栈维护 heading 层级，算出 path
#     leaf_sections: List[LeafSection] = []
#     stack: List[Tuple[int, str]] = []  # (level, text)

#     for (h_i, body_start, body_end, level, text, is_setext) in sections:
#         # 更新栈：弹出 >= 当前 level 的 heading
#         while stack and stack[-1][0] >= level:
#             stack.pop()
#         stack.append((level, text))

#         # 判断是否 leaf：在 (h_i, body_end) 区间内是否有更深 level 的 heading
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
#         # ★ 新增：过滤掉“See also / References / Cited sources / External links”之类的 leaf
#         if _norm_heading_title(text) in SKIP_SECTION_TITLES:
#             print(f"[Section 解析] 跳过尾部导航类 section: {text!r}")
#             continue
#         # 组 path：以 title 为 root，后接 stack 文本（去掉 level1 里的 title 重复）
#         path = [title]
#         for lv, tx in stack:
#             if lv == 1:
#                 continue
#             path.append(tx)

#         body_lines = lines[body_start:body_end]
#         body = "\n".join(body_lines).strip()
#         if body:
#             leaf_sections.append(LeafSection(path=path, body=body))

#     print(f"[Section 解析] 发现 leaf sections: {len(leaf_sections)}")
#     return leaf_sections
def parse_leaf_sections(wiki_text: str, wiki_title: str) -> List[LeafSection]:
    """
    解析 markdown，找出所有叶子 section
    
    新格式规则:
    - Title\n===== 视为 level 1
    - Section\n----- 视为 level 2
    - ### xxx 视为 level 3
    - #### xxx 视为 level 4
    
    叶子 section 判断:
    - 如果一个 section 后面没有更深层级的 subsection，则为叶子
    - wiki_title 到第一个 section 之间的内容也算一个叶子 section
    """
    lines = wiki_text.splitlines()
    n = len(lines)

    # 收集所有 heading: (行号, level, 标题文本, is_setext)
    headings = []
    i = 0
    
    while i < n:
        line = lines[i]
        
        # ===== 风格 (Title, level 1)
        if i + 1 < n and re.match(r'^[^\s].*$', line) and re.match(r'^\s*={3,}\s*$', lines[i+1]):
            text = line.strip()
            headings.append((i, 1, text, True))
            i += 2
            continue
        
        # ----- 风格 (Section, level 2)
        if i + 1 < n and re.match(r'^[^\s].*$', line) and re.match(r'^\s*-{3,}\s*$', lines[i+1]):
            text = line.strip()
            headings.append((i, 2, text, True))
            i += 2
            continue
        
        # ATX 风格: ### xxx (level = #的个数)
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

    # ★ 新增：处理 wiki_title 到第一个 section 的内容作为首个叶子 section
    sections = []
    
    # 如果第一个 heading 是 title (level 1)
    if headings and headings[0][1] == 1:
        title = headings[0][2]
        title_line_idx = headings[0][0]
        
        # ★ 检查 title 是否需要跳过
        if _should_skip_section(title):
            print(f"[Section 解析] 跳过 title section: {title!r}")
            # 从第二个 heading 开始处理
            start_idx = 1
        else:
            # title 内容：从 title 下一行到第一个 section (或文件末尾)
            if len(headings) > 1:
                body_start = title_line_idx + 2  # 跳过 title 和 ===== 行
                body_end = headings[1][0]  # 第二个 heading 开始
            else:
                body_start = title_line_idx + 2
                body_end = n
            
            # 收集 title section
            sections.append((title_line_idx, body_start, body_end, 1, title, True))
            
            # 从第二个 heading 开始处理
            start_idx = 1
    else:
        # 如果没有 title，用传入的 wiki_title
        title = wiki_title
        start_idx = 0

    # 处理剩余的 sections
    for idx in range(start_idx, len(headings)):
        h_i, level, text, is_setext = headings[idx]
        
        # 计算 body 范围
        if is_setext:
            body_start = h_i + 2  # 跳过标题行和下划线
        else:
            body_start = h_i + 1  # 跳过 ### 行
        
        # body_end: 到下一个 heading 或文件末尾
        if idx + 1 < len(headings):
            body_end = headings[idx + 1][0]
        else:
            body_end = n
        
        sections.append((h_i, body_start, body_end, level, text, is_setext))

    # ★ 判断叶子 section + 构建路径
    leaf_sections: List[LeafSection] = []
    stack: List[Tuple[int, str]] = []  # (level, text)
    
    for (h_i, body_start, body_end, level, text, is_setext) in sections:
        # 更新栈：弹出 >= 当前 level 的 heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        
        # ★ 判断是否为叶子：body 范围内是否有更深层级的 heading
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
            continue  # 不是叶子，跳过
        
        # ★ 过滤掉导航类 section 和 "XXX - Wikipedia" 格式的 section
        if _should_skip_section(text):
            print(f"[Section 解析] 跳过 section: {text!r}")
            continue
        
        # ★ 构建路径：[title, section1, subsection1, ...]
        path = [title]
        for lv, tx in stack:
            if lv == 1:  # 跳过 level 1 (title 已经加入)
                continue
            path.append(tx)
        
        # 提取 body 内容
        body_lines = lines[body_start:body_end]
        body = "\n".join(body_lines).strip()
        
        if body:
            leaf_sections.append(LeafSection(path=path, body=body))
    
    print(f"[Section 解析] 发现 {len(leaf_sections)} 个叶子 sections")
    return leaf_sections

# ====================== LLM：在 leaf section 中抽 sentence + statement ======================

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
        # # 轻微校验：sentence 必须出现在 body 中
        # if sent not in section_body:
        #     print(f"  [警告] sentence 不在 body 中，丢弃: {sent[:60]}...")
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

# ====================== 主流程：按 topic 抽取 triple ======================

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
        print(f"[跳过] {topic_dir.name}: 未找到 Wiki Markdown 文件")
        return

    wiki_text = wiki_md.read_text(encoding="utf-8", errors="ignore")
    wiki_text = strip_wikipedia_title_header(wiki_text)  # ★ 在这里清洗掉 “*- Wikipedia” 顶部块
    wiki_title = wiki_md.stem.replace("_", " ")
    # 统计整篇里的 [[N]] 数量
    total_cites = len(re.findall(r"\[\[(\d+)\]\]", wiki_text))
    print(f"[DEBUG] 整篇文章脚注标记数量: {total_cites}")

    resolver = ReferenceResolver(topic_dir)
    leaf_sections = parse_leaf_sections(wiki_text, wiki_title)

    # 统计所有 leaf.body 里的 [[N]] 数
    cites_in_leaves = sum(
        len(re.findall(r"\[\[(\d+)\]\]", leaf.body)) for leaf in leaf_sections
    )
    print(f"[DEBUG] 所有 leaf section 中脚注标记总数: {cites_in_leaves}")
    print(f"\n{'#'*80}")
    print(f"# 主题: {wiki_title} ({topic_dir.name})")
    print(f"# 文件: {wiki_md.name} ({len(wiki_text)} chars)")
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
        print(f"  [Leaf DEBUG] LLM 返回 triples 数: {len(triples_llm)}")

        # 粗看前几个 sentence 长什么样
        for t in triples_llm[:5]:
            s = t.get("sentence", "")
            print("    [例句] ", s[:200].replace("\n", " "))
            print("    [例句-原始 [[N]] 计数] ", len(re.findall(r"\[\[(\d+)\]\]", s)))
            print("    [例句-[N] 计数] ", len(re.findall(r"\[(\d+)\]", s)))
        if not triples_llm:
            print("  [Leaf] LLM 未返回任何 triple，跳过")
            continue

        valid_triples = []
        invalid_triples = []

        for t in triples_llm:
            sentence = t["sentence"]
            statement = t["statement"]
            # 只保留带 citation 的句子（否则对我们没用）
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
                "citation_keys": missing if not all_ok else [],  # 这里存 missing 其实没啥必要
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
        print(f"[Topic 小结] {wiki_title}: valid_triples={total_valid}, invalid_triples={total_invalid}")

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
    ap = argparse.ArgumentParser(description="阶段一：从 wiki md 抽取 (sentence, statement, refs) triple")
    ap.add_argument("--raw-dir", type=str, required=True,
                    help="原始 wiki topic 根目录（每个子目录一个 topic）")
    ap.add_argument("--out-valid", type=str, required=True,
                    help="有效 triple（全部脚注都解析成功）的 JSONL 输出路径")
    ap.add_argument("--out-invalid", type=str, required=True,
                    help="无效 triple 的 JSONL 输出路径")
    args = ap.parse_args()

    raw_root = Path(args.raw_dir)
    if not raw_root.exists():
        print(f"[错误] raw-dir 不存在: {raw_root}")
        return

    out_valid_path = Path(args.out_valid)
    out_valid_path.parent.mkdir(parents=True, exist_ok=True)
    out_invalid_path = Path(args.out_invalid)
    out_invalid_path.parent.mkdir(parents=True, exist_ok=True)

    # 判断 raw_root 是「category」还是「单个 topic」
    has_article_md = any(
        f for f in raw_root.glob("*.md")
        if f.name != "README.md"
    )

    if has_article_md:
        # 单 topic 模式：raw_root 自己就是一个 topic 目录
        topic_dirs = [raw_root]
        print(f"[模式] 单 topic 目录: {raw_root}")
    else:
        # category 模式：枚举子目录
        topic_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
        print(f"[模式] category 目录: {raw_root}，包含 {len(topic_dirs)} 个 topic")

    print(f"[开始] 共 {len(topic_dirs)} 个 topic")

    with out_valid_path.open("w", encoding="utf-8") as fv, \
        out_invalid_path.open("w", encoding="utf-8") as fi:
        for idx, topic_dir in enumerate(topic_dirs, 1):
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(topic_dirs)}] {topic_dir.name}")
            print(f"{'='*80}")
            process_topic(raw_root, topic_dir, fv, fi)
            fv.flush()
            fi.flush()
    print("\n[完成] triple 抽取结束")
    print(f"  有效 triple 文件: {out_valid_path}")
    print(f"  无效 triple 文件: {out_invalid_path}")

if __name__ == "__main__":
    main()
