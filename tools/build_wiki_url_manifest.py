#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_wiki_url_manifest.py

Generate a CSV manifest of Wikipedia article URLs at scale, ready for your run_ref_scraper.sh:
CSV schema: category,title,url

Features
- Pull pages from seed categories (BFS over subcategories with configurable depth)
- Optional focus on Featured Articles (FA) and/or Good Articles (GA) categories
- Filter pages by quality signals (min number of <ref> tags, min characters, optional Infobox presence)
- Optionally sample per-category (top-N by ref count)
- Multi-language support (default enwiki)
- Friendly rate limiting & retry

Usage examples:
  python build_wiki_url_manifest.py --topics "Computer science,Biology,History" \
      --per-topic 800 --depth 2 --min-refs 8 --min-chars 3000 --out urls.csv

  # Focus on FA/GA only (great for high-quality groundtruth)
  python build_wiki_url_manifest.py --fa --ga --per-topic 500 --min-refs 10 --out urls_fa_ga.csv

  # Read topics from file (one per line)
  python build_wiki_url_manifest.py --topics-file topics_seed.txt --per-topic 1000 --depth 1 --out urls.csv

Notes:
- Requires: requests, tqdm, mwparserfromhell (optional, for more robust infobox detection)
- Wikipedia API etiquette: keep requests modest; this script rate-limits by default.
"""

import argparse
import csv
import random
import re
import sys
import time
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

_session = None
_limiter = None

try:
    import requests
except ImportError:
    print("Please pip install requests tqdm mwparserfromhell (optional).", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x

try:
    import mwparserfromhell as mwp
    HAS_MWP = True
except Exception:
    HAS_MWP = False

API_BASE = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "GraphRAG-in-the-wild/0.1 (contact: you@example.com)"

FA_CATEGORY = "Category:Featured articles"
GA_CATEGORY = "Category:Good articles"

REF_RE = re.compile(r"<ref\b", re.IGNORECASE)
CITE_TEMPLATE_RE = re.compile(r"{{\s*cite\s*[|}]", re.IGNORECASE)
INFOBOX_RE = re.compile(r"{{\s*infobox[\s_|}]", re.IGNORECASE)

@dataclass
class PageStats:
    title: str
    refs: int
    chars: int
    has_infobox: bool

class RateLimiter:
    def __init__(self, rps: float = 5.0):
        import threading, time
        self.min_interval = 1.0 / max(0.1, rps)
        self.lock = threading.Lock()
        self.last = 0.0
    def wait(self):
        import time
        with self.lock:
            now = time.time()
            delta = now - self.last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self.last = time.time()
def _get_session():
    global _session
    if _session is not None:
        return _session
    s = requests.Session()
    # 读环境里的 HTTP(S)_PROXY / REQUESTS_CA_BUNDLE
    s.trust_env = True
    # 强化重试（含 429/5xx）
    retry = Retry(
        total=_GLOBAL_HTTP_CFG.get("max_retries", 6),
        connect=_GLOBAL_HTTP_CFG.get("max_retries", 6),
        read=_GLOBAL_HTTP_CFG.get("max_retries", 6),
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # 显式代理（优先命令行，其次系统环境）
    proxy = _GLOBAL_HTTP_CFG.get("proxy") or ""
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    _session = s
    return s
def _get_limiter():
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(_GLOBAL_HTTP_CFG.get("rps", 2.0))
    return _limiter

def mw_api(lang: str, params: Dict, retries: int = None, rps: float = None) -> Dict:
    """
    带限速/重试/代理的 MediaWiki API 请求。
    若网络不可达，会抛出 ConnectionError；在 main() 最早做一次连通性自检更友好。
    """
    url = API_BASE.format(lang=lang)
    s = _get_session()
    _get_limiter().wait()
    timeout = _GLOBAL_HTTP_CFG.get("timeout", 30.0)
    headers = {"User-Agent": "GraphRAG-in-the-wild/0.2 (+contact-you@example.com)"}
    r = s.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def normalize_category(cat: str) -> str:
    cat = cat.strip()
    if not cat:
        return ""
    if not cat.lower().startswith("category:"):
        cat = "Category:" + cat
    return cat

def iter_category_members(lang: str, category: str, include_subcats: bool = True) -> Tuple[List[str], List[str]]:
    """Return (pages, subcats) for a category (non-recursive)."""
    pages, subcats = [], []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmtype": "page|subcat" if include_subcats else "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = mw_api(lang, params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            if m.get("ns") == 0:
                pages.append(m["title"])
            elif m.get("ns") == 14 and include_subcats:
                subcats.append(m["title"])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return pages, subcats

def bfs_collect_pages(lang: str, root_category: str, depth: int = 1, max_pages: Optional[int] = None) -> List[str]:
    """BFS categories up to `depth`, collecting article titles (ns0)."""
    root = normalize_category(root_category)
    seen_cats: Set[str] = set([root])
    pages: List[str] = []
    q = [(root, 0)]
    while q:
        cat, d = q.pop(0)
        pgs, subs = iter_category_members(lang, cat, include_subcats=True)
        pages.extend(pgs)
        if d < depth:
            for sc in subs:
                if sc not in seen_cats:
                    seen_cats.add(sc)
                    q.append((sc, d + 1))
        if max_pages and len(pages) >= max_pages:
            pages = pages[:max_pages]
            break
    # dedup
    return sorted(list(set(pages)))

def get_page_wikitext(lang: str, title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
        "formatversion": "2",
    }
    data = mw_api(lang, params)
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return ""
    revs = pages[0].get("revisions", [])
    if not revs:
        return ""
    slot = revs[0].get("slots", {}).get("main", {})
    return slot.get("content", "") or ""

def compute_stats_from_wikitext(wikitext: str) -> Tuple[int, bool]:
    """Return (ref_count, has_infobox)."""
    ref_count = len(REF_RE.findall(wikitext)) + len(CITE_TEMPLATE_RE.findall(wikitext))
    if HAS_MWP:
        try:
            wt = mwp.parse(wikitext)
            has_infobox = any(str(t.name).strip().lower().startswith("infobox") for t in wt.filter_templates())
        except Exception:
            has_infobox = bool(INFOBOX_RE.search(wikitext))
    else:
        has_infobox = bool(INFOBOX_RE.search(wikitext))
    return ref_count, has_infobox

def fetch_page_stats(lang: str, title: str) -> PageStats:
    wt = get_page_wikitext(lang, title)
    refs, has_infobox = compute_stats_from_wikitext(wt)
    return PageStats(title=title, refs=refs, chars=len(wt), has_infobox=has_infobox)

def gather_topic_pages(lang: str, topic: str, depth: int, per_topic: int,
                       min_refs: int, min_chars: int, require_infobox: bool,
                       boost_fa: bool, boost_ga: bool) -> List[PageStats]:
    # Collect raw candidates via BFS
    candidates = bfs_collect_pages(lang, topic, depth=depth, max_pages=None)
    # Optionally merge Featured/Good subtrees for the topic
    extra_cats = []
    if boost_fa:
        extra_cats.append(FA_CATEGORY)
    if boost_ga:
        extra_cats.append(GA_CATEGORY)
    for cat in extra_cats:
        # Narrow FA/GA to the topic by taking subcats that contain the topic token (rough heuristic)
        _, subcats = iter_category_members(lang, cat, include_subcats=True)
        for sc in subcats:
            token = topic.split(":", 1)[-1].strip().lower()
            if token and token in sc.lower():
                candidates.extend(bfs_collect_pages(lang, sc, depth=0))

    candidates = sorted(list(set(candidates)))
    # Fetch stats with a small worker pool
    out: List[PageStats] = []
    lock = Lock()
    q = Queue()
    for t in candidates:
        q.put(t)

    def worker():
        while True:
            try:
                t = q.get_nowait()
            except Exception:
                break
            try:
                ps = fetch_page_stats(lang, t)
                if ps.refs >= min_refs and ps.chars >= min_chars and (ps.has_infobox or not require_infobox):
                    with lock:
                        out.append(ps)
            except Exception:
                pass
            finally:
                q.task_done()

    workers = min(16, max(4, len(candidates) // 250 or 4))
    threads = [Thread(target=worker, daemon=True) for _ in range(workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Sort by reference richness and length to pick top-N (diverse signals could be added)
    out.sort(key=lambda x: (x.refs, x.chars), reverse=True)
    if per_topic and len(out) > per_topic:
        out = out[:per_topic]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=str, default="",
                    help="Comma-separated seed topics (e.g., 'Computer science,Biology,History'). "
                         "Each becomes 'Category:<topic>' if not already prefixed.")
    ap.add_argument("--topics-file", type=str, default="",
                    help="File with one topic per line")
    ap.add_argument("--lang", type=str, default="en", help="Wikipedia language (default: en)")
    ap.add_argument("--depth", type=int, default=1, help="BFS depth over subcategories (default: 1)")
    ap.add_argument("--per-topic", type=int, default=1000, help="Max pages per topic after filtering")
    ap.add_argument("--min-refs", type=int, default=8, help="Minimum number of <ref> (or {{cite }}) occurrences")
    ap.add_argument("--min-chars", type=int, default=3000, help="Minimum characters in wikitext")
    ap.add_argument("--require-infobox", action="store_true", help="Keep only pages with an Infobox")
    ap.add_argument("--fa", action="store_true", help="Emphasize Featured articles subtrees")
    ap.add_argument("--ga", action="store_true", help="Emphasize Good articles subtrees")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    ap.add_argument("--proxy", type=str, default="",
                help="可选代理: 形如 http://user:pass@host:port 或 socks5h://host:port")
    ap.add_argument("--timeout", type=float, default=30.0, help="单次请求超时秒数")
    ap.add_argument("--max-retries", type=int, default=6, help="请求重试次数")
    ap.add_argument("--rps", type=float, default=2.0, help="API 频率限制：每秒请求数")
    ap.add_argument("--api", type=str, default="https://{lang}.wikipedia.org/w/api.php",
                    help="自定义 MediaWiki API 端点模板")
    global API_BASE, _GLOBAL_HTTP_CFG
    API_BASE = args.api
    _GLOBAL_HTTP_CFG = {
        "proxy": args.proxy.strip(),
        "timeout": float(args.timeout),
        "max_retries": int(args.max_retries),
        "rps": float(args.rps),
    }
    args = ap.parse_args()

    topics: List[str] = []
    if args.topics:
        topics.extend([x.strip() for x in args.topics.split(",") if x.strip()])
    if args.topics_file:
        with open(args.topics_file, "r", encoding="utf-8") as f:
            topics.extend([line.strip() for line in f if line.strip()])
    if not topics:
        print("No topics provided. Use --topics or --topics-file.", file=sys.stderr)
        sys.exit(2)

    topics = [normalize_category(t) for t in topics]
    print(f"[Info] Topics: {topics}")

    all_rows: List[Tuple[str,str,str]] = []
    for topic in topics:
        print(f"[Info] Gathering pages for {topic} ...")
        stats = gather_topic_pages(
            lang=args.lang,
            topic=topic,
            depth=args.depth,
            per_topic=args.per_topic,
            min_refs=args.min_refs,
            min_chars=args.min_chars,
            require_infobox=args.require_infobox,
            boost_fa=args.fa,
            boost_ga=args.ga
        )
        # Write rows
        for ps in stats:
            url = f"https://{args.lang}.wikipedia.org/wiki/{quote(ps.title.replace(' ', '_'))}"
            # category field should be a clean label (strip "Category:")
            cat_label = topic.split(":", 1)[-1]
            all_rows.append((cat_label, ps.title, url))
        print(f"[Info] {topic}: kept {len(stats)} pages")

    # Dedup rows by URL
    seen = set()
    deduped = []
    for cat, title, url in all_rows:
        if url not in seen:
            seen.add(url)
            deduped.append((cat, title, url))

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "title", "url"])
        for row in deduped:
            w.writerow(row)

    print(f"[Done] Wrote {len(deduped)} rows to {args.out}")

if __name__ == "__main__":
    main()
