#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
post_filter_ref_pages.py

Features (offline post-processing pipeline):
1. Traverse all reference/reference_pages/*.md for each article in corpus
2. Call LLM to filter each ref page:
   - keep=True: considered "valid reference source", retain
   - keep=False: considered "junk/error page/index/empty content", enter repair flow
3. For keep=False pages:
   - Use the md file title to fuzzy match corresponding entry in references.jsonl
   - If the ref has archive_url:
       * Call fetch_reference_pages.py with archive-mode=direct + fetcher=jina to force re-fetch using original url (--force + --no-skip-exists)
       * Run LLM filter again on newly fetched md
           - If passed: keep new md, mark llm_filter_status = "refetched_ok" in references.jsonl
           - If still failed: delete md, mark llm_filter_status = "refetched_still_bad_dropped"
   - Otherwise (no archive_url):
       * Delete md directly, mark llm_filter_status = "dropped_without_refetch"

New features:
- Support concurrent LLM calls: parameter --llm-workers, default 1 (serial)
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# -----------------------
# LLM Related
# -----------------------

def _truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncation not currently used, but utility function kept for future needs."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def call_quality_model(
    markdown_text: str,
    api_base: str,
    api_key: str,
    model: str,
    wiki_title: str,
    ref_title: str,
    page_title: str,
    timeout: int = 60,
) -> Tuple[bool, str, str]:
    """
    Call LLM to determine if a ref page is a "valid reference source".

    This will provide the model with:
      - Wikipedia article title wiki_title
      - Reference title from references.jsonl (ref_title)
      - The md page's own title (page_title)
      - Full text of the md (Markdown, absolutely no truncation)

    Returns:
        keep: bool       # True=keep, False=considered junk/error page
        category: str    # Model classification label (e.g. ok/404/index/login/...)
        reason: str      # Brief explanation
    """
    # No truncation: use full text directly
    excerpt = markdown_text if markdown_text is not None else ""
    excerpt = excerpt.strip()

    system_prompt = (
        "You are a web page quality filter assistant. We have scraped some web pages from the internet "
        "as reference source pages for Wikipedia articles, and the content has been converted to Markdown. "
        "Your task is to determine whether a given page can serve as a useful reference source for the Wikipedia article.\n\n"
        "You will see:\n"
        "1) The title of the Wikipedia article (WIKI_TITLE)\n"
        "2) The title of the reference entry (REF_TITLE)\n"
        "3) The full Markdown content of the reference page (PAGE_TITLE + FULL_MARKDOWN)\n\n"
        "Cases where keep=true (be lenient):\n"
        "- The page contains substantive text content related to the Wikipedia topic/reference entry, even if it is short, poorly formatted, or mixed with navigation or ads;\n"
        "- It is a news report, encyclopedia entry, paper, official statement, blog, forum post, etc. - as long as there is body text related to the topic, it is useful;\n"
        "- Even if there are many irrelevant elements (navigation bars, sidebars, recommended links), as long as there is some clear body text information, it counts.\n\n"
        "Cases where keep=false (be strict):\n"
        "- Clearly a 404 page, error page, or access denied page with only a few lines of error messages;\n"
        "- Pages requiring login/subscription/purchase to view, where the main content is completely invisible (only login or subscription prompts);\n"
        "- Pure search result pages, site directories, pages with only link lists or navigation, without any substantive body text;\n"
        "- Completely blank or containing only minimal meaningless characters.\n\n"
        "Note:\n"
        "- Do not mark as false just because the content appears simple/short/poorly written - as long as there is some information related to the topic, keep=true;\n"
        "- Exact match with WIKI_TITLE / REF_TITLE is not required - as long as it is roughly related to the topic and useful, it counts.\n\n"
        "Output must be a JSON object with no additional explanatory text. Fields:\n"
        '{ \"keep\": true or false, \"category\": \"ok/404/index/login/empty/other\", \"reason\": \"Brief explanation (<=30 words)\" }'
    )

    user_prompt = (
        "Below is the relevant information for a Wikipedia reference page. Please evaluate it according to the system prompt.\n\n"
        f"WIKI_TITLE: {wiki_title}\n"
        f"REF_TITLE: {ref_title}\n"
        f"PAGE_TITLE: {page_title}\n\n"
        "Below is the full Markdown content of the page:\n"
        "--------------------\n"
        f"{excerpt}\n"
        "--------------------\n"
        "Please output only a JSON object, without any other text."
    )

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        # Model failed, be conservative: keep it
        return True, "model_error", f"Model call failed: {type(e).__name__}"

    # Try to extract JSON from return text
    try:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError("JSON object not found")
        obj = json.loads(m.group(0))
        keep = bool(obj.get("keep", True))
        category = str(obj.get("category", "unknown"))
        reason = str(obj.get("reason", ""))
        return keep, category, reason
    except Exception as e:
        # Parse failed, also keep to avoid accidental deletion
        return True, "parse_error", f"Failed to parse model output: {type(e).__name__}"


# -----------------------
# references.jsonl related
# -----------------------

def load_references(jsonl_path: Path) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return refs
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                refs.append(json.loads(line))
            except json.JSONDecodeError:
                refs.append({"raw": line, "_decode_error": True})
    return refs


def save_references(jsonl_path: Path, refs: List[Dict[str, Any]]) -> None:
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for r in refs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(jsonl_path)


def normalize_title_for_match(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


def find_best_ref_index(page_title: str, refs: List[Dict[str, Any]], min_score: float = 0.4) -> Optional[int]:
    norm_page = normalize_title_for_match(page_title)
    if not norm_page:
        return None
    page_tokens = norm_page.split()

    best_idx: Optional[int] = None
    best_score = 0.0

    for i, ref in enumerate(refs):
        t = str(ref.get("title", "") or "")
        norm_ref = normalize_title_for_match(t)
        if not norm_ref:
            continue
        ref_tokens = norm_ref.split()
        score = jaccard(page_tokens, ref_tokens)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None or best_score < min_score:
        return None
    return best_idx


def extract_title_from_md(md_path: Path, text: Optional[str] = None) -> str:
    """
    Prefer extracting title from first line starting with '# ';
    otherwise fallback to filename (with some de-slugging).
    """
    if text is None:
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    # fallback: filename without extension
    name = md_path.stem
    name = name.replace("_", " ")
    return name


# -----------------------
# Traverse reference_pages
# -----------------------

def iter_reference_pages(root_dir: Path):
    """
    Traverse all reference/reference_pages directories, yield (md_path, references.jsonl path)
    """
    for ref_pages_dir in root_dir.rglob("reference_pages"):
        if not ref_pages_dir.is_dir():
            continue
        reference_dir = ref_pages_dir.parent
        jsonl_path = reference_dir / "references.jsonl"
        if not jsonl_path.exists():
            continue
        for md_path in sorted(ref_pages_dir.glob("*.md")):
            yield md_path, jsonl_path


# -----------------------
# LLM concurrent processing stage
# -----------------------

def run_llm_triple_check_for_page(
    task: Dict[str, Any],
    api_base: str,
    api_key: str,
    model: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Perform "triple judgment" LLM calls on a single ref page.
    Only when all three judgments are keep=False is the page considered invalid.

    Input task contains:
      - md_path
      - jsonl_path
      - rel_str
      - wiki_title
      - page_title
      - md_text

    Output result contains:
      - keep: bool
      - cat_label: str
      - reason: str
      - judgments: List[Dict] - records each judgment result
      - plus original task fields
    """
    md_path: Path = task["md_path"]
    jsonl_path: Path = task["jsonl_path"]
    rel_str: str = task["rel_str"]
    wiki_title: str = task["wiki_title"]
    page_title: str = task["page_title"]
    md_text: str = task["md_text"]

    # Try to find ref_title from references.jsonl
    refs = load_references(jsonl_path)
    ref_title = ""
    if refs:
        idx0 = find_best_ref_index(page_title, refs)
        if idx0 is not None:
            ref_title = str(refs[idx0].get("title", "") or "")

    judgments = []
    
    # First judgment
    keep1, cat1, reason1 = call_quality_model(
        markdown_text=md_text,
        api_base=api_base,
        api_key=api_key,
        model=model,
        wiki_title=wiki_title,
        ref_title=ref_title,
        page_title=page_title,
    )
    judgments.append({"round": 1, "keep": keep1, "category": cat1, "reason": reason1})
    
    if verbose:
        print(f"[INFO] (LLM-1) {rel_str} -> keep={keep1}, category={cat1}, reason={reason1}")

    # If first judgment is keep=True, return keep directly
    if keep1:
        result = dict(task)
        result["keep"] = True
        result["cat_label"] = cat1
        result["reason"] = reason1
        result["judgments"] = judgments
        return result

    # First judgment is False, proceed with second judgment
    keep2, cat2, reason2 = call_quality_model(
        markdown_text=md_text,
        api_base=api_base,
        api_key=api_key,
        model=model,
        wiki_title=wiki_title,
        ref_title=ref_title,
        page_title=page_title,
    )
    judgments.append({"round": 2, "keep": keep2, "category": cat2, "reason": reason2})
    
    if verbose:
        print(f"[INFO] (LLM-2) {rel_str} -> keep={keep2}, category={cat2}, reason={reason2}")

    # If second judgment is keep=True, final decision is keep
    if keep2:
        if verbose:
            print(f"[INFO] Triple judgment: first invalid, second valid, keeping page: {rel_str}")
        result = dict(task)
        result["keep"] = True
        result["cat_label"] = f"{cat1}|{cat2}"
        result["reason"] = f"1st: {reason1}; 2nd: {reason2} (flip_to_keep)"
        result["judgments"] = judgments
        return result

    # First two are both False, proceed with third judgment
    keep3, cat3, reason3 = call_quality_model(
        markdown_text=md_text,
        api_base=api_base,
        api_key=api_key,
        model=model,
        wiki_title=wiki_title,
        ref_title=ref_title,
        page_title=page_title,
    )
    judgments.append({"round": 3, "keep": keep3, "category": cat3, "reason": reason3})
    
    if verbose:
        print(f"[INFO] (LLM-3) {rel_str} -> keep={keep3}, category={cat3}, reason={reason3}")

    # Determine final result
    if keep3:
        # Third judgment is valid, final decision is keep
        if verbose:
            print(f"[INFO] Triple judgment: first two invalid, third valid, keeping page: {rel_str}")
        result = dict(task)
        result["keep"] = True
        result["cat_label"] = f"{cat1}|{cat2}|{cat3}"
        result["reason"] = f"1st: {reason1}; 2nd: {reason2}; 3rd: {reason3} (flip_to_keep)"
        result["judgments"] = judgments
        return result
    else:
        # All three judgments are False, consider page invalid
        if verbose:
            print(f"[INFO] Triple judgment: all three invalid, discarding page: {rel_str}")
        result = dict(task)
        result["keep"] = False
        result["cat_label"] = f"{cat1}|{cat2}|{cat3}"
        result["reason"] = f"1st: {reason1}; 2nd: {reason2}; 3rd: {reason3} (all_false)"
        result["judgments"] = judgments
        return result


# -----------------------
# Main logic
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        default="./corpus",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="LLM API base_url (OpenAI compatible /v1)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name for filtering",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="Model API key (can also use env var OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--only-category",
        default=None,
        help="Only process a specific top-level category (e.g. ai_and_ml), optional",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of ref pages to process (global), for debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations, don't actually delete/re-fetch",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more details",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=10,
        help="Number of concurrent LLM threads, default 1 (serial), can set to 4/8 based on API QPS",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Must provide model key via --api-key or env var OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    root_dir = Path(args.root_dir).resolve()
    fetch_script = (Path(__file__).parent / "fetch_reference_pages.py").resolve()

    # -----------------------
    # Stage 1: Collect all ref pages to process
    # -----------------------
    tasks: List[Dict[str, Any]] = []
    scanned = 0

    for md_path, jsonl_path in iter_reference_pages(root_dir):
        rel = md_path.relative_to(root_dir)
        parts = rel.parts
        if len(parts) < 4:
            continue
        category_name = parts[0]
        article_slug = parts[1]
        wiki_title = article_slug.replace("_", " ")

        if args.only_category and category_name != args.only_category:
            continue

        try:
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Failed to read {md_path}: {e}")
            continue

        page_title = extract_title_from_md(md_path, md_text)
        rel_str = str(rel)

        tasks.append({
            "md_path": md_path,
            "jsonl_path": jsonl_path,
            "rel_str": rel_str,
            "category_name": category_name,
            "wiki_title": wiki_title,
            "page_title": page_title,
            "md_text": md_text,
        })

        scanned += 1
        if args.max_pages is not None and scanned >= args.max_pages:
            break

    if not tasks:
        print("[INFO] No ref pages found to process, exiting.")
        return

    print(f"[INFO] Total ref pages pending LLM cleaning: {len(tasks)}")
    print(f"[INFO] Using triple judgment mechanism: only discarded when all three are False")
    if args.llm_workers > 1:
        print(f"[INFO] Using concurrent LLM threads: {args.llm_workers}")
    else:
        print("[INFO] LLM calls executing serially (--llm-workers=1)")

    # -----------------------
    # Stage 2: LLM concurrent cleaning (triple judgment)
    # -----------------------
    llm_results: List[Dict[str, Any]] = []

    if args.llm_workers <= 1:
        # Serial
        for t in tasks:
            res = run_llm_triple_check_for_page(
                t,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                verbose=args.verbose,
            )
            llm_results.append(res)
    else:
        # Concurrent
        with ThreadPoolExecutor(max_workers=args.llm_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_llm_triple_check_for_page,
                    t,
                    args.api_base,
                    args.api_key,
                    args.model,
                    args.verbose,
                ): t
                for t in tasks
            }
            for future in as_completed(future_to_task):
                res = future.result()
                llm_results.append(res)

    # Sort by relative path
    llm_results.sort(key=lambda r: r["rel_str"])

    # -----------------------
    # Stage 3: Execute subsequent operations based on triple judgment results
    # -----------------------
    invalid = 0
    refetched_ok = 0
    refetched_dropped = 0
    dropped_direct = 0

    for res in llm_results:
        md_path: Path = res["md_path"]
        jsonl_path: Path = res["jsonl_path"]
        rel_str: str = res["rel_str"]
        wiki_title: str = res["wiki_title"]
        page_title: str = res["page_title"]
        keep: bool = res["keep"]
        reason: str = res["reason"]
        judgments: List[Dict] = res.get("judgments", [])

        # If still True after triple judgment, no processing needed
        if keep:
            continue

        # All three judged as False
        invalid += 1
        print(f"[BAD] Detected invalid ref page (all three judgments False): {rel_str}")
        print(f"  Judgment details: {reason}")

        # Subsequent processing logic remains unchanged
        refs = load_references(jsonl_path)
        if not refs:
            print(f"  [WARN] {jsonl_path} is empty, skipping")
            continue

        idx = find_best_ref_index(page_title, refs)
        if idx is None:
            print(f"  [WARN] Cannot match to entry in references.jsonl")
            if not args.dry_run and md_path.exists():
                md_path.unlink()
                dropped_direct += 1
            continue

        ref = refs[idx]
        ref_title = ref.get("title", "")
        archive_url = ref.get("archive_url") or ""
        has_archive = bool(archive_url)

        print(f"  [MATCH] references.jsonl row {idx+1}: title='{ref_title}', has_archive={has_archive}")

        # Case A: has archive_url -> re-fetch
        if has_archive:
            print("  [ACTION] archive_url exists, attempting to re-fetch")
            if not args.dry_run:
                cmd = [
                    sys.executable,
                    str(fetch_script),
                    "--references", str(jsonl_path),
                    "--output-dir", str(md_path.parent),
                    "--start", str(idx + 1),
                    "--end", str(idx + 1),
                    "--archive-mode", "direct",
                    "--fetcher", "jina",
                    "--force",
                    "--no-skip-exists",
                ]
                if args.verbose:
                    cmd.append("--verbose")
                subprocess.run(cmd, check=True)

                try:
                    new_md_text = md_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    new_md_text = ""

                new_page_title = extract_title_from_md(md_path, new_md_text)
                keep2, cat2, reason2 = call_quality_model(
                    markdown_text=new_md_text,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    wiki_title=wiki_title,
                    ref_title=ref_title,
                    page_title=new_page_title,
                )
                print(f"  [REFETCH_CHECK] new content judgment keep={keep2}, reason={reason2}")

                refs2 = load_references(jsonl_path)
                if idx < len(refs2):
                    ref2 = refs2[idx]
                    ref2["llm_filter_status"] = "refetched_ok" if keep2 else "refetched_still_bad_dropped"
                    ref2["llm_filter_reason"] = reason2
                    save_references(jsonl_path, refs2)

                if keep2:
                    refetched_ok += 1
                else:
                    if md_path.exists():
                        md_path.unlink()
                    refetched_dropped += 1
            else:
                print("  [DRY-RUN] No actual operation")

        # Case B: no archive_url -> discard directly
        else:
            print("  [ACTION] No archive_url, discarding directly")
            if not args.dry_run:
                refs = load_references(jsonl_path)
                if idx < len(refs):
                    ref = refs[idx]
                    ref["llm_filter_status"] = "dropped_without_refetch"
                    ref["llm_filter_reason"] = reason
                    save_references(jsonl_path, refs)
                if md_path.exists():
                    md_path.unlink()
            dropped_direct += 1

    print("\n========== Summary ==========")
    print(f"Total ref pages scanned: {scanned}")
    print(f"Pages with all three judgments False: {invalid}")
    print(f"  Re-fetched and valid: {refetched_ok}")
    print(f"  Re-fetched but still invalid: {refetched_dropped}")
    print(f"  Directly discarded: {dropped_direct}")
    print("======================================")



if __name__ == "__main__":
    main()
