#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
post_filter_ref_pages.py

功能（离线后处理一条龙）：
1. 遍历 corpus 下所有 article 的 reference/reference_pages/*.md
2. 调用 LLM 过滤每个 ref 页面：
   - keep=True 认为是“正常引用原文”，保留
   - keep=False 认为是“垃圾/错误页/索引/空内容”等，进入修复流程
3. 对于 keep=False 的页面：
   - 用 md 文件里的标题，在 references.jsonl 中做模糊匹配找到对应条目
   - 若该条 ref 存在 archive_url：
       * 调用 fetch_reference_pages.py，用 archive-mode=direct + fetcher=jina 强制用原始 url 重抓（--force + --no-skip-exists）
       * 对新抓到的 md 再跑一次 LLM 过滤
           - 如果通过：保留新 md，并在 references.jsonl 上打上 llm_filter_status = "refetched_ok"
           - 如果仍不过：删除 md，标记 llm_filter_status = "refetched_still_bad_dropped"
   - 否则（没有 archive_url）：
       * 直接删除 md，标记 llm_filter_status = "dropped_without_refetch"

新增：
- 支持并发调用 LLM：参数 --llm-workers，默认 1（串行）
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
# LLM 相关
# -----------------------

def _truncate_text(text: str, max_chars: int = 6000) -> str:
    """目前不再使用截断，但保留工具函数以备后续需要。"""
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
    调用 LLM 判断一个 ref 页面是否是“正常引用原文”。

    这里会把：
      - 维基百科条目标题 wiki_title
      - references.jsonl 中该条 reference 的 title（ref_title）
      - 该 md 页面自身的标题（page_title）
      - 该 md 的全文（Markdown，绝对不截断）
    一并给到模型。

    返回:
        keep: bool       # True=保留, False=认为是垃圾/错误页
        category: str    # 模型分类标签（如 ok/404/index/login/...）
        reason: str      # 简短中文说明
    """
    # ✅ 不再截断：直接使用全文
    excerpt = markdown_text if markdown_text is not None else ""
    excerpt = excerpt.strip()

    system_prompt = (
        "你是一个网页质量过滤助手。我们从互联网上抓取了一些网页作为维基百科条目的参考文献原文页面，"
        "内容已经被转成 Markdown。你的任务是判断给定页面是否“可以作为这个维基条目的有用参考来源”。\n\n"
        "你会看到：\n"
        "1) 维基条目的标题(WIKI_TITLE)\n"
        "2) 参考文献条目的标题(REF_TITLE)\n"
        "3) 该参考页面 Markdown 的全文内容(PAGE_TITLE + FULL_MARKDOWN)\n\n"
        "判断为 keep=true 的情况（尽量宽松）：\n"
        "- 页面包含与该维基主题/参考条目相关的实质性文字内容，即使篇幅不长、排版混乱、夹杂导航或广告；\n"
        "- 是新闻报道、百科条目、论文、官方声明、博客、论坛帖子等，只要有与主题相关的正文，都算有用；\n"
        "- 即使有很多无关元素（导航栏、侧边栏、推荐链接），只要有一部分清晰的正文信息即可。\n\n"
        "判断为 keep=false 的情况（严格）：\n"
        "- 明显是 404 页面、错误页、访问被拒绝，仅有几行错误提示；\n"
        "- 需要登录/订阅/购买才能看的页面，主体内容完全不可见（只有登录或订阅提示）；\n"
        "- 纯搜索结果页、站点目录、仅有链接列表或导航，没有任何实质正文；\n"
        "- 完全空白或者只有极少量无意义字符。\n\n"
        "注意：\n"
        "- 不要因为内容看起来“简单/短/写得不好”就判为 false，只要有一些与主题相关的信息就应 keep=true；\n"
        "- 不要求和 WIKI_TITLE / REF_TITLE 完全匹配，只要在主题上大致相关且有用即可。\n\n"
        "输出必须是一个 JSON 对象，不要附加任何解释文字。字段：\n"
        '{ \"keep\": true 或 false, \"category\": \"ok/404/index/login/empty/other\", \"reason\": \"用中文简要说明(<=30字)\" }'
    )

    user_prompt = (
        "下面是一个维基参考文献页面的相关信息，请按照系统提示进行判断。\n\n"
        f"WIKI_TITLE: {wiki_title}\n"
        f"REF_TITLE: {ref_title}\n"
        f"PAGE_TITLE: {page_title}\n\n"
        "下面是该页面的 Markdown 全文：\n"
        "--------------------\n"
        f"{excerpt}\n"
        "--------------------\n"
        "请只输出一个 JSON 对象，不要输出其它任何文字。"
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
        # 模型挂了就保守点：先保留
        return True, "model_error", f"调用模型失败: {type(e).__name__}"

    # 从返回文本中尽量抠出 JSON
    try:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError("未找到 JSON 对象")
        obj = json.loads(m.group(0))
        keep = bool(obj.get("keep", True))
        category = str(obj.get("category", "unknown"))
        reason = str(obj.get("reason", ""))
        return keep, category, reason
    except Exception as e:
        # 解析失败，同样先保留，避免误删
        return True, "parse_error", f"解析模型输出失败: {type(e).__name__}"


# -----------------------
# references.jsonl 相关
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
    优先从 Markdown 第一行以 '# ' 开头的标题获取；
    否则退回到文件名（反 slug 一下）。
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
    # fallback：文件名去掉扩展名
    name = md_path.stem
    name = name.replace("_", " ")
    return name


# -----------------------
# 遍历 reference_pages
# -----------------------

def iter_reference_pages(root_dir: Path):
    """
    遍历所有 reference/reference_pages 目录，yield (md_path, references.jsonl path)
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
# LLM 并发清洗阶段
# -----------------------

def run_llm_triple_check_for_page(
    task: Dict[str, Any],
    api_base: str,
    api_key: str,
    model: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    对单个 ref 页面做"三次判定" LLM 调用。
    只有三次都判定为 keep=False 时，才认为页面无效。

    输入 task 包含：
      - md_path
      - jsonl_path
      - rel_str
      - wiki_title
      - page_title
      - md_text

    输出 result 包含：
      - keep: bool
      - cat_label: str
      - reason: str
      - judgments: List[Dict] - 记录每次判定的结果
      - 以及原 task 的字段
    """
    md_path: Path = task["md_path"]
    jsonl_path: Path = task["jsonl_path"]
    rel_str: str = task["rel_str"]
    wiki_title: str = task["wiki_title"]
    page_title: str = task["page_title"]
    md_text: str = task["md_text"]

    # 尝试从 references.jsonl 中找到 ref_title
    refs = load_references(jsonl_path)
    ref_title = ""
    if refs:
        idx0 = find_best_ref_index(page_title, refs)
        if idx0 is not None:
            ref_title = str(refs[idx0].get("title", "") or "")

    judgments = []
    
    # 第一次判定
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

    # 如果第一次判定为 keep=True，直接返回保留
    if keep1:
        result = dict(task)
        result["keep"] = True
        result["cat_label"] = cat1
        result["reason"] = reason1
        result["judgments"] = judgments
        return result

    # 第一次判定为 False，进行第二次判定
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

    # 如果第二次判定为 keep=True，最终保留
    if keep2:
        if verbose:
            print(f"[INFO] 三次判定：第一次无效，第二次有效，最终保留该页面：{rel_str}")
        result = dict(task)
        result["keep"] = True
        result["cat_label"] = f"{cat1}|{cat2}"
        result["reason"] = f"1st: {reason1}; 2nd: {reason2} (flip_to_keep)"
        result["judgments"] = judgments
        return result

    # 前两次都判定为 False，进行第三次判定
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

    # 判断最终结果
    if keep3:
        # 第三次判定为有效，最终保留
        if verbose:
            print(f"[INFO] 三次判定：前两次无效，第三次有效，最终保留该页面：{rel_str}")
        result = dict(task)
        result["keep"] = True
        result["cat_label"] = f"{cat1}|{cat2}|{cat3}"
        result["reason"] = f"1st: {reason1}; 2nd: {reason2}; 3rd: {reason3} (flip_to_keep)"
        result["judgments"] = judgments
        return result
    else:
        # 三次都判定为 False，认为页面无效
        if verbose:
            print(f"[INFO] 三次判定：三次都判为无效，最终丢弃该页面：{rel_str}")
        result = dict(task)
        result["keep"] = False
        result["cat_label"] = f"{cat1}|{cat2}|{cat3}"
        result["reason"] = f"1st: {reason1}; 2nd: {reason2}; 3rd: {reason3} (all_false)"
        result["judgments"] = judgments
        return result


# -----------------------
# 主逻辑
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        default="./corpus",
        help="数据集根目录",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="LLM API base_url (OpenAI 兼容 /v1)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="用于过滤的模型名",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="模型 API key（也可用环境变量 OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--only-category",
        default=None,
        help="仅处理某个一级 category（如 ai_and_ml），可选",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="最多处理多少个 ref 页面（全局），用于调试",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要进行的操作，不真正删除/重抓",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更多细节",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=10,
        help="并发调用 LLM 的线程数，默认 1（串行），可根据接口 QPS 设置为 4/8 等",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] 必须通过 --api-key 或环境变量 OPENAI_API_KEY 提供模型 key", file=sys.stderr)
        sys.exit(1)

    root_dir = Path(args.root_dir).resolve()
    fetch_script = (Path(__file__).parent / "fetch_reference_pages.py").resolve()

    # -----------------------
    # 阶段 1：收集所有要处理的 ref 页面
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
            print(f"[WARN] 读取 {md_path} 失败: {e}")
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
        print("[INFO] 未找到任何 ref 页面可处理，直接退出。")
        return

    print(f"[INFO] 待 LLM 清洗的 ref 页面总数: {len(tasks)}")
    print(f"[INFO] 使用三次判定机制：只有三次都判为 False 才会丢弃")
    if args.llm_workers > 1:
        print(f"[INFO] 使用并发 LLM 线程数: {args.llm_workers}")
    else:
        print("[INFO] LLM 调用串行执行（--llm-workers=1）")

    # -----------------------
    # 阶段 2：LLM 并发清洗（三次判定）
    # -----------------------
    llm_results: List[Dict[str, Any]] = []

    if args.llm_workers <= 1:
        # 串行
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
        # 并发
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

    # 按相对路径排序
    llm_results.sort(key=lambda r: r["rel_str"])

    # -----------------------
    # 阶段 3：根据三次判定结果执行后续操作
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

        # 如果三次判定后仍为 True，无需处理
        if keep:
            continue

        # 三次都判定为 False
        invalid += 1
        print(f"[BAD] 检测到无效 ref 页面（三次判定都为 False）：{rel_str}")
        print(f"  判定详情: {reason}")

        # 后续处理逻辑保持不变
        refs = load_references(jsonl_path)
        if not refs:
            print(f"  [WARN] {jsonl_path} 为空，跳过")
            continue

        idx = find_best_ref_index(page_title, refs)
        if idx is None:
            print(f"  [WARN] 无法匹配到 references.jsonl 中的条目")
            if not args.dry_run and md_path.exists():
                md_path.unlink()
                dropped_direct += 1
            continue

        ref = refs[idx]
        ref_title = ref.get("title", "")
        archive_url = ref.get("archive_url") or ""
        has_archive = bool(archive_url)

        print(f"  [MATCH] references.jsonl 第 {idx+1} 行：title='{ref_title}', has_archive={has_archive}")

        # 情况 A：有 archive_url -> 重抓
        if has_archive:
            print("  [ACTION] 存在 archive_url，尝试重新抓取")
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
                print(f"  [REFETCH_CHECK] 新内容判定 keep={keep2}, reason={reason2}")

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
                print("  [DRY-RUN] 不实际操作")

        # 情况 B：无 archive_url -> 直接丢弃
        else:
            print("  [ACTION] 无 archive_url，直接丢弃")
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

    print("\n========== 总结 ==========")
    print(f"扫描 ref 页面总数: {scanned}")
    print(f"三次判定都为 False 的页面数: {invalid}")
    print(f"  重抓后判定为有效: {refetched_ok}")
    print(f"  重抓后仍无效: {refetched_dropped}")
    print(f"  直接丢弃: {dropped_direct}")
    print("======================================")



if __name__ == "__main__":
    main()
