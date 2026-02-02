#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 human_performance.jsonl 与 reference_pages 中的 md 文件关联起来，
生成包含完整参考文献内容的新 JSONL 文件。

输入:
  - --input: human_performance.jsonl 文件路径
  - --topic-dir: 包含 reference 目录的主题目录 (如 Donald Trump)
  - --out: 输出的 enriched JSONL 文件路径

输出:
  - 每条记录包含原始的 question, ref_urls, predanswer，
    以及新增的 ref_docs 字段，包含每个 URL 对应的 md 文件全文
"""

import os
import re
import json
import argparse
import unicodedata
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional


class ReferenceResolver:
    """
    从 qa_generator.py 移植的引用解析器：
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
        unmatched_titles_info: List[tuple] = []

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
            still_unmatched: List[tuple] = []

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

            print(f"  [参考解析器] 相似度匹配: {sim_matched} 个额外匹配")

        print(f"[参考解析器] 总共匹配 URL: {matched_count}/{len(self.url_to_title)}")

    def get_file_for_url(self, url: str) -> Optional[Path]:
        """根据 URL 返回对应的 MD 文件路径"""
        url = url.split("#")[0]  # 去掉锚点
        meta = self.url_meta.get(url)
        if meta:
            return meta.get("file")
        return None

    def get_title_for_url(self, url: str) -> Optional[str]:
        """根据 URL 返回对应的标题"""
        url = url.split("#")[0]
        return self.url_to_title.get(url)


def resolve_ref_urls_to_docs(resolver: ReferenceResolver, ref_urls: List[str]) -> List[Dict]:
    """
    根据 ref_urls，加载对应的 MD 内容：
    返回形如 [{"url":..., "title":..., "content":..., "matched": True/False}, ...]
    """
    refs = []
    seen_keys = set()
    
    for url in ref_urls or []:
        url_clean = url.split("#")[0]
        if url_clean in seen_keys:
            continue
        seen_keys.add(url_clean)
        
        title = resolver.get_title_for_url(url_clean)
        file_path = resolver.get_file_for_url(url_clean)
        
        ref_doc = {
            "url": url,
            "title": title or "",
            "content": "",
            "matched": False
        }
        
        if file_path and file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                ref_doc["content"] = content
                ref_doc["matched"] = True
            except Exception as e:
                print(f"  [警告] 无法读取文件 {file_path}: {e}")
        
        refs.append(ref_doc)
    
    return refs


def main():
    parser = argparse.ArgumentParser(
        description="将 human_performance.jsonl 与 reference_pages 中的 md 文件关联"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入的 human_performance.jsonl 文件路径"
    )
    parser.add_argument(
        "--topic-dir", "-t",
        type=str,
        required=True,
        help="包含 reference 目录的主题目录"
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        required=True,
        help="输出的 enriched JSONL 文件路径"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    topic_dir = Path(args.topic_dir)
    output_path = Path(args.out)
    
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    if not topic_dir.exists():
        print(f"错误: 主题目录不存在: {topic_dir}")
        return
    
    print(f"[开始] 加载参考解析器...")
    resolver = ReferenceResolver(topic_dir)
    
    print(f"\n[处理] 读取 {input_path}...")
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  [警告] 无法解析 JSON: {e}")
    
    print(f"  读取到 {len(records)} 条记录")
    
    print(f"\n[关联] 为每条记录匹配参考文献...")
    enriched_records = []
    total_refs = 0
    matched_refs = 0
    
    for i, record in enumerate(records, 1):
        ref_urls = record.get("ref_urls", [])
        ref_docs = resolve_ref_urls_to_docs(resolver, ref_urls)
        
        # 统计
        total_refs += len(ref_docs)
        matched_refs += sum(1 for r in ref_docs if r["matched"])
        
        # 构建新记录
        enriched_record = {
            "question": record.get("question", ""),
            "ref_urls": ref_urls,
            "predanswer": record.get("predanswer", ""),
            "ref_docs": ref_docs
        }
        enriched_records.append(enriched_record)
        
        print(f"  [{i}/{len(records)}] 问题: {record.get('question', '')[:50]}... "
              f"| 引用: {len(ref_docs)}, 匹配: {sum(1 for r in ref_docs if r['matched'])}")
    
    print(f"\n[统计] 总引用数: {total_refs}, 成功匹配: {matched_refs} ({matched_refs/total_refs*100:.1f}%)")
    
    print(f"\n[保存] 写入 {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[完成] 已生成 {len(enriched_records)} 条 enriched 记录到 {output_path}")


if __name__ == "__main__":
    main()
