#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量抓取引用来源页面

功能:
  读取 references.jsonl (由 extract_references.py 生成)，按指定行号范围抓取其中的 url，
  利用 goliath 和 jina_scraping 工具获取页面内容并保存为 Markdown。

抓取策略:
  - 可通过 --fetcher 指定先手抓取器: goliath 或 jina
  - 可通过 --archive-mode 控制是否使用 archive_url:
      auto   : (默认) 若存在 archive_url 先抓 archive，再抓原始 url
      direct : 忽略 archive_url，直接抓原始 url；若低质量且存在 archive_url，则用 archive_url 作为一次性补位再校验
  - 若先手为 goliath（默认，成本优先）：
      1) 在 auto 模式下按 archive→main 顺序尝试；在 direct 模式下只尝试 main
      2) goliath 失败且为可回退错误时，fallback 到 jina
  - 若先手为 jina（带重试保护）：
      1) 对当前 URL 连续 --jina-retries 次 jina 抓取，仍失败再切 goliath

使用示例:
  python fetch_reference_pages.py \
    --references /path/to/wiki_data/Joe Biden/reference/references.jsonl \
    --output-dir /path/to/wiki_data/Joe Biden/reference/ref_pages \
    --start 1 --end 100 --fetcher jina --jina-retries 5 --archive-mode direct
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List, Tuple

from jina_scraping import WebScrapingJinaTool, DEFAULT_API_KEY, slugify  # type: ignore
from goliath import build_default_goliath_tool, GoliathTool  # type: ignore


def save_markdown(
  reference: Dict[str, Any],
  fetched: Dict[str, Any],
  output_dir: str,
  line_no: int,
  skip_exists: bool
) -> str:
  """保存引用抓取结果为纯净 Markdown。"""
  os.makedirs(output_dir, exist_ok=True)
  title = reference.get('title') or fetched.get('title') or 'Untitled'
  slug = slugify(title)
  filename = f"{slug}.md"
  path = os.path.join(output_dir, filename)

  if skip_exists and os.path.exists(path):
    return path

  parts: List[str] = []
  parts.append(f"# {title}\n")

  if 'error' in fetched:
    parts.append(f"<!-- fetch error: {fetched['error']} -->\n")
  else:
    content = fetched.get('content', '').rstrip() + '\n'
    parts.append(content)

  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(parts))

  return path


def load_references(jsonl_path: str) -> List[Dict[str, Any]]:
  refs: List[Dict[str, Any]] = []
  with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
      line_stripped = line.strip()
      if not line_stripped:
        continue
      try:
        data = json.loads(line_stripped)
      except json.JSONDecodeError:
        data = {'error': 'json_decode_error', 'raw': line_stripped}
      refs.append(data)
  return refs


def save_references(jsonl_path: str, refs: List[Dict[str, Any]]):
  tmp_path = jsonl_path + '.tmp'
  with open(tmp_path, 'w', encoding='utf-8') as f:
    for r in refs:
      f.write(json.dumps(r, ensure_ascii=False) + '\n')
  os.replace(tmp_path, jsonl_path)


def is_low_quality(content: str) -> Tuple[bool, str]:
  """判断内容是否为需要过滤的低质量页面。

  规则:
    1. 包含已知 404 / 错误页特征字符串之一
    2. 总行数 < 5 且每行词数 < 100（此处按更宽松的实现：行数 < 10 且所有行词数 < 50）

  返回 (需要过滤, 原因)
  """
  if not content:
    return True, 'empty_content'
  patterns = [
    '404 | Fox News',
    'It seems you clicked on a bad link',
    'Something has gone wrong',
    '404 Not Found',
    'Page Not Found',
    'Access Denied',
    'Permission Denied',
  ]
  for p in patterns:
    if p in content:
      return True, f'match_pattern:{p}'

  # 宽松版低质量判断
  lines = [l.strip() for l in content.splitlines() if l.strip()]
  if len(lines) < 13:
    all_short = True
    for line in lines:
      if len(line.split()) >= 50:
        all_short = False
        break
    if all_short:
      return True, 'line_count<10_and_all_lines<50words'
  return False, ''


def is_goliath_error(error_msg: str) -> bool:
  """判断是否为 goliath 的常见错误类型，这些错误应该使用 jina 重试"""
  if not error_msg:
    return False
  goliath_error_patterns = [
    'timeout', 'connection', 'network', 'selenium', 'browser',
    'webdriver', 'javascript', 'render', 'chrome', 'proxy',
    # 可按需继续扩充: '403', '429', 'captcha', 'forbidden', 'cloudflare', 'blocked', 'bot'
  ]
  error_lower = error_msg.lower()
  return any(pattern in error_lower for pattern in goliath_error_patterns)


def main():
  parser = argparse.ArgumentParser(
    description='批量抓取 references.jsonl 中的 URL 内容并保存为 Markdown'
  )
  parser.add_argument(
    '--references',
    required=True,
    help='references.jsonl 路径'
  )
  parser.add_argument(
    '--output-dir',
    required=True,
    help='输出目录，用于存放抓取的引用页面 Markdown'
  )
  parser.add_argument(
    '--start',
    type=int,
    default=1,
    help='起始行号(1-based)'
  )
  parser.add_argument(
    '--end',
    type=int,
    default=None,
    help='结束行号(1-based, 包含)'
  )
  parser.add_argument(
    '--api-key',
    dest='api_key',
    default=None,
    help='Jina API Key (仅 fallback 时使用或作为先手时使用)'
  )
  parser.add_argument(
    '--fetcher',
    choices=['goliath', 'jina'],
    default='goliath',
    help='先手抓取工具: goliath(默认) 或 jina'
  )
  parser.add_argument(
    '--jina-retries',
    type=int,
    default=5,
    help='当 --fetcher=jina 时，对同一 URL 的连续重试次数，超过该次数仍失败则切换到 goliath'
  )
  parser.add_argument(
    '--archive-mode',
    choices=['auto', 'direct'],
    default='direct',
    help='auto=先 archive 后 main；direct=忽略 archive_url，只抓原始 url；若低质量则尝试 archive 作为补位'
  )
  parser.add_argument(
    '--skip-exists',
    dest='skip_exists',
    action='store_true',
    default=True,
    help='已存在文件则跳过 (默认)'
  )
  parser.add_argument(
    '--no-skip-exists',
    dest='skip_exists',
    action='store_false',
    help='即使存在也覆盖'
  )
  parser.add_argument(
    '--verbose',
    action='store_true',
    help='输出详细抓取尝试日志'
  )
  parser.add_argument(
    '--force',
    action='store_true',
    help='忽略 scraped 标记，强制重新抓取并覆盖 fetcher_used'
  )
  parser.add_argument(
    '--record-attempts',
    action='store_true',
    help='在引用条目中写入 attempt_log 详细尝试结果'
  )
  args = parser.parse_args()

  primary = args.fetcher            # 'goliath' or 'jina'
  archive_mode = args.archive_mode  # 'auto' or 'direct'

  # 准备工具
  g_tool = build_default_goliath_tool()

  api_key = args.api_key or os.environ.get('JINA_API_KEY') or DEFAULT_API_KEY
  if not api_key.startswith('Bearer '):
    api_key = f'Bearer {api_key}'
  jina_tool = WebScrapingJinaTool(api_key)

  def normalize(data: dict) -> dict:
    return {
      'title': data.get('title', ''),
      'content': data.get('content', ''),
      'publish_time': data.get('publish_time', '') or data.get('fetched_publish_time', '')
    }

  def try_goliath(url: str):
    try:
      d = g_tool(url)
      if not d.get('success') or d.get('error'):
        return None, d.get('error', 'goliath_error')
      return normalize(d), None
    except Exception as e:
      return None, str(e)

  def try_jina(url: str):
    try:
      d = jina_tool(url)
      if 'error' in d:
        return None, d.get('error', 'jina_error')
      return normalize(d), None
    except Exception as e:
      return None, str(e)

  def try_jina_with_retries(
    url: str,
    url_type: str,
    retries: int
  ) -> Tuple[Any, Any, List[Tuple[str, str, str, bool, str]]]:
    attempts_local: List[Tuple[str, str, str, bool, str]] = []
    last_err = None
    for i in range(1, retries + 1):
      if args.verbose:
        print(f"       尝试 jina({url_type}) 第 {i}/{retries} 次")
      norm, err = try_jina(url)
      attempts_local.append(
        (f'jina_{url_type}_try{i}', 'jina', url, err is None, '' if err is None else err)
      )
      if err is None:
        return norm, None, attempts_local
      last_err = err
    return None, last_err, attempts_local

  # NEW: 抽出“单一 URL 抓取”的公共逻辑，便于 direct 模式下的二段补位
  def fetch_one(url_type: str, url_to_try: str):
    """按当前先手/重试策略抓取单一 URL，返回 (norm, used_label, attempts)"""
    attempts: List[Tuple[str, str, str, bool, str]] = []
    if primary == 'goliath':
      if args.verbose:
        print(f"       尝试 goliath({url_type})")
      norm, err = try_goliath(url_to_try)
      attempts.append(
        (f'goliath_{url_type}', 'goliath', url_to_try, err is None, '' if err is None else err)
      )
      if err is None:
        used_label = f'goliath({url_type})' if url_type == 'archive' else 'goliath'
        return norm, used_label, attempts
      if is_goliath_error(err):
        if args.verbose:
          print(f"       goliath 失败 ({err})，fallback 到 jina({url_type})")
        norm_j, err_j = try_jina(url_to_try)
        attempts.append(
          (f'jina_{url_type}', 'jina', url_to_try, err_j is None, '' if err_j is None else err_j)
        )
        if err_j is None:
          used_label = f'jina({url_type})' if url_type == 'archive' else 'jina'
          return norm_j, used_label, attempts
    else:
      norm_j, err_j, j_attempts = try_jina_with_retries(url_to_try, url_type, args.jina_retries)
      attempts.extend(j_attempts)
      if err_j is None:
        used_label = f'jina({url_type})' if url_type == 'archive' else 'jina'
        return norm_j, used_label, attempts
      if args.verbose:
        print(f"       jina 连续 {args.jina_retries} 次失败，切换到 goliath({url_type})")
      norm_g, err_g = try_goliath(url_to_try)
      attempts.append(
        (f'goliath_{url_type}', 'goliath', url_to_try, err_g is None, '' if err_g is None else err_g)
      )
      if err_g is None:
        used_label = f'goliath({url_type})' if url_type == 'archive' else 'goliath'
        return norm_g, used_label, attempts
    return None, 'failed', attempts

  def fetch_with_fallback(ref: dict):
    """根据先手抓取器与 archive_mode 执行抓取策略，返回: (norm, used_label, attempts)"""
    url_main = ref.get('url')
    archive_url = ref.get('archive_url')
    attempts_all: List[Tuple[str, str, str, bool, str]] = []

    # URL 尝试序列
    if archive_mode == 'auto' and archive_url:
      sequence: List[Tuple[str, str]] = [('archive', archive_url), ('main', url_main)]
    else:
      sequence = [('main', url_main)]

    for url_type, url_to_try in sequence:
      if args.verbose:
        print(f"    -> 处理 {url_type} url: {url_to_try}")
      norm, used_label, attempts = fetch_one(url_type, url_to_try)
      attempts_all.extend(attempts)
      if norm is not None:
        return norm, used_label, attempts_all

    return None, 'failed', attempts_all

  refs = load_references(args.references)
  total = 0
  success = 0
  failed = 0
  start = args.start if args.start >= 1 else 1
  end = args.end if args.end is not None else len(refs)
  end = min(end, len(refs))

  # 统计 fallback 情况
  fallback_count = 0
  goliath_error_count = 0

  print(
    f"[INFO] 抓取策略: 先手={primary}（"
    f"{'成本优先' if primary == 'goliath' else f'Jina重试{args.jina_retries}次后切Goliath'}）"
  )
  print(f"[INFO] Archive模式: {archive_mode}")
  print(f"[INFO] 处理范围: {start}-{end} (共 {end - start + 1} 个)")

  for idx in range(start, end + 1):  # idx 是 1-based 行号
    ref = refs[idx - 1]
    total += 1
    url = ref.get('url')
    archive_url = ref.get('archive_url')
    if not url:
      failed += 1
      print(f"[line {idx}] 缺少 url，跳过")
      continue
    if ref.get('scraped') is True and not args.force:
      print(f"[line {idx}] 已爬取，跳过 (可用 --force 重新抓取)")
      continue

    print(f"[line {idx}] 抓取 {url} ...")
    max_filter_retry = 3
    filter_attempts = 0
    aggregate_attempt_log: List[Dict[str, Any]] = []
    tried_archive_fill = False  # NEW: direct模式下的“归档补位”只做一次

    while True:
      norm, used, attempts = fetch_with_fallback(ref)

      # 统计是否发生 fallback 以及是否有 goliath 错误
      used_fallback = any((a[1] != primary) and a[3] for a in attempts)
      had_goliath_error = any((a[1] == 'goliath') and (not a[3]) for a in attempts)
      if had_goliath_error:
        goliath_error_count += 1
      if used_fallback:
        fallback_count += 1

      if norm is None:
        failed += 1
        if args.record_attempts:
          aggregate_attempt_log.extend([
            {
              'round': filter_attempts + 1,
              'step': a[0],
              'engine': a[1],
              'url': a[2],
              'success': a[3],
              'error': a[4],
            }
            for a in attempts
          ])
          ref['attempt_log'] = aggregate_attempt_log
        err_msg = '; '.join(f"{a[0]}:{a[4]}" for a in attempts if a[4]) or 'unknown_error'
        print(f"  失败(fetcher_used={used}): {err_msg} (不保存文件)")
        break

      if args.record_attempts:
        aggregate_attempt_log.extend([
          {
            'round': filter_attempts + 1,
            'step': a[0],
            'engine': a[1],
            'url': a[2],
            'success': a[3],
            'error': a[4],
          }
          for a in attempts
        ])

      # 质量校验
      filt, reason = is_low_quality(norm.get('content', ''))
      if filt:
        # NEW: direct 模式下、存在 archive_url、且尚未尝试过“归档补位”，则改抓 archive 再校验
        if (archive_mode == 'direct') and archive_url and (not tried_archive_fill):
          tried_archive_fill = True
          print("  低质量内容，direct模式启用归档补位 -> 抓取 archive_url 再校验")
          norm2, used2, attempts2 = fetch_one('archive', archive_url)

          # 统计/记录
          used_fallback2 = any((a[1] != primary) and a[3] for a in attempts2)
          had_goliath_error2 = any((a[1] == 'goliath') and (not a[3]) for a in attempts2)
          if had_goliath_error2:
            goliath_error_count += 1
          if used_fallback2:
            fallback_count += 1
          if args.record_attempts:
            aggregate_attempt_log.extend([
              {
                'round': filter_attempts + 1,
                'step': a[0],
                'engine': a[1],
                'url': a[2],
                'success': a[3],
                'error': a[4],
              }
              for a in attempts2
            ])

          if norm2 is None:
            # 归档抓取失败，视为处理完毕但不保存
            ref['scraped'] = True
            ref['fetcher_used'] = used + '->archive(fetch_failed)'
            ref['filter_reason'] = f'direct_low_quality_then_archive_fetch_failed:{reason}'
            success += 1
            print("  归档补位抓取失败，标记 scraped=true 不保存文件")
            break

          # 对归档内容做质量校验
          filt2, reason2 = is_low_quality(norm2.get('content', ''))
          if filt2:
            ref['scraped'] = True
            ref['fetcher_used'] = used2 + '(low_quality)'
            ref['filter_reason'] = f'direct_low_quality_then_archive_low_quality:{reason2}'
            success += 1
            print(f"  归档内容仍低质量({reason2})，标记 scraped=true 不保存文件")
            break

          # 归档内容可接受 -> 保存并完成
          success += 1
          ref['scraped'] = True
          ref['fetcher_used'] = used2
          if args.record_attempts:
            ref['attempt_log'] = aggregate_attempt_log
          out_path = save_markdown(ref, norm2, args.output_dir, idx, args.skip_exists)
          print(f"  保存 -> {out_path} (fetcher_used={used2}) [direct模式归档补位成功]")
          break

        # 非 direct 模式或已尝试过归档补位：沿用原重试上限
        filter_attempts += 1
        print(f"  低质量内容(原因={reason}) 第 {filter_attempts} 次")
        if filter_attempts < max_filter_retry:
          print("  -> 重试抓取...")
          continue
        else:
          ref['scraped'] = True
          ref['fetcher_used'] = used + '(low_quality)' if used else 'low_quality'
          ref['filter_reason'] = reason
          if args.record_attempts:
            ref['attempt_log'] = aggregate_attempt_log
          success += 1
          print(f"  达到最大重试，标记 scraped=true 不保存文件")
          break

      else:
        # 内容可接受，保存
        success += 1
        ref['scraped'] = True
        ref['fetcher_used'] = used
        if args.record_attempts:
          ref['attempt_log'] = aggregate_attempt_log
        out_path = save_markdown(ref, norm, args.output_dir, idx, args.skip_exists)
        fallback_msg = ""
        if used_fallback:
          fallback_msg = " [FALLBACK成功]"
        elif had_goliath_error and not used_fallback:
          fallback_msg = " [GOLIATH错误但FALLBACK失败]"
        print(f"  保存 -> {out_path} (fetcher_used={used}){fallback_msg}")
        break

    if total % 10 == 0:
      save_references(args.references, refs)

  save_references(args.references, refs)

  print(f"\n完成: total={total} success={success} failed={failed}")
  print(f"Fallback统计: goliath错误={goliath_error_count}, fallback成功={fallback_count}")

  from collections import Counter
  used_counter = Counter(r.get('fetcher_used') for r in refs if r.get('scraped'))
  if used_counter:
    print("\n成功来源统计:")
    for k, v in used_counter.most_common():
      if k:
        print(f"  {k}: {v}")


if __name__ == '__main__':
  main()
