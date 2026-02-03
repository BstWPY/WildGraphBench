#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Extraction Script

Features:
1. Use WebScrapingJinaTool from jina_scraping to scrape web pages and save as Markdown (creating directory/title/reference/ structure)
2. Parse reference links from the References section in saved Markdown
3. Generate reference/references.jsonl, each line containing a reference item with:
   - url
   - is_external (whether it's an external link)
   - jumpup (if contains footnote markers like "\u2191" or "jump up", provide object info, otherwise empty string)
   - title (estimated reference title)
   - date (if parseable)

Note:
Since Jina scraping returns refined text rather than raw HTML, this script uses heuristic parsing,
which may not 100% restore complex Wikipedia reference structures; can be improved later with MediaWiki API for more structured data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import List, Dict, Any, Optional

from jina_scraping import WebScrapingJinaTool, save_markdown, DEFAULT_API_KEY, slugify  # type: ignore
from urllib.parse import urlparse, unquote

# Retry configuration
MAX_RETRIES = 10       # Maximum 10 retries

# URL extraction regex: matches http/https URLs until whitespace, parenthesis, quote, or bracket
URL_REGEX = re.compile(r'(https?://[^\s\)\]\><"\']+)')
DATE_ISO_REGEX = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
DATE_YMD_REGEX = re.compile(r'\b(\d{4})[\./年-](\d{1,2})[\./月-](\d{1,2})日?\b')
MONTH_NAME_REGEX = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
)
RETRIEVED_DATE_REGEX = re.compile(
    r'(Retrieved\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\.?',
    re.IGNORECASE
)

# Jump/footnote pattern (narrowed: only remove single **[^](...)** or ^ with adjacent anchor links, not consuming following text)
JUMP_CARET_PATTERN = re.compile(r'(\*\*\[\^]\([^)]*\)\*\*)')  # **[^](...)**
JUMP_INLINE_ANCHORS_PATTERN = re.compile(
    r'(\[[^\]]+\]\(https?://[^)]+cite_ref[^)]*\))'
)  # a,b,c anchors
JUMP_PHRASE_PATTERN = re.compile(r'^\s*\^?\s*Jump up to:?', re.IGNORECASE)
MD_LINK_REGEX = re.compile(r'\[([^\]]+)\]\((https?://[^)]+)\)')

def url_to_slug_title(url: str) -> str:
    """Derive a title string from Wiki URL for directory naming.

    Example:
        https://en.wikipedia.org/wiki/Al_Jazeera_Media_Network
        -> 'Al Jazeera Media Network'
    """
    path = urlparse(url).path
    last = path.rsplit('/', 1)[-1]  # Get the last segment
    last = unquote(last)            # Handle %20 etc.
    title = last.replace('_', ' ')
    return title or 'page'

def scrape_with_retry(scraper, url: str) -> Optional[Dict]:
  """Web scraping with retry mechanism

  Args:
      scraper: WebScrapingJinaTool instance
      url: URL to scrape

  Returns:
      Scrape result or None
  """
  for attempt in range(MAX_RETRIES):
    try:
      print(f"🔄 Attempting scrape (attempt {attempt + 1}/{MAX_RETRIES}): {url}")

      # Call scraper without timeout parameter
      data = scraper(url)

      if data and data.get('content'):
        content_length = len(data.get('content', ''))
        print(f"✅ Scrape successful: {content_length} characters")
        return data
      else:
        print(f"⚠️  Scrape returned empty content")
    except Exception as e:
      error_msg = str(e)
      print(f"❌ Scrape failed (attempt {attempt + 1}): {error_msg}")

      if attempt < MAX_RETRIES - 1:
        # Simple wait strategy
        wait_time = (attempt + 1) * 10  # 10-second incremental wait
        print(f"⏳ Waiting {wait_time} seconds before retry...")
        time.sleep(wait_time)
      else:
        print(f"🚫 All retries exhausted")

  return None


def safe_url_to_title(url: str) -> str:
  """Safely extract title from URL, handling special characters"""
  try:
    import urllib.parse

    # Decode URL
    decoded_url = urllib.parse.unquote(url)
    # Extract last part as title
    title = decoded_url.split('/')[-1]
    # Replace underscores with spaces
    title = title.replace('_', ' ')
    # Remove unsafe filename characters
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    return title.strip()
  except Exception as e:
    print(f"⚠️  URL title extraction failed {url}: {e}")
    return "Unknown_Title"


def create_error_placeholder(url: str, output_dir: str, error_msg: str) -> tuple[str, str]:
  """Create error placeholder file

  Args:
      url: Failed URL
      output_dir: Output directory
      error_msg: Error message

  Returns:
      (markdown_path, jsonl_path) tuple
  """
  title = safe_url_to_title(url)
  error_dir = os.path.join(output_dir, "Error")
  os.makedirs(error_dir, exist_ok=True)

  # Create error Markdown file
  error_md_path = os.path.join(error_dir, "Error.md")
  with open(error_md_path, 'w', encoding='utf-8') as f:
    f.write("# Error\n\n")
    f.write(f"Scrape failed: {url}\n\n")
    f.write(f"Reason: {error_msg}\n\n")
    f.write("Suggestion: Check network connection or retry later\n")

  # Create empty reference file
  reference_dir = os.path.join(error_dir, 'reference')
  os.makedirs(reference_dir, exist_ok=True)
  jsonl_path = os.path.join(reference_dir, 'references.jsonl')

  with open(jsonl_path, 'w', encoding='utf-8') as f:
    # Create empty file
    pass

  return error_md_path, jsonl_path


def parse_references_block(markdown_text: str) -> List[str]:
  """Locate References and Bibliography sections and return their raw lines.

  Supports the following structures:
  1. References\n----------\n### Citations\n<entries>
  2. # References / ## References format
  3. Bibliography / ## Bibliography format
  4. Untitled reference list at end of page (Tell es-Sakan pattern)
  5. Numbered reference list (1. ^ Jump up to: ... pattern)
  """
  lines = markdown_text.splitlines()
  n = len(lines)
  ref_start_idx: Optional[int] = None
  bib_start_idx: Optional[int] = None
  citations_anchor_idx: Optional[int] = None
  numbered_refs_start: Optional[int] = None

  for i, ln in enumerate(lines):
    stripped = ln.strip()
    low = stripped.lower()

    # Find References section (including Chinese "参考文献")
    if low in {"references", "参考文献"}:
      ref_start_idx = i
      if i + 1 < n and re.fullmatch(r'-{3,}', lines[i + 1].strip()):
        pass
    # Markdown heading format - References
    if low.startswith('#') and 'references' in low:
      ref_start_idx = i

    # Find Bibliography section (including Chinese "书目", "参考书目")
    if low in {"bibliography", "书目", "参考书目"}:
      bib_start_idx = i
      if i + 1 < n and re.fullmatch(r'-{3,}', lines[i + 1].strip()):
        pass
    # Markdown heading format - Bibliography
    if low.startswith('#') and 'bibliography' in low:
      bib_start_idx = i

    if '### citations' in low:
      citations_anchor_idx = i
      if ref_start_idx is None:
        ref_start_idx = i

    # Find numbered reference list start (e.g. "1. ^ Jump up to:" or "1. **^**")
    if numbered_refs_start is None and re.match(
      r'^\s*1\.\s*[\^\*]*\s*(Jump up|[\*\^])', stripped, re.IGNORECASE
    ):
      numbered_refs_start = i

  # Collect all relevant content
  collected: List[str] = []

  # Process References section
  if ref_start_idx is not None:
    start_collect = (citations_anchor_idx + 1) if citations_anchor_idx is not None else (ref_start_idx + 1)
    end_collect = min(
      x
      for x in [bib_start_idx, numbered_refs_start, n]
      if x is not None and x > ref_start_idx
    )

    for j in range(start_collect, end_collect):
      if j >= n:
        break
      ln = lines[j]
      stripped = ln.strip()
      if re.match(r'^#{1,3} ', stripped):
        low = stripped.lower()
        if not ('reference' in low or 'citation' in low or 'bibliography' in low):
          break
      if stripped.lower().startswith(('external links', 'see also', 'notes')):
        if 'bibliography' not in stripped.lower():
          break
      collected.append(ln)

  # Process Bibliography section
  if bib_start_idx is not None:
    start_collect = bib_start_idx + 1
    end_collect = (
      numbered_refs_start if numbered_refs_start and numbered_refs_start > bib_start_idx else n
    )

    for j in range(start_collect, end_collect):
      if j >= n:
        break
      ln = lines[j]
      stripped = ln.strip()
      if re.match(r'^#{1,3} ', stripped):
        low = stripped.lower()
        if not ('bibliography' in low or 'reference' in low or 'citation' in low):
          break
      if stripped.lower().startswith(('external links', 'see also', 'notes')):
        break
      collected.append(ln)

  # Process numbered reference list
  if numbered_refs_start is not None:
    for j in range(numbered_refs_start, n):
      ln = lines[j]
      stripped = ln.strip()

      # If it's a numbered reference format, continue collecting
      if re.match(r'^\s*\d+\.\s*[\^\*]*\s*(Jump up|\*)', stripped, re.IGNORECASE):
        collected.append(ln)
      # If it's an empty line, skip
      elif not stripped:
        collected.append(ln)
      # If a new section heading is encountered, stop
      elif re.match(r'^#{1,3} ', stripped):
        break
      # If obviously other content is encountered, stop
      elif stripped.lower().startswith(('external links', 'see also', 'notes')):
        break
      # Otherwise consider it a continuation of the reference
      else:
        collected.append(ln)

  # If no formal References/Bibliography section found and no numbered references, look for end-of-page reference list
  if ref_start_idx is None and bib_start_idx is None and numbered_refs_start is None:
    print("🔍 No standard reference format found, trying fallback strategy...")
    # Original end-of-page reference detection logic
    ref_lines_from_end = []
    for i in range(n - 1, -1, -1):
      line = lines[i].strip()
      if not line:
        continue
      if line.startswith('*') and '[' in line and '](' in line:
        ref_lines_from_end.insert(0, lines[i])
      elif 'wikimedia' in line.lower() and '[' in line:
        ref_lines_from_end.insert(0, lines[i])
      elif line.startswith('#') or 'edit section' in line.lower():
        break
      elif (
        not line.startswith('*')
        and '[' not in line
        and len(line) > 20
      ):
        break

    if ref_lines_from_end:
      print(f"📋 Fallback strategy found {len(ref_lines_from_end)} possible reference lines")
      collected.extend(ref_lines_from_end)

    print(f"📊 Parse result: collected {len(collected)} reference lines")

  return collected


def group_reference_entries(ref_lines: List[str]) -> List[str]:
  """Group reference block by empty lines or numbered/list starts, output each reference text."""
  entries: List[str] = []
  buffer: List[str] = []

  def flush():
    if buffer:
      # Merge and compress whitespace
      merged = ' '.join(l.strip() for l in buffer if l.strip())
      if merged:
        entries.append(merged)
      buffer.clear()

  for ln in ref_lines:
    stripped = ln.strip()
    if not stripped:
      flush()
      continue
    # Typical numbered formats [1] or 1. or - prefix, start new entry
    if re.match(r'^\s*(\[[0-9]+\]|[0-9]+[.)]|[-*])\s+', ln):
      flush()
      buffer.append(stripped)
    else:
      buffer.append(stripped)
  flush()
  return entries


def extract_date(text: str) -> Optional[str]:
  # Prioritize matching Retrieved format (including trailing period)
  m = RETRIEVED_DATE_REGEX.search(text)
  if m:
    phrase = m.group(1)
    # Ensure ends with period
    return phrase if phrase.endswith('.') else phrase + '.'
  # ISO
  m = DATE_ISO_REGEX.search(text)
  if m:
    return m.group(1)
  # YYYY-MM-DD or with Chinese separators
  m = DATE_YMD_REGEX.search(text)
  if m:
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
  # Month name (publication date) return directly
  m = MONTH_NAME_REGEX.search(text)
  if m:
    return m.group(0)
  return None


def build_reference_items(ref_entries: List[str]) -> List[Dict[str, Any]]:
  """Parse reference entries -> structured fields.

  Rule summary (from ref_cases.md):

   1. Remove all "Jump up" / a b c footnote anchor links (including cite_ref / cite_note / caret ^ forms).
   2. If the first "content" link is an in-page anchor (#CITEREF / #cite_ref) and the entire entry has no external links or Archive links,
      discard the entire entry (pure in-page bibliography/Works cited reference).
   3. title = first *content* link text (strip quotes / leading & trailing whitespace).
       - If the link URL is still wikipedia.org and an [Archived](archive_url) link exists, title is still taken from the first link, but final url uses the Archived link.
   4. If [Archived](archive_url) exists, keep archive_url field, but final scraping uses the original non-Archived link (url field); no longer replace with archive address.
   5. author:
       - If (Month Day, Year) publication date exists: non-empty text before the date (after removing jump anchors and extra punctuation).
       - If no publication date: if short text (<= 12 words) ending with period appears before the first title link, treat as author. e.g., "United States Congress.".
   6. sources: collect all non-'Archived' link texts after the title; includes media names, publications, ISSN/ISBN and their number links; maintain dedupe order.
      If only 1 source, still put in a list.
   7. Filtering:
       - No scrapable url (no external http(s) links and no archive links) => discard (e.g., pure bibliography: _Promises to Keep: ..._ with no links).
       - Only internal #CITEREF/#cite_ref links => discard.
   8. Normalization:
       - Strip leading/trailing Chinese/English quotes and emphasis symbols _ * from title.
       - Remove duplicate whitespace.
   9. Dates:
       - publish_date: first (Month Day, Year) pattern.
       - retrieved_date: 'Retrieved Month Day, Year'.
  10. Still keep is_external: whether the final url used is external (non-wikipedia.org).
  11. When author is empty, if sources exist, backfill author with the first source (organization as author).
  """
  items: List[Dict[str, Any]] = []
  month_names = (
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
  )
  publish_date_pat = re.compile(r'\((' + month_names + r')\s+\d{1,2},\s+\d{4}\)')
  retrieved_pat = re.compile(
    r'Retrieved\s+' + month_names + r'\s+\d{1,2},\s+\d{4}\.?',
    re.IGNORECASE
  )

  def is_jump_token(text: str, url: str) -> bool:
    t = text.lower().strip('_* \'"')
    if 'jump' in t:  # Jump up / Jump up to
      return True
    # cite_ref anchors + single-letter label (a,b,c, etc.)
    if 'cite_ref' in url and re.fullmatch(r'[a-z]', t):
      return True
    # caret marker
    if t in {'^'}:
      return True
    return False

  for raw_entry in ref_entries:
    entry = raw_entry.strip()
    if not entry:
      continue
    # Remove line number prefixes "1.", "[1]", "-" etc.
    entry = re.sub(r'^\s*(\[[0-9]+\]|[0-9]+[.)]|[-*])\s+', '', entry)
    # Remove leading Jump up phrase
    entry = JUMP_PHRASE_PATTERN.sub('', entry)
    # Remove caret jump components **[^](...)**
    entry = JUMP_CARET_PATTERN.sub(' ', entry)
    # Remove cite_ref anchor link collections (a,b,c...)
    entry = JUMP_INLINE_ANCHORS_PATTERN.sub(' ', entry)
    entry = re.sub(r'\s+', ' ', entry).strip()

    md_links = MD_LINK_REGEX.findall(entry)
    if not md_links:
      # No links at all => possibly a book (no url) -> discard
      continue

    # Filter out pure footnote/jump/single letter anchors
    def is_pure_anchor(txt: str, url: str) -> bool:
      t = txt.strip().lower().strip('_*"')
      if not url:
        return True
      if 'cite_ref' in url or 'cite_note' in url:
        return True
      if re.match(r'^[a-z]$', t):
        return True
      if 'jump' in t:
        return True
      return False

    content_links = [(t, u) for (t, u) in md_links if not is_pure_anchor(t, u)]
    if not content_links:
      continue

    # Identify publication date & Retrieved
    m_pub = publish_date_pat.search(entry)
    m_ret = retrieved_pat.search(entry)
    publish_date = m_pub.group(0).strip('()') if m_pub else ''
    retrieved_date = m_ret.group(0) if m_ret else ''
    if retrieved_date and not retrieved_date.endswith('.'):
      retrieved_date += '.'

    def clean_name(s: str) -> str:
      s = re.sub(r'\s+', ' ', s).strip()
      # Remove leading symbols ^ * _ and extra punctuation
      s = re.sub(r'^[\^*_\s]+', '', s)
      s = s.strip()
      # Avoid producing empty string
      return s

    # Author candidate region
    author_segment = ''
    if m_pub:
      author_segment = entry[:m_pub.start()].strip()
    else:
      # No publication date: truncate to first external (non-wikipedia) link or before first quoted title link
      first_ext_idx = None
      for t, u in content_links:
        if 'wikipedia.org' not in u:
          # Position based on full text search
          pos = entry.find('[' + t + '](')  
          if pos != -1:
            first_ext_idx = pos
            break
      if first_ext_idx is not None:
        author_segment = entry[:first_ext_idx].strip()
      else:
        # If no external link, use text before the first link as author
        first_link_text = content_links[0][0]
        pos = entry.find('[' + first_link_text + '](')
        author_segment = entry[:pos].strip() if pos != -1 else ''

    # Clear remaining jump/anchor links in author segment
    if author_segment:
      author_segment = JUMP_CARET_PATTERN.sub(' ', author_segment)
      author_segment = JUMP_INLINE_ANCHORS_PATTERN.sub(' ', author_segment)
      author_segment = re.sub(
        r'(\[[^\]]+\]\(https?://[^)]+\))',
        lambda m: re.sub(r'^\[|\]\([^)]*\)$', '', m.group(0)),
        author_segment
      )
      author_segment = re.sub(
        r'\[[^\]]+\]\(https?://[^)]+\)',
        lambda m: re.sub(r'^\[|\]\([^)]*\)$', '', m.group(0)),
        author_segment
      )
      author_segment = re.sub(r'\s+', ' ', author_segment).strip()
      # Remove trailing period
      if author_segment.endswith('.'):
        author_segment = author_segment[:-1].strip()

    author = author_segment
    # If author is too long (> 15 words), treat as noise and discard
    if author and len(author.split()) > 15:
      author = ''

    # Determine title link: priority strategy
    title_link = None
    # 1. Among links after publication date, prefer text wrapped in quotes and is external link
    after_pub_pos = m_pub.end() if m_pub else (len(author_segment) if author_segment else 0)
    for t, u in content_links:
      pos = entry.find('[' + t + '](')
      if pos < after_pub_pos:
        continue
      txt = t.strip()
      if ('wikipedia.org' not in u) and re.match(r'^".*"$|^".*"$', txt):
        title_link = (t, u)
        break
    # 2. External link (after date)
    if not title_link:
      for t, u in content_links:
        pos = entry.find('[' + t + '](')
        if pos < after_pub_pos:
          continue
        if 'wikipedia.org' not in u:
          title_link = (t, u)
          break
    # 3. Any (after date)
    if not title_link:
      for t, u in content_links:
        pos = entry.find('[' + t + '](')
        if pos >= after_pub_pos:
          title_link = (t, u)
          break
    # 4. Fallback: first link not in author segment
    if not title_link:
      for t, u in content_links:
        pos = entry.find('[' + t + '](')
        if not author_segment or pos >= len(author_segment):
          title_link = (t, u)
          break
    if not title_link:
      continue
    title_text, title_url = title_link

    # Find Archived link (record but don't use as main url)
    archive_url = None
    for t, u in content_links:
      if t.lower() == 'archived':
        archive_url = u
        break

    # Extract source: first non-Archived, non-same URL link text after title link
    title_pos = entry.find('[' + title_text + '](')
    source = ''
    for t, u in content_links:
      if t == title_text and u == title_url:
        continue
      pos = entry.find('[' + t + '](')
      if pos < title_pos:
        continue
      if t.lower() == 'archived':
        continue
      if t.strip() == title_text.strip():
        continue
      source = t.strip('_* ')
      break

    # If source not found or same as author, try to parse plain text media name after title
    if not source or source == author:
      link_markdown = f'[{title_text}]({title_url})'
      link_end = entry.find(link_markdown)
      if link_end != -1:
        link_end += len(link_markdown)
        tail = entry[link_end:]
        # Truncate to before Archived / Retrieved
        cut_idx = len(tail)
        for kw in ['Archived', 'Retrieved']:
          kpos = tail.find(kw)
          if kpos != -1 and kpos < cut_idx:
            cut_idx = kpos
        tail_section = tail[:cut_idx]
        # Italic media _..._
        m_italic = re.search(r'_(\s*[^_]{2,}?)_', tail_section)
        candidate = ''
        if m_italic:
          candidate = m_italic.group(1).strip()
        else:
          # First capitalized/uppercase word phrase ending with period (NPR. / Associated Press.)
          m_acro = re.match(
            r'\s*([A-Z][A-Za-z&\.]*?(?:\s+[A-Z][A-Za-z&\.]*?){0,4})\.(?:\s|$)',
            tail_section
          )
          if m_acro:
            cand = m_acro.group(1).strip()
            # Filter out Retrieved / Archived misjudgment
            if cand.lower() not in {'retrieved', 'archived'}:
              candidate = cand
        if not candidate:
          # Capture like "NPR." appearing after title link
          m_npr = re.search(r'\b(NPR)\.(?:\s|$)', tail_section)
          if m_npr:
            candidate = m_npr.group(1)
        if candidate and candidate != author:
          source = candidate.strip('_* ')

    # Filter internal works cited: if title is still wikipedia link and no external links
    if 'wikipedia.org' in title_url and not any(
      'wikipedia.org' not in u for _, u in content_links
    ):
      continue

    # Clean title
    title = title_text.strip().strip('"""').strip('_* ')
    title = re.sub(r'\s+', ' ', title)

    # Author/source mutual complement
    if not author and source:
      author = source
    if not source and author:
      source = author

    author = clean_name(author)
    source = clean_name(source)
    # If still empty after cleaning and mutual complement exists
    if not author and source:
      author = source
    if not source and author:
      source = author

    is_external = ('wikipedia.org' not in title_url)
    item: Dict[str, Any] = {
      'title': title,
      'url': title_url,
      'is_external': is_external,
    }
    if author:
      item['author'] = author
    if source:
      item['source'] = source
    if archive_url:
      item['archive_url'] = archive_url
    if publish_date:
      item['publish_date'] = publish_date
    if retrieved_date:
      item['retrieved_date'] = retrieved_date
    # scraped flag is added when writing out
    items.append(item)

  return items


def write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w', encoding='utf-8') as f:
    for it in items:
      # Initialize scraped mark
      if 'scraped' not in it:
        it['scraped'] = False
      f.write(json.dumps(it, ensure_ascii=False) + '\n')


def main():
  parser = argparse.ArgumentParser(description='Scrape webpage and extract references to generate JSONL')
  parser.add_argument('--url', required=True, help='Target webpage URL (base URL)')
  parser.add_argument('--output_dir', required=True, help='Output root directory')
  parser.add_argument('--api-key', dest='api_key', default=None, help='Jina API Key (optional)')
  args = parser.parse_args()

  print(f"🌐 Starting to process URL: {args.url}")
  print(f"📁 Output directory: {args.output_dir}")
  print(f"🔄 Retry count: {MAX_RETRIES}")

  api_key = args.api_key or os.environ.get('JINA_API_KEY') or DEFAULT_API_KEY
  if not api_key.startswith('Bearer '):
    api_key = f'Bearer {api_key}'

  scraper = WebScrapingJinaTool(api_key)
  # Use URL to derive a "stable title" to ensure directory naming matches run_ref_scraper
  url_title = url_to_slug_title(args.url)

  # Use scraping with retries
  data = scrape_with_retry(scraper, args.url)

  if not data or not data.get('content'):
    print("❌ Scrape failed, unable to get page content")

    # Create error placeholder
    error_md_path, jsonl_path = create_error_placeholder(
      args.url,
      args.output_dir,
      "Network timeout or service unavailable"
    )

    print(f"📄 Error page: {error_md_path}")
    print(f"📋 Empty reference file: {jsonl_path}")
    return

  # Save Markdown
  try:
    md_path = save_markdown(data, args.output_dir, slug=url_title)
    print(f"📄 Markdown saved: {md_path}")
  except Exception as e:
    print(f"❌ Failed to save Markdown: {e}")
    return

  # Read markdown content
  try:
    with open(md_path, 'r', encoding='utf-8') as f:
      markdown_text = f.read()
  except Exception as e:
    print(f"❌ Failed to read Markdown file: {e}")
    return

  reference_dir = os.path.join(os.path.dirname(md_path), 'reference')
  os.makedirs(reference_dir, exist_ok=True)
  jsonl_path = os.path.join(reference_dir, 'references.jsonl')

  if os.path.exists(jsonl_path):
    print(f'📋 Reference file already exists, skipping: {jsonl_path}')
    return

  print("🔍 Starting to parse references...")
  ref_lines = parse_references_block(markdown_text)

  if not ref_lines:
    print("⚠️  No reference content found, creating empty file")
    write_jsonl([], jsonl_path)
    print(f"📋 Empty reference file: {jsonl_path}")
    return

  ref_entries = group_reference_entries(ref_lines)
  print(f"📝 Got {len(ref_entries)} reference entries after grouping")

  if not ref_entries:
    print("⚠️  Reference entries are empty, creating empty file")
    write_jsonl([], jsonl_path)
    print(f"📋 Empty reference file: {jsonl_path}")
    return

  items = build_reference_items(ref_entries)
  print(f"🔗 Parsed {len(items)} valid references")

  # Deduplication: keep only one copy of completely identical entries
  if items:
    seen = set()
    deduped = []
    for it in items:
      key = json.dumps(it, sort_keys=True, ensure_ascii=False)
      if key in seen:
        continue
      seen.add(key)
      deduped.append(it)
    removed = len(items) - len(deduped)
    if removed > 0:
      print(f'🔄 Deduplication: removed {removed} duplicate references (original {len(items)} -> kept {len(deduped)})')
    items = deduped

  write_jsonl(items, jsonl_path)

  print("✅ Processing complete!")
  print(f"📄 Markdown: {md_path}")
  print(f"📋 Reference file: {jsonl_path} ({len(items)} entries)")

  if not items:
    print("⚠️  Warning: No reference content parsed")
    print("💡 Suggestion: Check if page contains References section or adjust parsing strategy")


if __name__ == '__main__':
  main()
