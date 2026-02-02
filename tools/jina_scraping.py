import logging
import os
import re
import argparse
from datetime import datetime
import requests
from typing import Dict, Any

# Set up basic logging
logger = logging.getLogger('WebScrapingJinaTool')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Get API key from environment variable JINA_API_KEY
# Sign up at https://jina.ai/reader to get your API key
DEFAULT_API_KEY = os.environ.get("JINA_API_KEY", "YOUR_JINA_API_KEY_HERE")


class WebScrapingJinaTool:
    def __init__(self, api_key: str, timeout: int = 180):
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                # "Accept": "application/json",
                "Authorization": self.api_key,
                "X-Timeout": "120000",
                "X-With-Generated-Alt": "true",
                "X-Return-Format": "markdown"
            }
            response = requests.get(jina_url, headers=headers, timeout=self.timeout)

            if response.status_code != 200:
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

            response_dict = response.json()

            return {
                'url': response_dict['data'].get('url', url),
                'title': response_dict['data'].get('title', 'Untitled'),
                'description': response_dict['data'].get('description', ''),
                'content': response_dict['data'].get('content', ''),
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            logger.error(str(e))
            return {
                'url': url,
                'title': 'Error',
                'description': '',
                'content': '',
                'publish_time': 'unknown',
                'error': str(e)
            }


def slugify(text: str, max_length: int = 80) -> str:
    """Convert text to a filesystem-safe slug (保留原有大小写)."""
    text = text.strip()
    # 压缩多余空白为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去掉下划线（可根据需要保留）
    text = text.replace('_', ' ')
    # 仅允许 字母/数字/空格/连字符
    text = re.sub(r'[^A-Za-z0-9\- ]+', '', text)
    # 压缩多余空格（再次，避免非法清理后留下的双空格）
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip(' -')
    if not text:
        text = 'page'
    if len(text) > max_length:
        text = text[:max_length].rstrip('-')
    return text


def save_markdown(data: Dict[str, Any], output_dir: str, slug: str | None = None) -> str:
    """Save scraped data as a markdown file inside a title-named folder; create reference subfolder.

    Structure:
        output_dir/
            <TitleSlug>/
                <TitleSlug>.md
                reference/
    Returns markdown file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 如果调用方给了 slug（比如从 URL 推出来），就优先用它；
    # 否则维持原逻辑，用抓取结果里的标题。
    if slug is not None:
        base_name = slugify(slug)
    else:
        base_name = slugify(data.get('title') or 'page')

    # Create page directory; avoid collision by suffix counter
    page_dir = os.path.join(output_dir, base_name)
    counter = 1
    original_page_dir = page_dir
    while os.path.exists(page_dir) and not os.path.isdir(page_dir):  # name occupied by file
        page_dir = f"{original_page_dir}-{counter}"
        counter += 1

    # If directory exists we still reuse it (append new file only if file would clash)
    if not os.path.exists(page_dir):
        os.makedirs(page_dir, exist_ok=True)

    # Ensure reference subfolder
    reference_dir = os.path.join(page_dir, 'reference')
    os.makedirs(reference_dir, exist_ok=True)

    filename = f"{base_name}.md"
    path = os.path.join(page_dir, filename)
    if os.path.exists(path):
        logger.info(f"Markdown 已存在，直接复用: {path}")
        return path

    lines = []
    # Optional front matter style metadata
    lines.append('---')
    lines.append("title: " + data.get('title', 'Untitled').replace('\n', ' '))
    lines.append(f"source_url: {data.get('url','')}")
    lines.append(f"publish_time: {data.get('publish_time','unknown')}")
    if 'error' in data:
        lines.append(f"error: {data['error']}")
    lines.append('---\n')
    lines.append(f"# {data.get('title','Untitled')}\n")
    description = data.get('description')
    if description:
        lines.append(f"> {description.strip()}\n")
    content = data.get('content', '').rstrip() + '\n'
    lines.append(content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return path


def main():
    parser = argparse.ArgumentParser(
        description='Scrape a webpage via Jina Reader and save as Markdown.'
    )
    parser.add_argument(
        '--url',
        default='https://en.wikipedia.org/wiki/Persecution_of_Muslims',
        help='目标网页 URL'
    )
    parser.add_argument(
        '--output_dir',
        default='./output',
        help='Markdown 输出目录'
    )
    parser.add_argument(
        '--api-key',
        dest='api_key',
        default=None,
        help='可选：显式指定 Jina API Key (会覆盖环境变量 JINA_API_KEY)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1200,
        help='超时时间（秒）'
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('JINA_API_KEY') or DEFAULT_API_KEY
    if not api_key.startswith('Bearer '):  # allow providing raw token
        api_key = f'Bearer {api_key}'

    scraper = WebScrapingJinaTool(api_key=api_key, timeout=args.timeout)
    logger.info(f"Scraping {args.url} ...")
    data = scraper(args.url)

    output_path = save_markdown(data, args.output_dir)
    if 'error' in data:
        logger.warning(f"Saved (with error) to {output_path}: {data['error']}")
    else:
        logger.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
