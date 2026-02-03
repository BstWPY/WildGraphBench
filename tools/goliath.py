import json
import time
import requests
import logging
import base64
import random
import os
import re
from typing import Dict, Any, Optional

# Default configuration - users should set their own API URL
# You can use Jina AI Reader API (https://jina.ai/reader) or similar services
DEFAULT_SPIDER_API_URL = os.environ.get("SPIDER_API_URL", "YOUR_SPIDER_API_URL_HERE")
DEFAULT_SPIDER_TIMEOUT = 120
DEFAULT_MAX_RETRY = 2

# Setup logging
logger = logging.getLogger('SpiderTool')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SpiderTool:
    """Web scraping tool based on spider-api-gateway"""

    def __init__(
        self,
        api_url: str = DEFAULT_SPIDER_API_URL,
        timeout: int = DEFAULT_SPIDER_TIMEOUT,
        max_retry: int = DEFAULT_MAX_RETRY,
        enable_cache: bool = True,
        enable_oversea: bool = True,
        debug: bool = False,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self.max_retry = max_retry
        self.enable_cache = enable_cache
        self.enable_oversea = enable_oversea
        self.debug = debug

    def retrieve(
        self,
        url: str,
        content: str = "string",  # Default value, can be customized
    ) -> Dict[str, Any]:
        """
        Scrape and parse web page content

        Args:
            url: URL to scrape
            content: Content parameter (adjust according to API documentation)

        Returns:
            Dictionary containing scrape results
        """
        for attempt in range(self.max_retry):
            request_id = (
                f"spider_retrieve_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            )

            # Build payload based on curl command
            payload = {
                "content": content,
                "enable_cache": self.enable_cache,
                "enable_oversea": self.enable_oversea,
                "url": url,
            }

            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
            }

            try:
                logger.info(f"Scraping: {url}")
                if self.debug:
                    logger.debug(
                        f"Request payload: {json.dumps(payload, ensure_ascii=False, indent=2)}"
                    )
                    logger.debug(
                        f"Request headers: {json.dumps(headers, ensure_ascii=False, indent=2)}"
                    )

                response = requests.post(
                    self.api_url,
                    json=payload,  # Use json parameter instead of data
                    headers=headers,
                    timeout=self.timeout,
                )

                if self.debug:
                    logger.debug(f"HTTP status code: {response.status_code}")
                    logger.debug(f"Response headers: {dict(response.headers)}")

                return self._handle_response(response, url, request_id)

            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout (attempt {attempt + 1}/{self.max_retry}): {e}")
                if attempt < self.max_retry - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                continue

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error (attempt {attempt + 1}/{self.max_retry}): {e}")
                if attempt < self.max_retry - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(
                    f"Other error (attempt {attempt + 1}/{self.max_retry}): {type(e).__name__}: {e}"
                )
                if attempt < self.max_retry - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                continue

        return {
            "success": False,
            "error": f"All {self.max_retry} retry attempts failed",
            "url": url,
        }

    def _handle_response(
        self,
        response: requests.Response,
        url: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """Handle response with detailed debug output"""
        try:
            # First check HTTP status code
            if response.status_code != 200:
                error_msg = f"HTTP error: {response.status_code} - {response.reason}"
                if self.debug:
                    logger.error(f"Response content: {response.text}")
                return {
                    "success": False,
                    "error": error_msg,
                    "http_status": response.status_code,
                    "response_text": response.text,
                    "url": url,
                    "request_id": request_id,
                }

            # Try to parse JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing failed: {e}"
                logger.error(f"{error_msg}, raw response: {response.text[:1000]}")
                return {
                    "success": False,
                    "error": error_msg,
                    "response_text": response.text,
                    "url": url,
                    "request_id": request_id,
                }

            # Print full response for debugging
            if self.debug:
                logger.debug(
                    f"Full API response: {json.dumps(response_data, ensure_ascii=False, indent=2)}"
                )

            # Analyze response structure
            response_keys = list(response_data.keys())
            logger.info(f"Response contains fields: {response_keys}")

            # Check if response is successful (adjust based on actual API response format)
            success_indicators = [
                response_data.get("success") is True,
                response_data.get("status") == "success",
                response_data.get("code") == 200,
                "data" in response_data,
                "content" in response_data,
                "result" in response_data,
            ]

            if any(success_indicators):
                # Extract content (adjust field names based on actual API response format)
                content = ""
                title = ""
                description = ""

                # Try different field names
                if "data" in response_data:
                    data_field = response_data["data"]
                    if isinstance(data_field, dict):
                        content = (
                            data_field.get("content", "")
                            or data_field.get("text", "")
                            or data_field.get("markdown", "")
                        )
                        title = data_field.get("title", "")
                        description = data_field.get("description", "")
                    elif isinstance(data_field, str):
                        content = data_field

                elif "content" in response_data:
                    content = response_data["content"]
                    title = response_data.get("title", "")
                    description = response_data.get("description", "")

                elif "result" in response_data:
                    result_field = response_data["result"]
                    if isinstance(result_field, dict):
                        content = result_field.get("content", "") or result_field.get(
                            "text", ""
                        )
                        title = result_field.get("title", "")
                        description = result_field.get("description", "")
                    elif isinstance(result_field, str):
                        content = result_field

                # If still no content, try to extract directly from response
                if not content:
                    for key in ["text", "markdown", "html"]:
                        if key in response_data and response_data[key]:
                            content = response_data[key]
                            break

                logger.info(f"✅ Successfully extracted content, length: {len(content)} chars")

                return {
                    "success": True,
                    "result": {
                        "content": content,
                        "title": title,
                        "description": description,
                        "url": url,
                    },
                    "request_id": request_id,
                    "raw_response_keys": response_keys,
                    "url": url,
                }

            # Failure case
            error_details = []

            # Check common error fields
            if "error" in response_data:
                error_details.append(f"API error: {response_data['error']}")
            if "message" in response_data:
                error_details.append(f"Message: {response_data['message']}")
            if "status" in response_data:
                error_details.append(f"Status: {response_data['status']}")
            if "code" in response_data:
                error_details.append(f"Error code: {response_data['code']}")

            # Combine error messages
            if error_details:
                error_msg = "API returned failure status: " + " | ".join(error_details)
            else:
                error_msg = f"Unknown API response format, response fields: {response_keys}"

            # If response is small, include full content
            if len(str(response_data)) < 2000:
                error_msg += f" | Full response: {json.dumps(response_data, ensure_ascii=False)}"

            logger.error(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "raw_response": response_data,
                "response_keys": response_keys,
                "url": url,
                "request_id": request_id,
            }

        except Exception as e:
            error_msg = f"Error parsing response: {type(e).__name__}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "raw_text": response.text,
                "url": url,
                "request_id": request_id,
            }

    def __call__(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Simplified interface call method

        Args:
            url: URL to scrape
            **kwargs: Additional parameters

        Returns:
            Scraping result
        """
        try:
            response_dict = self.retrieve(url, **kwargs)
            if response_dict.get("success"):
                result = response_dict.get("result", {})
                return {
                    'success': True,
                    'url': result.get("url", url),
                    'title': result.get("title", ""),
                    'description': result.get("description", ""),
                    'content': result.get("content", ""),
                    'request_id': response_dict.get("request_id", "")
                }
            else:
                return {
                    'success': False,
                    'url': url,
                    'title': '',
                    'content': '',
                    'error': response_dict.get("error", "Unknown error"),
                    'request_id': response_dict.get("request_id", "")
                }
        except Exception as e:
            logger.error(f"Scrape failed: {e}")
            return {
                'success': False,
                'url': url,
                'title': '',
                'content': '',
                'error': f"Call exception: {type(e).__name__}: {e}"
            }


def build_default_spider_tool(debug: bool = False) -> SpiderTool:
    """Provide a reusable default instance"""
    return SpiderTool(
        api_url=DEFAULT_SPIDER_API_URL,
        timeout=DEFAULT_SPIDER_TIMEOUT,
        max_retry=DEFAULT_MAX_RETRY,
        enable_cache=True,
        enable_oversea=True,
        debug=debug,
    )


def test_spider_api():
    """Test the new spider API"""
    test_urls = [
        "https://en.wikipedia.org/wiki/ChatGPT",
    ]

    print("🚀 Testing Spider API...")

    # Create tool instance
    tool = SpiderTool(debug=True)

    success_count = 0
    total_count = len(test_urls)

    for i, url in enumerate(test_urls, 1):
        print("\n" + "=" * 80)
        print(f"📋 Test {i}/{total_count}: {url}")
        print("=" * 80)

        start_time = time.time()
        result = tool(url)
        end_time = time.time()

        if result.get("success"):
            content_len = len(result.get("content", ""))
            print("✅ Success!")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Content length: {content_len} chars")
            print(f"   Time: {end_time - start_time:.2f}s")
            print(f"   Content preview: {result.get('content', '')[:200]}...")
            success_count += 1
        else:
            error = result.get("error", "Unknown error")
            print(f"❌ Failed: {error}")
            print(f"   Time: {end_time - start_time:.2f}s")

        # Add delay between URLs
        if i < total_count:
            print("⏳ Waiting 2 seconds...")
            time.sleep(2)

    print(f"\n🎉 Testing complete! Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")


# Maintain compatibility: provide aliases so old code still works
GoliathTool = SpiderTool
build_default_goliath_tool = build_default_spider_tool


if __name__ == "__main__":
    print("=== Testing new Spider API web scraping functionality ===")

    # Directly test single URL
    tool = build_default_spider_tool(debug=True)
    result = tool("https://www.bbc.co.uk/pressoffice/pressreleases/stories/2008/03_march/07/ob.shtml")

    print("\n📊 Single URL test result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Save result to file
    if result.get("success"):
        output_dir = os.environ.get("SPIDER_OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)

        title = result.get("title", "untitled").replace("/", "_").replace("\\", "_")
        content = result.get("content", "")

        filename = f"spider_test_{title}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {result.get('title', 'Untitled')}\n\n")
            f.write(f"**URL**: {result.get('url', '')}\n\n")
            f.write(f"**Description**: {result.get('description', '')}\n\n")
            f.write("---\n\n")
            f.write(content)

        print(f"✅ Content saved to: {filepath}")

    print("\n" + "=" * 80)
    test_spider_api()
