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

# 设置日志
logger = logging.getLogger('SpiderTool')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SpiderTool:
    """基于spider-api-gateway的网页爬取工具"""

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
        content: str = "string",  # 默认值，可以自定义
    ) -> Dict[str, Any]:
        """
        爬取和解析网页内容

        Args:
            url: 要爬取的网址
            content: 内容参数（根据API文档调整）

        Returns:
            包含爬取结果的字典
        """
        for attempt in range(self.max_retry):
            request_id = (
                f"spider_retrieve_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            )

            # 根据curl命令构建payload
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
                logger.info(f"正在爬取: {url}")
                if self.debug:
                    logger.debug(
                        f"请求载荷: {json.dumps(payload, ensure_ascii=False, indent=2)}"
                    )
                    logger.debug(
                        f"请求头: {json.dumps(headers, ensure_ascii=False, indent=2)}"
                    )

                response = requests.post(
                    self.api_url,
                    json=payload,  # 使用json参数而不是data
                    headers=headers,
                    timeout=self.timeout,
                )

                if self.debug:
                    logger.debug(f"HTTP状态码: {response.status_code}")
                    logger.debug(f"响应头: {dict(response.headers)}")

                return self._handle_response(response, url, request_id)

            except requests.exceptions.Timeout as e:
                logger.error(f"请求超时 (尝试 {attempt + 1}/{self.max_retry}): {e}")
                if attempt < self.max_retry - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)  # 指数退避
                    logger.info(f"等待 {wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)
                continue

            except requests.exceptions.ConnectionError as e:
                logger.error(f"连接错误 (尝试 {attempt + 1}/{self.max_retry}): {e}")
                if attempt < self.max_retry - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.info(f"等待 {wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(
                    f"其他错误 (尝试 {attempt + 1}/{self.max_retry}): {type(e).__name__}: {e}"
                )
                if attempt < self.max_retry - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.info(f"等待 {wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)
                continue

        return {
            "success": False,
            "error": f"所有 {self.max_retry} 次重试尝试均失败",
            "url": url,
        }

    def _handle_response(
        self,
        response: requests.Response,
        url: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """处理响应，输出详细的调试信息"""
        try:
            # 首先检查HTTP状态码
            if response.status_code != 200:
                error_msg = f"HTTP错误: {response.status_code} - {response.reason}"
                if self.debug:
                    logger.error(f"响应内容: {response.text}")
                return {
                    "success": False,
                    "error": error_msg,
                    "http_status": response.status_code,
                    "response_text": response.text,
                    "url": url,
                    "request_id": request_id,
                }

            # 尝试解析JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析失败: {e}"
                logger.error(f"{error_msg}, 原始响应: {response.text[:1000]}")
                return {
                    "success": False,
                    "error": error_msg,
                    "response_text": response.text,
                    "url": url,
                    "request_id": request_id,
                }

            # 打印完整响应用于调试
            if self.debug:
                logger.debug(
                    f"完整API响应: {json.dumps(response_data, ensure_ascii=False, indent=2)}"
                )

            # 分析响应结构
            response_keys = list(response_data.keys())
            logger.info(f"响应包含字段: {response_keys}")

            # 检查响应是否成功（根据实际API响应格式调整）
            success_indicators = [
                response_data.get("success") is True,
                response_data.get("status") == "success",
                response_data.get("code") == 200,
                "data" in response_data,
                "content" in response_data,
                "result" in response_data,
            ]

            if any(success_indicators):
                # 提取内容（根据实际API响应格式调整字段名）
                content = ""
                title = ""
                description = ""

                # 尝试不同的字段名
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

                # 如果还是没有内容，尝试直接从响应中提取
                if not content:
                    for key in ["text", "markdown", "html"]:
                        if key in response_data and response_data[key]:
                            content = response_data[key]
                            break

                logger.info(f"✅ 成功提取内容，长度: {len(content)} 字符")

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

            # 失败情况
            error_details = []

            # 检查常见的错误字段
            if "error" in response_data:
                error_details.append(f"API错误: {response_data['error']}")
            if "message" in response_data:
                error_details.append(f"消息: {response_data['message']}")
            if "status" in response_data:
                error_details.append(f"状态: {response_data['status']}")
            if "code" in response_data:
                error_details.append(f"错误代码: {response_data['code']}")

            # 组合错误信息
            if error_details:
                error_msg = "API返回失败状态: " + " | ".join(error_details)
            else:
                error_msg = f"未知的API响应格式，响应字段: {response_keys}"

            # 如果响应很小，包含完整内容
            if len(str(response_data)) < 2000:
                error_msg += f" | 完整响应: {json.dumps(response_data, ensure_ascii=False)}"

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
            error_msg = f"解析响应时出错: {type(e).__name__}: {e}"
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
        调用接口的简化方法

        Args:
            url: 要爬取的网址
            **kwargs: 其他参数

        Returns:
            爬取结果
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
            logger.error(f"爬取失败: {e}")
            return {
                'success': False,
                'url': url,
                'title': '',
                'content': '',
                'error': f"调用异常: {type(e).__name__}: {e}"
            }


def build_default_spider_tool(debug: bool = False) -> SpiderTool:
    """提供一个可复用的默认实例"""
    return SpiderTool(
        api_url=DEFAULT_SPIDER_API_URL,
        timeout=DEFAULT_SPIDER_TIMEOUT,
        max_retry=DEFAULT_MAX_RETRY,
        enable_cache=True,
        enable_oversea=True,
        debug=debug,
    )


def test_spider_api():
    """测试新的spider API"""
    test_urls = [
        "https://en.wikipedia.org/wiki/ChatGPT",
    ]

    print("🚀 测试Spider API...")

    # 创建工具实例
    tool = SpiderTool(debug=True)

    success_count = 0
    total_count = len(test_urls)

    for i, url in enumerate(test_urls, 1):
        print("\n" + "=" * 80)
        print(f"📋 测试 {i}/{total_count}: {url}")
        print("=" * 80)

        start_time = time.time()
        result = tool(url)
        end_time = time.time()

        if result.get("success"):
            content_len = len(result.get("content", ""))
            print("✅ 成功！")
            print(f"   标题: {result.get('title', 'N/A')}")
            print(f"   内容长度: {content_len} 字符")
            print(f"   耗时: {end_time - start_time:.2f}秒")
            print(f"   内容预览: {result.get('content', '')[:200]}...")
            success_count += 1
        else:
            error = result.get("error", "Unknown error")
            print(f"❌ 失败: {error}")
            print(f"   耗时: {end_time - start_time:.2f}秒")

        # 在URL之间添加延迟
        if i < total_count:
            print("⏳ 等待2秒...")
            time.sleep(2)

    print(f"\n🎉 测试完成！成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")


# 保持兼容性：为了让旧代码仍然工作，提供别名
GoliathTool = SpiderTool
build_default_goliath_tool = build_default_spider_tool


if __name__ == "__main__":
    print("=== 测试新的Spider API网页爬取功能 ===")

    # 直接测试单个URL
    tool = build_default_spider_tool(debug=True)
    result = tool("https://www.bbc.co.uk/pressoffice/pressreleases/stories/2008/03_march/07/ob.shtml")

    print("\n📊 单个URL测试结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 保存结果到文件
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

        print(f"✅ 内容已保存到: {filepath}")

    print("\n" + "=" * 80)
    test_spider_api()
