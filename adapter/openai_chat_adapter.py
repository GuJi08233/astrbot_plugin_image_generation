from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Iterable
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import OPENAI_DEFAULT_BASE_URL
from ..core.types import GenerationRequest, ImageCapability


class OpenAIChatAdapter(BaseImageAdapter):
    """基于 OpenAI Chat Completions 流式响应的图像生成适配器。"""

    DEFAULT_BASE_URL = OPENAI_DEFAULT_BASE_URL
    DEFAULT_MODEL = "gemini-3.1-flash-image-landscape"
    IMAGE_URL_PATTERN = re.compile(
        r"https?://[^\s\"'<>]+?\.(?:jpg|jpeg|png|webp)(?:\?[^\s\"'<>]*)?",
        flags=re.IGNORECASE,
    )

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return ImageCapability.TEXT_TO_IMAGE | ImageCapability.IMAGE_TO_IMAGE

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)
        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = self._build_payload(request)

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        logger.debug(
            f"{prefix} 请求 -> {url}, model={payload['model']}, stream={payload['stream']}"
        )

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): {preview}"
                    )
                    return None, f"API 错误 ({resp.status})"

                image_urls = await self._extract_image_urls_from_stream(
                    resp, request.task_id
                )
                if not image_urls:
                    logger.error(f"{prefix} 流式响应中未找到图片 URL")
                    return None, "流式响应中未找到图片 URL"

                images = await self._download_images(image_urls, request.task_id)
                if not images:
                    return None, "检测到图片 URL，但下载失败"

                logger.info(
                    f"{prefix} 生成成功 (耗时: {duration:.2f}s, 图片数: {len(images)})"
                )
                return images, None
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {exc}")
            return None, str(exc)

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """构建请求载荷。"""
        if request.images:
            content: str | list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
            for image in request.images:
                b64_data = base64.b64encode(image.data).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.mime_type};base64,{b64_data}"
                        },
                    }
                )
        else:
            content = request.prompt

        return {
            "model": self.model or self.DEFAULT_MODEL,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }

    async def _extract_image_urls_from_stream(
        self,
        response: aiohttp.ClientResponse,
        task_id: str | None = None,
    ) -> list[str]:
        """从 SSE 流式响应中提取图片 URL。"""
        urls: list[str] = []
        fragments: list[str] = []
        prefix = self._get_log_prefix(task_id)

        async for raw_line in response.content:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith(":"):
                continue

            if not line.startswith("data:"):
                self._append_image_urls(line, urls)
                fragments.append(line)
                continue

            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                break

            self._append_image_urls(payload, urls)
            fragments.append(payload)

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            for value in self._walk_strings(data):
                self._append_image_urls(value, urls)
                fragments.append(value)

        if not urls and fragments:
            self._append_image_urls("\n".join(fragments), urls)

        if urls:
            logger.info(f"{prefix} 在流式响应中找到 {len(urls)} 个图片 URL")
        return urls

    async def _download_images(
        self, image_urls: Iterable[str], task_id: str | None = None
    ) -> list[bytes]:
        """下载全部图片 URL。"""
        images: list[bytes] = []
        for image_url in image_urls:
            image = await self._download_image(image_url, task_id)
            if image:
                images.append(image)
        return images

    async def _download_image(
        self, image_url: str, task_id: str | None = None
    ) -> bytes | None:
        """下载单张图片。"""
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                image_url,
                proxy=self.proxy,
                timeout=self._get_download_timeout(),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.error(f"{prefix} 下载图像失败 ({resp.status}): {image_url}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像异常: {exc}")
        return None

    @classmethod
    def _append_image_urls(cls, text: str, urls: list[str]) -> None:
        """从文本中提取图片 URL 并去重。"""
        for url in cls.IMAGE_URL_PATTERN.findall(text):
            if url not in urls:
                urls.append(url)

    @classmethod
    def _walk_strings(cls, value: Any) -> Iterable[str]:
        """递归提取 JSON 中的全部字符串字段。"""
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from cls._walk_strings(item)
            return
        if isinstance(value, list):
            for item in value:
                yield from cls._walk_strings(item)
