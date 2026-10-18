from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION, VOLCENGINE_ARK_DEFAULT_BASE_URL
from ..core.shared.logging import (
    safe_log_error_body,
    safe_log_mapping,
    safe_log_url,
)
from ..core.shared.types import GenerationRequest, ImageCapability, ImageData


class VolcengineArkAdapter(BaseImageAdapter):
    """Volcengine Ark Seedream image generation adapter."""

    DEFAULT_BASE_URL = VOLCENGINE_ARK_DEFAULT_BASE_URL

    # Seedream 5.0 Pro / Flash official sample pixel maps (1K / 1.5K / 2K).
    SIZE_MAPS: dict[str, dict[str, str]] = {
        "1K": {
            "1:1": "1024x1024",
            "4:3": "1152x864",
            "3:4": "864x1152",
            "16:9": "1424x800",
            "9:16": "800x1424",
            "3:2": "1248x832",
            "2:3": "832x1248",
            "21:9": "1568x672",
            "4:5": "864x1152",
            "5:4": "1152x864",
        },
        "1.5K": {
            "1:1": "1536x1536",
            "4:3": "1792x1344",
            "3:4": "1344x1792",
            "16:9": "2048x1152",
            "9:16": "1152x2048",
            "3:2": "1872x1248",
            "2:3": "1248x1872",
            "21:9": "2352x1008",
            "4:5": "1344x1792",
            "5:4": "1792x1344",
        },
        "2K": {
            "1:1": "2048x2048",
            "4:3": "2368x1776",
            "3:4": "1776x2368",
            "16:9": "2816x1584",
            "9:16": "1584x2816",
            "3:2": "2496x1664",
            "2:3": "1664x2496",
            "21:9": "3136x1344",
            "4:5": "1776x2368",
            "5:4": "2368x1776",
        },
    }

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one Volcengine Ark image generation request."""
        start_time = time.time()
        payload = self._build_payload(request)
        session = self._get_session()

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }
        url = self._endpoint_url()

        self._log_request_overview(request, url, payload=payload)
        self._log_debug_json("请求", payload, request.task_id)

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as resp:
                duration = time.time() - start_time
                self._log_response_status(request, resp.status, duration)
                if resp.status != 200:
                    error_text = await resp.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    self._log_api_error(request, resp.status, duration, error_text)
                    return None, self._format_api_error_message(
                        resp.status,
                        error_text,
                    )

                data = await self._read_response_json(resp, request.task_id)
                return await self._extract_images(data, request.task_id)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            self._log_request_exception(request, duration, exc)
            return None, safe_log_error_body(exc)

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the Volcengine Ark image generation request payload."""
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "prompt": request.prompt,
            "response_format": "b64_json",
        }

        size = self._resolve_size(request)
        if size:
            payload["size"] = size

        if request.images:
            image_values = [self._to_data_url(image) for image in request.images]
            payload["image"] = (
                image_values[0] if len(image_values) == 1 else image_values
            )

        self._add_extra_options(payload)
        return payload

    def _is_agent_plan(self) -> bool:
        """Return whether the Agent Plan endpoint should be used."""
        mode = str(self.config.extra.get("api_mode") or "api").strip().lower()
        return mode in {"agent_plan", "plan"}

    def _endpoint_url(self) -> str:
        """Return a usable Ark image generation endpoint URL.

        Standard API uses ``/api/v3``; Agent Plan uses the dedicated
        ``/api/plan/v3`` prefix with a plan-specific API key.
        """
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        for suffix in (
            "/api/plan/v3/images/generations",
            "/api/v3/images/generations",
            "/api/plan/v3",
            "/api/v3",
        ):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        api_path = "/api/plan/v3" if self._is_agent_plan() else "/api/v3"
        return f"{base}{api_path}/images/generations"

    def _model_name(self) -> str:
        """Return the active model name."""
        return self.model or "doubao-seedream-5-0-pro-260628"

    def _model_name_lower(self) -> str:
        """Return the lower-cased active model name."""
        return self._model_name().lower()

    def _is_seedream_pro(self) -> bool:
        """Return whether the active model is Seedream 5.0 Pro."""
        model = self._model_name_lower()
        return "seedream-5-0-pro" in model or "seedream-5.0-pro" in model

    def _resolve_size(self, request: GenerationRequest) -> str | None:
        """Resolve the Ark size parameter from resolution and aspect ratio.

        Seedream accepts either a resolution tier (e.g. ``2K``) or exact
        ``widthxheight`` pixels. When only resolution is set, send the tier.
        When both resolution and aspect ratio are set, send mapped pixels.
        """
        resolution_unspecified = (
            not request.resolution or request.resolution == UNSPECIFIED_OPTION
        )
        aspect_unspecified = (
            not request.aspect_ratio or request.aspect_ratio == UNSPECIFIED_OPTION
        )

        if resolution_unspecified and aspect_unspecified:
            return None

        if not resolution_unspecified and aspect_unspecified:
            return self._normalize_resolution(request.resolution)

        if resolution_unspecified:
            return None

        resolution = self._normalize_resolution(request.resolution)
        aspect_ratio = request.aspect_ratio
        resolution_map = self.SIZE_MAPS.get(resolution) or self.SIZE_MAPS["2K"]
        return resolution_map.get(aspect_ratio, resolution_map["1:1"])

    def _normalize_resolution(self, resolution: str | None) -> str:
        """Map plugin resolution values to the sizes supported by Seedream 5.0."""
        value = resolution or "2K"
        return value if value in {"1K", "1.5K", "2K"} else "2K"

    def _to_data_url(self, image: ImageData) -> str:
        """Convert image data to a Volcengine Ark-compatible data URL."""
        mime_type = (image.mime_type or "image/png").lower()
        b64_data = base64.b64encode(image.data).decode("ascii")
        return f"data:{mime_type};base64,{b64_data}"

    def _add_extra_options(self, payload: dict[str, Any]) -> None:
        """Add optional Volcengine Ark generation parameters.

        Seedream 5.0 Pro and Flash do not support image-set generation or web
        search, so those fields are intentionally never sent.
        """
        extra = self.config.extra

        payload["watermark"] = self._coerce_bool(extra.get("watermark"), default=False)

        optimize_mode = str(extra.get("optimize_prompt_mode") or "").strip()
        if optimize_mode in {"standard", "fast"}:
            # Official docs: Seedream 5.0 Flash only supports standard.
            if optimize_mode == "fast" and not self._is_seedream_pro():
                optimize_mode = "standard"
            payload["optimize_prompt_options"] = {"mode": optimize_mode}

        output_format = str(extra.get("output_format") or "").strip().lower()
        if output_format in {"png", "jpeg"}:
            payload["output_format"] = output_format

    async def _extract_images(
        self, response: dict[str, Any], task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """Extract image bytes from a Volcengine Ark response."""
        if response_error := response.get("error"):
            if isinstance(response_error, dict):
                message = response_error.get("message") or response_error.get("code")
                return None, f"API 错误: {message}"
            return None, f"API 错误: {response_error}"

        data_items = response.get("data")
        if not isinstance(data_items, list):
            return None, f"响应中未找到 data 字段: {safe_log_mapping(response)}"

        images: list[bytes] = []
        errors: list[str] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue

            if item_error := item.get("error"):
                errors.append(self._format_item_error(item_error))
                continue

            if b64_json := item.get("b64_json"):
                try:
                    images.append(base64.b64decode(str(b64_json)))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"{self._get_log_prefix(task_id)} Base64 解码失败: {safe_log_error_body(exc)}"
                    )
                continue

            url = item.get("url")
            if isinstance(url, str) and url:
                if data := await self._download_image(url, task_id):
                    images.append(data)

        if images:
            return images, None
        if errors:
            return None, "; ".join(errors)
        return None, "未找到有效的图片数据"

    def _format_item_error(self, error: Any) -> str:
        """Format an item-level image generation error."""
        if not isinstance(error, dict):
            return str(error)
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or "").strip()
        if code and message:
            return f"{code}: {message}"
        return message or code or str(error)

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """Download a temporary image URL returned by Volcengine Ark."""
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url, proxy=self.proxy, timeout=self._get_download_timeout()
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.error(
                    f"{prefix} 下载图像失败 ({resp.status}): {safe_log_url(url)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像异常: {safe_log_error_body(exc)}")
        return None

    def _coerce_bool(self, value: Any, *, default: bool) -> bool:
        """Safely coerce a boolean setting."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on", "开启", "启用"}:
                return True
            if lowered in {"false", "0", "no", "off", "关闭", "禁用"}:
                return False
        return default
