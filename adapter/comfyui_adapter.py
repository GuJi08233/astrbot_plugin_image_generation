"""ComfyUI workflow adapter using the HTTP prompt API."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import aiohttp
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import (
    PLUGIN_NAME,
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
    UNSPECIFIED_OPTION,
)
from ..core.shared.logging import (
    format_cn_log_fields,
    safe_log_error_body,
    safe_log_text,
)
from ..core.shared.types import (
    AdapterConfig,
    GenerationRequest,
    ImageCapability,
    ImageData,
)

TEXT_INPUT_NAMES = ("prompt", "text", "value", "string", "on_false")
TEXT_LINK_NAMES = (
    "prompt",
    "text",
    "value",
    "string",
    "on_false",
    "on_true",
    "source",
    "positive",
)
MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}
MEGAPIXELS_BY_RESOLUTION = {
    "1K": 1.0,
    "1.5K": 2.25,
    "2K": 4.0,
    "4K": 16.0,
}


class ComfyUIAdapter(BaseImageAdapter):
    """Run a user-provided ComfyUI API workflow over HTTP."""

    requires_api_key = False
    DEFAULT_BASE_URL = "http://127.0.0.1:8188"
    DEFAULT_POLL_INTERVAL = 2
    WORKFLOW_ERROR_PREFIX = "工作流参数无效"

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._object_info_cache: dict[str, dict[str, Any]] = {}

    def get_capabilities(self) -> ImageCapability:
        """Return provider capabilities selected in the configuration."""
        capabilities = self._get_configured_capabilities()
        if not capabilities:
            capabilities = ImageCapability.TEXT_TO_IMAGE
        return capabilities

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Submit one ComfyUI workflow and wait for its output images.

        Args:
            request: Image generation request to execute.

        Returns:
            Generated image bytes or a user-facing error message.
        """
        workflow, error = self._load_workflow()
        if error:
            return None, error

        if error := self._inject_prompt(workflow, request.prompt):
            return None, error

        await self._apply_generation_options(workflow, request)

        error = await self._apply_reference_images(workflow, request)
        if error:
            return None, error

        prompt_id, error = await self._submit_prompt(workflow, request)
        if error:
            return None, error

        return await self._wait_for_result(prompt_id, request)

    # Workflow preparation.

    def _load_workflow(self) -> tuple[dict[str, Any] | None, str | None]:
        """Load and validate the uploaded API-format workflow JSON."""
        raw_file = self._workflow_file_value()
        if not raw_file:
            return None, (
                "未上传 ComfyUI 工作流文件，请上传导出的 API 格式 JSON"
                f"（{self.WORKFLOW_ERROR_PREFIX}）"
            )

        path = self._resolve_workflow_path(raw_file)
        if path is None:
            return None, (
                f"工作流文件路径无效（{self.WORKFLOW_ERROR_PREFIX}），请重新上传"
            )
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            return None, (
                f"工作流文件读取失败（{self.WORKFLOW_ERROR_PREFIX}）: "
                f"{safe_log_error_body(exc, 300)}"
            )
        source = str(path)

        try:
            workflow = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, (
                f"工作流 JSON 解析失败（{self.WORKFLOW_ERROR_PREFIX}）: {exc}"
            )

        if not isinstance(workflow, dict) or not workflow:
            return None, f"工作流不是有效的节点对象（{self.WORKFLOW_ERROR_PREFIX}）"
        for node_id, node in workflow.items():
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("class_type"), str)
                or not isinstance(node.get("inputs"), dict)
            ):
                return None, (
                    f"工作流节点 {node_id} 格式无效（{self.WORKFLOW_ERROR_PREFIX}）"
                )

        logger.debug(
            f"{self._get_log_prefix()} 已加载 ComfyUI 工作流: "
            + format_cn_log_fields(来源=source, 节点数=len(workflow))
        )
        return workflow, None

    def _workflow_file_value(self) -> str:
        """Extract the first uploaded workflow file path from config."""
        raw = self.config.extra.get("workflow_file")
        if not isinstance(raw, list):
            return ""
        values = [
            item.strip()
            for item in raw
            if isinstance(item, str)
            and item.strip()
            and item.strip().lower().endswith(".json")
        ]
        if not values:
            return ""
        if len(values) > 1:
            logger.warning(
                f"{self._get_log_prefix()} 配置了 {len(values)} 个工作流文件，"
                f"仅使用第一个: {safe_log_text(values[0], 200)}"
            )
        return values[0]

    def _resolve_workflow_path(self, raw: str) -> Path | None:
        """Resolve an uploaded workflow path inside the plugin data directory."""
        path = Path(raw)
        if path.is_absolute():
            return None
        plugin_data_dir = (Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME).resolve()
        target = (plugin_data_dir / path).resolve()
        try:
            target.relative_to(plugin_data_dir)
        except ValueError:
            return None
        return target

    async def _apply_generation_options(
        self,
        workflow: dict[str, Any],
        request: GenerationRequest,
    ) -> None:
        """Apply optional aspect-ratio and resolution settings to the workflow."""
        aspect_ratio = self._option_value(request.aspect_ratio)
        resolution = self._option_value(request.resolution)
        if not aspect_ratio and not resolution:
            return

        aspect_node_id = self._resolve_node_id(
            workflow,
            "aspect_ratio_node_id",
            lambda node: isinstance(node["inputs"].get("aspect_ratio"), str),
        )
        if aspect_node_id:
            inputs = workflow[aspect_node_id]["inputs"]
            if aspect_ratio:
                inputs["aspect_ratio"] = await self._resolve_enum_option(
                    workflow[aspect_node_id],
                    "aspect_ratio",
                    aspect_ratio,
                )
            if resolution and isinstance(inputs.get("megapixels"), (int, float)):
                inputs["megapixels"] = self._megapixels(resolution)

        size_node_id = self._resolve_node_id(
            workflow,
            "width_node_id",
            lambda node: (
                isinstance(node["inputs"].get("width"), (int, float))
                and isinstance(node["inputs"].get("height"), (int, float))
            ),
        )
        if not size_node_id:
            return

        size = self._resolve_size(aspect_ratio, resolution)
        if not size:
            return
        width, height = size
        inputs = workflow[size_node_id]["inputs"]
        inputs["width"] = width
        inputs["height"] = height

    def _resolve_node_id(
        self,
        workflow: dict[str, Any],
        config_key: str,
        predicate: Callable[[dict[str, Any]], bool],
    ) -> str | None:
        """Resolve an explicit node id or auto-detect the first matching node."""
        for node_id in self._split_ids(self.config.extra.get(config_key)):
            if node_id in workflow:
                return node_id
        if self._split_ids(self.config.extra.get(config_key)):
            return None
        for node_id, node in workflow.items():
            if predicate(node):
                return node_id
        return None

    async def _resolve_enum_option(
        self,
        node: dict[str, Any],
        input_name: str,
        requested: str,
    ) -> str:
        """Pick the workflow enum option that matches a requested value."""
        options = await self._get_enum_options(
            str(node.get("class_type") or ""),
            input_name,
        )
        requested_lower = requested.lower()
        for option in options:
            if option.lower().startswith(requested_lower):
                return option
        for option in options:
            if requested_lower in option.lower():
                return option

        current = node["inputs"].get(input_name)
        if isinstance(current, str) and requested_lower in current.lower():
            return current
        return requested

    async def _get_enum_options(
        self,
        class_type: str,
        input_name: str,
    ) -> list[str]:
        """Return enum options for one node input from ComfyUI object info."""
        if not class_type:
            return []
        if class_type not in self._object_info_cache:
            url = f"{self.base_url}/object_info/{quote(class_type, safe='')}"
            try:
                async with self._get_session().get(
                    url,
                    proxy=self.proxy,
                    timeout=self._get_timeout(),
                ) as response:
                    if 200 <= response.status < 300:
                        data = await response.json()
                        node_info = (
                            data.get(class_type) if isinstance(data, dict) else None
                        )
                        if isinstance(node_info, dict):
                            self._object_info_cache[class_type] = node_info
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    f"{self._get_log_prefix()} 获取节点定义失败: {safe_log_error_body(exc, 200)}"
                )

        node_info = self._object_info_cache.get(class_type) or {}
        input_defs = node_info.get("input")
        if not isinstance(input_defs, dict):
            return []
        for section in ("required", "optional"):
            definitions = input_defs.get(section)
            if not isinstance(definitions, dict):
                continue
            spec = definitions.get(input_name)
            if not isinstance(spec, list) or not spec:
                continue
            if isinstance(spec[0], list):
                return [str(value) for value in spec[0]]
            if str(spec[0]).upper() == "COMBO" and len(spec) > 1:
                options = spec[1].get("options") if isinstance(spec[1], dict) else None
                if isinstance(options, list):
                    return [str(value) for value in options]
        return []

    def _megapixels(self, resolution: str) -> float:
        """Map a resolution tier to a megapixel value.

        Args:
            resolution: Requested resolution tier, such as ``1K``.

        Returns:
            The matching megapixel value, defaulting to 1.0.
        """
        return MEGAPIXELS_BY_RESOLUTION.get(resolution, 1.0)

    def _resolve_size(
        self,
        aspect_ratio: str | None,
        resolution: str | None,
    ) -> tuple[int, int] | None:
        """Resolve a width-by-height pair for latent-size workflow nodes."""
        mapped = self._size_map_value(resolution, aspect_ratio)
        if mapped:
            return mapped

        ratio = aspect_ratio or "1:1"
        tier = resolution or "1K"
        base = RESOLUTION_2K_MAP if tier in {"2K", "4K"} else RESOLUTION_1K_MAP
        raw = base.get(ratio)
        if not raw:
            return self._computed_size(ratio, tier)
        width, height = (int(part) for part in raw.split("x", 1))
        if tier == "1.5K":
            width, height = int(width * 1.5), int(height * 1.5)
        elif tier == "4K":
            width, height = width * 2, height * 2
        return self._round_size(width, height)

    def _size_map_value(
        self,
        resolution: str | None,
        aspect_ratio: str | None,
    ) -> tuple[int, int] | None:
        """Read a configured width-by-height size-map entry.

        Args:
            resolution: Requested resolution tier.
            aspect_ratio: Requested aspect ratio.

        Returns:
            The first matching configured size, or None.
        """
        raw_map = self.config.extra.get("size_map_json")
        if isinstance(raw_map, str):
            try:
                size_map = json.loads(raw_map)
            except json.JSONDecodeError:
                logger.warning(
                    f"{self._get_log_prefix()} size_map_json JSON 解析失败，已忽略"
                )
                return None
        else:
            size_map = raw_map
        if not isinstance(size_map, dict):
            return None

        keys = [
            f"{resolution}:{aspect_ratio}" if resolution and aspect_ratio else None,
            resolution,
            aspect_ratio,
            "default",
        ]
        for key in keys:
            value = size_map.get(key) if key else None
            if isinstance(value, str) and "x" in value:
                width, _, height = value.partition("x")
                try:
                    return self._round_size(int(width), int(height))
                except ValueError:
                    continue
        return None

    def _computed_size(self, ratio: str, tier: str) -> tuple[int, int] | None:
        """Compute a pixel size from an aspect ratio and megapixel tier."""
        try:
            left, right = (float(part) for part in ratio.split(":", 1))
        except ValueError:
            return None
        if left <= 0 or right <= 0:
            return None

        area = self._megapixels(tier) * 1_000_000
        width = int((area * left / right) ** 0.5)
        height = int(width * right / left)
        return self._round_size(width, height)

    def _round_size(self, width: int, height: int) -> tuple[int, int]:
        """Round a pixel size down to a multiple of 8."""
        return max(64, width // 8 * 8), max(64, height // 8 * 8)

    # Prompt injection.

    def _inject_prompt(self, workflow: dict[str, Any], prompt: str) -> str | None:
        """Write the user prompt into the resolved workflow text input."""
        target = self._resolve_prompt_target(workflow)
        if not target:
            return (
                "未找到可写入提示词的工作流节点"
                f"（{self.WORKFLOW_ERROR_PREFIX}），请配置 prompt_node_id"
            )

        node_id, input_name = target
        value = workflow[node_id]["inputs"].get(input_name)
        if isinstance(value, str):
            workflow[node_id]["inputs"][input_name] = prompt
            return None
        if self._is_link(value):
            self._inject_text_upstream(workflow, value, prompt, set())
            return None
        return (
            f"工作流节点 {node_id}.{input_name} 不是文本输入"
            f"（{self.WORKFLOW_ERROR_PREFIX}）"
        )

    def _resolve_prompt_target(
        self,
        workflow: dict[str, Any],
    ) -> tuple[str, str] | None:
        """Find the text-encode node that should receive the user prompt."""
        configured = self._split_ids(self.config.extra.get("prompt_node_id"))
        if configured:
            for node_id in configured:
                node = workflow.get(node_id)
                if not node:
                    continue
                input_name = str(self.config.extra.get("prompt_input") or "").strip()
                if input_name and input_name in node["inputs"]:
                    return node_id, input_name
                for candidate in ("prompt", "text", "value"):
                    if candidate in node["inputs"]:
                        return node_id, candidate
                return node_id, "prompt"

        for node in workflow.values():
            positive = node["inputs"].get("positive")
            if self._is_link(positive):
                target = self._follow_text_target(workflow, positive, set())
                if target:
                    return target

        for node_id, node in workflow.items():
            class_name = str(node.get("class_type") or "").lower().replace("_", "")
            if "textencode" not in class_name:
                continue
            for candidate in ("prompt", "text"):
                if candidate in node["inputs"]:
                    return node_id, candidate
        return None

    def _follow_text_target(
        self,
        workflow: dict[str, Any],
        link: list[Any],
        visited: set[str],
    ) -> tuple[str, str] | None:
        """Follow a link upstream until a text input is found."""
        source_id = str(link[0])
        if source_id in visited:
            return None
        visited.add(source_id)
        node = workflow.get(source_id)
        if not node:
            return None

        class_name = str(node.get("class_type") or "").lower().replace("_", "")
        for candidate in ("prompt", "text"):
            if candidate in node["inputs"]:
                return source_id, candidate
        if "textencode" in class_name:
            return source_id, "prompt"

        for name in ("source", "on_true", "on_false", "value", "string", "prompt"):
            value = node["inputs"].get(name)
            if self._is_link(value):
                target = self._follow_text_target(workflow, value, visited)
                if target:
                    return target
        return None

    def _inject_text_upstream(
        self,
        workflow: dict[str, Any],
        link: list[Any],
        prompt: str,
        visited: set[str],
    ) -> None:
        """Write the prompt into literal text inputs reached through a link."""
        source_id = str(link[0])
        if source_id in visited:
            return
        visited.add(source_id)
        node = workflow.get(source_id)
        if not node:
            return

        inputs = node["inputs"]
        for name in TEXT_INPUT_NAMES:
            if isinstance(inputs.get(name), str):
                inputs[name] = prompt
        for name in TEXT_LINK_NAMES:
            value = inputs.get(name)
            if self._is_link(value):
                self._inject_text_upstream(workflow, value, prompt, visited)

    # Reference images.

    async def _apply_reference_images(
        self,
        workflow: dict[str, Any],
        request: GenerationRequest,
    ) -> str | None:
        """Upload reference images and point LoadImage nodes at them."""
        if not request.images:
            return None

        node_ids = self._split_ids(self.config.extra.get("image_node_id"))
        if not node_ids:
            node_ids = [
                node_id
                for node_id, node in workflow.items()
                if "loadimage"
                in str(node.get("class_type") or "").lower().replace("_", "")
                and "image" in node["inputs"]
            ]
        if not node_ids:
            return (
                "工作流未包含 LoadImage 节点"
                f"（{self.WORKFLOW_ERROR_PREFIX}），请配置 image_node_id"
            )

        for node_id, image in zip(node_ids, request.images):
            node = workflow.get(node_id)
            if not node:
                return f"工作流节点 {node_id} 不存在（{self.WORKFLOW_ERROR_PREFIX}）"
            uploaded_name, error = await self._upload_image(image, request)
            if error:
                return error
            node["inputs"]["image"] = uploaded_name
        return None

    async def _upload_image(
        self,
        image: ImageData,
        request: GenerationRequest,
    ) -> tuple[str, str | None]:
        """Upload one reference image to ComfyUI's input directory."""
        start_time = time.monotonic()
        mime_type = (image.mime_type or "image/png").lower()
        suffix = MIME_SUFFIXES.get(mime_type, ".png")
        filename = f"astrbot_{uuid.uuid4().hex[:12]}{suffix}"
        form = aiohttp.FormData()
        form.add_field(
            "image",
            image.data,
            filename=filename,
            content_type=mime_type,
        )
        form.add_field("type", "input")
        form.add_field("overwrite", "true")

        url = f"{self.base_url}/upload/image"
        self._log_request_overview(request, url, method="POST", form_fields=["image"])

        try:
            async with self._get_session().post(
                url,
                data=form,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                self._log_response_status(request, response.status, duration)
                body = await response.text()
                if not 200 <= response.status < 300:
                    self._log_api_error(request, response.status, duration, body)
                    return "", self._format_api_error_message(response.status, body)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    return "", "ComfyUI 图片上传响应无效"
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            self._log_request_exception(request, duration, exc, label="图片上传异常")
            return "", safe_log_error_body(exc)

        name = data.get("name") if isinstance(data, dict) else None
        if not isinstance(name, str) or not name:
            return "", "ComfyUI 图片上传响应缺少文件名"
        subfolder = data.get("subfolder") if isinstance(data, dict) else None
        if isinstance(subfolder, str) and subfolder:
            return f"{subfolder}/{name}", None
        return name, None

    # Submission and polling.

    async def _submit_prompt(
        self,
        workflow: dict[str, Any],
        request: GenerationRequest,
    ) -> tuple[str, str | None]:
        """Queue one workflow and return the ComfyUI prompt id."""
        start_time = time.monotonic()
        url = f"{self.base_url}/prompt"
        payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}
        headers = {"Content-Type": "application/json"}
        api_key = self._get_current_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._log_request_overview(request, url, payload=payload)
        if self.debug_request_logging:
            logger.debug(
                f"{self._get_log_prefix(request.task_id)} ComfyUI 工作流节点: "
                f"{', '.join(workflow.keys())}"
            )

        try:
            async with self._get_session().post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                self._log_response_status(request, response.status, duration)
                body = await response.text()
                if not 200 <= response.status < 300:
                    self._log_api_error(request, response.status, duration, body)
                    workflow_error = self._format_workflow_error(body)
                    if workflow_error:
                        return "", f"{self.WORKFLOW_ERROR_PREFIX}: {workflow_error}"
                    return "", self._format_api_error_message(response.status, body)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    return "", "ComfyUI /prompt 响应不是有效 JSON"
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            self._log_request_exception(request, duration, exc)
            return "", safe_log_error_body(exc)

        prompt_id = data.get("prompt_id") if isinstance(data, dict) else None
        node_errors = data.get("node_errors") if isinstance(data, dict) else None
        if isinstance(node_errors, dict) and node_errors:
            workflow_error = self._format_workflow_error(json.dumps(data))
            return "", f"{self.WORKFLOW_ERROR_PREFIX}: {workflow_error}"
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            return "", "ComfyUI /prompt 响应缺少 prompt_id"
        logger.debug(
            f"{self._get_log_prefix(request.task_id)} ComfyUI 任务已入队: "
            + format_cn_log_fields(prompt_id=safe_log_text(prompt_id, 120))
        )
        return prompt_id.strip(), None

    def _format_workflow_error(self, body: str) -> str:
        """Extract concise node validation messages from a /prompt error body."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return ""
        if not isinstance(data, dict):
            return ""

        messages: list[str] = []
        node_errors = data.get("node_errors")
        if isinstance(node_errors, dict):
            for node_id, info in node_errors.items():
                if not isinstance(info, dict):
                    continue
                class_type = str(info.get("class_type") or "").strip()
                for item in info.get("errors") or []:
                    if not isinstance(item, dict):
                        continue
                    message = str(item.get("message") or "").strip()
                    details = str(item.get("details") or "").strip()
                    text = f"{message}: {details}".strip(": ") if details else message
                    if text:
                        messages.append(f"{node_id} ({class_type}): {text}")

        if not messages and data.get("error"):
            messages.append(str(data["error"]))
        return "；".join(messages)[:600]

    async def _wait_for_result(
        self,
        prompt_id: str,
        request: GenerationRequest,
    ) -> tuple[list[bytes] | None, str | None]:
        """Poll ComfyUI history until the queued workflow finishes."""
        deadline = time.monotonic() + max(1, self.timeout)
        interval = self._poll_interval()
        url = f"{self.base_url}/history/{prompt_id}"

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return await self._handle_timeout(prompt_id, request)

            try:
                poll_start = time.monotonic()
                async with self._get_session().get(
                    url,
                    proxy=self.proxy,
                    timeout=aiohttp.ClientTimeout(total=max(1, min(remaining, 30))),
                ) as response:
                    if not 200 <= response.status < 300:
                        error_text = await response.text()
                        self._log_api_error(
                            request,
                            response.status,
                            time.monotonic() - poll_start,
                            error_text,
                        )
                        await asyncio.sleep(min(interval, remaining))
                        continue
                    data = await response.json()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    f"{self._get_log_prefix(request.task_id)} ComfyUI 轮询暂时失败: "
                    f"{safe_log_error_body(exc, 200)}"
                )
                await asyncio.sleep(min(interval, remaining))
                continue

            entry = data.get(prompt_id) if isinstance(data, dict) else None
            if isinstance(entry, dict):
                status = entry.get("status")
                if isinstance(status, dict) and status.get("status_str") == "error":
                    return None, self._format_execution_error(status)
                if status is None or status.get("completed", True):
                    outputs = entry.get("outputs")
                    return await self._download_outputs(
                        outputs if isinstance(outputs, dict) else {},
                        request,
                    )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return await self._handle_timeout(prompt_id, request)
            await asyncio.sleep(min(interval, remaining))

    async def _handle_timeout(
        self,
        prompt_id: str,
        request: GenerationRequest,
    ) -> tuple[None, str]:
        """Build the timeout result and optionally cancel the prompt."""
        cancel_note = "，任务仍在 ComfyUI 中运行"
        if self._cancel_on_timeout():
            cancelled = await self._cancel_prompt(prompt_id, request)
            cancel_note = "，已尝试取消任务" if cancelled else "，取消任务失败"
        return None, (
            f"ComfyUI 生成超时（{int(max(1, self.timeout))} 秒）{cancel_note}，"
            "请检查超时参数"
        )

    def _cancel_on_timeout(self) -> bool:
        """Return whether timed-out tasks should be cancelled."""
        value = self.config.extra.get("cancel_on_timeout", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off"}
        return bool(value)

    async def _cancel_prompt(
        self,
        prompt_id: str,
        request: GenerationRequest,
    ) -> bool:
        """Cancel a queued or running prompt without affecting other jobs."""
        url = f"{self.base_url}/api/jobs/{quote(prompt_id, safe='')}/cancel"
        self._log_request_overview(request, url, method="POST")
        start_time = time.monotonic()
        try:
            async with self._get_session().post(
                url,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                self._log_response_status(request, response.status, duration)
                body = await response.text()
                if 200 <= response.status < 300:
                    logger.info(
                        f"{self._get_log_prefix(request.task_id)} "
                        f"已尝试取消超时任务: {safe_log_text(prompt_id, 120)}"
                    )
                    return True
                if response.status in {404, 405}:
                    logger.debug(
                        f"{self._get_log_prefix(request.task_id)} "
                        f"取消接口不可用，改为移出排队: {response.status}"
                    )
                    return await self._dequeue_prompt(prompt_id, request)
                self._log_api_error(
                    request,
                    response.status,
                    duration,
                    body,
                    label="取消任务错误",
                )
        except Exception as exc:  # noqa: BLE001
            self._log_request_exception(
                request,
                time.monotonic() - start_time,
                exc,
                label="取消任务异常",
            )
        return False

    async def _dequeue_prompt(
        self,
        prompt_id: str,
        request: GenerationRequest,
    ) -> bool:
        """Remove a pending prompt from the ComfyUI queue."""
        url = f"{self.base_url}/queue"
        payload = {"delete": [prompt_id]}
        self._log_request_overview(request, url, method="POST", payload=payload)
        start_time = time.monotonic()
        try:
            async with self._get_session().post(
                url,
                json=payload,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                self._log_response_status(request, response.status, duration)
                body = await response.text()
                if 200 <= response.status < 300:
                    logger.info(
                        f"{self._get_log_prefix(request.task_id)} "
                        f"已将超时任务移出排队: {safe_log_text(prompt_id, 120)}"
                    )
                    return True
                self._log_api_error(
                    request,
                    response.status,
                    duration,
                    body,
                    label="移出排队错误",
                )
        except Exception as exc:  # noqa: BLE001
            self._log_request_exception(
                request,
                time.monotonic() - start_time,
                exc,
                label="移出排队异常",
            )
        return False

    def _format_execution_error(self, status: dict[str, Any]) -> str:
        """Build a user-facing message from ComfyUI execution status."""
        detail = ""
        messages = status.get("messages")
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                event, payload = item[0], item[1]
                if event != "execution_error" or not isinstance(payload, dict):
                    continue
                node_id = payload.get("node_id")
                node_type = payload.get("node_type")
                message = payload.get("exception_message")
                detail = f"{node_id} ({node_type}): {message}"
                break
        if detail:
            return f"ComfyUI 执行失败，请检查工作流参数: {detail}"
        return "ComfyUI 执行失败，请检查工作流参数"

    async def _download_outputs(
        self,
        outputs: dict[str, Any],
        request: GenerationRequest,
    ) -> tuple[list[bytes] | None, str | None]:
        """Download every image produced by one finished workflow."""
        images: list[bytes] = []
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for image in node_output.get("images") or []:
                if not isinstance(image, dict):
                    continue
                filename = image.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                query = urlencode(
                    {
                        "filename": filename,
                        "subfolder": str(image.get("subfolder") or ""),
                        "type": str(image.get("type") or "output"),
                    }
                )
                url = f"{self.base_url}/view?{query}"
                try:
                    async with self._get_session().get(
                        url,
                        proxy=self.proxy,
                        timeout=self._get_download_timeout(),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            self._log_api_error(
                                request,
                                response.status,
                                0,
                                error_text,
                                label="结果下载错误",
                            )
                            return None, self._format_api_error_message(
                                response.status,
                                error_text,
                            )
                        images.append(await response.read())
                except Exception as exc:  # noqa: BLE001
                    self._log_request_exception(request, 0, exc, label="结果下载异常")
                    return None, safe_log_error_body(exc)

        if not images:
            return None, "ComfyUI 未返回图片结果"
        return images, None

    # Helpers.

    def _poll_interval(self) -> float:
        """Return a validated interval between history requests."""
        return float(
            self._positive_int("poll_interval", self.DEFAULT_POLL_INTERVAL, minimum=1)
        )

    def _positive_int(self, key: str, default: int, *, minimum: int) -> int:
        """Read a positive integer adapter setting with a safe fallback."""
        value = self.config.extra.get(key, default)
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else default

    def _option_value(self, value: str | None) -> str | None:
        """Return an option unless the request omits it."""
        if not value or value == UNSPECIFIED_OPTION:
            return None
        return value

    def _split_ids(self, raw: Any) -> list[str]:
        """Normalize a node-id setting from a string or list."""
        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, list):
            items = [str(item) for item in raw if isinstance(item, (str, int))]
        else:
            return []

        result: list[str] = []
        for item in items:
            for part in item.replace("，", ",").replace("\n", ",").split(","):
                node_id = part.strip()
                if node_id and node_id not in result:
                    result.append(node_id)
        return result

    def _is_link(self, value: Any) -> bool:
        """Return whether a workflow input is a node link."""
        return (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[1], int)
            and isinstance(value[0], (str, int))
            and str(value[0]).strip() != ""
        )
