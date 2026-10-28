"""Backend API handlers for the plugin Dashboard Page."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.web import (
    PluginUploadFile,
    error_response,
    file_response,
    json_response,
    request,
)
from PIL import Image

from ..adapters.generator import create_adapter
from ..generation.image_processor import GENERATED_IMAGE_EXTENSIONS
from ..generation.novelai_options import validate_novelai_options
from ..shared.logging import log_prefix, safe_log_text
from ..shared.types import AdapterType, ImageCapability
from ..tasks.models import GenerationTaskRecord
from .payload import PagePayloadMixin
from .stats import PageStatsMixin

PLUGIN_NAME = "astrbot_plugin_image_generation"
PAGE_PREVIEW_MAX_BYTES = 12 * 1024 * 1024
PAGE_PREVIEW_MAX_SIDE = 1024
LOG = log_prefix("Page")


class ImageGenerationPageAPI(PagePayloadMixin, PageStatsMixin):
    """Register and serve backend endpoints used by the Dashboard Page."""

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin

    def register(self) -> None:
        """Register backend APIs used by the plugin Dashboard Page."""
        page_routes = [
            ("/page/state", self.page_state, ["GET"], "Image generation Page state"),
            (
                "/page/stats",
                self.page_stats,
                ["GET"],
                "Image generation Page statistics",
            ),
            ("/page/tasks", self.page_tasks, ["GET"], "Image generation Page tasks"),
            (
                "/page/tasks/<task_id>",
                self.page_task_detail,
                ["GET"],
                "Image generation Page task detail",
            ),
            (
                "/page/tasks/<task_id>/cancel",
                self.page_cancel_task,
                ["POST"],
                "Cancel image generation Page task",
            ),
            (
                "/page/tasks/<task_id>/images/<image_index>/download",
                self.page_download_task_image,
                ["GET"],
                "Download generated image from Page",
            ),
            (
                "/page/tasks/<task_id>/images/<image_index>/preview",
                self.page_preview_task_image,
                ["GET"],
                "Preview generated image from Page",
            ),
            (
                "/page/gallery",
                self.page_gallery,
                ["GET"],
                "Image generation Page gallery",
            ),
            (
                "/page/reference/upload",
                self.page_upload_reference,
                ["POST"],
                "Upload reference image from Page",
            ),
            (
                "/page/generate",
                self.page_generate,
                ["POST"],
                "Submit image generation task from Page",
            ),
            (
                "/page/model",
                self.page_switch_model,
                ["POST"],
                "Switch the active image generation model from Page",
            ),
            (
                "/page/audit/test",
                self.page_test_audit,
                ["POST"],
                "Test safety audit from Page",
            ),
            (
                "/page/tasks/<task_id>/delete",
                self.page_delete_task,
                ["POST"],
                "Delete finished task from Page",
            ),
            (
                "/page/tasks/cleanup",
                self.page_cleanup_tasks,
                ["POST"],
                "Cleanup finished tasks from Page",
            ),
            (
                "/page/tasks/<task_id>/images/<image_index>/delete",
                self.page_delete_task_image,
                ["POST"],
                "Delete one generated image from Page",
            ),
            (
                "/page/tasks/<task_id>/reaudit",
                self.page_reaudit_task,
                ["POST"],
                "Re-audit generated images from Page",
            ),
        ]
        for route, handler, methods, description in page_routes:
            self.plugin.context.register_web_api(
                f"/{PLUGIN_NAME}{route}",
                handler,
                methods,
                description,
            )

    def _available_models(self) -> list[str]:
        """Return every selectable generation model in `provider/model` form.

        Returns:
            Model choices in the same format the Dashboard Page offers.
        """
        models: list[str] = []
        for provider in getattr(
            self.plugin.config_manager, "_all_provider_configs", []
        ):
            provider_name = provider.name.strip()
            for model in provider.available_models:
                models.append(f"{provider_name}/{model}" if provider_name else model)
        return models

    def _model_capabilities(self) -> dict[str, dict[str, Any]]:
        """Return the adapter capabilities declared by each selectable model.

        Returns:
            A mapping of `provider/model` to the capability flags the Page uses
            to enable or disable the per-model form controls, plus the
            negative-prompt delivery mode and NovelAI default dimensions.
        """
        capabilities: dict[str, dict[str, Any]] = {}
        for provider in getattr(
            self.plugin.config_manager, "_all_provider_configs", []
        ):
            try:
                provider_capabilities = create_adapter(provider).get_capabilities()
            except ValueError:
                # Unknown adapter types fail on submit; the Page falls back to
                # the capability flags of the active model in that case.
                continue
            flags = {
                "adapter_type": provider.type.value,
                "supports_image_to_image": bool(
                    provider_capabilities & ImageCapability.IMAGE_TO_IMAGE
                ),
                "supports_aspect_ratio": bool(
                    provider_capabilities & ImageCapability.ASPECT_RATIO
                ),
                "supports_resolution": bool(
                    provider_capabilities & ImageCapability.RESOLUTION
                ),
                "supports_negative_prompt": bool(
                    provider_capabilities & ImageCapability.NEGATIVE_PROMPT
                ),
                "supports_novelai": provider.type == AdapterType.NOVELAI,
                "negative_prompt_mode": (
                    provider.extra.get("negative_prompt_mode", "prompt")
                    if provider.type in {AdapterType.OPENAI, AdapterType.OPENAI_CHAT}
                    else "parameter"
                ),
            }
            if provider.type == AdapterType.NOVELAI:
                flags.update(
                    novelai_width=provider.extra.get("width", 832),
                    novelai_height=provider.extra.get("height", 1216),
                )
            provider_name = provider.name.strip()
            for model in provider.available_models:
                model_key = f"{provider_name}/{model}" if provider_name else model
                capabilities[model_key] = {**flags, "model": model}
        return capabilities

    async def page_state(self):
        """Return plugin state used to initialize the Dashboard Page.

        Returns:
            A JSON response with model, template, capability, and runtime state.
        """
        plugin = self.plugin
        adapter_config = plugin.config_manager.adapter_config
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )
        current_model = (
            f"{adapter_config.name}/{adapter_config.model}" if adapter_config else ""
        )
        return json_response(
            {
                "initialized": bool(plugin.generator and plugin.generator.adapter),
                "has_api_key": plugin.has_required_api_key(),
                "current_model": current_model,
                "available_models": self._available_models(),
                "model_capabilities": self._model_capabilities(),
                "default_image_count": plugin.config_manager.default_image_count,
                "max_image_count": plugin.config_manager.max_image_count,
                "default_aspect_ratio": plugin.config_manager.default_aspect_ratio,
                "default_resolution": plugin.config_manager.default_resolution,
                "supports_image_to_image": bool(
                    capabilities & ImageCapability.IMAGE_TO_IMAGE
                ),
                "supports_aspect_ratio": bool(
                    capabilities & ImageCapability.ASPECT_RATIO
                ),
                "supports_resolution": bool(capabilities & ImageCapability.RESOLUTION),
                "supports_negative_prompt": bool(
                    capabilities & ImageCapability.NEGATIVE_PROMPT
                ),
                "presets": [
                    {
                        "name": name,
                        "summary": safe_log_text(preset.prompt, 120),
                        "has_image": bool(preset.image),
                    }
                    for name, preset in plugin.config_manager.presets.items()
                ],
                "personas": [
                    {
                        "name": name,
                        "summary": safe_log_text(persona.prompt, 120),
                        "has_image": bool(persona.image),
                    }
                    for name, persona in plugin.config_manager.personas.items()
                ],
                "queue": {
                    "can_accept": plugin.task_manager.can_accept_generation_task(),
                    "max_running": plugin.config_manager.max_running_generation_tasks,
                    "max_queued": plugin.config_manager.max_queued_generation_tasks,
                },
                "history": {
                    "enabled": plugin.config_manager.enable_generation_task_history,
                    "limit": plugin.config_manager.generation_task_history_limit,
                    "retention_days": plugin.config_manager.generation_task_history_retention_days,
                },
            }
        )

    async def page_stats(self):
        """Return aggregated dashboard statistics for the overview page.

        Returns:
            A JSON response with summary cards and chart-ready distributions.
        """
        days_raw = str(request.query.get("days", "7") or "7").strip()
        try:
            days = max(0, int(days_raw))
        except ValueError:
            return error_response("days 必须是非负整数", status_code=400)
        return json_response(self._page_stats_payload(days=days))

    async def page_tasks(self):
        """Return generation tasks for the Dashboard Page.

        Returns:
            A JSON response containing filtered and paginated task summaries.
        """
        status_filter = str(request.query.get("status", "all") or "all").strip()
        source_filter = str(request.query.get("source", "") or "").strip().lower()
        model_filter = str(request.query.get("model", "") or "").strip().lower()
        keyword = str(request.query.get("keyword", "") or "").strip().lower()
        days_raw = str(request.query.get("days", "0") or "0").strip()
        try:
            days = max(0, int(days_raw))
        except ValueError:
            return error_response("days 必须是非负整数", status_code=400)
        limit = max(1, min(request.query.get("limit", 30, type=int), 200))
        offset = max(0, request.query.get("offset", 0, type=int))
        since = self._parse_page_datetime(request.query.get("since"))
        until = self._parse_page_datetime(request.query.get("until"))
        if days > 0 and since is None:
            since = datetime.now() - timedelta(days=days)

        # Scan full local history so the queue can page through all saved tasks.
        scan_limit = min(
            5000,
            max(
                self.plugin.config_manager.generation_task_history_limit,
                limit + offset,
                200,
            ),
        )
        records = self.plugin.task_manager.list_generation_tasks(
            include_finished=True,
            limit=scan_limit,
        )

        filtered: list[GenerationTaskRecord] = []
        model_names: set[str] = set()
        source_names: set[str] = set()
        for record in records:
            record_model = str(record.model or "").strip()
            record_source = str(record.source or "").strip()
            if record_model:
                model_names.add(record_model)
            if record_source:
                source_names.add(record_source)

            if status_filter == "active":
                if not record.is_active:
                    continue
            elif status_filter not in {"", "all"} and (
                record.status.value != status_filter
            ):
                continue

            if source_filter and source_filter not in record_source.lower():
                continue
            if model_filter and model_filter not in record_model.lower():
                continue

            record_time = record.finished_at or record.started_at or record.created_at
            if since and record_time and record_time < since:
                continue
            if until and record_time and record_time > until:
                continue

            if keyword:
                haystack = " ".join(
                    [
                        record.task_id,
                        record.prompt_summary or "",
                        record.prompt or "",
                        record.preset or "",
                        record.preset_label or "",
                        record.source or "",
                        record.model or "",
                        record.status.value,
                        record.unified_msg_origin or "",
                        record.message or "",
                        record.error or "",
                    ]
                ).lower()
                if keyword not in haystack:
                    continue
            filtered.append(record)

        total = len(filtered)
        page_records = filtered[offset : offset + limit]
        return json_response(
            {
                "tasks": [self._page_task_payload(record) for record in page_records],
                "total": total,
                "offset": offset,
                "limit": limit,
                "models": sorted(model_names),
                "sources": sorted(source_names),
            }
        )

    async def page_task_detail(self, task_id: str):
        """Return one generation task detail for the Dashboard Page.

        Args:
            task_id: Generation task ID from the route.

        Returns:
            A JSON response with task detail or an error response.
        """
        normalized_task_id = str(task_id or "").strip()
        record = self.plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return error_response(f"任务不存在: {normalized_task_id}", status_code=404)
        return json_response(
            {"task": self._page_task_payload(record, include_detail=True)}
        )

    async def page_cancel_task(self, task_id: str):
        """Cancel an active generation task from the Dashboard Page.

        Args:
            task_id: Generation task ID from the route.

        Returns:
            A JSON response with operation status.
        """
        normalized_task_id = str(task_id or "").strip()
        ok, message = self.plugin.task_manager.cancel_generation_task(
            normalized_task_id
        )
        status_code = 200 if ok else 400
        return json_response(
            {"ok": ok, "message": message, "task_id": normalized_task_id},
            status_code=status_code,
        )

    async def page_download_task_image(self, task_id: str, image_index: str):
        """Download one generated image file through a safe Page endpoint.

        Args:
            task_id: Generation task ID from the route.
            image_index: 1-based image index in the task result list.

        Returns:
            A file response or an error response.
        """
        normalized_task_id = str(task_id or "").strip()
        record = self.plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return error_response(f"任务不存在: {normalized_task_id}", status_code=404)
        path = self._result_image_path(record, image_index)
        if not path:
            return error_response("图片不存在或不可访问", status_code=404)
        return file_response(path, filename=path.name)

    async def page_preview_task_image(self, task_id: str, image_index: str):
        """Return a small inline preview payload for one generated image.

        Args:
            task_id: Generation task ID from the route.
            image_index: 1-based image index in the task result list.

        Returns:
            A JSON response with a data URL preview, or an error response.
        """
        normalized_task_id = str(task_id or "").strip()
        record = self.plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return error_response(f"任务不存在: {normalized_task_id}", status_code=404)
        path = self._result_image_path(record, image_index)
        if not path:
            return error_response("图片不存在或不可访问", status_code=404)
        size = path.stat().st_size
        if size > PAGE_PREVIEW_MAX_BYTES:
            return error_response(
                "图片过大，请改用下载查看",
                status_code=413,
                data={
                    "download_endpoint": (
                        f"page/tasks/{normalized_task_id}/images/{image_index}/download"
                    ),
                    "size": size,
                },
            )

        # Previews cross the iframe bridge as base64 strings, so shipping the
        # original 2K/4K file stalls both the dashboard and the page. Downscale
        # to a WebP that still fills the lightbox; downloads keep the original.
        def encode_preview() -> tuple[bytes, str]:
            try:
                with Image.open(path) as img:
                    img.thumbnail((PAGE_PREVIEW_MAX_SIDE, PAGE_PREVIEW_MAX_SIDE))
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGBA")
                    output = BytesIO()
                    img.save(output, format="WEBP", quality=85)
                    return output.getvalue(), "image/webp"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"{LOG} 预览图压缩失败，回退原图: {safe_log_text(str(exc))}"
                )
                return path.read_bytes(), self._page_image_mime_type(path)

        data, mime_type = await asyncio.to_thread(encode_preview)
        return json_response(
            {
                "task_id": normalized_task_id,
                "image_index": int(str(image_index).strip()),
                "filename": path.name,
                "mime_type": mime_type,
                "size": size,
                "data_url": (
                    f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
                ),
            }
        )

    async def page_gallery(self):
        """Return generated image gallery items from task history.

        Returns:
            A JSON response containing filtered gallery image entries.
        """
        status_filter = str(request.query.get("status", "all") or "all").strip()
        model_filter = str(request.query.get("model", "") or "").strip().lower()
        keyword = str(request.query.get("keyword", "") or "").strip().lower()
        days_raw = str(request.query.get("days", "0") or "0").strip()
        try:
            days = max(0, int(days_raw))
        except ValueError:
            return error_response("days 必须是非负整数", status_code=400)
        limit = max(1, min(request.query.get("limit", 60, type=int), 200))
        offset = max(0, request.query.get("offset", 0, type=int))
        available_only = str(
            request.query.get("available_only", "") or ""
        ).strip().lower() not in {"", "0", "false", "no"}
        since = self._parse_page_datetime(request.query.get("since"))
        until = self._parse_page_datetime(request.query.get("until"))
        if days > 0 and since is None:
            since = datetime.now() - timedelta(days=days)

        # Scan more task records than page size because one task can yield
        # multiple gallery images.
        scan_limit = min(1000, max((limit + offset) * 5, 200))
        records = self.plugin.task_manager.list_generation_tasks(
            include_finished=True,
            limit=scan_limit,
        )
        items: list[dict[str, Any]] = []
        model_names: set[str] = set()
        for record in records:
            record_model = str(record.model or "").strip()
            if record_model:
                model_names.add(record_model)
            if (
                status_filter not in {"", "all"}
                and record.status.value != status_filter
            ):
                continue
            if model_filter and model_filter not in record_model.lower():
                continue
            record_time = record.finished_at or record.started_at or record.created_at
            if since and record_time and record_time < since:
                continue
            if until and record_time and record_time > until:
                continue
            if keyword:
                haystack = " ".join(
                    [
                        record.task_id,
                        record.prompt_summary or "",
                        record.preset or "",
                        record.source or "",
                        record.model or "",
                        record.status.value,
                    ]
                ).lower()
                if keyword not in haystack:
                    continue
            items.extend(self._gallery_items_for_record(record))

        hidden_count = 0
        if available_only:
            hidden_count = sum(1 for item in items if not item.get("available"))
            items = [item for item in items if item.get("available")]

        total = len(items)
        page_items = items[offset : offset + limit]
        return json_response(
            {
                "items": page_items,
                "total": total,
                "offset": offset,
                "limit": limit,
                "available_count": sum(1 for item in items if item.get("available")),
                "hidden_count": hidden_count,
                "models": sorted(model_names),
            }
        )

    def _page_allowed_image_roots(self) -> list[Path]:
        """Return directories that Page-managed result files must live in."""
        return [
            self.plugin.image_temp_dir.resolve(),
            self.plugin.persisted_image_dir.resolve(),
            self.plugin.astrbot_temp_dir.resolve(),
        ]

    def _delete_result_files(self, paths: list[str]) -> int:
        """Delete result files that live under the allowed image roots."""
        removed = 0
        roots = self._page_allowed_image_roots()
        for path_value in paths:
            try:
                path = Path(path_value).resolve()
            except OSError:
                continue
            if not any(self._path_is_under(path, root) for root in roots):
                continue
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                logger.warning(
                    f"{LOG} 删除结果图片失败: {safe_log_text(str(path), 160)}"
                )
        return removed

    @staticmethod
    def _path_is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    async def page_delete_task(self, task_id: str):
        """Delete one finished task record and optionally its result files.

        Returns:
            A JSON response with the number of removed files.
        """
        payload = await request.json(default={})
        delete_files = True
        if isinstance(payload, dict) and "delete_files" in payload:
            delete_files = bool(payload.get("delete_files"))
        record = self.plugin.task_manager.get_generation_task(task_id)
        if not record:
            return error_response("任务不存在", status_code=404)
        if record.is_active:
            return error_response("任务仍在进行中，请先取消", status_code=400)
        removed_record = self.plugin.task_manager.delete_generation_task(task_id)
        if not removed_record:
            return error_response("任务删除失败", status_code=400)
        files_removed = (
            self._delete_result_files(removed_record.result_paths)
            if delete_files
            else 0
        )
        return json_response(
            {"ok": True, "task_id": task_id, "files_removed": files_removed}
        )

    async def page_cleanup_tasks(self):
        """Cleanup finished task records older than N days (0 = all finished).

        Returns:
            A JSON response with removed record and file counts.
        """
        payload = await request.json(default={})
        days = 0
        delete_files = True
        if isinstance(payload, dict):
            try:
                days = max(0, int(payload.get("days") or 0))
            except (TypeError, ValueError):
                days = 0
            if "delete_files" in payload:
                delete_files = bool(payload.get("delete_files"))
        cutoff = datetime.now() - timedelta(days=days) if days > 0 else None

        candidates = self.plugin.task_manager.list_generation_tasks(limit=1_000_000)
        removed = 0
        files_removed = 0
        for record in candidates:
            if record.is_active:
                continue
            if cutoff is not None:
                finished_at = (
                    record.finished_at or record.started_at or record.created_at
                )
                if finished_at >= cutoff:
                    continue
            deleted = self.plugin.task_manager.delete_generation_task(record.task_id)
            if not deleted:
                continue
            removed += 1
            if delete_files:
                files_removed += self._delete_result_files(deleted.result_paths)
        return json_response(
            {"ok": True, "removed": removed, "files_removed": files_removed}
        )

    async def page_delete_task_image(self, task_id: str, image_index: str):
        """Delete one generated image file and drop it from the task record.

        Returns:
            A JSON response with the remaining image count.
        """
        record = self.plugin.task_manager.get_generation_task(task_id)
        if not record:
            return error_response("任务不存在", status_code=404)
        if record.is_active:
            return error_response("任务仍在进行中，请先取消", status_code=400)
        try:
            index = int(str(image_index).strip())
        except (TypeError, ValueError):
            return error_response("图片序号无效", status_code=400)
        if index < 1 or index > len(record.result_paths):
            return error_response("图片序号无效", status_code=400)
        path_value = record.result_paths[index - 1]
        files_removed = self._delete_result_files([path_value])
        removed_path = self.plugin.task_manager.remove_generation_task_image(
            task_id, index
        )
        if removed_path is None:
            return error_response("图片删除失败", status_code=400)
        return json_response(
            {
                "ok": True,
                "task_id": task_id,
                "files_removed": files_removed,
                "remaining": len(record.result_paths),
            }
        )

    async def page_reaudit_task(self, task_id: str):
        """Re-run the image audit for a task's generated images.

        The verdict is returned for display only and never overwrites the
        audit history recorded when the task originally ran.

        Returns:
            A JSON response with the fresh audit verdict and details.
        """
        plugin = self.plugin
        record = plugin.task_manager.get_generation_task(task_id)
        if not record:
            return error_response("任务不存在", status_code=404)

        payload = await request.json(default={})
        image_index: int | None = None
        if isinstance(payload, dict) and payload.get("image_index") is not None:
            try:
                image_index = int(payload.get("image_index"))
            except (TypeError, ValueError):
                return error_response("图片序号无效", status_code=400)

        if image_index is not None:
            if image_index < 1 or image_index > len(record.result_paths):
                return error_response("图片序号无效", status_code=400)
            candidate_paths = [record.result_paths[image_index - 1]]
        else:
            candidate_paths = list(record.result_paths)

        roots = self._page_allowed_image_roots()
        image_paths: list[str] = []
        for path_value in candidate_paths:
            try:
                path = Path(path_value).resolve()
            except OSError:
                continue
            if (
                any(self._path_is_under(path, root) for root in roots)
                and path.is_file()
            ):
                image_paths.append(str(path))
        if not image_paths:
            return error_response("没有可复审的图片文件", status_code=400)

        settings = plugin.config_manager.safety_audit_settings
        (
            allowed,
            reason,
            audit_results,
        ) = await plugin.safety_auditor.audit_generated_images(
            prompt=record.prompt or record.prompt_summary or "",
            image_paths=image_paths,
            unified_msg_origin="webui:audit-test",
        )
        return json_response(
            {
                "ok": True,
                "task_id": task_id,
                "audited_count": len(image_paths),
                "enabled": {
                    "image_moderation": settings.image_audit.enable_moderation_audit,
                    "image_ai": settings.image_audit.enable_ai_audit,
                },
                "image_audit": {
                    "allowed": allowed,
                    "reason": reason or ("审核通过" if allowed else "审核未通过"),
                    "audit_results": audit_results,
                },
            }
        )

    async def page_test_audit(self):
        """Run prompt / image audits against current config for admin testing.

        Returns:
            A JSON response with audit verdicts and per-image audit details.
        """
        plugin = self.plugin
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        prompt = str(payload.get("prompt") or "").strip()
        upload_tokens = payload.get("reference_tokens") or []
        image_paths = [
            str(path)
            for token in upload_tokens
            if (path := self._uploaded_image_path(str(token)))
        ]
        if not prompt and not image_paths:
            return error_response("请至少提供测试提示词或测试图片", status_code=400)

        settings = plugin.config_manager.safety_audit_settings
        # Fixed test origin: never whitelisted by accident, so audits really run.
        test_origin = "webui:audit-test"
        response: dict[str, Any] = {
            "enabled": {
                "prompt_blocked_words": bool(settings.prompt_audit.blocked_words),
                "prompt_ai": settings.prompt_audit.enable_ai_audit,
                "image_moderation": settings.image_audit.enable_moderation_audit,
                "image_ai": settings.image_audit.enable_ai_audit,
            },
            "prompt_audit": None,
            "image_audit": None,
        }
        if prompt:
            allowed, reason = await plugin.safety_auditor.audit_prompt(
                prompt, test_origin
            )
            response["prompt_audit"] = {
                "allowed": allowed,
                "reason": reason or ("审核通过" if allowed else "审核未通过"),
            }
        if image_paths:
            (
                allowed,
                reason,
                audit_results,
            ) = await plugin.safety_auditor.audit_generated_images(
                prompt=prompt,
                image_paths=image_paths,
                unified_msg_origin=test_origin,
            )
            response["image_audit"] = {
                "allowed": allowed,
                "reason": reason or ("审核通过" if allowed else "审核未通过"),
                "audit_results": audit_results,
            }
        return json_response(response)

    async def page_upload_reference(self):
        """Upload and validate one reference image for Page generation.

        Returns:
            A JSON response containing an opaque upload token.
        """
        files = await request.files()
        upload = files.get("file")
        if not isinstance(upload, PluginUploadFile):
            return error_response("缺少上传文件", status_code=400)
        data = await upload.read()
        max_bytes = (
            self.plugin.config_manager.usage_settings.max_image_size_mb * 1024 * 1024
        )
        if not data:
            return error_response("上传文件为空", status_code=400)
        if len(data) > max_bytes:
            return error_response(
                f"参考图超过大小限制 ({self.plugin.config_manager.usage_settings.max_image_size_mb}MB)",
                status_code=400,
            )
        image_data = self.plugin.image_processor.validate_image_data(
            data,
            log_source=upload.filename or "Page upload",
        )
        if not image_data:
            return error_response("上传文件不是受支持的图片", status_code=400)
        suffix = GENERATED_IMAGE_EXTENSIONS.get(image_data.mime_type, ".png")
        digest = hashlib.md5(data).hexdigest()[:12]
        token = (
            f"upload_{int(asyncio.get_running_loop().time() * 1000)}_{digest}{suffix}"
        )
        target = (self.plugin.page_upload_dir / token).resolve()
        try:
            target.relative_to(self.plugin.page_upload_dir.resolve())
        except ValueError:
            return error_response("上传路径无效", status_code=400)
        self.plugin.page_upload_dir.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            output.write(data)
        return json_response(
            {
                "token": token,
                "filename": Path(upload.filename or token).name,
                "mime_type": image_data.mime_type,
                "size": len(data),
            }
        )

    def _parse_novelai_page_options(self, payload: dict[str, Any]) -> dict | None:
        """Resolve the page's typed NovelAI options and role-specific upload tokens.

        Args:
            payload: JSON generation request from the page.

        Returns:
            NovelAI options with safe local image paths, or no overrides.

        Raises:
            ValueError: Options, asset metadata, or an upload token is invalid.
        """
        options = payload.get("novelai")
        if options is not None and not isinstance(options, dict):
            raise ValueError("NovelAI 高级参数必须是 JSON 对象或 null")
        serialized = json.dumps(options or {}, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > 24 * 1024 * 1024:
            raise ValueError("NovelAI 高级参数总大小不能超过 24 MiB")
        options = json.loads(serialized)
        for key in ("i2i", "inpaint"):
            if key in options and not isinstance(options[key], dict):
                raise ValueError(f"NovelAI {key} 必须是 JSON 对象")
        for key, maximum in (
            ("characters", 6),
            ("character_references", 16),
            ("vibe_references", 16),
        ):
            if key not in options:
                continue
            entries = options[key]
            if (
                not isinstance(entries, list)
                or len(entries) > maximum
                or any(not isinstance(entry, dict) for entry in entries)
            ):
                raise ValueError(f"NovelAI {key} 必须是最多 {maximum} 项的对象数组")
        action = options.get("action")
        if action is not None and not isinstance(action, str):
            raise ValueError("NovelAI action 必须是 generate、img2img 或 infill")
        mask_token = payload.get("novelai_mask_token")
        if mask_token is not None:
            if not isinstance(mask_token, str) or not mask_token:
                raise ValueError("局部重绘蒙版 token 必须是非空字符串")
            if "i2i" in options or action in {"generate", "img2img"}:
                raise ValueError("当前生成模式不能使用局部重绘蒙版")
            mask_path = self._uploaded_image_path(mask_token)
            if not mask_path:
                raise ValueError("局部重绘蒙版不存在或已失效，请重新上传")
            options.setdefault("inpaint", {})["mask"] = str(mask_path)
        uploaded_references = payload.get("novelai_character_references")
        if uploaded_references is not None:
            if not isinstance(uploaded_references, list) or any(
                not isinstance(entry, dict) for entry in uploaded_references
            ):
                raise ValueError("角色精确参考必须是对象数组")
            references = options.setdefault("character_references", [])
            if len(references) + len(uploaded_references) > 16:
                raise ValueError("角色精确参考最多允许 16 项")
            for entry in uploaded_references:
                token = entry.get("token")
                if not isinstance(token, str) or not token:
                    raise ValueError("角色精确参考 token 必须是非空字符串")
                image_path = self._uploaded_image_path(token)
                if not image_path:
                    raise ValueError("角色精确参考不存在或已失效，请重新上传")
                references.append(
                    {
                        "image": str(image_path),
                        "type": entry.get("type", "character"),
                        "strength": entry.get("strength", 1.0),
                        "fidelity": entry.get("fidelity", 1.0),
                    }
                )
        return validate_novelai_options(options) or None

    async def page_generate(self):
        """Submit one image generation task from the Dashboard Page.

        Returns:
            A JSON response with submitted task information.
        """
        plugin = self.plugin
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        prompt = str(payload.get("prompt") or "").strip()
        negative_prompt = payload.get("negative_prompt")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            return error_response("反向提示词必须是字符串或 null", status_code=400)
        try:
            novelai = self._parse_novelai_page_options(payload)
        except (TypeError, ValueError) as exc:
            return error_response(str(exc), status_code=400)
        image_count = plugin.normalize_image_count(payload.get("image_count"))
        aspect_ratio = str(
            payload.get("aspect_ratio") or plugin.config_manager.default_aspect_ratio
        ).strip()
        resolution = str(
            payload.get("resolution") or plugin.config_manager.default_resolution
        ).strip()
        model = str(payload.get("model") or "").strip()
        presets = payload.get("presets") or None
        personas = payload.get("personas") or None
        # Validate against the model picked in the form, and only switch the
        # active model once the request is accepted, so rejected input never
        # changes the persisted model selection.
        adapter_config = plugin.config_manager.adapter_config
        current_model = (
            f"{adapter_config.name}/{adapter_config.model}" if adapter_config else ""
        )
        switch_model = bool(model) and model != current_model
        if switch_model:
            adapter_config = next(
                (
                    provider
                    for provider in getattr(
                        plugin.config_manager, "_all_provider_configs", []
                    )
                    if any(
                        model
                        == (
                            f"{provider.name.strip()}/{candidate}"
                            if provider.name.strip()
                            else candidate
                        )
                        for candidate in provider.available_models
                    )
                ),
                None,
            )
            if not adapter_config:
                return error_response(f"未知模型: {model}", status_code=400)
        is_novelai = bool(adapter_config and adapter_config.type == AdapterType.NOVELAI)
        if novelai and not is_novelai:
            return error_response("当前模型不支持 NovelAI 高级参数", status_code=400)
        upload_tokens = payload.get("reference_tokens")
        if upload_tokens is None:
            upload_tokens = []
        if is_novelai:
            # NovelAI uses one explicit source image; stale tokens must fail loudly
            # instead of silently turning an edit into text-to-image.
            if not isinstance(upload_tokens, list) or any(
                not isinstance(token, str) or not token for token in upload_tokens
            ):
                return error_response("源图 token 必须是字符串数组", status_code=400)
            reference_sources = []
            for token in upload_tokens:
                path = self._uploaded_image_path(token)
                if not path:
                    return error_response(
                        "源图不存在或已失效，请重新上传", status_code=400
                    )
                reference_sources.append(str(path))
            if novelai and novelai.get("action") == "generate":
                reference_sources = []
            if len(reference_sources) > 1:
                return error_response("NovelAI 最多使用一张源图", status_code=400)
        else:
            reference_sources = [
                str(path)
                for token in upload_tokens
                if (path := self._uploaded_image_path(str(token)))
            ]

        image_size = None
        has_width = "width" in payload
        has_height = "height" in payload
        if has_width != has_height:
            return error_response("图片宽度和高度必须同时指定", status_code=400)
        if (has_width or has_height) and not is_novelai:
            return error_response("仅 NovelAI 支持指定像素宽高", status_code=400)
        if is_novelai:
            raw_dimensions = (
                payload.get("width", adapter_config.extra.get("width", 832)),
                payload.get("height", adapter_config.extra.get("height", 1216)),
            )
            dimensions: list[int] = []
            try:
                for raw in raw_dimensions:
                    value = float(raw)
                    if (
                        isinstance(raw, bool)
                        or not math.isfinite(value)
                        or not value.is_integer()
                    ):
                        raise ValueError("Image dimensions must be finite integers")
                    dimensions.append(int(value))
            except (TypeError, ValueError, OverflowError):
                return error_response("图片宽高必须为正整数", status_code=400)
            width, height = dimensions
            if (
                not all(64 <= value <= 2048 and value % 64 == 0 for value in dimensions)
                or width * height > 3_145_728
            ):
                return error_response(
                    "图片宽高必须是 64–2048 内的 64 倍数，且总面积不超过 3,145,728 像素",
                    status_code=400,
                )
            image_size = (width, height)

        if switch_model:
            await plugin.switch_generation_model(model)

        # Page generation runs from Dashboard auth context. Prefer the signed-in
        # WebUI username so overview statistics can attribute tasks correctly.
        webui_username = str(getattr(request, "username", None) or "").strip()
        page_user_origin = (
            f"webui:{webui_username}" if webui_username else "webui:admin"
        )

        result = await plugin.public_api.submit_generation_task(
            prompt=prompt,
            negative_prompt=negative_prompt,
            novelai=novelai,
            source="Page",
            unified_msg_origin=page_user_origin,
            image_count=image_count,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_size=image_size,
            reference_image_sources=reference_sources,
            presets=presets,
            personas=personas,
            is_admin=True,
        )
        if not result.ok:
            return error_response(
                result.message,
                status_code=400,
                data={"code": result.code, "error": result.error},
            )
        record = plugin.task_manager.get_generation_task(result.task_id or "")
        return json_response(
            {
                "ok": True,
                "message": result.message,
                "task_id": result.task_id,
                "task": self._page_task_payload(record) if record else None,
            }
        )

    async def page_switch_model(self):
        """Switch the active generation model from the Dashboard Page.

        Persists the chosen model and reloads the runtime helpers so later
        requests use it, without submitting a generation task.

        Returns:
            A JSON response with the model now in use, or an error response.
        """
        plugin = self.plugin
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        model = str(payload.get("model") or "").strip()
        if not model:
            return error_response("缺少 model 参数", status_code=400)

        adapter_config = plugin.config_manager.adapter_config
        if model not in self._available_models():
            return error_response(f"未知模型: {model}", status_code=400)

        current_model = (
            f"{adapter_config.name}/{adapter_config.model}" if adapter_config else ""
        )
        if model != current_model:
            await plugin.switch_generation_model(model)
        return json_response(
            {
                "current_model": model,
                "switched": model != current_model,
            }
        )
