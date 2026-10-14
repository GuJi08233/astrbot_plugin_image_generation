"""LLM tool task result handling for image generation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.star import Context
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.db.po import Conversation
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import ProviderRequest

from ..config.manager import ConfigManager
from ..formatting.result import format_generation_failure_message
from ..tasks.models import GenerationTaskRecord, GenerationTaskStatus
from ..shared.logging import log_prefix, safe_log_error_body, safe_log_text
from ..tasks.manager import TaskManager
from .tools import DeliverySendMessageToUserTool

IMAGE_GENERATION_TASK_WOKE_SYSTEM_PROMPT = (
    "You are an autonomous proactive agent.\n\n"
    "You are awakened because an image generation task you initiated earlier has "
    "completed. The completed task payload is provided in the user message.\n"
    "# IMPORTANT RULES\n"
    "1. This is NOT a normal chat turn. Do NOT greet the user. Do NOT ask questions "
    "unless strictly necessary.\n"
    "2. Deliver the task result to the user with the `send_message_to_user` tool; "
    "otherwise the user will not see it.\n"
    "3. If the task succeeded, send every generated image exactly once, using "
    "type='image' and the provided local path. Send all images of this task in one "
    "call when possible, and do NOT add plain text components to that call.\n"
    "4. If the task failed, send the failure reason as a single `plain` component.\n"
    "5. Your final reply text is delivered to the user right after the images: write "
    "the user-facing wording there and keep it short. Do not repeat content you "
    "already sent with the tool, and do not write process wording such as 'done'.\n"
    "6. Each generated image must be delivered exactly once: never send the same "
    "image twice, and do not call `generate_image` again for this completed task.\n"
    "7. If generated images are attached in this request, review them before "
    "deciding the final wording. If images are not attached, use the provided paths.\n"
    "# IMAGE GENERATION TASK CONTEXT\n"
    "The completed task payload is provided in the user message."
)


class LLMResultHandler:
    """Handle LLM tool task submission text and task-result AI delivery."""

    def __init__(
        self,
        *,
        context: Context,
        config_manager: ConfigManager,
        task_manager: TaskManager,
        create_background_task: Callable[
            [Coroutine[Any, Any, Any], str | None], asyncio.Task
        ],
    ) -> None:
        self.context = context
        self.config_manager = config_manager
        self.task_manager = task_manager
        self._create_background_task = create_background_task

    def attach_task_wakeup(
        self,
        record: GenerationTaskRecord,
        *,
        source_event: AstrMessageEvent | None,
    ) -> None:
        """Attach an AI wakeup task to a completed LLM image generation task."""
        if not source_event:
            return

        def _on_done(current_record: GenerationTaskRecord) -> None:
            if current_record.status in {
                GenerationTaskStatus.CANCELLING,
                GenerationTaskStatus.CANCELLED,
            }:
                logger.debug(
                    f"{log_prefix('Task', record.task_id)} 生图任务已取消，跳过 AI 结果唤醒"
                )
                return
            self._create_background_task(
                self.wake_ai_for_generation_task_result(
                    task_id=record.task_id,
                    source_event=source_event,
                ),
                f"image_generation_ai_wakeup:{record.task_id}",
            )

        self.task_manager.add_generation_task_done_callback(record.task_id, _on_done)

    def format_tool_start_result(
        self,
        *,
        prompt: str,
        reference_image_count: int,
        preset: str | None,
        preset_label: str = "预设",
        presets: list[str] | None = None,
        personas: list[str] | None = None,
        aspect_ratio: str,
        resolution: str,
        image_count: int,
        task_id: str,
    ) -> str:
        """Format the successful LLM tool result for the agent."""
        model = ""
        if self.config_manager.adapter_config:
            model = (
                f"{self.config_manager.adapter_config.name}/"
                f"{self.config_manager.adapter_config.model}"
            )

        template_summary_parts = []
        preset_names = "、".join(presets or [])
        persona_names = "、".join(personas or [])
        if preset_names:
            template_summary_parts.append(f"预设={preset_names}")
        if persona_names:
            template_summary_parts.append(f"人设={persona_names}")
        if not template_summary_parts and preset:
            template_summary_parts.append(f"{preset_label}={preset}")

        lines = [
            "✅ 生图任务已提交，正在后台执行。",
            f"任务ID: {task_id}",
            "状态: 排队/运行中（尚未完成）",
            f"请求数量: {image_count}；结果图片数以任务完成为准",
            f"模式: {'图生图' if reference_image_count else '文生图'}",
            f"参考图: {reference_image_count}张",
            f"宽高比: {aspect_ratio}，分辨率: {resolution}",
        ]
        if model:
            lines.append(f"模型: {model}")
        if template_summary_parts:
            lines.append("模板: " + "，".join(template_summary_parts))
        if prompt:
            lines.append(f"提示词摘要: {safe_log_text(prompt, 120)}")

        lines.extend(
            [
                "",
                "内部指引:",
                "- 当前仅表示任务已提交，不能向用户声明图片已完成。",
                "- 等待插件唤醒并提供最终结果后，再通过 send_message_to_user 发送图片或失败原因。",
                "- 唤醒后图片调用只放图片组件，面向用户的说明写在最终回复里，不要重复发送。",
                "- 最终结果到达前，除非上下文确有需要，否则不要主动发送额外消息。",
                "- 不要重复调用 generate_image。",
            ]
        )
        return "\n".join(lines)

    def _build_task_result_payload(
        self,
        record: GenerationTaskRecord,
        *,
        supports_image_input: bool,
    ) -> dict[str, Any]:
        """Build structured task result context for the awakened AI."""
        request_stats = record.request_stats
        payload: dict[str, Any] = {
            "task_id": record.task_id,
            "source": record.source,
            "status": record.status.value,
            "status_label": record.status_label,
            "prompt": record.prompt_summary,
            "reference_image_count": record.reference_image_count,
            "requested_count": record.requested_count,
            "current_index": record.current_index,
            "aspect_ratio": record.aspect_ratio,
            "resolution": record.resolution,
            "preset_label": record.preset_label,
            "preset": record.preset or "",
            "result_count": record.result_count,
            "result_paths": record.result_paths,
            "request_stats": request_stats,
            "items": [
                {
                    "index": item.index,
                    "status": item.status.value,
                    "result_count": item.result_count,
                    "error": safe_log_text(item.error, 160) if item.error else "",
                    "retry_attempts": item.retry_attempts,
                    "max_retry_attempts": item.max_retry_attempts,
                }
                for item in sorted(
                    record.items.values(), key=lambda task_item: task_item.index
                )
            ],
            "error": record.error,
            "message": record.message,
            "images_attached_to_model": supports_image_input
            and bool(record.result_paths),
        }
        if record.duration_seconds is not None:
            payload["duration_seconds"] = round(record.duration_seconds, 2)
        return payload

    def _build_fallback_chain(self, record: GenerationTaskRecord) -> MessageChain:
        """Build a direct user-facing fallback message when AI handling fails."""
        chain = MessageChain()
        if record.error:
            chain.message(format_generation_failure_message(record.error))
            return chain

        if not record.result_paths:
            chain.message("❌ 生成失败: 未能获取生成图片")
            return chain

        for file_path in record.result_paths:
            chain.file_image(file_path)
        info_parts = [f"🧾 任务ID: {record.task_id}"]
        if record.result_count:
            info_parts.append(f"🖼️ 数量: {record.result_count}张")
        chain.message("\n" + "\n".join(info_parts))
        return chain

    async def _send_fallback(
        self,
        record: GenerationTaskRecord,
        *,
        reason: str,
    ) -> None:
        """Directly send task result if the awakened AI cannot handle it."""
        logger.warning(
            f"{log_prefix('Task', record.task_id)} AI 处理生图结果失败，改为直接发送: {safe_log_text(reason, 200)}"
        )
        sent = await self.context.send_message(
            record.unified_msg_origin,
            self._build_fallback_chain(record),
        )
        if sent:
            self.task_manager.mark_generation_task_delivered(
                record.task_id,
                paths=record.result_paths,
            )

    def _missing_delivered_images(
        self,
        record: GenerationTaskRecord,
        delivered_paths: list[str],
    ) -> list[str]:
        """Return task result images the awakened AI did not deliver.

        Args:
            record: Completed generation task record.
            delivered_paths: Image paths the AI tool delivered successfully.

        Returns:
            Result image paths that still need to be sent to the user.
        """
        normalized = {
            os.path.normcase(os.path.normpath(path)) for path in delivered_paths
        }
        return [
            path
            for path in record.result_paths
            if os.path.normcase(os.path.normpath(path)) not in normalized
        ]

    async def _send_missing_images(
        self,
        record: GenerationTaskRecord,
        image_paths: list[str],
    ) -> list[str]:
        """Send result images the AI left out, in configured batches.

        Args:
            record: Completed generation task record.
            image_paths: Result image paths still missing from the delivery.

        Returns:
            Image paths that were sent successfully.
        """
        max_per_message = max(1, self.config_manager.max_images_per_message)
        sent_paths: list[str] = []
        for start in range(0, len(image_paths), max_per_message):
            batch = image_paths[start : start + max_per_message]
            chain = MessageChain()
            for file_path in batch:
                chain.file_image(file_path)
            try:
                if await self.context.send_message(record.unified_msg_origin, chain):
                    sent_paths.extend(batch)
            except Exception as exc:
                logger.error(
                    f"{log_prefix('Task', record.task_id)} 补发生成图片失败: "
                    f"{safe_log_error_body(exc, 160)}",
                    exc_info=True,
                )
        if sent_paths:
            logger.info(
                f"{log_prefix('Task', record.task_id)} 已补发 {len(sent_paths)} 张未送达的生成图片"
            )
        return sent_paths

    async def _send_closing_note(
        self,
        record: GenerationTaskRecord,
        text: str,
        *,
        delivered_texts: list[str],
    ) -> None:
        """Send the proactive turn's closing note as a follow-up message.

        Args:
            record: Completed generation task record.
            text: The awakened AI's final reply text.
            delivered_texts: Plain texts already delivered by the tool in this run.
        """
        note = text.strip()
        if not note or note in delivered_texts:
            return
        try:
            if await self.context.send_message(
                record.unified_msg_origin,
                MessageChain().message(note),
            ):
                logger.debug(
                    f"{log_prefix('Task', record.task_id)} 已发送收尾文案: "
                    f"{safe_log_text(note, 80)}"
                )
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', record.task_id)} 发送收尾文案失败: "
                f"{safe_log_error_body(exc, 160)}",
                exc_info=True,
            )

    async def _save_task_note_to_history(
        self,
        *,
        record: GenerationTaskRecord,
        conversation: Conversation | None,
        note: str,
        token_usage: int | None = None,
    ) -> None:
        """Append the proactive turn note to the session conversation history.

        The history is re-read right before appending so a user turn that landed
        while the result was being delivered is not overwritten.

        Args:
            record: Completed generation task record.
            conversation: Conversation bound to the proactive turn.
            note: Assistant note describing the delivered task result.
            token_usage: Optional total token usage of the proactive turn.
        """
        if not conversation:
            return
        conversation_manager = self.context.conversation_manager
        latest = await conversation_manager.get_conversation(
            record.unified_msg_origin,
            conversation.cid,
        )
        try:
            history = json.loads(getattr(latest, "history", None) or "[]")
        except Exception:
            history = []
        if not isinstance(history, list):
            history = []
        history.append({"role": "assistant", "content": note})
        await conversation_manager.update_conversation(
            record.unified_msg_origin,
            conversation.cid,
            history=history,
            token_usage=token_usage,
        )

    async def wake_ai_for_generation_task_result(
        self,
        *,
        task_id: str,
        source_event: AstrMessageEvent,
    ) -> None:
        """Wake the main agent to process a completed image generation task."""
        record = self.task_manager.get_generation_task(task_id)
        if not record:
            logger.warning(
                f"{log_prefix('Task', task_id)} 生图任务记录不存在，无法唤醒 AI"
            )
            return
        if record.status == GenerationTaskStatus.CANCELLED:
            logger.debug(f"{log_prefix('Task', task_id)} 生图任务已取消，不唤醒 AI")
            return
        if record.unified_msg_origin != source_event.unified_msg_origin:
            await self._send_fallback(
                record,
                reason="任务会话与来源事件不一致",
            )
            return

        try:
            await self._run_generation_result_agent(
                record,
                source_event=source_event,
            )
        except Exception as exc:
            logger.error(
                f"{log_prefix('Task', task_id)} 唤醒 AI 处理生图结果异常: {exc}",
                exc_info=True,
            )
            if record.delivered:
                # The AI already delivered the result before the failure.
                logger.warning(
                    f"{log_prefix('Task', task_id)} 结果已发送给用户，跳过兜底发送"
                )
                return
            await self._send_fallback(record, reason=str(exc))

    async def _run_generation_result_agent(
        self,
        record: GenerationTaskRecord,
        *,
        source_event: AstrMessageEvent,
    ) -> None:
        """Run a proactive agent turn with the completed task result."""
        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
            _get_session_conv,
            build_main_agent,
        )

        session = MessageSession.from_str(record.unified_msg_origin)
        cron_event = CronMessageEvent(
            context=self.context,
            session=session,
            message=f"图像生成任务 {record.task_id} 已完成",
            sender_id=(
                str(source_event.get_self_id())
                if hasattr(source_event, "get_self_id")
                else "astrbot"
            ),
            sender_name="ImageGeneration",
            message_type=session.message_type,
        )
        cron_event.role = source_event.role
        cron_event.plugins_name = source_event.plugins_name

        cfg = self.context.get_config(umo=record.unified_msg_origin)
        provider_settings = cfg.get("provider_settings", {})
        tool_call_timeout = provider_settings.get("tool_call_timeout", 120)
        provider = self.context.get_using_provider(record.unified_msg_origin)
        supports_image_input = bool(
            provider
            and "image" in provider.provider_config.get("modalities", [])
            and record.result_paths
        )
        payload = self._build_task_result_payload(
            record,
            supports_image_input=supports_image_input,
        )

        req = ProviderRequest()
        req.conversation = await _get_session_conv(
            event=cron_event,
            plugin_context=self.context,
        )
        # 直接使用真实会话消息作为上下文：把历史压成文本塞进 system prompt 会让
        # system prompt 每轮都变，provider 的前缀缓存无法复用。
        history_context = json.loads(req.conversation.history or "[]")
        if history_context:
            req.contexts = history_context

        req.system_prompt += IMAGE_GENERATION_TASK_WOKE_SYSTEM_PROMPT
        req.prompt = (
            "Proceed according to your system instructions. "
            "Output using same language as previous conversation. "
            "Use `send_message_to_user` to deliver the image generation result now. "
            "If the task succeeded, send every generated image exactly once with "
            "type='image' and the provided path(s), without plain text in the same "
            "call. If it failed, send the failure reason as a single plain text. "
            "Put the user-facing wording in your final reply."
        )
        req.extra_user_content_parts.append(
            TextPart(
                text=(
                    "<image_generation_task_result>\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
                    "</image_generation_task_result>"
                )
            )
        )
        if supports_image_input:
            req.image_urls = list(record.result_paths)
        else:
            for path in record.result_paths:
                req.extra_user_content_parts.append(
                    TextPart(text=f"[Generated image path: {path}]")
                )

        delivery_tool = DeliverySendMessageToUserTool()
        req.func_tool = ToolSet()
        req.func_tool.add_tool(delivery_tool)

        config = MainAgentBuildConfig(
            tool_call_timeout=tool_call_timeout,
            llm_safety_mode=False,
            streaming_response=False,
            provider_settings=provider_settings,
            computer_use_runtime="none",
            add_cron_tools=False,
        )
        result = await build_main_agent(
            event=cron_event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )
        if not result:
            await self._send_fallback(record, reason="无法构建主 Agent")
            return

        # build_main_agent merges default tools from persona/plugin settings.
        # Trim tools before reset so this proactive turn can only deliver results.
        result.provider_request.func_tool = ToolSet()
        result.provider_request.func_tool.add_tool(delivery_tool)
        if result.reset_coro:
            await result.reset_coro

        runner = result.agent_runner
        closing_text = ""
        async for resp in runner.step_until_done(30):
            # 投递由工具完成；这里让本轮完整跑完，并接住模型的最终回复作为随图文案
            if resp.type != "llm_result":
                continue
            chain = resp.data.get("chain") if isinstance(resp.data, dict) else None
            text = chain.get_plain_text().strip() if chain else ""
            if text:
                closing_text = text

        llm_resp = runner.get_final_llm_resp()
        # 投递是否发生以投递工具自身的记录为准（它只在核心工具真正发送成功后写入）。
        if not delivery_tool.delivered:
            await self._send_fallback(
                record,
                reason="AI 未通过 send_message_to_user 发送结果",
            )
        else:
            missing_paths = self._missing_delivered_images(
                record,
                delivery_tool.delivered_paths,
            )
            resent_paths = (
                await self._send_missing_images(record, missing_paths)
                if missing_paths
                else []
            )
            self.task_manager.mark_generation_task_delivered(
                record.task_id,
                paths=[*delivery_tool.delivered_paths, *resent_paths],
            )

        # 图片投递完成后再补一句文案，形成「先图片、后文案」的两条消息
        await self._send_closing_note(
            record,
            closing_text,
            delivered_texts=delivery_tool.delivered_texts,
        )

        summary_note = (
            f"[ImageGenerationTask] task_id={record.task_id}, "
            f"status={record.status.value}, result_count={record.result_count}, "
            f"delivered={str(record.delivered).lower()}, "
            f"error={record.error or ''}"
        )
        if record.delivered:
            summary_note += (
                ", delivered_paths="
                f"{json.dumps(record.delivered_paths, ensure_ascii=False)}. "
                "This task result was already sent to the user; do not send it again."
            )
        if llm_resp and llm_resp.completion_text:
            summary_note += f" AI response: {llm_resp.completion_text}"
        with suppress(Exception):
            await self._save_task_note_to_history(
                record=record,
                conversation=req.conversation,
                note=summary_note,
                token_usage=llm_resp.usage.total
                if llm_resp and llm_resp.usage
                else None,
            )
