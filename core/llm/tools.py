"""LLM callable image generation tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from ..config.templates import (
    build_generation_prompt,
    extract_templates_from_prompt,
    find_named_entry,
    format_template_summary,
    parse_preset_prompt,
)
from ..config.templates import (
    normalize_name_items as _normalize_name_items,
)
from ..generation.novelai_assets import prepare_novelai_assets
from ..generation.novelai_options import (
    NOVELAI_NOISE_SCHEDULES,
    NOVELAI_SAMPLERS,
    validate_novelai_options,
)
from ..generation.reference_collector import (
    collect_tool_reference_images,
)
from ..generation.reference_collector import (
    normalize_string_items as _normalize_string_items,
)
from ..shared.constants import SUPPORTED_ASPECT_RATIOS, SUPPORTED_RESOLUTIONS
from ..shared.logging import (
    log_prefix,
    mask_sensitive,
    safe_log_error_body,
    safe_log_text,
)
from ..shared.types import ImageCapability
from ..tasks.ids import new_task_id
from ..tasks.models import GenerationTaskCreationError

ASPECT_RATIO_OPTIONS = list(SUPPORTED_ASPECT_RATIOS)
RESOLUTION_OPTIONS = list(SUPPORTED_RESOLUTIONS)
LOG = log_prefix("LLMTool")


def extract_event(context: ContextWrapper[AstrAgentContext] | dict[str, Any]) -> Any:
    """Extract AstrBot message event from different tool context wrappers."""
    wrapped_context = getattr(context, "context", None)
    if event := getattr(wrapped_context, "event", None):
        return event
    if event := getattr(context, "event", None):
        return event
    if isinstance(context, dict):
        return context.get("event")
    return None


def normalize_string_items(raw: Any) -> list[str]:
    """Normalize one or many string-like values from tool arguments."""
    return _normalize_string_items(raw)


def normalize_name_items(raw: Any) -> list[str]:
    """Normalize one or many preset/persona names from tool arguments."""
    return _normalize_name_items(raw)


def normalize_preset_query_category(category: str) -> str:
    """Normalize preset query category aliases."""
    normalized = category.strip().lower()
    if normalized in {
        "人设",
        "persona",
        "personas",
        "list_persona",
        "list_personas",
        "get_persona",
        "persona_list",
    }:
        return "persona"
    return "preset"


def normalize_preset_edit_action(action: str) -> str:
    """Normalize preset edit action aliases."""
    normalized = action.strip().lower()
    if normalized in {"添加", "新增", "保存", "save", "create", "add", "add_preset"}:
        return "create_preset"
    if normalized in {
        "删除",
        "移除",
        "remove",
        "del",
        "delete",
        "delete_preset",
    }:
        return "delete_preset"
    return normalized


def normalize_task_action(action: str) -> str:
    """Normalize image task management action aliases."""
    normalized = action.strip().lower()
    if normalized in {"", "列表", "查看列表", "list", "list_tasks", "tasks"}:
        return "list"
    if normalized in {"详情", "查看", "查询", "detail", "get", "show"}:
        return "detail"
    if normalized in {"取消", "cancel", "cancel_task"}:
        return "cancel"
    return normalized


def _parse_preset(
    plugin: Any,
    preset_names: Any,
    aspect_ratio: Any,
    resolution: Any,
) -> tuple[list[str], str, str, list[str], str | None]:
    """Apply one or more preset prompts and optional generation overrides."""
    names = normalize_name_items(preset_names)
    if not names:
        return [], str(aspect_ratio), str(resolution), [], None

    prompt_parts: list[str] = []
    matched_presets: list[str] = []
    for preset_name in names:
        matched_preset = find_named_entry(plugin.config_manager.presets, preset_name)
        if not matched_preset:
            return (
                [],
                str(aspect_ratio),
                str(resolution),
                [],
                f"❌ 预设不存在: {preset_name}",
            )

        preset_prompt, aspect_ratio, resolution = parse_preset_prompt(
            plugin.config_manager.presets[matched_preset],
            str(aspect_ratio),
            str(resolution),
        )

        if preset_prompt:
            prompt_parts.append(preset_prompt)
        matched_presets.append(matched_preset)

    return prompt_parts, str(aspect_ratio), str(resolution), matched_presets, None


def _parse_persona(
    plugin: Any,
    persona_names: Any,
) -> tuple[list[str], list[tuple[str, str]], list[str], str | None]:
    """Apply one or more persona prompts and reference images."""
    names = normalize_name_items(persona_names)
    if not names:
        return [], [], [], None

    prompt_parts: list[str] = []
    persona_images: list[tuple[str, str]] = []
    matched_personas: list[str] = []
    for persona_name in names:
        matched_persona = find_named_entry(
            plugin.config_manager.personas,
            persona_name,
        )
        if not matched_persona:
            return [], [], [], f"❌ 人设不存在: {persona_name}"

        persona = plugin.config_manager.personas[matched_persona]
        persona_prompt = persona.prompt.strip()
        if persona_prompt:
            prompt_parts.append(persona_prompt)
        if persona.image:
            persona_images.append((matched_persona, persona.image))
        matched_personas.append(matched_persona)

    return prompt_parts, persona_images, matched_personas, None


async def _start_generation_task(
    plugin: Any,
    event: Any,
    *,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    reference_images: Any = None,
    avatar_references: Any = None,
    persona_images: list[tuple[str, str]] | None = None,
    preset_or_persona: str | None = None,
    preset_label: str = "预设",
    presets: list[str] | None = None,
    personas: list[str] | None = None,
    image_count: int = 1,
    negative_prompt: str | None = None,
    novelai: dict[str, Any] | None = None,
) -> ToolExecResult:
    """Validate request, collect references, and schedule image generation.

    Args:
        plugin: Image generation plugin instance.
        event: Message event identifying the requesting session.
        prompt: Final positive prompt.
        aspect_ratio: Requested image aspect ratio.
        resolution: Requested image resolution.
        reference_images: Optional image references supplied by the tool.
        avatar_references: Optional user avatar references.
        persona_images: Optional persona reference images.
        preset_or_persona: Summary of selected templates.
        preset_label: Label describing the template summary.
        presets: Names of selected presets.
        personas: Names of selected personas.
        image_count: Requested number of generated images.
        negative_prompt: Per-request override; None inherits the provider
            default, while an empty string clears it for this request.
        novelai: Optional NovelAI sampling, character and image-role settings.

    Returns:
        The accepted task description or a validation error.
    """
    if not plugin.generator or not plugin.generator.adapter:
        return "❌ 生图生成器未初始化"

    if negative_prompt is not None:
        if not isinstance(negative_prompt, str):
            return "❌ 反向提示词必须为字符串"
        if not (
            plugin.generator.adapter.get_capabilities()
            & ImageCapability.NEGATIVE_PROMPT
        ):
            return "❌ 当前生图接口不支持反向提示词输入"

    if novelai is not None and not (
        plugin.generator.adapter.get_capabilities() & ImageCapability.NOVELAI_OPTIONS
    ):
        return "❌ 当前生图接口不支持 NovelAI 专属参数"

    image_count = plugin.normalize_image_count(image_count)
    is_usage_limit_admin = plugin.is_usage_limit_admin(event)
    if not plugin.has_required_api_key():
        masked_uid = mask_sensitive(event.unified_msg_origin)
        logger.debug(f"{LOG} 工具调用失败: 未配置 API Key (用户: {masked_uid})")
        return "❌ 未配置 API Key，无法生成图片"

    check_result = plugin.usage_manager.check_rate_limit(
        event.unified_msg_origin,
        is_admin=is_usage_limit_admin,
        requested_count=image_count,
        update_timestamp=False,
    )
    if isinstance(check_result, str):
        if check_result:
            masked_uid = mask_sensitive(event.unified_msg_origin)
            logger.info(f"{LOG} 工具调用触发限制: {check_result} (用户: {masked_uid})")
        return check_result

    audit_prompt = prompt
    if novelai:
        character_prompts = [
            character["prompt"]
            for character in novelai.get("characters", [])
            if character.get("enabled", True)
        ]
        audit_prompt = "\n".join([prompt, *character_prompts])
    prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
        audit_prompt,
        event.unified_msg_origin,
    )
    if not prompt_allowed:
        return f"❌ 提示词审核未通过: {prompt_reason}"

    if rejection := plugin.task_manager.get_generation_queue_rejection():
        code, message = rejection
        return f"❌ 生图任务提交失败: {message} ({code})"

    check_result = plugin.usage_manager.check_rate_limit(
        event.unified_msg_origin,
        is_admin=is_usage_limit_admin,
        requested_count=image_count,
    )
    if isinstance(check_result, str):
        if check_result:
            masked_uid = mask_sensitive(event.unified_msg_origin)
            logger.info(f"{LOG} 工具调用触发限制: {check_result} (用户: {masked_uid})")
        return check_result
    usage_reserved = True
    task_created = False

    task_id = new_task_id()
    try:
        capabilities = plugin.generator.adapter.get_capabilities()
        try:
            images_data = await collect_tool_reference_images(
                plugin.image_processor,
                event,
                capabilities=capabilities,
                reference_images=reference_images,
                avatar_references=avatar_references,
                persona_images=persona_images,
                task_id=task_id,
            )
        except Exception as exc:
            logger.error(
                f"{log_prefix('LLMTool', task_id)} 处理参考图失败: {safe_log_error_body(exc, 200)}",
                exc_info=True,
            )
            images_data = []

        reference_image_count = len(images_data)
        novelai = await prepare_novelai_assets(
            novelai,
            image_processor=plugin.image_processor,
            unified_msg_origin=event.unified_msg_origin,
        )

        try:
            plugin.create_generation_task(
                task_id=task_id,
                source="LLM工具",
                prompt=prompt,
                negative_prompt=negative_prompt,
                novelai=novelai,
                images_data=images_data,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                image_count=image_count,
                is_usage_limit_admin=is_usage_limit_admin,
                preset=preset_or_persona,
                preset_label=preset_label,
                presets=presets,
                personas=personas,
                source_event=event,
            )
            task_created = True
        except GenerationTaskCreationError:
            # create_generation_task() rolls back quota reservation on creation failure.
            usage_reserved = False
            raise
    except asyncio.CancelledError:
        if usage_reserved and not task_created:
            plugin.usage_manager.release_reserved_usage(
                event.unified_msg_origin,
                is_admin=is_usage_limit_admin,
                count=image_count,
            )
        raise
    except GenerationTaskCreationError as exc:
        return f"❌ 生图任务提交失败: {exc.message} ({exc.code})"
    except Exception as exc:
        if usage_reserved and not task_created:
            plugin.usage_manager.release_reserved_usage(
                event.unified_msg_origin,
                is_admin=is_usage_limit_admin,
                count=image_count,
            )
        logger.error(
            f"{log_prefix('LLMTool', task_id)} 生图任务提交前处理失败: {safe_log_error_body(exc, 200)}",
            exc_info=True,
        )
        return f"❌ 生图任务提交失败: {safe_log_error_body(exc, 160)}"

    return plugin.llm_result_handler.format_tool_start_result(
        prompt=prompt,
        reference_image_count=reference_image_count,
        preset=preset_or_persona,
        preset_label=preset_label,
        presets=presets,
        personas=personas,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        image_count=image_count,
        task_id=task_id,
    )


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """Unified image generation tool callable by the LLM."""

    name: str = "generate_image"
    description: str = (
        "使用生图模型生成或修改图片；支持普通生图、多预设、自拍、人像、头像和多个人设照片。"
        "当用户要求自拍/头像/人像/某个人设或角色出镜时，应优先填写 persona。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的额外提示词。设置 preset、persona 或有效的 NovelAI 角色提示词时可留空。开启提示词正文匹配后，正文中出现的已配置名称才会自动应用。",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "可选。本次希望避免的内容，与 prompt 一起组成同一次生图的完整请求。供应商配置决定将正反向词分开发送，还是合入同一次请求的提示词；不传时使用供应商默认值，传入文本时覆盖默认值，传空字符串时清空本次反向提示词。",
                },
                "novelai": {
                    "type": "object",
                    "description": "可选。仅 NovelAI 接口支持的本次生图参数，覆盖供应商默认值。正向词使用外层 prompt，反向词使用外层 negative_prompt。i2i 与 inpaint 互斥；嵌套 image/mask 可填网络 URL 或当前会话允许的本地图片路径；未填源图时使用本次收集的单张参考图。Vibe 使用已编码数据，不会自动调用收费的编码工具。",
                    "additionalProperties": False,
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["generate", "img2img", "infill"],
                            "description": "省略时按素材自动选择；generate 强制文生图，避免消息/人设参考图自动触发图生图。高级功能取决于模型支持。",
                        },
                        "steps": {"type": "integer", "minimum": 1, "maximum": 50},
                        "sampler": {
                            "type": "string",
                            "enum": list(NOVELAI_SAMPLERS),
                            "description": "采样器，如 k_euler_ancestral。",
                        },
                        "scale": {"type": "number", "minimum": 0, "maximum": 10},
                        "seed": {
                            "type": "integer",
                            "minimum": -1,
                            "maximum": 4294967295,
                            "description": "随机种子；-1 表示随机。",
                        },
                        "noise_schedule": {
                            "type": "string",
                            "enum": list(NOVELAI_NOISE_SCHEDULES),
                            "description": "噪声调度，如 karras、exponential、polyexponential。",
                        },
                        "cfg_rescale": {"type": "number", "minimum": 0, "maximum": 1},
                        "width": {
                            "type": "integer",
                            "minimum": 64,
                            "maximum": 2048,
                            "multipleOf": 64,
                            "description": "手动宽度，必须与 height 同时传入，宽高乘积不超过 3145728。",
                        },
                        "height": {
                            "type": "integer",
                            "minimum": 64,
                            "maximum": 2048,
                            "multipleOf": 64,
                            "description": "手动高度，必须与 width 同时传入。",
                        },
                        "sm": {
                            "type": "boolean",
                            "description": "SMEA，仅适用于支持的模型与采样器。",
                        },
                        "sm_dyn": {
                            "type": "boolean",
                            "description": "SMEA DYN，需同时启用 sm。",
                        },
                        "dynamic_thresholding": {
                            "type": "boolean",
                            "description": "动态阈值（Decrisp）。",
                        },
                        "variety_boost": {
                            "type": "boolean",
                            "description": "多样性增强（Variety+）。",
                        },
                        "autoSmea": {"type": "boolean"},
                        "legacy": {"type": "boolean"},
                        "legacy_uc": {"type": "boolean"},
                        "legacy_v3_extend": {"type": "boolean"},
                        "deliberate_euler_ancestral_bug": {"type": "boolean"},
                        "prefer_brownian": {"type": "boolean"},
                        "normalize_reference_strength_multiple": {"type": "boolean"},
                        "skip_cfg_above_sigma": {"type": "number", "minimum": 0},
                        "controlnet_strength": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "characters": {
                            "type": "array",
                            "description": "V4/V4.5/V5 的独立角色正反向词和位置。省略位置时让模型自动布局。",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "prompt": {"type": "string"},
                                    "negative_prompt": {"type": "string"},
                                    "enabled": {"type": "boolean"},
                                    "position": {
                                        "description": "A1 到 E5 的 5×5 网格位置，或 [x,y]，坐标范围 0–1。",
                                        "anyOf": [
                                            {"type": "null"},
                                            {
                                                "type": "string",
                                                "pattern": "^[A-E][1-5]$",
                                            },
                                            {
                                                "type": "array",
                                                "items": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1,
                                                },
                                                "minItems": 2,
                                                "maxItems": 2,
                                            },
                                        ],
                                    },
                                },
                                "required": ["prompt"],
                            },
                        },
                        "i2i": {
                            "type": "object",
                            "description": "图生图设置，不可与 inpaint 同时使用。",
                            "additionalProperties": False,
                            "properties": {
                                "image": {
                                    "type": "string",
                                    "description": "源图 URL 或允许的本地路径；省略时使用本次单张参考图。",
                                },
                                "strength": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "noise": {"type": "number", "minimum": 0, "maximum": 1},
                                "seed": {
                                    "type": "integer",
                                    "minimum": -1,
                                    "maximum": 4294967295,
                                },
                            },
                        },
                        "inpaint": {
                            "type": "object",
                            "description": "局部重绘；mask 白色重绘、黑色保留，会使用相应的 inpainting 模型。不可与 i2i 或 Vibe 同时使用。",
                            "additionalProperties": False,
                            "properties": {
                                "image": {
                                    "type": "string",
                                    "description": "源图 URL 或允许的本地路径；省略时使用本次单张参考图。",
                                },
                                "mask": {
                                    "type": "string",
                                    "description": "遮罩图 URL 或允许的本地路径，必须提供。",
                                },
                                "strength": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "seed": {
                                    "type": "integer",
                                    "minimum": -1,
                                    "maximum": 4294967295,
                                },
                                "color_correct": {"type": "boolean"},
                            },
                            "required": ["mask"],
                        },
                        "character_references": {
                            "type": "array",
                            "description": "精确角色/风格参考，仅支持相应的 V4.5 模型；不可与 Vibe 同时使用。",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "image": {
                                        "type": "string",
                                        "description": "参考图 URL 或允许的本地路径。",
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "character",
                                            "style",
                                            "character&style",
                                        ],
                                    },
                                    "fidelity": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "strength": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": ["image"],
                            },
                        },
                        "vibe_references": {
                            "type": "array",
                            "description": "Vibe 氛围参考，最多 16 个；V4/V4.5 传已编码的原始 Base64 encoding；仅 V3 可直接传 image。不会自动进行 Vibe 编码；V5 不支持。",
                            "maxItems": 16,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "encoding": {
                                        "type": "string",
                                        "description": "已有的 Vibe 编码 Base64，不是普通图片 Base64。",
                                    },
                                    "image": {
                                        "type": "string",
                                        "description": "仅 V3 可用的参考图 URL 或允许的本地路径，与 encoding 二选一。",
                                    },
                                    "information_extracted": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": 1,
                                    },
                                    "strength": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "oneOf": [
                                    {"required": ["encoding"]},
                                    {"required": ["image"]},
                                ],
                            },
                        },
                    },
                },
                "preset": {
                    "type": "string",
                    "description": "可选。使用一个或多个已配置的预设名称，多个名称可用空格分隔；会按顺序拼接预设提示词，并可继承预设中的宽高比和分辨率。",
                },
                "persona": {
                    "type": "string",
                    "description": "可选。使用一个或多个已配置的人设名称，多个名称可用空格分隔。当用户要求自拍、头像、人像、人设照片或某角色出镜时优先填写；会按顺序拼接人设描述，并在支持图生图时加入人设参考图。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比。如果不确定，请使用'不指定'。",
                    "enum": ASPECT_RATIO_OPTIONS,
                    "default": "不指定",
                },
                "resolution": {
                    "type": "string",
                    "description": "图片质量/分辨率。使用'不指定'时请求中不携带分辨率字段。",
                    "enum": RESOLUTION_OPTIONS,
                    "default": "不指定",
                },
                "image_count": {
                    "type": "integer",
                    "description": "本次任务要生成的图片数量。不填时使用插件默认出图数量；超过配置上限时会自动截断。",
                    "minimum": 1,
                    "default": 1,
                },
                "avatar_references": {
                    "type": "array",
                    "description": "可选。需要使用头像作为参考图时填写，'self' 表示机器人，'sender' 表示发送者，也可填写 QQ 号/用户 ID。",
                    "items": {"type": "string"},
                },
                "reference_images": {
                    "type": "array",
                    "description": "可选。参考图列表，支持 http(s) 网络图片 URL；本地图片仅允许当前会话 workspace和AstrBot temp 目录。",
                    "items": {"type": "string"},
                },
            },
            "required": [],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """Execute the image generation tool call."""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        try:
            novelai = validate_novelai_options(kwargs.get("novelai"))
        except (TypeError, ValueError) as exc:
            return f"❌ NovelAI 参数错误: {safe_log_error_body(exc, 160)}"

        prompt = str(kwargs.get("prompt", "") or "").strip()
        raw_aspect_ratio = kwargs.get("aspect_ratio")
        aspect_ratio = raw_aspect_ratio or plugin.config_manager.default_aspect_ratio
        resolution = (
            kwargs.get("resolution") or plugin.config_manager.default_resolution
        )

        preset_prompts, aspect_ratio, resolution, matched_presets, error = (
            _parse_preset(
                plugin,
                kwargs.get("preset"),
                aspect_ratio,
                resolution,
            )
        )
        if error:
            return error
        persona_prompts, persona_images, matched_personas, error = _parse_persona(
            plugin,
            kwargs.get("persona"),
        )
        if error:
            return error
        if plugin.config_manager.match_templates_in_prompt_body:
            (
                prompt,
                aspect_ratio,
                resolution,
                extra_preset_prompts,
                extra_persona_prompts,
                extra_presets,
                extra_personas,
                extra_persona_images,
            ) = extract_templates_from_prompt(
                prompt,
                plugin.config_manager.prompt_body_presets(),
                plugin.config_manager.prompt_body_personas(),
                aspect_ratio=str(aspect_ratio),
                resolution=str(resolution),
                skip_names=[*matched_presets, *matched_personas],
            )
            preset_prompts.extend(extra_preset_prompts)
            persona_prompts.extend(extra_persona_prompts)
            matched_presets.extend(extra_presets)
            matched_personas.extend(extra_personas)
            persona_images.extend(extra_persona_images)
        prompt = build_generation_prompt(
            preset_prompts=preset_prompts,
            persona_prompts=persona_prompts,
            extra_prompt=prompt,
        )
        if not prompt and not any(
            character.get("enabled", True) and character["prompt"].strip()
            for character in (novelai or {}).get("characters", [])
        ):
            return "❌ 请提供图片生成的提示词、预设或人设"

        preset_or_persona, preset_label = format_template_summary(
            matched_presets,
            matched_personas,
        )

        event = extract_event(context)
        if not event:
            logger.warning(f"{LOG} 工具调用上下文缺少事件。上下文类型: {type(context)}")
            return "❌ 无法获取当前消息上下文"

        return await _start_generation_task(
            plugin,
            event,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt"),
            novelai=novelai,
            aspect_ratio=str(aspect_ratio),
            resolution=str(resolution),
            reference_images=kwargs.get("reference_images"),
            avatar_references=kwargs.get("avatar_references"),
            persona_images=persona_images,
            preset_or_persona=preset_or_persona,
            preset_label=preset_label,
            presets=matched_presets,
            personas=matched_personas,
            image_count=kwargs.get("image_count")
            or plugin.config_manager.default_image_count,
        )


def _format_preset_detail(name: str, content: Any) -> str:
    """Format one preset's full content for query results."""
    content_text = str(content or "").strip()
    lines = [f"📋 预设详情: {name}"]
    if content_text.startswith("{"):
        try:
            preset_data = json.loads(content_text)
        except json.JSONDecodeError:
            lines.append("格式: 高级 JSON（解析失败，将按原文展示）")
            lines.append(f"内容: {content_text}")
            return "\n".join(lines)

        if isinstance(preset_data, dict):
            lines.append("格式: 高级 JSON")
            if prompt := str(preset_data.get("prompt", "") or "").strip():
                lines.append(f"提示词: {prompt}")
            if aspect_ratio := str(preset_data.get("aspect_ratio", "") or "").strip():
                lines.append(f"宽高比: {aspect_ratio}")
            if resolution := str(preset_data.get("resolution", "") or "").strip():
                lines.append(f"分辨率: {resolution}")
            if description := str(preset_data.get("description", "") or "").strip():
                lines.append(f"描述: {description}")
            lines.append(f"原始内容: {content_text}")
            return "\n".join(lines)

    lines.append("格式: 简单提示词")
    lines.append(f"内容: {content_text}")
    return "\n".join(lines)


def _format_persona_detail(name: str, persona: Any) -> str:
    """Format one persona's full content for query results."""
    lines = [f"👤 人设详情: {name}"]
    lines.append(f"提示词: {persona.prompt}")
    lines.append(f"参考图: {persona.image or '无'}")
    return "\n".join(lines)


def _validate_preset_content(content: str) -> str | None:
    """Validate preset content when it is written by an LLM tool."""
    if not content.startswith("{"):
        return None

    try:
        preset_data = json.loads(content)
    except json.JSONDecodeError as exc:
        return f"高级 JSON 预设格式错误: {exc}"

    if not isinstance(preset_data, dict):
        return "高级 JSON 预设必须是对象"
    if not str(preset_data.get("prompt", "") or "").strip():
        return "高级 JSON 预设必须包含非空 prompt 字段"

    aspect_ratio = str(preset_data.get("aspect_ratio", "") or "").strip()
    if aspect_ratio and aspect_ratio not in ASPECT_RATIO_OPTIONS:
        return f"高级 JSON 预设的 aspect_ratio 不支持: {aspect_ratio}"

    resolution = str(preset_data.get("resolution", "") or "").strip()
    if resolution and resolution not in RESOLUTION_OPTIONS:
        return f"高级 JSON 预设的 resolution 不支持: {resolution}"

    return None


@pydantic_dataclass
class PresetQueryTool(FunctionTool[AstrAgentContext]):
    """Preset and persona query tool callable by the LLM."""

    name: str = "query_image_presets"
    description: str = (
        "查询生图预设或人设。默认只返回名称列表；当需要了解某个预设或人设的具体内容时，"
        "填写 name 查看详情。查询到的预设名称可用于 generate_image 的 preset 参数，人设名称可用于 persona 参数。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "查询类别。preset 查询生图预设，persona 查询人设。默认 preset。",
                    "enum": ["preset", "persona"],
                    "default": "preset",
                },
                "name": {
                    "type": "string",
                    "description": "可选。填写后查看指定预设或人设的具体内容；留空时只返回名称列表。",
                },
            },
            "required": [],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """Execute the preset/persona query tool call."""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        event = extract_event(context)
        if (
            event
            and not plugin.usage_manager.is_limit_exempt(
                event.unified_msg_origin,
                is_admin=plugin.is_usage_limit_admin(event),
            )
            and plugin.usage_manager.is_session_blocked(event.unified_msg_origin)
        ):
            return plugin.config_manager.usage_settings.blacklist_block_message

        category = normalize_preset_query_category(
            str(kwargs.get("category", "preset") or "preset")
        )
        name = str(kwargs.get("name", "") or "").strip()

        if category == "persona":
            if name:
                matched_name = find_named_entry(plugin.config_manager.personas, name)
                if not matched_name:
                    return f"❌ 人设不存在: {name}"
                return _format_persona_detail(
                    matched_name, plugin.config_manager.personas[matched_name]
                )

            if not plugin.config_manager.personas:
                return "👤 当前没有人设"
            lines = ["👤 人设列表:"]
            for idx, (persona_name, persona) in enumerate(
                plugin.config_manager.personas.items(),
                1,
            ):
                image_mark = "有参考图" if persona.image else "无参考图"
                lines.append(f"{idx}. {persona_name} [{image_mark}]")
            return "\n".join(lines)

        if name:
            matched_name = find_named_entry(plugin.config_manager.presets, name)
            if not matched_name:
                return f"❌ 预设不存在: {name}"
            return _format_preset_detail(
                matched_name, plugin.config_manager.presets[matched_name]
            )

        if not plugin.config_manager.presets:
            return "📋 当前没有预设"
        lines = ["📋 预设列表:"]
        for idx, preset_name in enumerate(plugin.config_manager.presets, 1):
            lines.append(f"{idx}. {preset_name}")
        return "\n".join(lines)


@pydantic_dataclass
class PresetEditTool(FunctionTool[AstrAgentContext]):
    """Preset editing tool callable by the LLM."""

    name: str = "edit_image_presets"
    description: str = (
        "创建或删除生图预设。支持两种预设格式："
        "1) 简单格式：name + prompt，prompt 直接写普通提示词；"
        "2) 高级 JSON 格式：prompt 写 JSON 字符串，如 "
        '{"prompt":"提示词","aspect_ratio":"16:9","resolution":"2K","description":"描述"}。'
        "只编辑预设，不编辑人设。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "要执行的操作。create_preset 创建或覆盖预设，delete_preset 删除预设。",
                    "enum": ["create_preset", "delete_preset"],
                },
                "name": {
                    "type": "string",
                    "description": "预设名称。创建和删除均必填。",
                },
                "prompt": {
                    "type": "string",
                    "description": '创建预设时必填。可填写普通提示词，或填写高级 JSON 字符串：{"prompt":"提示词","aspect_ratio":"16:9","resolution":"2K","description":"描述"}。',
                },
            },
            "required": ["action", "name"],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """Execute the preset editing tool call."""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        event = extract_event(context)
        if (
            event
            and not plugin.usage_manager.is_limit_exempt(
                event.unified_msg_origin,
                is_admin=plugin.is_usage_limit_admin(event),
            )
            and plugin.usage_manager.is_session_blocked(event.unified_msg_origin)
        ):
            return plugin.config_manager.usage_settings.blacklist_block_message

        action = normalize_preset_edit_action(
            str(kwargs.get("action", "create_preset") or "")
        )
        name = str(kwargs.get("name", "") or "").strip()
        if not name:
            return "❌ 请提供预设名称"

        if action == "create_preset":
            prompt = str(kwargs.get("prompt", "") or "").strip()
            if not prompt:
                return "❌ 请提供预设内容"
            if error := _validate_preset_content(prompt):
                return f"❌ {error}"
            plugin.config_manager.save_preset(name, prompt)
            return f"✅ 预设已保存: {name}"

        if action == "delete_preset":
            matched_name = find_named_entry(plugin.config_manager.presets, name)
            if not matched_name:
                return f"❌ 预设不存在: {name}"
            plugin.config_manager.delete_preset(matched_name)
            return f"✅ 预设已删除: {matched_name}"

        return "❌ 不支持的操作，请使用 create_preset 或 delete_preset"


@pydantic_dataclass
class ImageTaskTool(FunctionTool[AstrAgentContext]):
    """Image task management tool callable by the LLM."""

    name: str = "manage_image_tasks"
    description: str = (
        "管理当前会话的生图任务。list 只列出正在进行的任务；"
        "detail 可查看仍保留记录的进行中或已结束任务；"
        "cancel 只能取消仍在排队/运行/取消中的任务。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "操作类型。list 查看任务列表；detail 查看任务详情；cancel 取消任务。默认 list。",
                    "enum": ["list", "detail", "cancel"],
                    "default": "list",
                },
                "task": {
                    "type": "string",
                    "description": "任务编号或任务ID。detail 和 cancel 时填写；编号来自 list 返回的正在进行任务列表。已结束任务只能用任务ID查看详情。",
                },
            },
            "required": [],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """Execute the image task management tool call."""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        event = extract_event(context)
        if not event:
            logger.warning(
                f"{LOG} 任务工具调用上下文缺少事件。上下文类型: {type(context)}"
            )
            return "❌ 无法获取当前消息上下文"

        unified_msg_origin = event.unified_msg_origin
        action = normalize_task_action(str(kwargs.get("action", "list") or "list"))
        task_ref = str(kwargs.get("task", "") or "").strip()

        if action == "list":
            records = plugin.task_manager.list_generation_tasks(
                unified_msg_origin=unified_msg_origin,
                include_finished=False,
                limit=10,
            )
            return plugin.format_task_list(records)

        if action == "detail":
            if not task_ref:
                return "❌ 请提供要查看的任务编号或任务ID"
            record = plugin.resolve_task_reference(
                unified_msg_origin,
                task_ref,
                include_finished=True,
            )
            if not record:
                if task_ref.isdigit():
                    return (
                        "❌ 未找到编号对应的进行中任务；已结束任务请使用完整任务ID查看"
                    )
                return f"❌ 任务不存在或已被清理: {task_ref}"
            return plugin.format_task_detail(record)

        if action == "cancel":
            if not task_ref:
                active_records = plugin.task_manager.list_generation_tasks(
                    unified_msg_origin=unified_msg_origin,
                    include_finished=False,
                    limit=5,
                )
                if active_records:
                    return "❌ 请提供要取消的编号或任务ID\n" + plugin.format_task_list(
                        active_records
                    )
                return "📭 当前没有可取消的生图任务"

            record = plugin.resolve_active_task_reference(unified_msg_origin, task_ref)
            if not record:
                return f"❌ 未找到对应的进行中任务，请检查编号或任务ID: {task_ref}"
            _, message = plugin.task_manager.cancel_generation_task(
                record.task_id,
                unified_msg_origin=unified_msg_origin,
            )
            logger.debug(
                f"{log_prefix('Task', record.task_id)} LLM 工具请求取消任务: "
                f"用户={mask_sensitive(unified_msg_origin)}，结果={safe_log_text(message)}"
            )
            return message

        return "❌ 不支持的操作，请使用 list、detail 或 cancel"


def adjust_tool_parameters(
    tool: FunctionTool[AstrAgentContext], capabilities: ImageCapability
) -> None:
    """Adjust tool parameters dynamically based on adapter capabilities."""
    props = tool.parameters.get("properties", {})

    unsupported_parameters = (
        (ImageCapability.ASPECT_RATIO, "aspect_ratio", "宽高比"),
        (ImageCapability.RESOLUTION, "resolution", "分辨率"),
        (ImageCapability.NEGATIVE_PROMPT, "negative_prompt", "反向提示词"),
        (ImageCapability.NOVELAI_OPTIONS, "novelai", "NovelAI 专属参数"),
    )
    for capability, key, label in unsupported_parameters:
        if not (capabilities & capability) and props.pop(key, None) is not None:
            logger.debug(f"{LOG} 适配器不支持{label}，已从工具参数中移除")

    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        for key in ("avatar_references", "reference_images"):
            props.pop(key, None)
        logger.debug(f"{LOG} 适配器不支持参考图，已从工具参数中移除参考图相关参数")


__all__ = (
    "ImageGenerationTool",
    "ImageTaskTool",
    "PresetEditTool",
    "PresetQueryTool",
    "adjust_tool_parameters",
)
