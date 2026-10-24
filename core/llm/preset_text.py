"""Preset and persona text helpers for LLM tools.

These functions only normalize aliases, format query results, and validate
preset content. Tool classes and their schemas stay in ``tools.py``.
"""

from __future__ import annotations

import json
from typing import Any

from ..shared.constants import SUPPORTED_ASPECT_RATIOS, SUPPORTED_RESOLUTIONS

ASPECT_RATIO_OPTIONS = list(SUPPORTED_ASPECT_RATIOS)
RESOLUTION_OPTIONS = list(SUPPORTED_RESOLUTIONS)


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


def _format_preset_detail(name: str, preset: Any) -> str:
    """Format one preset's content for query results."""
    lines = [f"📋 预设详情: {name}"]
    lines.append(f"提示词: {preset.prompt}")
    if preset.aspect_ratio:
        lines.append(f"宽高比: {preset.aspect_ratio}")
    if preset.resolution:
        lines.append(f"分辨率: {preset.resolution}")
    lines.append(f"参考图: {preset.image or '无'}")
    return "\n".join(lines)


def _format_persona_detail(name: str, persona: Any) -> str:
    """Format one persona's full content for query results."""
    lines = [f"👤 人设详情: {name}"]
    lines.append(f"提示词: {persona.prompt}")
    lines.append(f"参考图: {persona.image or '无'}")
    return "\n".join(lines)


def validate_preset_options(aspect_ratio: str, resolution: str) -> str | None:
    """Validate preset size options and return an error message when invalid."""
    if aspect_ratio and aspect_ratio not in ASPECT_RATIO_OPTIONS:
        return f"预设的 aspect_ratio 不支持: {aspect_ratio}"
    if resolution and resolution not in RESOLUTION_OPTIONS:
        return f"预设的 resolution 不支持: {resolution}"
    return None


def _validate_preset_content(content: str) -> str | None:
    """Validate legacy JSON preset content written by an LLM tool."""
    if not content.startswith("{"):
        return None

    try:
        preset_data = json.loads(content)
    except json.JSONDecodeError as exc:
        return f"预设 JSON 格式错误: {exc}"

    if not isinstance(preset_data, dict):
        return "预设 JSON 必须是对象"
    if not str(preset_data.get("prompt", "") or "").strip():
        return "预设 JSON 必须包含非空 prompt 字段"

    return validate_preset_options(
        str(preset_data.get("aspect_ratio", "") or "").strip(),
        str(preset_data.get("resolution", "") or "").strip(),
    )
