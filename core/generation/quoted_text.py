"""Text material from quoted messages and merged-forward cards."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import astrbot.api.message_components as Comp
from astrbot.api import logger

from ..shared.constants import MAX_QUOTED_TEXT_CHARS
from ..shared.logging import log_prefix, safe_log_error_body

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

LOG = log_prefix("QuotedText")

# Framework parsers render non-text segments as placeholders; they must not end up
# in an image prompt.
_PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:Image|Video|Forward Message|File:[^\]]*|图片|视频|文件[^\]]*)\]",
    re.IGNORECASE,
)
_BLANK_RUN_PATTERN = re.compile(r"[ \t]{2,}")


class _SupportState:
    """Module-level state shared by the lazy core imports."""

    warned = False


def _warn_core_support_unavailable(exc: Exception) -> None:
    """Log missing quoted-message support in AstrBot core once.

    The quoted-message helpers live in a relatively new core module. Plugins may
    run on an older core, so the imports stay lazy and degrade to a single
    warning instead of failing every command.

    Args:
        exc: Import error raised while loading the core helpers.

    """
    if _SupportState.warned:
        return
    _SupportState.warned = True
    logger.warning(
        f"{LOG} 当前 AstrBot 版本不支持引用消息解析，引用与转发文字将被忽略: {exc}"
    )


def has_quoted_text_source(event: AstrMessageEvent) -> bool:
    """Return whether the message may carry quoted or forwarded text.

    This is a component-level probe: it never parses content and never calls a
    platform API, so callers can use it before deciding to reject a request.

    Args:
        event: Incoming message event.

    Returns:
        Whether the message chain holds a reply, node or forward component.

    """
    message = getattr(getattr(event, "message_obj", None), "message", None)
    if not message:
        return False

    candidate_types = tuple(
        candidate
        for candidate in (
            getattr(Comp, "Reply", None),
            getattr(Comp, "Nodes", None),
            getattr(Comp, "Node", None),
            getattr(Comp, "Forward", None),
        )
        if candidate is not None
    )
    if not candidate_types:
        return False
    return any(isinstance(component, candidate_types) for component in message)


async def collect_quoted_text(event: AstrMessageEvent) -> str:
    """Collect text material from the quoted message and merged-forward cards.

    Every source is parsed one level deep: nested forwards and nested node
    chains are not entered. Only text is collected, images are ignored.

    Args:
        event: Incoming message event.

    Returns:
        Cleaned text material, or an empty string when nothing usable was found.

    """
    message = getattr(getattr(event, "message_obj", None), "message", None)
    if not message:
        return ""

    parts: list[str] = []
    parts.extend(await _collect_reply_text(event))
    parts.extend(_collect_inline_node_text(message))
    if not any(part.strip() for part in parts):
        fetched = await _fetch_forward_card_text(event)
        if fetched:
            parts.append(fetched)

    return _finalize_text(parts)


async def _collect_reply_text(event: AstrMessageEvent) -> list[str]:
    """Read the text of the quoted message through the core parser.

    Args:
        event: Incoming message event.

    Returns:
        Collected quoted text, or an empty list when there is nothing to read.

    """
    try:
        from astrbot.core.utils.quoted_message import extract_quoted_message_text
        from astrbot.core.utils.quoted_message.chain_parser import ReplyChainParser
        from astrbot.core.utils.quoted_message.settings import SETTINGS
    except ImportError as exc:
        _warn_core_support_unavailable(exc)
        return []

    reply = ReplyChainParser.find_first_reply_component(event)
    if reply is None:
        return []

    # 只解析一层：转发层数与拉取次数都锁到 1
    settings = SETTINGS.with_overrides(
        {"max_forward_node_depth": 1, "max_forward_fetch": 1}
    )
    try:
        text = await extract_quoted_message_text(event, reply, settings)
    except Exception as exc:
        logger.warning(f"{LOG} 引用消息文字解析失败: {safe_log_error_body(exc)}")
        return []
    if not text or not text.strip():
        # 纯图片/文件引用等没有文字的情况，属于正常结果
        logger.debug(f"{LOG} 引用消息没有可用文字，将忽略引用内容")
        return []
    return [text]


def _collect_inline_node_text(message: list) -> list[str]:
    """Read first-level text from inline ``Node``/``Nodes`` components.

    Args:
        message: Message component chain of the incoming message.

    Returns:
        One text entry per node, prefixed with the sender name when known.

    """
    node_cls = getattr(Comp, "Node", None)
    nodes_cls = getattr(Comp, "Nodes", None)
    if node_cls is None and nodes_cls is None:
        return []

    parts: list[str] = []
    for component in message:
        if nodes_cls is not None and isinstance(component, nodes_cls):
            nodes = list(getattr(component, "nodes", None) or [])
        elif node_cls is not None and isinstance(component, node_cls):
            nodes = [component]
        else:
            continue

        for node in nodes:
            # 只解析一层：content 里若还嵌 Node/Nodes，不再深入
            segments = [
                str(getattr(segment, "text", "") or "")
                for segment in (getattr(node, "content", None) or [])
                if isinstance(segment, Comp.Plain)
            ]
            text = " ".join(segment for segment in segments if segment.strip()).strip()
            if not text:
                continue
            sender = str(
                getattr(node, "name", "") or getattr(node, "uin", "") or ""
            ).strip()
            parts.append(f"{sender}: {text}" if sender else text)
    return parts


async def _fetch_forward_card_text(event: AstrMessageEvent) -> str:
    """Fetch one level of text from a ``Forward`` component carrying only an id.

    Args:
        event: Incoming message event.

    Returns:
        Fetched card text, or an empty string when nothing could be read.

    """
    forward_cls = getattr(Comp, "Forward", None)
    if forward_cls is None:
        return ""

    forward_id = ""
    for component in getattr(event.message_obj, "message", None) or []:
        if not isinstance(component, forward_cls):
            continue
        forward_id = str(getattr(component, "id", "") or "").strip()
        if forward_id:
            break
    if not forward_id:
        return ""

    try:
        from astrbot.core.utils.quoted_message.chain_parser import (
            OneBotPayloadParser,
        )
        from astrbot.core.utils.quoted_message.onebot_client import OneBotClient
        from astrbot.core.utils.quoted_message.settings import SETTINGS
    except ImportError as exc:
        _warn_core_support_unavailable(exc)
        return ""

    try:
        client = OneBotClient(event, settings=SETTINGS)
        payload = await client.get_forward_msg(forward_id)
        if not payload:
            logger.warning(
                f"{LOG} 合并转发内容拉取失败，将忽略卡片文字（get_forward_msg 无返回）"
            )
            return ""
        # parse_get_forward_payload 只解一层，递归由调用方决定
        parsed = OneBotPayloadParser(settings=SETTINGS).parse_get_forward_payload(
            payload
        )
    except Exception as exc:
        logger.warning(f"{LOG} 合并转发文字解析失败: {safe_log_error_body(exc)}")
        return ""
    return str(parsed.get("text") or "").strip()


def _finalize_text(parts: list[str]) -> str:
    """Clean placeholders, drop duplicate lines and cap the collected length.

    Args:
        parts: Raw text collected from each source.

    Returns:
        Cleaned text material, possibly truncated, or an empty string.

    """
    lines: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or not part.strip():
            continue
        text = _PLACEHOLDER_PATTERN.sub(" ", part)
        text = _BLANK_RUN_PATTERN.sub(" ", text)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)

    joined = "\n".join(lines).strip()
    if len(joined) > MAX_QUOTED_TEXT_CHARS:
        joined = joined[:MAX_QUOTED_TEXT_CHARS].rstrip()
        logger.debug(f"{LOG} 引用与转发文字超过 {MAX_QUOTED_TEXT_CHARS} 字符，已截断")
    return joined
