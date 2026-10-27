"""Resolve NovelAI image roles without mixing them with img2img references."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ..shared.types import ImageData

if TYPE_CHECKING:
    from .image_processor import ImageProcessor


async def prepare_novelai_assets(
    novelai: dict[str, Any] | None,
    *,
    image_processor: ImageProcessor,
    unified_msg_origin: str | None,
) -> dict[str, Any] | None:
    """Resolve each source, mask, and reference while preserving its role and order.

    Args:
        novelai: Validated per-task options, with image sources or ImageData values.
        image_processor: Existing image reader enforcing allowed paths and formats.
        unified_msg_origin: Session whose workspace may contain source images.

    Returns:
        Independent options with image sources replaced by validated ImageData.

    Raises:
        ValueError: An explicitly supplied image is missing, invalid, or too large.
    """
    if novelai is None:
        return None
    prepared = deepcopy(novelai)
    entries: list[tuple[dict[str, Any], str, str]] = []
    for mode in ("i2i", "inpaint"):
        if options := prepared.get(mode):
            for field in ("image", "mask"):
                if field in options:
                    entries.append((options, field, f"{mode}.{field}"))
    for role in ("character_references", "vibe_references"):
        for index, options in enumerate(prepared.get(role, [])):
            if "image" in options:
                entries.append((options, "image", f"{role}[{index}].image"))
    if not entries:
        return prepared

    workspace_dir = image_processor.workspace_dir_for_origin(unified_msg_origin)
    for options, field, label in entries:
        source = options[field]
        if isinstance(source, ImageData):
            data = source
        else:
            data = await image_processor.download_image(
                source, workspace_dir=workspace_dir
            )
        if not isinstance(data, ImageData):
            raise ValueError(f"NovelAI {label}: image could not be read or validated")
        if (
            not data.data
            or len(data.data) > image_processor._max_image_size_mb * 1024**2
        ):
            raise ValueError(
                f"NovelAI {label}: image exceeds the configured size limit"
            )
        validated = await asyncio.to_thread(
            image_processor.validate_image_data,
            data.data,
            source_url=data.source_url,
            log_source=f"NovelAI {label}",
        )
        if not isinstance(validated, ImageData):
            raise ValueError(f"NovelAI {label}: image could not be read or validated")
        options[field] = validated
    return prepared
