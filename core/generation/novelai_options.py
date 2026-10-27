"""Validate request-scoped NovelAI options without contacting a provider."""

from __future__ import annotations

import base64
import binascii
import math
import re
from copy import deepcopy
from typing import Any

from ..shared.types import ImageData

NOVELAI_SAMPLERS = (
    "k_euler",
    "k_euler_ancestral",
    "k_dpm_2",
    "k_dpm_2_ancestral",
    "k_dpmpp_2m",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_sde",
    "ddim",
)
NOVELAI_NOISE_SCHEDULES = ("karras", "exponential", "polyexponential")
NOVELAI_BOOLEAN_OPTIONS = {
    "sm",
    "sm_dyn",
    "autoSmea",
    "dynamic_thresholding",
    "variety_boost",
    "legacy",
    "legacy_uc",
    "legacy_v3_extend",
    "deliberate_euler_ancestral_bug",
    "prefer_brownian",
    "normalize_reference_strength_multiple",
    "straight_alpha",
    "tag_hint_transparent_background",
}
NOVELAI_NUMBER_OPTIONS = {
    "width": (64, 2048, True),
    "height": (64, 2048, True),
    "steps": (1, 50, True),
    "seed": (-1, 2**32 - 1, True),
    "scale": (0, 10, False),
    "cfg_rescale": (0, 1, False),
    "skip_cfg_above_sigma": (0, None, False),
    "controlnet_strength": (0, 1, False),
}


def validate_novelai_options(value: Any) -> dict[str, Any] | None:
    """Validate and copy supported native image-generation options.

    Args:
        value: A request option object, or None to retain provider defaults.

    Returns:
        An independent copy preserving resolved ImageData objects, or None.

    Raises:
        ValueError: An option has an unknown field, invalid shape, or unsafe value.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("novelai must be an object")
    allowed = (
        set(NOVELAI_NUMBER_OPTIONS)
        | NOVELAI_BOOLEAN_OPTIONS
        | {
            "sampler",
            "noise_schedule",
            "action",
            "characters",
            "i2i",
            "inpaint",
            "character_references",
            "vibe_references",
        }
    )
    if unknown := set(value) - allowed:
        raise ValueError(
            f"Unknown novelai fields: {', '.join(sorted(map(str, unknown)))}"
        )
    # One validation pass covers repeated scalar and image fields at all levels.
    fields: list[tuple[str, Any, str, Any]] = []
    for key, item in value.items():
        if key in NOVELAI_BOOLEAN_OPTIONS:
            fields.append((key, item, "bool", None))
        elif key in NOVELAI_NUMBER_OPTIONS:
            fields.append((key, item, "number", NOVELAI_NUMBER_OPTIONS[key]))
        elif key in {"sampler", "noise_schedule", "action"}:
            choices = {
                "sampler": NOVELAI_SAMPLERS,
                "noise_schedule": NOVELAI_NOISE_SCHEDULES,
                "action": ("generate", "img2img", "infill"),
            }[key]
            fields.append((key, item, "choice", choices))
    if "i2i" in value and "inpaint" in value:
        raise ValueError("i2i and inpaint are mutually exclusive")
    if ("width" in value) != ("height" in value):
        raise ValueError("width and height must be provided together")
    for mode in ("i2i", "inpaint"):
        if mode not in value:
            continue
        block = value[mode]
        mode_fields = {"image", "strength", "seed"}
        mode_fields |= {"noise"} if mode == "i2i" else {"mask", "color_correct"}
        if not isinstance(block, dict) or set(block) - mode_fields:
            raise ValueError(f"{mode} must be an object with supported fields")
        if mode == "inpaint" and "mask" not in block:
            raise ValueError("inpaint.mask is required")
        if value.get("action") not in (None, "img2img" if mode == "i2i" else "infill"):
            raise ValueError(f"action conflicts with {mode}")
        for key, item in block.items():
            kind = (
                "image"
                if key in {"image", "mask"}
                else ("bool" if key == "color_correct" else "number")
            )
            bounds = (-1, 2**32 - 1, True) if key == "seed" else (0, 1, False)
            fields.append((f"{mode}.{key}", item, kind, bounds))
    if value.get("action") == "infill" and "inpaint" not in value:
        raise ValueError("infill requires an inpaint object and mask")
    for key in ("characters", "character_references", "vibe_references"):
        if key not in value:
            continue
        entries = value[key]
        if not isinstance(entries, list):
            raise ValueError(f"{key} must be an array")
        if key == "vibe_references" and len(entries) > 16:
            raise ValueError("At most 16 Vibe references are supported")
        for index, entry in enumerate(entries):
            label = f"{key}[{index}]"
            allowed_fields = {
                "characters": {"prompt", "negative_prompt", "position", "enabled"},
                "character_references": {"image", "type", "fidelity", "strength"},
                "vibe_references": {
                    "encoding",
                    "image",
                    "information_extracted",
                    "strength",
                },
            }[key]
            if not isinstance(entry, dict) or set(entry) - allowed_fields:
                raise ValueError(f"{label} must contain only supported fields")
            if key == "characters" and "prompt" not in entry:
                raise ValueError(f"{label}.prompt is required")
            if key == "character_references" and "image" not in entry:
                raise ValueError(f"{label}.image is required")
            if key == "vibe_references" and (
                ("encoding" in entry) == ("image" in entry)
            ):
                raise ValueError(f"{label} requires exactly one encoding or image")
            for field, item in entry.items():
                path = f"{label}.{field}"
                if field == "position":
                    if item is None:
                        continue
                    if isinstance(item, str) and re.fullmatch(r"[A-E][1-5]", item):
                        continue
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        raise ValueError(f"{path} must be a coordinate pair or A1-E5")
                    fields.extend(
                        (path, coordinate, "number", (0, 1, False))
                        for coordinate in item
                    )
                elif field in {"prompt", "negative_prompt"}:
                    fields.append((path, item, "str", None))
                elif field == "enabled":
                    fields.append((path, item, "bool", None))
                elif field == "image":
                    fields.append((path, item, "image", None))
                elif field == "type":
                    fields.append(
                        (
                            path,
                            item,
                            "choice",
                            ("character", "style", "character&style"),
                        )
                    )
                elif field == "encoding":
                    fields.append((path, item, "encoding", None))
                else:
                    if field == "information_extracted" and item == 0:
                        raise ValueError(f"{path} must be greater than zero")
                    fields.append((path, item, "number", (0, 1, False)))
    encoding_bytes = 0
    for path, item, kind, bounds in fields:
        if kind == "bool" and not isinstance(item, bool):
            raise ValueError(f"{path} must be a boolean")
        if kind == "str" and not isinstance(item, str):
            raise ValueError(f"{path} must be a string")
        if kind == "choice" and item not in bounds:
            raise ValueError(f"{path} must be one of {', '.join(bounds)}")
        if kind == "image" and not (
            isinstance(item, ImageData) or isinstance(item, str) and bool(item.strip())
        ):
            raise ValueError(f"{path} must be an image source or resolved ImageData")
        if kind == "number":
            minimum, maximum, integer = bounds
            try:
                finite = math.isfinite(item)
            except (TypeError, OverflowError):
                finite = False
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not finite
                or item < minimum
                or maximum is not None
                and item > maximum
                or integer
                and not float(item).is_integer()
            ):
                raise ValueError(f"{path} is outside its supported numeric range")
            if path in {"width", "height"} and item % 64:
                raise ValueError(f"{path} must be a multiple of 64")
        if kind == "encoding":
            if not isinstance(item, str) or not item:
                raise ValueError(f"{path} must be nonempty encoded Vibe Base64")
            if len(item) > 256 * 1024 * 1024:
                raise ValueError("Vibe encodings exceed the 192 MiB limit")
            try:
                decoded = base64.b64decode(item, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(f"{path} must be valid Base64") from exc
            if not decoded or decoded.startswith(
                (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF")
            ):
                raise ValueError(
                    f"{path} must contain encoded Vibe features, not a raw image"
                )
            encoding_bytes += len(decoded)
    if encoding_bytes > 192 * 1024 * 1024:
        raise ValueError("Vibe encodings exceed the 192 MiB limit")
    if "width" in value and value["width"] * value["height"] > 3_145_728:
        raise ValueError("Dimensions must have area <= 3145728")
    if value.get("character_references") and value.get("vibe_references"):
        raise ValueError("Precise Reference and Vibe cannot be used together")
    if "inpaint" in value and value.get("vibe_references"):
        raise ValueError("Inpaint and Vibe cannot be used together")
    return deepcopy(value)
