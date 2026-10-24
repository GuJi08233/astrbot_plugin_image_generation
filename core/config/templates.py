"""Prompt template helpers for presets and personas."""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger

from ..generation.reference_collector import normalize_string_items
from ..shared.logging import log_prefix
from .models import PersonaTemplate, PresetTemplate
from .validator import TEMPLATE_KEY_FIELD

LOG = log_prefix("Config")


def build_generation_prompt(
    *,
    preset_prompts: list[str] | None = None,
    persona_prompts: list[str] | None = None,
    extra_prompt: str = "",
) -> str:
    """Build a structured generation prompt from multiple sources.

    Args:
        preset_prompts: Prompt fragments from matched presets.
        persona_prompts: Prompt fragments from matched personas.
        extra_prompt: User-provided prompt for the current request.

    Returns:
        Original text for single-source prompts, otherwise a lightweight
        sectioned prompt that keeps source intent clear for the model.
    """
    preset_parts = [part.strip() for part in preset_prompts or [] if part.strip()]
    persona_parts = [part.strip() for part in persona_prompts or [] if part.strip()]
    extra_text = str(extra_prompt or "").strip()

    sections: list[tuple[str, list[str]]] = []
    if persona_parts:
        sections.append(("人物设定", persona_parts))
    if preset_parts:
        sections.append(("预设提示词", preset_parts))
    if extra_text:
        sections.append(("附加提示词", [extra_text]))

    if not sections:
        return ""
    if len(sections) == 1 and len(sections[0][1]) == 1:
        return sections[0][1][0]

    blocks: list[str] = []
    for title, parts in sections:
        blocks.append(f"[{title}]\n" + "\n".join(parts))
    return "\n\n".join(blocks).strip()


def normalize_name_items(raw: Any) -> list[str]:
    """Normalize one or many preset/persona names from tool arguments."""
    names: list[str] = []
    seen: set[str] = set()
    for item in normalize_string_items(raw):
        for name in item.split():
            normalized = name.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(normalized)
    return names


def find_named_entry(entries: dict[str, Any], token: str) -> str | None:
    """Find an entry by exact or case-insensitive name."""
    if token in entries:
        return token
    lowered_token = token.lower()
    for name in entries:
        if name.lower() == lowered_token:
            return name
    return None


def template_image_label(kind: str, name: str) -> str:
    """Return a readable label for a template reference image."""
    return f"{'人设' if kind == 'persona' else '预设'}「{name}」"


def resolve_named_templates(
    names: list[str],
    entries: dict[str, Any],
    *,
    kind: str,
    aspect_ratio: str,
    resolution: str,
) -> tuple[list[str], str, str, list[str], list[tuple[str, str]], str | None]:
    """Resolve one kind of explicit template name.

    Args:
        names: Names already split and ordered by the caller.
        entries: Preset or persona templates for ``kind``.
        kind: ``preset`` or ``persona``.
        aspect_ratio: Current aspect ratio, updated by preset templates.
        resolution: Current resolution, updated by preset templates.

    Returns:
        Prompt fragments, aspect ratio, resolution, matched names, template
        reference images, and a plain missing-name error. Persona resolution
        leaves ratio and resolution unchanged, and the first preset that
        declares a size wins.
    """
    prompts: list[str] = []
    matched_names: list[str] = []
    template_images: list[tuple[str, str]] = []
    ratio_from_preset = False
    resolution_from_preset = False
    label = "人设" if kind == "persona" else "预设"

    for name in names:
        matched_name = find_named_entry(entries, name)
        if not matched_name:
            return [], aspect_ratio, resolution, [], [], f"{label}不存在: {name}"

        template = entries[matched_name]
        if template_prompt := str(getattr(template, "prompt", "") or "").strip():
            prompts.append(template_prompt)
        if kind == "preset":
            if not ratio_from_preset and (
                ratio := str(getattr(template, "aspect_ratio", "") or "").strip()
            ):
                aspect_ratio = ratio
                ratio_from_preset = True
            if not resolution_from_preset and (
                size := str(getattr(template, "resolution", "") or "").strip()
            ):
                resolution = size
                resolution_from_preset = True
        if image := str(getattr(template, "image", "") or "").strip():
            template_images.append((template_image_label(kind, matched_name), image))
        matched_names.append(matched_name)

    return prompts, aspect_ratio, resolution, matched_names, template_images, None


def _ascii_token_boundary_ok(text: str, start: int, end: int) -> bool:
    """Return whether an ASCII template name sits on a token boundary.

    CJK names are matched as substrings. ASCII names must not match inside a
    longer ASCII word, so ``cat`` does not match ``category``.
    """
    name = text[start:end]
    if not name.isascii():
        return True
    glued_to_previous = (
        start > 0
        and text[start - 1].isascii()
        and text[start - 1].isalnum()
        and name[0].isalnum()
    )
    glued_to_next = (
        end < len(text)
        and text[end].isascii()
        and text[end].isalnum()
        and name[-1].isalnum()
    )
    return not glued_to_previous and not glued_to_next


def extract_templates_from_prompt(
    prompt: str,
    presets: dict[str, PresetTemplate],
    personas: dict[str, PersonaTemplate],
    *,
    aspect_ratio: str,
    resolution: str,
    skip_names: list[str] | None = None,
) -> tuple[
    str,
    str,
    str,
    list[str],
    list[str],
    list[str],
    list[str],
    list[tuple[str, str]],
]:
    """Match configured preset/persona names anywhere in prompt text.

    Longer names win at the same position. Each template is applied at most
    once. Scanning is linear in prompt length times template count and stays
    cheap for typical plugin-scale lists.

    Args:
        prompt: User-provided prompt that may contain template names.
        presets: Configured preset templates.
        personas: Configured persona templates.
        aspect_ratio: Current aspect ratio, updated by preset templates.
        resolution: Current resolution, updated by preset templates.
        skip_names: Template names already applied by explicit parameters.

    Returns:
        Remaining extra prompt, aspect ratio, resolution, preset prompt
        fragments, persona prompt fragments, matched preset names, matched
        persona names, and template reference images in scan order.
    """
    extra_source = str(prompt or "").strip()
    empty: tuple[
        str,
        str,
        str,
        list[str],
        list[str],
        list[str],
        list[str],
        list[tuple[str, str]],
    ] = ("", aspect_ratio, resolution, [], [], [], [], [])
    if not extra_source:
        return empty

    needles: dict[str, tuple[str, str]] = {}
    for name in personas:
        key = name.strip().lower()
        if key:
            needles[key] = ("persona", name)
    for name in presets:
        key = name.strip().lower()
        if key:
            needles[key] = ("preset", name)
    ordered_needles = sorted(
        ((key, kind, original) for key, (kind, original) in needles.items()),
        key=lambda item: (-len(item[0]), item[0]),
    )
    if not ordered_needles:
        return extra_source, aspect_ratio, resolution, [], [], [], [], []

    skip = {name.strip().lower() for name in skip_names or [] if str(name).strip()}
    lowered = extra_source.lower()
    preset_prompts: list[str] = []
    persona_prompts: list[str] = []
    matched_presets: list[str] = []
    matched_personas: list[str] = []
    template_images: list[tuple[str, str]] = []
    ratio_from_preset = False
    resolution_from_preset = False
    leftover_chars: list[str] = []
    index = 0
    length = len(extra_source)
    while index < length:
        found: tuple[str, str, int] | None = None
        for key, kind, original in ordered_needles:
            end = index + len(key)
            if end > length or lowered[index:end] != key:
                continue
            if not _ascii_token_boundary_ok(extra_source, index, end):
                continue
            found = (kind, original, end)
            break
        if found is None:
            leftover_chars.append(extra_source[index])
            index += 1
            continue

        kind, original, end = found
        if original.lower() in skip:
            index = end
            continue
        skip.add(original.lower())
        if kind == "preset":
            preset = presets[original]
            if preset_prompt := preset.prompt.strip():
                preset_prompts.append(preset_prompt)
            if not ratio_from_preset and (ratio := preset.aspect_ratio.strip()):
                aspect_ratio = ratio
                ratio_from_preset = True
            if not resolution_from_preset and (size := preset.resolution.strip()):
                resolution = size
                resolution_from_preset = True
            if preset.image:
                template_images.append(
                    (template_image_label("preset", original), preset.image)
                )
            matched_presets.append(original)
        else:
            persona = personas[original]
            persona_prompt = persona.prompt.strip()
            if persona_prompt:
                persona_prompts.append(persona_prompt)
            if persona.image:
                template_images.append(
                    (template_image_label("persona", original), persona.image)
                )
            matched_personas.append(original)
        index = end

    extra = extra_source
    if matched_presets or matched_personas:
        extra = " ".join("".join(leftover_chars).split())
    return (
        extra,
        aspect_ratio,
        resolution,
        preset_prompts,
        persona_prompts,
        matched_presets,
        matched_personas,
        template_images,
    )


def parse_preset_content(content: Any) -> tuple[str, str, str]:
    """Parse legacy preset content into prompt and optional size overrides.

    Args:
        content: Preset content, either a plain prompt or a JSON object with
            ``prompt``, ``aspect_ratio`` and ``resolution`` fields.

    Returns:
        Prompt text, aspect ratio, and resolution. Unparsable JSON keeps the
        raw content as the prompt so no preset is silently dropped.
    """
    raw_content = str(content or "").strip()
    if not raw_content.startswith("{"):
        return raw_content, "", ""

    try:
        preset_data = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content, "", ""

    if not isinstance(preset_data, dict):
        return raw_content, "", ""

    return (
        str(preset_data.get("prompt", "") or "").strip(),
        str(preset_data.get("aspect_ratio", "") or "").strip(),
        str(preset_data.get("resolution", "") or "").strip(),
    )


def format_template_summary(
    matched_presets: list[str],
    matched_personas: list[str],
) -> tuple[str | None, str]:
    """Format matched preset/persona names for task metadata."""
    if matched_presets and matched_personas:
        return (
            "；".join(
                (
                    f"预设: {'、'.join(matched_presets)}",
                    f"人设: {'、'.join(matched_personas)}",
                )
            ),
            "预设/人设",
        )
    if matched_presets:
        return "、".join(matched_presets), "预设"
    if matched_personas:
        return "、".join(matched_personas), "人设"
    return None, "预设"


class ConfigTemplateStoreMixin:
    """Mixin for reading and writing preset/persona template config."""

    def _load_presets(
        self,
        presets_config: Any,
        preset_templates_config: Any,
    ) -> dict[str, PresetTemplate]:
        """Load prompt presets from legacy and template configs.

        Legacy ``名称:提示词`` entries are still read for upgrade compatibility,
        while structured ``preset_templates`` entries win on name conflicts and
        keep the legacy order for unchanged names.

        Args:
            presets_config: Legacy preset string list config value.
            preset_templates_config: Structured preset template config value.

        Returns:
            Merged preset templates keyed by name.
        """
        presets: dict[str, PresetTemplate] = {}
        if isinstance(presets_config, list):
            for preset_str in presets_config:
                if not isinstance(preset_str, str) or ":" not in preset_str:
                    continue
                name, content = preset_str.split(":", 1)
                name = name.strip()
                prompt, aspect_ratio, resolution = parse_preset_content(content)
                if name and (prompt or aspect_ratio or resolution):
                    presets[name] = PresetTemplate(
                        name=name,
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                    )

        if isinstance(preset_templates_config, list):
            for item in preset_templates_config:
                if not isinstance(item, dict):
                    continue

                name = str(item.get("preset_name") or item.get("name") or "").strip()
                prompt = str(
                    item.get("preset_prompt") or item.get("prompt") or ""
                ).strip()
                aspect_ratio = str(
                    item.get("preset_aspect_ratio") or item.get("aspect_ratio") or ""
                ).strip()
                resolution = str(
                    item.get("preset_resolution") or item.get("resolution") or ""
                ).strip()
                image = self._parse_file_value(
                    item.get("preset_image")
                    or item.get("image")
                    or item.get("reference_image")
                )
                if name and (prompt or image or aspect_ratio or resolution):
                    presets[name] = PresetTemplate(
                        name=name,
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                        image=image,
                    )
        return presets

    def migrate_legacy_presets(self) -> int:
        """Copy legacy presets into the template config and return the count.

        Legacy entries stay untouched so a downgrade keeps working, and only
        names without a template entry are copied. The config file is saved once
        when anything was copied.

        Returns:
            Number of presets copied into ``preset_templates``.
        """
        prompt_templates_cfg = self._get_writable_prompt_templates_config()
        legacy_presets = prompt_templates_cfg.get("presets")
        if not isinstance(legacy_presets, list):
            return 0

        templates = prompt_templates_cfg.get("preset_templates")
        if not isinstance(templates, list):
            templates = []
        known_names = {
            str(item.get("preset_name") or item.get("name") or "").strip().lower()
            for item in templates
            if isinstance(item, dict)
        }

        copied = 0
        for preset_str in legacy_presets:
            if not isinstance(preset_str, str) or ":" not in preset_str:
                continue
            name, content = preset_str.split(":", 1)
            name = name.strip()
            if not name or name.lower() in known_names:
                continue

            prompt, aspect_ratio, resolution = parse_preset_content(content)
            if not (prompt or aspect_ratio or resolution):
                continue

            templates.append(
                {
                    TEMPLATE_KEY_FIELD: "preset",
                    "preset_name": name,
                    "preset_prompt": prompt,
                    "preset_aspect_ratio": aspect_ratio,
                    "preset_resolution": resolution,
                    "preset_image": [],
                }
            )
            known_names.add(name.lower())
            copied += 1

        if copied:
            prompt_templates_cfg["preset_templates"] = templates
            self._config.save_config()
        return copied

    def _get_writable_prompt_templates_config(self) -> dict[str, Any]:
        """Return the grouped prompt-template config for command-side updates."""
        value = self._config.setdefault("prompt_templates", {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} prompt_templates 配置格式错误，已重置为空对象")
        value = {}
        self._config["prompt_templates"] = value
        return value

    def _save_presets_config(self) -> None:
        """Persist presets into the structured preset template config."""
        prompt_templates_cfg = self._get_writable_prompt_templates_config()
        prompt_templates_cfg["preset_templates"] = [
            {
                TEMPLATE_KEY_FIELD: "preset",
                "preset_name": preset.name,
                "preset_prompt": preset.prompt,
                "preset_aspect_ratio": preset.aspect_ratio,
                "preset_resolution": preset.resolution,
                # The file field is stored as a list, matching the dashboard.
                "preset_image": [preset.image] if preset.image else [],
            }
            for preset in self._plugin_config.presets.values()
        ]
        self._config.save_config()

    def _load_personas(self, personas_config: Any) -> dict[str, PersonaTemplate]:
        """Load configured persona templates."""
        personas: dict[str, PersonaTemplate] = {}
        if not isinstance(personas_config, list):
            return personas

        for item in personas_config:
            if not isinstance(item, dict):
                continue

            name = str(item.get("persona_name") or item.get("name") or "").strip()
            prompt = str(item.get("persona_prompt") or item.get("prompt") or "").strip()
            image = self._parse_file_value(
                item.get("persona_image")
                or item.get("image")
                or item.get("reference_image")
            )
            if name and (prompt or image):
                personas[name] = PersonaTemplate(name=name, prompt=prompt, image=image)
        return personas

    def _parse_file_value(self, raw: Any) -> str:
        """Extract the first usable file path or URL from a file config value."""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            for item in raw:
                if parsed := self._parse_file_value(item):
                    return parsed
            return ""
        if isinstance(raw, dict):
            for key in ("path", "file", "url", "name"):
                if parsed := self._parse_file_value(raw.get(key)):
                    return parsed
        return ""

    def save_preset(self, name: str, content: str) -> None:
        """Save a prompt preset from simple or advanced JSON content.

        Args:
            name: Preset name to create or overwrite.
            content: Plain prompt, or a JSON object with ``prompt``,
                ``aspect_ratio`` and ``resolution`` fields.
        """
        prompt, aspect_ratio, resolution = parse_preset_content(content)
        self.save_preset_template(
            name=name,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

    def save_preset_template(
        self,
        *,
        name: str,
        prompt: str,
        aspect_ratio: str = "",
        resolution: str = "",
        image: str = "",
    ) -> None:
        """Save a structured preset template and return nothing."""
        self._plugin_config.presets[name] = PresetTemplate(
            name=name,
            prompt=str(prompt or "").strip(),
            aspect_ratio=str(aspect_ratio or "").strip(),
            resolution=str(resolution or "").strip(),
            image=str(image or "").strip(),
        )
        self._save_presets_config()

    def delete_preset(self, name: str) -> bool:
        """Delete a prompt preset and return whether it existed.

        The hidden legacy entry with the same name is removed as well, so a
        deleted preset does not reappear from the compatibility config.
        """
        if name not in self._plugin_config.presets:
            return False
        del self._plugin_config.presets[name]

        prompt_templates_cfg = self._get_writable_prompt_templates_config()
        legacy_presets = prompt_templates_cfg.get("presets")
        if isinstance(legacy_presets, list):
            lowered_name = name.strip().lower()
            remaining = [
                item
                for item in legacy_presets
                if not (
                    isinstance(item, str)
                    and item.split(":", 1)[0].strip().lower() == lowered_name
                )
            ]
            if len(remaining) != len(legacy_presets):
                prompt_templates_cfg["presets"] = remaining

        self._save_presets_config()
        return True


__all__ = (
    "ConfigTemplateStoreMixin",
    "build_generation_prompt",
    "extract_templates_from_prompt",
    "find_named_entry",
    "format_template_summary",
    "normalize_name_items",
    "parse_preset_content",
    "resolve_named_templates",
    "template_image_label",
)
