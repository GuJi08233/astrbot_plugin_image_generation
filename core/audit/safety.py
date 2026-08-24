"""Safety audit module for prompt and generated images."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import aiohttp

from astrbot.api import logger
from astrbot.api.star import Context

from ..config.manager import ConfigManager
from ..config.models import ImageAuditSettings
from ..shared.constants import DEFAULT_MODERATION_TIMEOUT_SECONDS
from ..shared.logging import log_prefix, safe_log_text

LOG = log_prefix("SafetyAudit")


class SafetyAuditor:
    """Audits prompts and generated images."""

    PROMPT_PLACEHOLDER = "{prompt}"
    AUDIT_RESULT_TAGS = ("audit_result", "result", "output", "json")

    def __init__(self, context: Context, config_manager: ConfigManager):
        self._context = context
        self._config_manager = config_manager

    async def audit_prompt(
        self, prompt: str, unified_msg_origin: str
    ) -> tuple[bool, str]:
        if self._is_umo_whitelisted(unified_msg_origin):
            return True, ""

        settings = self._config_manager.safety_audit_settings.prompt_audit

        hit = self._match_blocked_word(prompt, settings.blocked_words)
        if hit:
            return False, f"命中屏蔽词: {hit}"

        if not settings.enable_ai_audit:
            return True, ""

        review_prompt = self._build_review_prompt(
            settings.ai_prompt,
            prompt,
            append_prompt_if_missing_placeholder=True,
        )
        return await self._audit_with_model(
            unified_msg_origin=unified_msg_origin,
            review_prompt=review_prompt,
            provider_id=settings.ai_provider_id,
            max_retry_attempts=settings.max_retry_attempts,
            image_urls=None,
        )

    async def audit_generated_images(
        self,
        prompt: str,
        image_paths: list[str],
        unified_msg_origin: str,
    ) -> tuple[bool, str]:
        if self._is_umo_whitelisted(unified_msg_origin):
            return True, ""

        settings = self._config_manager.safety_audit_settings.image_audit

        if settings.enable_moderation_audit:
            allowed, reason = await self._audit_with_moderation_api(
                image_paths, settings
            )
            if not allowed:
                return False, reason

        if not settings.enable_ai_audit:
            return True, ""

        review_prompt = self._build_review_prompt(
            settings.ai_prompt,
            prompt,
            append_prompt_if_missing_placeholder=False,
        )
        return await self._audit_with_model(
            unified_msg_origin=unified_msg_origin,
            review_prompt=review_prompt,
            provider_id=settings.ai_provider_id,
            max_retry_attempts=settings.max_retry_attempts,
            image_urls=image_paths,
        )

    def _is_umo_whitelisted(self, unified_msg_origin: str) -> bool:
        umo = unified_msg_origin.strip()
        if not umo:
            return False
        return umo in self._config_manager.safety_audit_settings.umo_whitelist

    def _build_review_prompt(
        self,
        template: str,
        prompt: str,
        *,
        append_prompt_if_missing_placeholder: bool,
    ) -> str:
        review_prompt = template.strip()
        prompt = prompt.strip()

        if not review_prompt:
            review_prompt = (
                "请根据输入内容完成安全审核。"
                '仅输出 JSON：{"allow": true/false, "reason": "简短原因"}。'
            )

        if self.PROMPT_PLACEHOLDER in review_prompt:
            return review_prompt.replace(
                self.PROMPT_PLACEHOLDER,
                self._format_prompt_placeholder(review_prompt, prompt),
            )

        if not append_prompt_if_missing_placeholder or not prompt:
            return review_prompt

        # Preserve legacy prompt-audit configs by appending the prompt when no placeholder exists.
        return f"{review_prompt}\n\n用户提示词：\n{prompt}"

    def _format_prompt_placeholder(self, template: str, prompt: str) -> str:
        """Format user prompt for insertion into a review prompt template."""
        if "<![CDATA[" in template:
            return prompt.replace("]]>", "]]]]><![CDATA[>")
        return prompt

    async def _audit_with_model(
        self,
        *,
        unified_msg_origin: str,
        review_prompt: str,
        provider_id: str,
        max_retry_attempts: int,
        image_urls: list[str] | None,
    ) -> tuple[bool, str]:
        provider = None
        if provider_id:
            provider = self._context.get_provider_by_id(provider_id)
            if not provider:
                logger.warning(
                    f"{LOG} 未找到审核 Provider ID: {safe_log_text(provider_id)}，将回退到当前会话模型"
                )

        if provider is None:
            provider = self._context.get_using_provider(unified_msg_origin)

        if not provider:
            msg = "安全审核异常：未找到可用审核模型"
            logger.warning(f"{LOG} {msg}")
            return False, msg

        attempts = max(1, max_retry_attempts)
        last_reason = "安全审核异常：模型返回为空"
        for attempt in range(1, attempts + 1):
            try:
                response = await provider.text_chat(
                    prompt=review_prompt,
                    image_urls=image_urls or [],
                    persist=False,
                )
                completion_text = (response.completion_text or "").strip()
                decision, reason = self._parse_audit_response(completion_text)
                if self._is_retryable_audit_reason(reason) and attempt < attempts:
                    last_reason = reason
                    logger.warning(
                        f"{LOG} 审核模型返回无法判定，准备重试: {attempt}/{attempts}，原因={safe_log_text(reason, 160)}"
                    )
                    continue
                return decision, reason
            except Exception as exc:
                last_reason = f"安全审核异常：模型调用失败 - {str(exc)[:180]}"
                if attempt < attempts:
                    logger.warning(
                        f"{LOG} 审核模型调用失败，准备重试: {attempt}/{attempts}，错误={safe_log_text(str(exc), 160)}",
                        exc_info=True,
                    )
                    continue
                logger.warning(f"{LOG} {last_reason}", exc_info=True)
                return False, last_reason

        return False, last_reason

    def _is_retryable_audit_reason(self, reason: str) -> bool:
        """Return whether an audit result should be retried."""
        return reason.startswith("安全审核异常：")

    async def _audit_with_moderation_api(
        self,
        image_paths: list[str],
        settings: ImageAuditSettings,
    ) -> tuple[bool, str]:
        if not image_paths:
            return True, ""

        api_key = settings.moderation_api_key.strip()
        if not api_key:
            msg = "安全审核异常：未配置 Moderation API 密钥"
            logger.warning(f"{LOG} {msg}")
            return False, msg

        url = settings.moderation_api_base.strip().rstrip("/") + "/moderations"
        model = settings.moderation_model.strip()
        proxy = settings.moderation_proxy.strip() or None
        rules = self._collect_moderation_rules(settings)
        total = len(image_paths)

        # Gitee AI 的 /moderations 每次请求只接受一张图片，逐张审核。
        for index, path in enumerate(image_paths, start=1):
            try:
                data_uri = self._image_file_to_data_uri(path)
            except OSError as exc:
                msg = f"安全审核异常：读取待审核图片失败 - {str(exc)[:180]}"
                logger.warning(f"{LOG} {msg}")
                return False, msg

            payload: dict[str, object] = {
                "model": model,
                "input": [{"type": "image_url", "image_url": {"url": data_uri}}],
            }
            position = f"第 {index} 张图片" if total > 1 else "图片"
            allowed, reason = await self._moderate_single_image(
                url=url,
                api_key=api_key,
                payload=payload,
                proxy=proxy,
                max_retry_attempts=settings.max_retry_attempts,
                rules=rules,
                position=position,
            )
            if not allowed:
                return False, reason
        return True, "审核通过"

    async def _moderate_single_image(
        self,
        *,
        url: str,
        api_key: str,
        payload: dict[str, object],
        proxy: str | None,
        max_retry_attempts: int,
        rules: list[tuple[str, float | None]],
        position: str,
    ) -> tuple[bool, str]:
        attempts = max(1, max_retry_attempts)
        last_reason = "安全审核异常：Moderation 接口未返回结果"
        for attempt in range(1, attempts + 1):
            try:
                response = await self._post_moderation(url, api_key, payload, proxy)
                hits = self._parse_moderation_result(response, rules)
                if hits:
                    return False, f"{position}命中 Moderation 拦截: {', '.join(hits)}"
                return True, "审核通过"
            except Exception as exc:
                last_reason = (
                    f"安全审核异常：Moderation 接口调用失败 - {str(exc)[:180]}"
                )
                if attempt < attempts:
                    logger.warning(
                        f"{LOG} Moderation 审核失败，准备重试: {attempt}/{attempts}，"
                        f"错误={safe_log_text(str(exc), 160)}"
                    )
                    continue
                logger.warning(f"{LOG} {last_reason}", exc_info=True)
        return False, last_reason

    async def _post_moderation(
        self,
        url: str,
        api_key: str,
        payload: dict[str, object],
        proxy: str | None,
    ) -> dict[str, object]:
        timeout = aiohttp.ClientTimeout(total=DEFAULT_MODERATION_TIMEOUT_SECONDS)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url, json=payload, headers=headers, proxy=proxy
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {body[:160]}")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise RuntimeError("Moderation 返回不是 JSON 对象")
        return data

    def _image_file_to_data_uri(self, path: str) -> str:
        from ..generation.image_utils import detect_mime_type

        data = Path(path).read_bytes()
        mime = detect_mime_type(data)
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _parse_moderation_result(
        self,
        response: dict[str, object],
        rules: list[tuple[str, float | None]],
    ) -> list[str]:
        """Return blocked category hits from a single-image moderation response."""
        results = response.get("results")
        if not isinstance(results, list) or len(results) != 1:
            actual = len(results) if isinstance(results, list) else "无"
            raise RuntimeError(f"Moderation 返回结果数量异常: 期望 1，实际 {actual}")
        result = results[0]
        if not isinstance(result, dict):
            raise RuntimeError("Moderation 返回结果格式异常")
        return self._collect_moderation_hits(result, rules)

    def _collect_moderation_rules(
        self, settings: ImageAuditSettings
    ) -> list[tuple[str, float | None]]:
        """Combine slider thresholds (0 = disabled) with the extra category rules."""
        rules: list[tuple[str, float | None]] = [
            (name, threshold)
            for name, threshold in (
                ("porn", settings.moderation_porn_threshold),
                ("hentai", settings.moderation_hentai_threshold),
                ("sexy", settings.moderation_sexy_threshold),
                ("drawings", settings.moderation_drawings_threshold),
                ("neutral", settings.moderation_neutral_threshold),
            )
            if threshold > 0
        ]
        rules.extend(
            self._parse_moderation_rules(settings.moderation_blocked_categories)
        )
        return rules

    def _parse_moderation_rules(
        self, blocked_categories: list[str]
    ) -> list[tuple[str, float | None]]:
        """Parse category rules like "porn" or "sexy:0.8" (score threshold)."""
        rules: list[tuple[str, float | None]] = []
        for raw in blocked_categories:
            entry = str(raw).strip().replace("：", ":")
            if not entry:
                continue
            name, _, threshold_text = entry.partition(":")
            name = name.strip().lower()
            if not name:
                continue
            threshold: float | None = None
            threshold_text = threshold_text.strip()
            if threshold_text:
                try:
                    threshold = float(threshold_text)
                except ValueError:
                    logger.warning(
                        f"{LOG} 无法解析 Moderation 拦截阈值，按类别布尔判定处理: "
                        f"{safe_log_text(entry, 80)}"
                    )
            rules.append((name, threshold))
        return rules

    def _collect_moderation_hits(
        self,
        result: dict[str, object],
        rules: list[tuple[str, float | None]],
    ) -> list[str]:
        raw_categories = result.get("categories")
        raw_scores = result.get("category_scores")
        categories = {
            str(key).lower(): value
            for key, value in (
                raw_categories.items() if isinstance(raw_categories, dict) else ()
            )
        }
        scores = {
            str(key).lower(): value
            for key, value in (
                raw_scores.items() if isinstance(raw_scores, dict) else ()
            )
        }

        hits: list[str] = []

        if self._to_bool(result.get("flagged")):
            flagged_names = [
                name for name, value in categories.items() if self._to_bool(value)
            ]
            if flagged_names:
                hits.extend(
                    self._format_moderation_hit(name, scores) for name in flagged_names
                )
            else:
                hits.append("flagged")

        for name, threshold in rules:
            if threshold is None:
                matched = bool(self._to_bool(categories.get(name)))
            else:
                score = scores.get(name)
                matched = isinstance(score, (int, float)) and score >= threshold
            if matched:
                hit = self._format_moderation_hit(name, scores)
                if hit not in hits:
                    hits.append(hit)
        return hits

    @staticmethod
    def _format_moderation_hit(name: str, scores: dict[str, object]) -> str:
        score = scores.get(name)
        if isinstance(score, (int, float)):
            return f"{name}({score:.2f})"
        return name

    def _match_blocked_word(self, prompt: str, blocked_words: list[str]) -> str:
        content = prompt.lower()
        for word in blocked_words:
            if word and word.lower() in content:
                return word
        return ""

    def _parse_audit_response(self, text: str) -> tuple[bool, str]:
        if not text:
            return False, "安全审核异常：模型返回为空"

        payload = self._extract_json(text)
        if payload is not None:
            allow = self._to_bool(self._first_present(payload, "allow", "allowed"))
            reason = str(
                self._first_present(payload, "reason", "message", "detail") or ""
            ).strip()
            if allow is not None:
                return allow, reason or ("审核通过" if allow else "审核未通过")

        lowered = text.lower()
        reject_tokens = ("reject", "deny", "forbid", "不通过", "违规", "拒绝", "不允许")
        allow_tokens = ("allow", "pass", "safe", "通过", "安全", "允许")

        if any(token in lowered for token in reject_tokens):
            return False, text[:120]
        if any(token in lowered for token in allow_tokens):
            return True, text[:120]

        return False, f"安全审核异常：无法判定审核结果，原始返回: {text[:120]}"

    def _extract_json(self, text: str) -> dict[str, object] | None:
        for candidate in self._json_candidates(text):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _json_candidates(self, text: str) -> list[str]:
        """Return likely JSON objects from model output."""
        text = text.strip()
        if not text:
            return []

        candidates = [text]
        candidates.extend(self._extract_fenced_blocks(text))
        candidates.extend(self._extract_tagged_blocks(text))
        candidates.extend(self._extract_balanced_json_objects(text))

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(normalized)
        return unique_candidates

    def _extract_fenced_blocks(self, text: str) -> list[str]:
        pattern = r"```(?:json|JSON)?\s*([\s\S]*?)\s*```"
        return [match.group(1).strip() for match in re.finditer(pattern, text)]

    def _extract_tagged_blocks(self, text: str) -> list[str]:
        candidates: list[str] = []
        for tag in self.AUDIT_RESULT_TAGS:
            pattern = rf"<{tag}\b[^>]*>\s*([\s\S]*?)\s*</{tag}>"
            candidates.extend(
                match.group(1).strip()
                for match in re.finditer(pattern, text, flags=re.IGNORECASE)
            )
        return candidates

    def _extract_balanced_json_objects(self, text: str) -> list[str]:
        candidates: list[str] = []
        in_string = False
        escape = False
        depth = 0
        start = -1

        for index, char in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
                continue

            if char != "}" or depth == 0:
                continue

            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : index + 1])
                start = -1
        return candidates

    def _first_present(self, payload: dict[str, object], *keys: str) -> object:
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _to_bool(self, value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "allow", "pass", "通过", "允许"}:
                return True
            if lowered in {"false", "0", "no", "reject", "deny", "拒绝", "不通过"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None
