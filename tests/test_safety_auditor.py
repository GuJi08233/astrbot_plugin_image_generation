from __future__ import annotations

import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class _Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


ROOT = Path(__file__).resolve().parents[1]
astrbot_module = types.ModuleType("astrbot")
astrbot_api_module = types.ModuleType("astrbot.api")
astrbot_api_module.logger = _Logger()
astrbot_api_star_module = types.ModuleType("astrbot.api.star")
astrbot_api_star_module.Context = object
astrbot_core_module = types.ModuleType("astrbot.core")
astrbot_core_config_module = types.ModuleType("astrbot.core.config")
astrbot_core_config_astrbot_config_module = types.ModuleType(
    "astrbot.core.config.astrbot_config"
)
astrbot_core_config_astrbot_config_module.AstrBotConfig = dict
sys.modules.setdefault("astrbot", astrbot_module)
sys.modules.setdefault("astrbot.api", astrbot_api_module)
sys.modules.setdefault("astrbot.api.star", astrbot_api_star_module)
sys.modules.setdefault("astrbot.core", astrbot_core_module)
sys.modules.setdefault("astrbot.core.config", astrbot_core_config_module)
sys.modules.setdefault(
    "astrbot.core.config.astrbot_config",
    astrbot_core_config_astrbot_config_module,
)

plugin_module = types.ModuleType("astrbot_plugin_image_generation")
plugin_module.__path__ = [str(ROOT)]
core_module = types.ModuleType("astrbot_plugin_image_generation.core")
core_module.__path__ = [str(ROOT / "core")]
core_audit_module = types.ModuleType("astrbot_plugin_image_generation.core.audit")
core_audit_module.__path__ = [str(ROOT / "core" / "audit")]
core_config_module = types.ModuleType("astrbot_plugin_image_generation.core.config")
core_config_module.__path__ = [str(ROOT / "core" / "config")]
core_generation_module = types.ModuleType(
    "astrbot_plugin_image_generation.core.generation"
)
core_generation_module.__path__ = [str(ROOT / "core" / "generation")]
core_shared_module = types.ModuleType("astrbot_plugin_image_generation.core.shared")
core_shared_module.__path__ = [str(ROOT / "core" / "shared")]
sys.modules.setdefault("astrbot_plugin_image_generation", plugin_module)
sys.modules.setdefault("astrbot_plugin_image_generation.core", core_module)
sys.modules.setdefault("astrbot_plugin_image_generation.core.audit", core_audit_module)
sys.modules.setdefault(
    "astrbot_plugin_image_generation.core.config", core_config_module
)
sys.modules.setdefault(
    "astrbot_plugin_image_generation.core.generation", core_generation_module
)
sys.modules.setdefault(
    "astrbot_plugin_image_generation.core.shared", core_shared_module
)

SafetyAuditor = importlib.import_module(
    "astrbot_plugin_image_generation.core.audit.safety"
).SafetyAuditor
ImageAuditSettings = importlib.import_module(
    "astrbot_plugin_image_generation.core.config.models"
).ImageAuditSettings


def _moderation_result(
    *,
    flagged: bool = False,
    categories: dict[str, bool] | None = None,
    scores: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "flagged": flagged,
        "categories": categories or {},
        "category_scores": scores or {},
    }


def _response(*results: dict[str, object]) -> dict[str, object]:
    return {"id": "test", "model": "nsfw-classifier", "results": list(results)}


class SafetyAuditorModerationTest(unittest.TestCase):
    def setUp(self):
        self.auditor = SafetyAuditor(context=object(), config_manager=object())
        self.settings = ImageAuditSettings(
            enable_moderation_audit=True,
            moderation_api_key="test-key",
            max_retry_attempts=2,
        )

    def _run_audit(self, response, settings=None, side_effect=None):
        post = AsyncMock(return_value=response, side_effect=side_effect)
        with (
            patch.object(self.auditor, "_post_moderation", post),
            patch.object(
                self.auditor,
                "_image_file_to_data_uri",
                return_value="data:image/png;base64,AA==",
            ),
        ):
            return (
                asyncio.run(
                    self.auditor._audit_with_moderation_api(
                        ["img.png"], settings or self.settings
                    )
                ),
                post,
            )

    def test_allows_unflagged_result(self):
        (allowed, reason), _ = self._run_audit(_response(_moderation_result()))
        self.assertTrue(allowed)
        self.assertEqual(reason, "审核通过")

    def test_blocks_flagged_result_with_category_details(self):
        response = _response(
            _moderation_result(
                flagged=True,
                categories={"porn": True, "neutral": False},
                scores={"porn": 0.93},
            )
        )
        (allowed, reason), _ = self._run_audit(response)
        self.assertFalse(allowed)
        self.assertIn("porn(0.93)", reason)

    def test_blocks_flagged_result_without_categories(self):
        (allowed, reason), _ = self._run_audit(
            _response(_moderation_result(flagged=True))
        )
        self.assertFalse(allowed)
        self.assertIn("flagged", reason)

    def test_blocked_category_rule_hits_boolean(self):
        self.settings.moderation_blocked_categories = ["Sexy"]
        response = _response(
            _moderation_result(categories={"sexy": True}, scores={"sexy": 0.4})
        )
        (allowed, reason), _ = self._run_audit(response)
        self.assertFalse(allowed)
        self.assertIn("sexy(0.40)", reason)

    def test_blocked_category_threshold_rule(self):
        self.settings.moderation_blocked_categories = ["sexy:0.8"]
        blocked = _response(
            _moderation_result(categories={"sexy": False}, scores={"sexy": 0.85})
        )
        (allowed, reason), _ = self._run_audit(blocked)
        self.assertFalse(allowed)
        self.assertIn("sexy(0.85)", reason)

        passed = _response(
            _moderation_result(categories={"sexy": False}, scores={"sexy": 0.5})
        )
        (allowed, _), _ = self._run_audit(passed)
        self.assertTrue(allowed)

    def test_multiple_images_requested_one_by_one_and_reports_position(self):
        post = AsyncMock(
            side_effect=[
                _response(_moderation_result()),
                _response(
                    _moderation_result(flagged=True, categories={"hentai": True})
                ),
            ]
        )
        with (
            patch.object(self.auditor, "_post_moderation", post),
            patch.object(
                self.auditor,
                "_image_file_to_data_uri",
                return_value="data:image/png;base64,AA==",
            ),
        ):
            allowed, reason = asyncio.run(
                self.auditor._audit_with_moderation_api(
                    ["a.png", "b.png"], self.settings
                )
            )
        self.assertFalse(allowed)
        self.assertIn("第 2 张图片", reason)
        self.assertEqual(post.await_count, 2)
        for call in post.await_args_list:
            self.assertEqual(len(call.args[2]["input"]), 1)

    def test_result_count_mismatch_fails_closed_with_retry(self):
        (allowed, reason), post = self._run_audit(_response())
        self.assertFalse(allowed)
        self.assertIn("安全审核异常", reason)
        self.assertEqual(post.await_count, 2)

    def test_http_error_fails_closed_after_retries(self):
        (allowed, reason), post = self._run_audit(
            None, side_effect=RuntimeError("HTTP 500: boom")
        )
        self.assertFalse(allowed)
        self.assertIn("Moderation 接口调用失败", reason)
        self.assertEqual(post.await_count, 2)

    def test_missing_api_key_fails_closed(self):
        settings = ImageAuditSettings(
            enable_moderation_audit=True, moderation_api_key="  "
        )
        allowed, reason = asyncio.run(
            self.auditor._audit_with_moderation_api(["img.png"], settings)
        )
        self.assertFalse(allowed)
        self.assertIn("未配置 Moderation API 密钥", reason)

    def test_empty_image_list_passes(self):
        allowed, reason = asyncio.run(
            self.auditor._audit_with_moderation_api([], self.settings)
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_slider_threshold_blocks_when_score_reached(self):
        self.settings.moderation_sexy_threshold = 0.8
        blocked = _response(
            _moderation_result(categories={"sexy": False}, scores={"sexy": 0.85})
        )
        (allowed, reason), _ = self._run_audit(blocked)
        self.assertFalse(allowed)
        self.assertIn("sexy(0.85)", reason)

        passed = _response(
            _moderation_result(categories={"sexy": False}, scores={"sexy": 0.5})
        )
        (allowed, _), _ = self._run_audit(passed)
        self.assertTrue(allowed)

    def test_slider_threshold_zero_is_disabled(self):
        self.settings.moderation_porn_threshold = 0.0
        response = _response(
            _moderation_result(categories={"porn": False}, scores={"porn": 0.99})
        )
        (allowed, _), _ = self._run_audit(response)
        self.assertTrue(allowed)

    def test_slider_thresholds_combine_with_extra_rules(self):
        self.settings.moderation_hentai_threshold = 0.7
        self.settings.moderation_blocked_categories = ["porn:0.5"]
        rules = self.auditor._collect_moderation_rules(self.settings)
        self.assertEqual(rules, [("hentai", 0.7), ("porn", 0.5)])

    def test_rule_parsing_supports_fullwidth_colon_and_invalid_threshold(self):
        rules = self.auditor._parse_moderation_rules(
            ["porn：0.6", "hentai:abc", " ", "SEXY"]
        )
        self.assertEqual(rules, [("porn", 0.6), ("hentai", None), ("sexy", None)])


if __name__ == "__main__":
    unittest.main()
