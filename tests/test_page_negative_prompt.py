from __future__ import annotations

import importlib
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class PageNegativePromptTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load the page API without starting AstrBot or accessing real services."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_page_negative_prompt_test_plugin"
        modules = {}
        for suffix in (
            "",
            "core",
            "core.page",
            "core.shared",
            "core.tasks",
            "core.generation",
        ):
            name = f"{namespace}.{suffix}" if suffix else namespace
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.split(".")))]
            modules[name] = module
        api = types.ModuleType("astrbot.api")
        api.logger = logging.getLogger(__name__)
        web = types.ModuleType("astrbot.api.web")
        web.PluginUploadFile = object
        web.request = SimpleNamespace(json=AsyncMock(), username="test-admin")
        web.json_response = lambda data: data
        web.error_response = lambda message, **kwargs: {"error": message, **kwargs}
        web.file_response = MagicMock()
        modules.update({"astrbot.api": api, "astrbot.api.web": web})
        with patch.dict(sys.modules, modules):
            page_module = importlib.import_module(f"{namespace}.core.page.api")
            shared = importlib.import_module(f"{namespace}.core.shared.types")
        self.request = web.request
        self.AdapterType = shared.AdapterType
        self.AdapterConfig = shared.AdapterConfig
        provider = shared.AdapterConfig(
            type=shared.AdapterType.SILICONFLOW,
            name="Provider",
            model="image",
            available_models=["image"],
        )
        config = SimpleNamespace(
            adapter_config=provider,
            _all_provider_configs=[provider],
            default_image_count=1,
            max_image_count=4,
            default_aspect_ratio="不指定",
            default_resolution="不指定",
            presets={},
            personas={},
            max_running_generation_tasks=1,
            max_queued_generation_tasks=4,
            enable_generation_task_history=False,
            generation_task_history_limit=50,
            generation_task_history_retention_days=7,
            save_model_setting=MagicMock(),
        )
        self.plugin = SimpleNamespace(
            config_manager=config,
            normalize_image_count=MagicMock(return_value=1),
            public_api=SimpleNamespace(
                submit_generation_task=AsyncMock(
                    return_value=SimpleNamespace(
                        ok=True, message="accepted", task_id="test-task"
                    )
                )
            ),
            task_manager=SimpleNamespace(
                get_generation_task=MagicMock(return_value=None),
                can_accept_generation_task=MagicMock(return_value=True),
            ),
            generator=SimpleNamespace(
                adapter=SimpleNamespace(
                    get_capabilities=lambda: shared.ImageCapability.TEXT_TO_IMAGE
                )
            ),
            has_required_api_key=lambda: True,
        )
        self.page = page_module.ImageGenerationPageAPI(self.plugin)

    async def test_generation_preserves_inherit_override_and_clear_values(self):
        """Forward the three request modes without changing provider defaults."""
        for fields, expected in (
            ({}, None),
            ({"negative_prompt": None}, None),
            ({"negative_prompt": "blur, watermark"}, "blur, watermark"),
            ({"negative_prompt": ""}, ""),
        ):
            with self.subTest(fields=fields):
                self.request.json.return_value = {
                    "prompt": "white cat",
                    "model": "Provider/image",
                    **fields,
                }
                result = await self.page.page_generate()
                self.assertTrue(result["ok"])
                self.assertEqual(
                    self.plugin.public_api.submit_generation_task.call_args.kwargs[
                        "negative_prompt"
                    ],
                    expected,
                )
        self.assertEqual(self.plugin.config_manager.adapter_config.extra, {})

    async def test_invalid_negative_prompt_is_rejected_before_model_changes(self):
        """Reject non-string JSON values before changing the selected model."""
        for value in (False, 1, [], {}):
            with self.subTest(value=value):
                self.request.json.return_value = {
                    "prompt": "white cat",
                    "model": "Other/image",
                    "negative_prompt": value,
                }
                result = await self.page.page_generate()
                self.assertEqual(result["status_code"], 400)
        self.plugin.normalize_image_count.assert_not_called()
        self.plugin.config_manager.save_model_setting.assert_not_called()
        self.plugin.public_api.submit_generation_task.assert_not_awaited()

    async def test_state_declares_negative_support_for_each_adapter(self):
        """Expose support for configured models and the current fallback model."""
        config = self.plugin.config_manager
        config._all_provider_configs = [
            self.AdapterConfig(
                type=adapter_type,
                name=adapter_type.value,
                available_models=["image"],
            )
            for adapter_type in (
                self.AdapterType.NOVELAI,
                self.AdapterType.SILICONFLOW,
                self.AdapterType.MODELSCOPE,
                self.AdapterType.OPENAI,
                self.AdapterType.OPENAI_CHAT,
                self.AdapterType.GEMINI,
            )
        ]
        result = await self.page.page_state()
        settings = result["model_settings"]
        for adapter_type in (
            self.AdapterType.NOVELAI,
            self.AdapterType.SILICONFLOW,
            self.AdapterType.MODELSCOPE,
            self.AdapterType.OPENAI,
            self.AdapterType.OPENAI_CHAT,
        ):
            self.assertTrue(
                settings[f"{adapter_type.value}/image"]["supports_negative_prompt"]
            )
        self.assertFalse(settings["gemini/image"]["supports_negative_prompt"])
        self.assertEqual(settings["openai/image"]["negative_prompt_mode"], "prompt")
        self.assertEqual(
            settings["openai_chat/image"]["negative_prompt_mode"], "prompt"
        )
        self.assertTrue(settings["Provider/image"]["supports_negative_prompt"])

    async def test_openai_pair_preserves_prompt_and_negative_request_modes(self):
        """Forward one prompt pair without changing provider defaults or modes."""
        for adapter_type in (self.AdapterType.OPENAI, self.AdapterType.OPENAI_CHAT):
            for mode in ("prompt", "parameter"):
                provider = self.AdapterConfig(
                    type=adapter_type,
                    name="Provider",
                    model="image",
                    available_models=["image"],
                    extra={
                        "negative_prompt": "default negative",
                        "negative_prompt_mode": mode,
                    },
                )
                self.plugin.config_manager.adapter_config = provider
                self.plugin.config_manager._all_provider_configs = [provider]
                state = await self.page.page_state()
                self.assertEqual(
                    state["model_settings"]["Provider/image"]["negative_prompt_mode"],
                    mode,
                )
                for value in (None, "watermark, text", ""):
                    with self.subTest(adapter=adapter_type, mode=mode, value=value):
                        self.request.json.return_value = {
                            "prompt": "white cat in sunlight",
                            "negative_prompt": value,
                            "model": "Provider/image",
                        }
                        result = await self.page.page_generate()
                        self.assertTrue(result["ok"])
                        submitted = self.plugin.public_api.submit_generation_task.call_args.kwargs
                        self.assertEqual(submitted["prompt"], "white cat in sunlight")
                        self.assertEqual(submitted["negative_prompt"], value)
                        self.assertIsNone(submitted["novelai"])
                self.assertEqual(
                    provider.extra,
                    {
                        "negative_prompt": "default negative",
                        "negative_prompt_mode": mode,
                    },
                )


if __name__ == "__main__":
    unittest.main()
