from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import aiohttp


class NegativePromptAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load adapters in isolation and prohibit real provider requests."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_negative_prompt_adapter_test_plugin"
        modules = {}
        for suffix in (
            "",
            "adapter",
            "core",
            "core.adapters",
            "core.shared",
            "core.generation",
        ):
            name = f"{namespace}.{suffix}" if suffix else namespace
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.split(".")))]
            modules[name] = module
        api = types.ModuleType("astrbot.api")
        api.logger = logging.getLogger(__name__)
        modules["astrbot.api"] = api
        with patch.dict(sys.modules, modules):
            self.shared = importlib.import_module(f"{namespace}.core.shared.types")
            self.adapters = []
            for module_name, class_name, adapter_type in (
                ("novelai_adapter", "NovelAIAdapter", self.shared.AdapterType.NOVELAI),
                (
                    "siliconflow_adapter",
                    "SiliconFlowAdapter",
                    self.shared.AdapterType.SILICONFLOW,
                ),
                (
                    "modelscope_adapter",
                    "ModelScopeAdapter",
                    self.shared.AdapterType.MODELSCOPE,
                ),
            ):
                adapter_module = importlib.import_module(
                    f"{namespace}.adapter.{module_name}"
                )
                self.adapters.append(
                    getattr(adapter_module, class_name)(
                        self.shared.AdapterConfig(
                            type=adapter_type,
                            extra={"negative_prompt": "provider default"},
                        )
                    )
                )
        session_patch = patch.object(
            aiohttp,
            "ClientSession",
            side_effect=AssertionError("Real HTTP sessions are forbidden in tests"),
        )
        session_patch.start()
        self.addCleanup(session_patch.stop)

    def test_inherit_override_and_explicit_clear(self):
        for adapter in self.adapters:
            for override, expected in (
                (None, "provider default"),
                ("request override", "request override"),
                ("", ""),
            ):
                with self.subTest(adapter=type(adapter).__name__, override=override):
                    payload = adapter._build_payload(
                        self.shared.GenerationRequest(
                            prompt="a cat", negative_prompt=override
                        )
                    )
                    parameters = payload.get("parameters", payload)
                    self.assertEqual(parameters["negative_prompt"], expected)
                    if "v4_negative_prompt" in parameters:
                        self.assertEqual(
                            parameters["v4_negative_prompt"]["caption"]["base_caption"],
                            expected,
                        )
                    self.assertEqual(
                        adapter.config.extra["negative_prompt"], "provider default"
                    )
                    self.assertTrue(
                        adapter.get_capabilities()
                        & self.shared.ImageCapability.NEGATIVE_PROMPT
                    )

    def test_empty_provider_defaults_keep_existing_payload_behavior(self):
        for adapter in self.adapters:
            with self.subTest(adapter=type(adapter).__name__):
                adapter.config.extra["negative_prompt"] = ""
                payload = adapter._build_payload(
                    self.shared.GenerationRequest(prompt="a cat")
                )
                if "parameters" in payload:
                    self.assertEqual(payload["parameters"]["negative_prompt"], "")
                else:
                    self.assertNotIn("negative_prompt", payload)

    async def test_concurrent_request_overrides_do_not_leak_into_other_requests(self):
        for adapter in self.adapters:
            overrides = [None, "", "watermark", "blur"] * 3
            payloads = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        adapter._build_payload,
                        self.shared.GenerationRequest(
                            prompt="a cat", negative_prompt=override
                        ),
                    )
                    for override in overrides
                )
            )
            for override, payload in zip(overrides, payloads, strict=True):
                parameters = payload.get("parameters", payload)
                self.assertEqual(
                    parameters["negative_prompt"],
                    "provider default" if override is None else override,
                )
            self.assertEqual(
                adapter.config.extra["negative_prompt"], "provider default"
            )


if __name__ == "__main__":
    unittest.main()
