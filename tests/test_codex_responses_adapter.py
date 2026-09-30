from __future__ import annotations

import base64
import importlib
import json
import logging
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp


class CodexResponsesAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load the adapter in an isolated namespace and replace HTTP with a fake."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_codex_responses_test_plugin"
        modules = {}
        for suffix in ("", "adapter", "core", "core.adapters", "core.shared"):
            name = f"{namespace}.{suffix}" if suffix else namespace
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.split(".")))]
            modules[name] = module
        api = types.ModuleType("astrbot.api")
        api.logger = logging.getLogger(__name__)
        modules["astrbot.api"] = api

        # Restore all module entries after import so other plugin tests stay isolated.
        with patch.dict(sys.modules, modules):
            adapter_module = importlib.import_module(
                f"{namespace}.adapter.codex_responses_adapter"
            )
            shared = importlib.import_module(f"{namespace}.core.shared.types")
        self.GenerationRequest = shared.GenerationRequest
        self.ImageData = shared.ImageData
        self.adapter = adapter_module.CodexResponsesAdapter(
            shared.AdapterConfig(
                type=shared.AdapterType.CODEX_RESPONSES,
                base_url="https://images.example.invalid",
                api_keys=["test-key"],
                model="test-model",
            )
        )
        self.image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+a0WQAAAAASUVORK5CYII="
        )
        self.response = MagicMock()
        self.response.status = 200
        self.response.headers = {"Content-Type": "application/json"}
        self.response.read = AsyncMock(
            return_value=json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "image_generation_call",
                            "status": "completed",
                            "result": self.image_base64,
                        }
                    ],
                }
            ).encode()
        )
        self.session = MagicMock()
        self.session.post.return_value.__aenter__.return_value = self.response
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.object(self.adapter, "_get_session", return_value=self.session)
        )
        patches.enter_context(
            patch.object(
                aiohttp,
                "ClientSession",
                side_effect=AssertionError("Real HTTP sessions are forbidden in tests"),
            )
        )

    async def test_bare_subject_requests_image_tool_and_decodes_response(self):
        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(error)
        self.assertEqual(images, [base64.b64decode(self.image_base64)])
        self.session.post.assert_called_once()
        self.assertEqual(
            self.session.post.call_args.args,
            ("https://images.example.invalid/codex/responses",),
        )
        payload = self.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["input"], "原神")
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertEqual(
            payload["tools"], [{"type": "image_generation", "output_format": "png"}]
        )

    async def test_reference_images_preserve_order_and_mime_with_forced_image_tool(
        self,
    ):
        references = [
            self.ImageData(data=b"first-reference", mime_type="image/jpeg"),
            self.ImageData(data=b"second-reference", mime_type="image/webp"),
        ]
        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神", images=references)
        )

        self.assertIsNone(error)
        self.assertEqual(images, [base64.b64decode(self.image_base64)])
        self.session.post.assert_called_once()
        payload = self.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertEqual(
            payload["input"],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "原神"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,Zmlyc3QtcmVmZXJlbmNl",
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/webp;base64,c2Vjb25kLXJlZmVyZW5jZQ==",
                        },
                    ],
                }
            ],
        )

    async def test_http_rejection_of_tool_choice_is_reported(self):
        self.response.status = 400
        self.response.read.return_value = json.dumps(
            {"error": {"message": "unsupported tool_choice"}}
        ).encode()
        self.adapter.show_user_error_details = True

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(images)
        self.assertIn("400", error)
        self.assertIn("unsupported tool_choice", error)
        self.session.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
