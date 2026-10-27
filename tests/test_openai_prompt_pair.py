from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
import sys
import types
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp


class OpenAIPromptPairTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load the adapters independently and forbid external HTTP sessions."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_openai_prompt_pair_test_plugin"
        modules = {}
        for suffix in ("", "adapter", "core", "core.adapters", "core.shared"):
            name = f"{namespace}.{suffix}" if suffix else namespace
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.split(".")))]
            modules[name] = module
        api = types.ModuleType("astrbot.api")
        api.logger = logging.getLogger(__name__)
        modules["astrbot.api"] = api
        with patch.dict(sys.modules, modules):
            self.shared = importlib.import_module(f"{namespace}.core.shared.types")
            image_module = importlib.import_module(
                f"{namespace}.adapter.openai_adapter"
            )
            chat_module = importlib.import_module(
                f"{namespace}.adapter.openai_chat_adapter"
            )
        self.adapters = {
            self.shared.AdapterType.OPENAI: image_module.OpenAIAdapter,
            self.shared.AdapterType.OPENAI_CHAT: chat_module.OpenAIChatAdapter,
        }
        self.models = ("gpt-image-2", "gpt-image-2.5", "z-image-turbo")
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(
            patch.object(
                aiohttp,
                "ClientSession",
                side_effect=AssertionError("Real HTTP sessions are forbidden in tests"),
            )
        )

    def _adapter(self, adapter_type, model="gpt-image-2", **extra):
        """Create an adapter using isolated request-scoped configuration.

        Args:
            adapter_type: Images or Chat Completions adapter type.
            model: Model name forwarded without any capability inference here.
            **extra: Provider settings for prompt-pair behavior and legacy fields.

        Returns:
            A fresh adapter with mock credentials and no HTTP session.
        """
        return self.adapters[adapter_type](
            self.shared.AdapterConfig(
                type=adapter_type,
                model=model,
                api_keys=["test-key"],
                base_url="https://images.example.invalid",
                extra=extra,
            )
        )

    def test_prompt_mode_keeps_both_parts_together_for_every_model_and_endpoint(self):
        for adapter_type in self.adapters:
            for model in self.models:
                with self.subTest(adapter=adapter_type, model=model):
                    adapter = self._adapter(adapter_type, model, prompt_prefix="Draw: ")
                    request = self.shared.GenerationRequest(
                        prompt="a white cat", negative_prompt="watermark, blur"
                    )
                    payload = adapter._build_payload(request)
                    text = (
                        payload.get("prompt")
                        or payload["messages"][0]["content"][0]["text"]
                    )
                    self.assertIn("[Content to generate]\na white cat", text)
                    self.assertIn("[Content to avoid]\nwatermark, blur", text)
                    self.assertNotIn("negative_prompt", payload)
                    self.assertEqual(payload["model"], model)
                    if adapter_type == self.shared.AdapterType.OPENAI_CHAT:
                        self.assertTrue(text.startswith("Draw: "))
                    self.assertTrue(
                        adapter.get_capabilities()
                        & self.shared.ImageCapability.NEGATIVE_PROMPT
                    )
                    self.assertEqual(request.prompt, "a white cat")
                    self.assertEqual(request.negative_prompt, "watermark, blur")

    def test_parameter_mode_sends_native_field_without_changing_positive_prompt(self):
        for adapter_type in self.adapters:
            for model in self.models:
                with self.subTest(adapter=adapter_type, model=model):
                    adapter = self._adapter(
                        adapter_type,
                        model,
                        negative_prompt_mode="parameter",
                        prompt_prefix="",
                    )
                    payload = adapter._build_payload(
                        self.shared.GenerationRequest(
                            prompt="a cat", negative_prompt="a dog"
                        )
                    )
                    text = (
                        payload.get("prompt")
                        or payload["messages"][0]["content"][0]["text"]
                    )
                    self.assertEqual(text, "a cat")
                    self.assertEqual(payload["negative_prompt"], "a dog")

    def test_provider_default_and_explicit_clear_preserve_three_state_semantics(self):
        for adapter_type in self.adapters:
            for mode in ("prompt", "parameter"):
                adapter = self._adapter(
                    adapter_type,
                    negative_prompt_mode=mode,
                    negative_prompt="provider negative",
                    prompt_prefix="",
                )
                for override in (None, "request negative", ""):
                    with self.subTest(
                        adapter=adapter_type, mode=mode, override=override
                    ):
                        payload = adapter._build_payload(
                            self.shared.GenerationRequest(
                                prompt="a cat", negative_prompt=override
                            )
                        )
                        text = (
                            payload.get("prompt")
                            or payload["messages"][0]["content"][0]["text"]
                        )
                        expected = "provider negative" if override is None else override
                        if mode == "parameter":
                            self.assertEqual(text, "a cat")
                            self.assertEqual(payload["negative_prompt"], expected)
                        else:
                            self.assertNotIn("negative_prompt", payload)
                            if expected:
                                self.assertIn(expected, text)
                            else:
                                self.assertEqual(text, "a cat")
                        self.assertEqual(
                            adapter.config.extra["negative_prompt"], "provider negative"
                        )

    def test_absent_negative_prompt_leaves_existing_prompt_format_unchanged(self):
        for adapter_type in self.adapters:
            for mode in ("prompt", "parameter"):
                with self.subTest(adapter=adapter_type, mode=mode):
                    adapter = self._adapter(
                        adapter_type, negative_prompt_mode=mode, prompt_prefix=""
                    )
                    payload = adapter._build_payload(
                        self.shared.GenerationRequest(prompt="a cat")
                    )
                    self.assertNotIn("negative_prompt", payload)
                    text = (
                        payload.get("prompt")
                        or payload["messages"][0]["content"][0]["text"]
                    )
                    self.assertEqual(text, "a cat")

    def test_chat_preserves_legacy_extra_fields_until_new_pair_settings_take_effect(
        self,
    ):
        legacy = {
            "negative_prompt": "legacy negative",
            "messages": [{"role": "user", "content": "legacy template"}],
            "temperature": 0.4,
        }
        for mode in ("prompt", "parameter"):
            with self.subTest(mode=mode):
                adapter = self._adapter(
                    self.shared.AdapterType.OPENAI_CHAT,
                    negative_prompt_mode=mode,
                    extra_body_json=json.dumps(legacy),
                )
                payload = adapter._build_payload(
                    self.shared.GenerationRequest(prompt="a cat")
                )
                self.assertEqual(payload["negative_prompt"], "legacy negative")
                self.assertEqual(payload["messages"], legacy["messages"])
                self.assertEqual(payload["temperature"], 0.4)

    def test_chat_pair_takes_precedence_over_legacy_negative_and_message_templates(
        self,
    ):
        legacy = {
            "negative_prompt": "legacy negative",
            "messages": [{"role": "user", "content": "legacy template"}],
            "temperature": 0.4,
        }
        for mode in ("prompt", "parameter"):
            for override in (None, "request negative", ""):
                for extra_field in ("extra_body", "extra_body_json"):
                    with self.subTest(
                        mode=mode, override=override, extra_field=extra_field
                    ):
                        adapter = self._adapter(
                            self.shared.AdapterType.OPENAI_CHAT,
                            negative_prompt_mode=mode,
                            negative_prompt="provider negative",
                            prompt_prefix="",
                            **{
                                extra_field: legacy
                                if extra_field == "extra_body"
                                else json.dumps(legacy)
                            },
                        )
                        payload = adapter._build_payload(
                            self.shared.GenerationRequest(
                                prompt="a cat",
                                negative_prompt=override,
                                images=[
                                    self.shared.ImageData(
                                        b"reference bytes", "image/png"
                                    )
                                ],
                            )
                        )
                        content = payload["messages"][0]["content"]
                        self.assertEqual(content[0]["type"], "text")
                        self.assertIn("a cat", content[0]["text"])
                        self.assertEqual(content[1]["type"], "image_url")
                        self.assertNotIn("legacy template", content[0]["text"])
                        self.assertEqual(payload["temperature"], 0.4)
                        if mode == "prompt":
                            self.assertNotIn("negative_prompt", payload)
                        else:
                            self.assertEqual(
                                payload["negative_prompt"],
                                "provider negative" if override is None else override,
                            )
                        self.assertEqual(legacy["negative_prompt"], "legacy negative")
                        self.assertEqual(
                            legacy["messages"][0]["content"], "legacy template"
                        )

    def test_chat_explicit_clear_removes_legacy_negative_without_new_default(self):
        for mode in ("prompt", "parameter"):
            with self.subTest(mode=mode):
                adapter = self._adapter(
                    self.shared.AdapterType.OPENAI_CHAT,
                    negative_prompt_mode=mode,
                    prompt_prefix="",
                    extra_body={
                        "negative_prompt": "legacy negative",
                        "messages": [{"role": "user", "content": "legacy template"}],
                    },
                )
                payload = adapter._build_payload(
                    self.shared.GenerationRequest(prompt="a cat", negative_prompt="")
                )
                self.assertEqual(payload["messages"][0]["content"][0]["text"], "a cat")
                if mode == "prompt":
                    self.assertNotIn("negative_prompt", payload)
                else:
                    self.assertEqual(payload["negative_prompt"], "")
                self.assertEqual(
                    adapter.config.extra["extra_body"]["negative_prompt"],
                    "legacy negative",
                )

    async def test_multipart_edits_use_the_pair_for_both_gpt_image_model_names(self):
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.json = AsyncMock(
            return_value={"data": [{"b64_json": base64.b64encode(b"result").decode()}]}
        )
        session = MagicMock()
        session.post.return_value.__aenter__.return_value = response
        for model in self.models[:2]:
            for mode in ("prompt", "parameter"):
                for override in (None, "request negative", ""):
                    with self.subTest(model=model, mode=mode, override=override):
                        adapter = self._adapter(
                            self.shared.AdapterType.OPENAI,
                            model,
                            negative_prompt_mode=mode,
                            negative_prompt="provider negative",
                        )
                        request = self.shared.GenerationRequest(
                            prompt="edit a cat",
                            negative_prompt=override,
                            images=[self.shared.ImageData(b"reference", "image/png")],
                        )
                        with patch.object(
                            adapter, "_get_session", return_value=session
                        ):
                            images, error = await adapter._generate_once(request)
                        self.assertIsNone(error)
                        self.assertEqual(images, [b"result"])
                        self.assertEqual(
                            session.post.call_args.args[0],
                            "https://images.example.invalid/v1/images/edits",
                        )
                        form = session.post.call_args.kwargs["data"]
                        fields = {field[0]["name"]: field[2] for field in form._fields}
                        self.assertEqual(fields["image[]"], b"reference")
                        self.assertEqual(fields["model"], model)
                        expected = "provider negative" if override is None else override
                        if mode == "parameter":
                            self.assertEqual(fields["prompt"], "edit a cat")
                            self.assertEqual(fields["negative_prompt"], expected)
                        else:
                            self.assertNotIn("negative_prompt", fields)
                            self.assertIn("edit a cat", fields["prompt"])
                            if expected:
                                self.assertIn(expected, fields["prompt"])
                            else:
                                self.assertEqual(fields["prompt"], "edit a cat")

    async def test_concurrent_pairs_do_not_mutate_provider_defaults_or_requests(self):
        for adapter_type in self.adapters:
            for mode in ("prompt", "parameter"):
                adapter = self._adapter(
                    adapter_type,
                    negative_prompt_mode=mode,
                    negative_prompt="provider negative",
                    prompt_prefix="",
                )
                config_before = deepcopy(adapter.config.extra)
                requests = [
                    self.shared.GenerationRequest(
                        prompt=f"cat {index}", negative_prompt=negative
                    )
                    for index, negative in enumerate((None, "one", "", "two") * 3)
                ]
                before = deepcopy(requests)
                payloads = await asyncio.gather(
                    *(
                        asyncio.to_thread(adapter._build_payload, request)
                        for request in requests
                    )
                )
                for request, payload in zip(requests, payloads, strict=True):
                    text = (
                        payload.get("prompt")
                        or payload["messages"][0]["content"][0]["text"]
                    )
                    expected = (
                        "provider negative"
                        if request.negative_prompt is None
                        else request.negative_prompt
                    )
                    self.assertIn(request.prompt, text)
                    if mode == "parameter":
                        self.assertEqual(payload["negative_prompt"], expected)
                    elif expected:
                        self.assertTrue(text.endswith(expected))
                    else:
                        self.assertEqual(text, request.prompt)
                self.assertEqual(requests, before)
                self.assertEqual(adapter.config.extra, config_before)


if __name__ == "__main__":
    unittest.main()
