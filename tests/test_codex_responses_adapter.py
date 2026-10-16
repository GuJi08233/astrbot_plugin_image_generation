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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp


class CodexResponsesAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load the adapter in an isolated namespace and replace HTTP with a fake."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_codex_responses_test_plugin"
        modules = {}
        for suffix in (
            "",
            "adapter",
            "core",
            "core.adapters",
            "core.config",
            "core.shared",
        ):
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
            provider_parser = importlib.import_module(
                f"{namespace}.core.config.provider_parser"
            )
        self.AdapterType = shared.AdapterType
        self.provider_parser = provider_parser.ConfigProviderParserMixin()
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

    def _set_sse(self, *events, chunks=None, fail_after_chunks=False):
        """Replace the response body with an incrementally consumed SSE stream.

        Args:
            *events: JSON event objects or literal SSE data values.
            chunks: Optional raw transport chunks for testing SSE framing.
            fail_after_chunks: Raise if the adapter reads past the supplied events.
        """
        if chunks is None:
            chunks = [
                (
                    "data: "
                    + (event if isinstance(event, str) else json.dumps(event))
                    + "\n\n"
                ).encode()
                for event in events
            ]

        async def iterate_chunks():
            for chunk in chunks:
                yield chunk
            if fail_after_chunks:
                raise AssertionError("The completed event must stop stream consumption")

        self.response.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        self.response.content.iter_any.side_effect = iterate_chunks
        self.response.read.side_effect = AssertionError(
            "SSE responses must be consumed incrementally"
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
        self.assertIs(payload["stream"], True)
        self.assertEqual(
            self.session.post.call_args.kwargs["headers"]["Accept"],
            "text/event-stream, application/json, image/*",
        )
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertEqual(
            payload["tools"], [{"type": "image_generation", "output_format": "png"}]
        )

    async def test_disabled_streaming_text_to_image_decodes_json(self):
        self.adapter.config.extra["enable_streaming"] = False

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(error)
        self.assertEqual(images, [base64.b64decode(self.image_base64)])
        self.session.post.assert_called_once()
        payload = self.session.post.call_args.kwargs["json"]
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["input"], "原神")
        self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
        self.assertEqual(
            payload["tools"], [{"type": "image_generation", "output_format": "png"}]
        )
        self.assertEqual(
            self.session.post.call_args.kwargs["headers"]["Accept"],
            "application/json, image/*",
        )
        self.response.read.assert_awaited_once()
        self.response.content.iter_any.assert_not_called()

    async def test_disabled_streaming_reference_images_preserve_input_and_decode_json(
        self,
    ):
        self.adapter.config.extra["enable_streaming"] = False
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
        self.assertIs(payload["stream"], False)
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
        self.assertEqual(
            self.session.post.call_args.kwargs["headers"]["Accept"],
            "application/json, image/*",
        )

    async def test_streaming_option_is_normalized_to_a_boolean(self):
        for value, expected in (
            (True, True),
            ("true", True),
            ("enabled", True),
            (" FALSE ", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
            (0, False),
            (1, True),
        ):
            with self.subTest(value=value):
                self.adapter.config.extra["enable_streaming"] = value
                self.session.post.reset_mock()

                images, error = await self.adapter._generate_once(
                    self.GenerationRequest(prompt="原神")
                )

                self.assertIsNone(error)
                self.assertEqual(images, [base64.b64decode(self.image_base64)])
                self.session.post.assert_called_once()
                self.assertIs(
                    self.session.post.call_args.kwargs["json"]["stream"], expected
                )
                self.assertEqual(
                    self.session.post.call_args.kwargs["headers"]["Accept"],
                    "text/event-stream, application/json, image/*"
                    if expected
                    else "application/json, image/*",
                )

    async def test_disabled_streaming_still_decodes_sse_by_content_type(self):
        self.adapter.config.extra["enable_streaming"] = False
        self._set_sse(
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "image_generation_call",
                            "status": "completed",
                            "result": self.image_base64,
                        }
                    ],
                },
            },
            fail_after_chunks=True,
        )

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(error)
        self.assertEqual(images, [base64.b64decode(self.image_base64)])
        self.session.post.assert_called_once()
        self.assertIs(self.session.post.call_args.kwargs["json"]["stream"], False)
        self.assertEqual(
            self.session.post.call_args.kwargs["headers"]["Accept"],
            "application/json, image/*",
        )
        self.response.read.assert_not_awaited()

    async def test_disabled_streaming_http_rejection_does_not_switch_modes(self):
        self.adapter.config.extra["enable_streaming"] = False
        self.response.status = 400
        self.response.read.return_value = json.dumps(
            {"error": {"message": "stream must be true"}}
        ).encode()

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(images)
        self.assertIn("400", error)
        self.session.post.assert_called_once()
        self.assertIs(self.session.post.call_args.kwargs["json"]["stream"], False)

    def test_provider_parser_defaults_streaming_and_preserves_explicit_values(self):
        for provider_item, expected in (
            ({}, True),
            ({"enable_streaming": False}, False),
            ({"enable_streaming": True}, True),
        ):
            with self.subTest(provider_item=provider_item):
                extra = self.provider_parser._parse_provider_extra(
                    self.AdapterType.CODEX_RESPONSES, provider_item
                )

                self.assertIs(extra["enable_streaming"], expected)
                self.assertEqual(extra["endpoint"], "/codex/responses")

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

    async def test_binary_image_response_remains_supported(self):
        image = base64.b64decode(self.image_base64)
        self.response.headers = {"Content-Type": "image/png"}
        self.response.read.return_value = image

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(error)
        self.assertEqual(images, [image])

    async def test_completed_response_replaces_done_items_and_stops_reading(self):
        self._set_sse(
            {
                "type": "response.image_generation_call.partial_image",
                "partial_image_b64": base64.b64encode(b"preview" * 8).decode(),
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "ig_test",
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": base64.b64encode(b"intermediate" * 8).decode(),
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "id": "ig_test",
                            "type": "image_generation_call",
                            "status": "completed",
                            "result": self.image_base64,
                        }
                    ],
                },
            },
            fail_after_chunks=True,
        )

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(error)
        self.assertEqual(images, [base64.b64decode(self.image_base64)])

    async def test_large_multiline_sse_survives_crlf_and_split_utf8(self):
        """Exercise transport boundaries without relying on a private SSE parser."""
        image = base64.b64decode(self.image_base64) * 800
        response = {
            "status": "completed",
            "metadata": {"subject": "猫"},
            "output": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": base64.b64encode(image).decode(),
                }
            ],
        }
        body = (
            ": keepalive\r\n\r\n"
            "event: response.completed\r\n"
            'data: {"type": "response.completed",\r\n'
            f'data: "response": {json.dumps(response, ensure_ascii=False)}}}\r\n\r\n'
        ).encode()
        unicode_boundary = body.index("猫".encode()) + 1
        boundaries = [2, unicode_boundary, unicode_boundary + 1, len(body) - 3]
        chunks = []
        previous = 0
        for boundary in boundaries:
            chunks.append(body[previous:boundary])
            previous = boundary
        chunks.append(body[previous:])
        self.assertGreater(len(chunks[-2]), 65536)
        self._set_sse(chunks=chunks)

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="猫")
        )

        self.assertIsNone(error)
        self.assertEqual(images, [image])

    async def test_done_images_are_recovered_on_done_marker_or_eof(self):
        done = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "ig_test",
                "type": "image_generation_call",
                "status": "completed",
                "result": self.image_base64,
            },
        }
        for ending in ([], ["[DONE]"]):
            with self.subTest(ending=ending):
                self._set_sse(done, done, *ending)

                images, error = await self.adapter._generate_once(
                    self.GenerationRequest(prompt="原神")
                )

                self.assertIsNone(error)
                self.assertEqual(images, [base64.b64decode(self.image_base64)])

    async def test_cr_only_and_split_crlf_finish_without_waiting_for_eof(self):
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": self.image_base64,
                }
            ],
        }
        for newline in (b"\r", b"\r\n"):
            with self.subTest(newline=newline):
                body = newline.join(
                    [
                        b"event: response.created",
                        b'data: {"type": "response.created", "response": {}}',
                        b"",
                        b"event: response.completed",
                        b'data: {"type": "response.completed",',
                        f'data: "response": {json.dumps(response)}}}'.encode(),
                        b"",
                        b"",
                    ]
                )
                chunks = [body]
                if newline == b"\r\n":
                    chunks = []
                    for line in body.splitlines(keepends=True):
                        chunks.extend([line[:-1], line[-1:]])
                self._set_sse(chunks=chunks, fail_after_chunks=True)

                images, error = await self.adapter._generate_once(
                    self.GenerationRequest(prompt="原神")
                )

                self.assertIsNone(error)
                self.assertEqual(images, [base64.b64decode(self.image_base64)])

    async def test_stream_cancellation_propagates_to_the_caller(self):
        self._set_sse()

        async def cancelled_chunks():
            yield b": keepalive\n\n"
            raise asyncio.CancelledError

        self.response.content.iter_any.side_effect = cancelled_chunks

        with self.assertRaises(asyncio.CancelledError):
            await self.adapter._generate_once(self.GenerationRequest(prompt="原神"))

    async def test_stream_errors_override_already_finished_image_items(self):
        done = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "image_generation_call",
                "status": "completed",
                "result": self.image_base64,
            },
        }
        failures = [
            {"type": "error", "code": "server_error", "message": "stream rejected"},
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"message": "stream rejected"},
                    "output": [done["item"]],
                },
            },
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [done["item"]],
                },
            },
        ]
        for failure in failures:
            with self.subTest(event_type=failure["type"]):
                self._set_sse(done, failure)

                images, error = await self.adapter._generate_once(
                    self.GenerationRequest(prompt="原神")
                )

                self.assertIsNone(images)
                self.assertTrue(error)

    async def test_partial_images_do_not_make_unfinished_streams_successful(self):
        partial = {
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": self.image_base64,
        }
        endings = [
            [],
            ["[DONE]"],
            [
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": []},
                }
            ],
        ]
        for ending in endings:
            with self.subTest(ending=ending):
                self._set_sse(partial, *ending)

                images, error = await self.adapter._generate_once(
                    self.GenerationRequest(prompt="原神")
                )

                self.assertIsNone(images)
                self.assertTrue(error)

    async def test_invalid_json_event_is_reported(self):
        self._set_sse('{"type": "response.completed", "response":')

        images, error = await self.adapter._generate_once(
            self.GenerationRequest(prompt="原神")
        )

        self.assertIsNone(images)
        self.assertTrue(error)


if __name__ == "__main__":
    unittest.main()
