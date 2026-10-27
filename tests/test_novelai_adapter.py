from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import io
import json
import logging
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from PIL import Image


class NovelAIAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load only adapter dependencies and forbid real HTTP sessions."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_novelai_test_plugin"
        modules = {}
        for suffix in (
            "",
            "adapter",
            "core",
            "core.adapters",
            "core.generation",
            "core.shared",
        ):
            name = f"{namespace}.{suffix}" if suffix else namespace
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.split(".")))]
            modules[name] = module
        api = types.ModuleType("astrbot.api")
        api.logger = logging.getLogger(__name__)
        modules["astrbot.api"] = api
        with patch.dict(sys.modules, modules):
            adapter_module = importlib.import_module(
                f"{namespace}.adapter.novelai_adapter"
            )
            shared = importlib.import_module(f"{namespace}.core.shared.types")
            generator_module = importlib.import_module(
                f"{namespace}.core.adapters.generator"
            )

        self.GenerationRequest = shared.GenerationRequest
        self.ImageGenerator = generator_module.ImageGenerator
        self.ImageData = shared.ImageData
        self.ImageCapability = shared.ImageCapability
        self.adapter = adapter_module.NovelAIAdapter(
            shared.AdapterConfig(
                type=shared.AdapterType.NOVELAI,
                base_url="https://images.example.invalid",
                api_keys=["first-test-key", "second-test-key"],
                model="nai-diffusion-4-5-full",
                max_retry_attempts=3,
                capability_options={
                    "text_to_image": True,
                    "image_to_image": True,
                    "aspect_ratio": True,
                    "resolution": True,
                },
            )
        )
        image_buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color="red").save(image_buffer, format="PNG")
        self.image_bytes = image_buffer.getvalue()
        self.image_base64 = base64.b64encode(self.image_bytes).decode("ascii")
        self.response = MagicMock()
        self.response.status = 200
        self.response.headers = {"Content-Type": "application/json"}
        self.response_data = {"images": [self.image_base64]}
        self.response.json = AsyncMock(return_value=self.response_data)
        self.response.read = AsyncMock(
            return_value=json.dumps(self.response_data).encode()
        )
        self.response.text = AsyncMock(return_value=json.dumps(self.response_data))
        self.quote_response = MagicMock()
        self.quote_response.status = 200
        self.quote_response.headers = {"Content-Type": "application/json"}
        self.quote_response.json = AsyncMock(return_value={"total_gems": 0})
        self.quote_response.read = AsyncMock(return_value=b'{"total_gems": 0}')
        self.quote_response.text = AsyncMock(return_value='{"total_gems": 0}')
        self.generation_post = MagicMock()
        self.generation_post.return_value.__aenter__.return_value = self.response
        self.quote_post = MagicMock()
        self.quote_post.return_value.__aenter__.return_value = self.quote_response
        self.session = MagicMock()
        self.session.post.side_effect = lambda url, **kwargs: (
            self.quote_post(url, **kwargs)
            if url.endswith("/ai/generate-image/quote")
            else self.generation_post(url, **kwargs)
        )
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

    def _set_response(self, data, *, status=200, quote=False):
        """Set every supported fake JSON body access path consistently.

        Args:
            data: JSON response body supplied by the fake provider.
            status: HTTP response status code.
            quote: Whether to update the quotation response instead of generation.
        """
        response = self.quote_response if quote else self.response
        response.status = status
        response.json.return_value = data
        response.read.return_value = json.dumps(data).encode()
        response.text.return_value = json.dumps(data)

    async def test_text_to_image_uses_native_json_and_decodes_valid_image(self):
        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.error)
        self.assertEqual(result.images, [self.image_bytes])
        self.generation_post.assert_called_once()
        self.assertEqual(
            self.generation_post.call_args.args,
            ("https://images.example.invalid/v1/nai/generate-image",),
        )
        payload = self.generation_post.call_args.kwargs["json"]
        self.assertEqual(payload["input"], "1girl")
        self.assertEqual(payload["model"], "nai-diffusion-4-5-full")
        self.assertEqual(payload["action"], "generate")
        parameters = payload["parameters"]
        self.assertEqual(parameters["width"], 832)
        self.assertEqual(parameters["height"], 1216)
        self.assertEqual(parameters["steps"], 28)
        self.assertEqual(parameters["n_samples"], 1)
        self.assertEqual(parameters["sampler"], "k_euler_ancestral")
        self.assertEqual(parameters["scale"], 5)
        self.assertEqual(parameters["noise_schedule"], "karras")
        self.assertEqual(parameters["cfg_rescale"], 0)
        self.assertEqual(parameters["negative_prompt"], "")
        self.assertNotIn("image", parameters)
        self.assertIsInstance(parameters["seed"], int)
        self.assertGreaterEqual(parameters["seed"], 0)
        self.assertLessEqual(parameters["seed"], 4294967295)
        headers = self.generation_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer first-test-key")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertTrue(headers["Idempotency-Key"])

    async def test_request_negative_prompt_matches_quote_and_generation(self):
        self.adapter.config.extra["negative_prompt"] = "provider default"
        for override, expected in (
            (None, "provider default"),
            ("request override", "request override"),
            ("", ""),
        ):
            with self.subTest(override=override):
                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl", negative_prompt=override)
                )

                self.assertIsNone(result.error)
                quote = self.quote_post.call_args.kwargs["json"]
                generation = self.generation_post.call_args.kwargs["json"]
                self.assertEqual(quote, generation)
                self.assertEqual(generation["parameters"]["negative_prompt"], expected)
                self.assertEqual(
                    generation["parameters"]["v4_negative_prompt"]["caption"][
                        "base_caption"
                    ],
                    expected,
                )
                self.assertEqual(
                    self.adapter.config.extra["negative_prompt"], "provider default"
                )

    async def test_image_to_image_encodes_reference_and_native_parameters(self):
        self.adapter.config.extra.update(
            {"img2img_strength": 0.55, "img2img_noise": 0.15, "seed": 12345}
        )
        request = self.GenerationRequest(
            prompt="change hair color",
            images=[self.ImageData(data=self.image_bytes, mime_type="image/png")],
        )

        result = await self.adapter.generate(request)

        self.assertIsNone(result.error)
        payload = self.generation_post.call_args.kwargs["json"]
        self.assertEqual(payload["action"], "img2img")
        parameters = payload["parameters"]
        self.assertNotIn("data:", parameters["image"])
        reference_bytes = base64.b64decode(parameters["image"], validate=True)
        with Image.open(io.BytesIO(reference_bytes)) as reference:
            self.assertEqual(reference.format, "PNG")
            self.assertEqual(reference.size, (832, 1216))
            self.assertEqual(reference.convert("RGB").getpixel((416, 608)), (255, 0, 0))
        self.assertEqual(parameters["strength"], 0.55)
        self.assertEqual(parameters["noise"], 0.15)
        self.assertEqual(parameters["seed"], 12345)

    def test_image_to_image_defaults_and_v4_caption_structure(self):
        self.adapter.config.extra["negative_prompt"] = "lowres, bad anatomy"
        payload = self.adapter._build_payload(
            self.GenerationRequest(
                prompt="1girl, blue hair",
                images=[self.ImageData(data=self.image_bytes, mime_type="image/png")],
            )
        )

        self.assertEqual(payload["action"], "img2img")
        parameters = payload["parameters"]
        self.assertEqual(parameters["strength"], 0.7)
        self.assertEqual(parameters["noise"], 0)
        self.assertEqual(
            parameters["v4_prompt"]["caption"],
            {"base_caption": "1girl, blue hair", "char_captions": []},
        )
        self.assertEqual(
            parameters["v4_negative_prompt"]["caption"],
            {"base_caption": "lowres, bad anatomy", "char_captions": []},
        )
        self.assertIs(parameters["v4_prompt"]["use_coords"], False)
        self.assertIs(parameters["v4_prompt"]["use_order"], True)
        self.assertIs(parameters["v4_negative_prompt"]["legacy_uc"], False)
        self.assertEqual(parameters["params_version"], 3)
        self.assertIs(parameters["use_coords"], False)
        self.assertIs(parameters["legacy_v3_extend"], False)
        self.assertIs(parameters["legacy_uc"], False)

    def test_non_v4_models_do_not_receive_v4_prompt_structures(self):
        self.adapter.model = "nai-diffusion-3"

        payload = self.adapter._build_payload(self.GenerationRequest(prompt="1girl"))

        self.assertNotIn("v4_prompt", payload["parameters"])
        self.assertNotIn("v4_negative_prompt", payload["parameters"])

    async def test_multiple_references_are_rejected_before_http(self):
        reference = self.ImageData(data=self.image_bytes, mime_type="image/png")

        result = await self.adapter.generate(
            self.GenerationRequest(prompt="1girl", images=[reference, reference])
        )

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.session.post.assert_not_called()

    async def test_invalid_reference_is_rejected_before_http(self):
        result = await self.adapter.generate(
            self.GenerationRequest(
                prompt="1girl",
                images=[self.ImageData(data=b"not-an-image", mime_type="image/png")],
            )
        )

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.session.post.assert_not_called()

    async def test_invalid_config_parameters_are_rejected_before_http(self):
        invalid_values = (
            ("width", 63),
            ("width", 833),
            ("width", 2112),
            ("height", 63),
            ("height", 1217),
            ("height", 2112),
            ("steps", 0),
            ("steps", 51),
            ("steps", 1.5),
            ("scale", -0.1),
            ("scale", 10.1),
            ("scale", float("nan")),
            ("cfg_rescale", -0.1),
            ("cfg_rescale", 1.1),
            ("img2img_strength", -0.1),
            ("img2img_strength", 1.1),
            ("img2img_noise", -0.1),
            ("img2img_noise", 1.1),
            ("seed", -2),
            ("seed", 4294967296),
            ("seed", 1.5),
        )
        for name, value in invalid_values:
            with self.subTest(parameter=name, value=value):
                self.adapter.config.extra.clear()
                self.adapter.config.extra[name] = value
                self.session.post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(
                        prompt="1girl",
                        images=[
                            self.ImageData(data=self.image_bytes, mime_type="image/png")
                        ],
                    )
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.session.post.assert_not_called()

    async def test_excessive_pixel_area_is_rejected_before_http(self):
        self.adapter.config.extra.update({"width": 2048, "height": 2048})

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.session.post.assert_not_called()

    def test_manual_dimensions_and_numeric_boundaries_remain_valid(self):
        for options in (
            {"width": 64, "height": 64, "steps": 1, "scale": 0, "seed": 0},
            {
                "width": 2048,
                "height": 1536,
                "steps": 50,
                "scale": 10,
                "seed": 4294967295,
                "cfg_rescale": 1,
            },
        ):
            with self.subTest(options=options):
                self.adapter.config.extra = options

                payload = self.adapter._build_payload(
                    self.GenerationRequest(
                        prompt="1girl",
                        image_size=(options["width"], options["height"]),
                    )
                )

                for name, value in options.items():
                    self.assertEqual(payload["parameters"][name], value)

    def test_automatic_aspect_ratios_only_use_standard_dimensions(self):
        expected_sizes = {
            None: (832, 1216),
            "不指定": (832, 1216),
            "1:1": (1024, 1024),
            "2:3": (832, 1216),
            "3:2": (1216, 832),
            "3:4": (832, 1216),
            "4:3": (1216, 832),
            "4:5": (832, 1216),
            "5:4": (1216, 832),
            "9:16": (832, 1216),
            "16:9": (1216, 832),
            "21:9": (1216, 832),
        }
        allowed_sizes = {(512, 512), (832, 1216), (1216, 832), (1024, 1024)}
        for ratio, expected_size in expected_sizes.items():
            with self.subTest(ratio=ratio):
                payload = self.adapter._build_payload(
                    self.GenerationRequest(prompt="1girl", aspect_ratio=ratio)
                )

                parameters = payload["parameters"]
                size = (parameters["width"], parameters["height"])
                self.assertIn(size, allowed_sizes)
                self.assertEqual(size, expected_size)

    def test_provider_dimensions_map_to_standard_sizes(self):
        for configured, expected in (
            ((512, 512), (512, 512)),
            ((896, 1152), (832, 1216)),
            ((1152, 896), (1216, 832)),
            ((768, 768), (512, 512)),
            ((960, 960), (1024, 1024)),
        ):
            with self.subTest(configured=configured):
                self.adapter.config.extra.update(
                    {"width": configured[0], "height": configured[1]}
                )

                parameters = self.adapter._build_payload(
                    self.GenerationRequest(prompt="1girl")
                )["parameters"]

                self.assertEqual((parameters["width"], parameters["height"]), expected)

    def test_square_dimensions_follow_configured_pixel_area(self):
        for configured, expected in (
            ((512, 512), (512, 512)),
            ((1024, 1024), (1024, 1024)),
            ((832, 1216), (1024, 1024)),
        ):
            with self.subTest(configured=configured):
                self.adapter.config.extra.update(
                    {"width": configured[0], "height": configured[1]}
                )

                parameters = self.adapter._build_payload(
                    self.GenerationRequest(prompt="1girl", aspect_ratio="1:1")
                )["parameters"]

                self.assertEqual((parameters["width"], parameters["height"]), expected)

    async def test_manual_dimensions_override_aspect_ratio_and_provider_defaults(self):
        self.adapter.config.extra.update({"width": 833, "height": 999})

        result = await self.adapter.generate(
            self.GenerationRequest(
                prompt="1girl", aspect_ratio="16:9", image_size=(896, 1152)
            )
        )

        self.assertIsNone(result.error)
        quote_payload = self.quote_post.call_args.kwargs["json"]
        generation_payload = self.generation_post.call_args.kwargs["json"]
        self.assertEqual(quote_payload, generation_payload)
        parameters = generation_payload["parameters"]
        self.assertEqual((parameters["width"], parameters["height"]), (896, 1152))

    async def test_all_standard_manual_sizes_match_quoted_and_generated_dimensions(
        self,
    ):
        for size in ((512, 512), (832, 1216), (1216, 832), (1024, 1024)):
            with self.subTest(size=size):
                self.quote_post.reset_mock()
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(
                        prompt="1girl", aspect_ratio="16:9", image_size=size
                    )
                )

                self.assertIsNone(result.error)
                self.quote_post.assert_called_once()
                self.generation_post.assert_called_once()
                quote_payload = self.quote_post.call_args.kwargs["json"]
                generation_payload = self.generation_post.call_args.kwargs["json"]
                self.assertEqual(quote_payload, generation_payload)
                parameters = generation_payload["parameters"]
                self.assertEqual((parameters["width"], parameters["height"]), size)

    async def test_orchestrator_preserves_manual_dimensions_for_quote_and_generation(
        self,
    ):
        generator = self.ImageGenerator.__new__(self.ImageGenerator)
        generator.adapter = self.adapter

        result = await generator.generate_preconverted(
            self.GenerationRequest(
                prompt="1girl", aspect_ratio="16:9", image_size=(896, 1152)
            ),
            images=[],
        )

        self.assertIsNone(result.error)
        quote_payload = self.quote_post.call_args.kwargs["json"]
        generation_payload = self.generation_post.call_args.kwargs["json"]
        self.assertEqual(quote_payload, generation_payload)
        parameters = generation_payload["parameters"]
        self.assertEqual((parameters["width"], parameters["height"]), (896, 1152))

    async def test_invalid_manual_dimensions_are_rejected_before_http(self):
        for size in (
            (),
            (512,),
            (512, 512, 512),
            "512x512",
            (True, 512),
            (512, False),
            (512.0, 512),
            (512, "512"),
            (0, 512),
            (63, 512),
            (833, 1216),
            (512, 1217),
            (2112, 512),
            (2048, 2048),
        ):
            with self.subTest(size=size):
                self.session.post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl", image_size=size)
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.session.post.assert_not_called()

    async def test_invalid_aspect_ratio_is_rejected_before_http(self):
        requests = [
            self.GenerationRequest(prompt="1girl", aspect_ratio=value)
            for value in ("0:1", "1:0", "-1:1", "invalid", "inf:1", "nan:1")
        ]
        for request in requests:
            with self.subTest(request=request):
                self.session.post.reset_mock()

                result = await self.adapter.generate(request)

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.session.post.assert_not_called()

    async def test_global_resolution_does_not_override_configured_dimensions(self):
        self.adapter.config.extra.update({"width": 1024, "height": 1024})

        result = await self.adapter.generate(
            self.GenerationRequest(prompt="1girl", resolution="4K")
        )

        self.assertIsNone(result.error)
        parameters = self.generation_post.call_args.kwargs["json"]["parameters"]
        self.assertEqual(parameters["width"], 1024)
        self.assertEqual(parameters["height"], 1024)

    def test_capabilities_include_image_editing_and_ratio_without_resolution(self):
        capabilities = self.adapter.get_capabilities()

        self.assertTrue(capabilities & self.ImageCapability.TEXT_TO_IMAGE)
        self.assertTrue(capabilities & self.ImageCapability.IMAGE_TO_IMAGE)
        self.assertTrue(capabilities & self.ImageCapability.ASPECT_RATIO)
        self.assertFalse(capabilities & self.ImageCapability.RESOLUTION)

    def test_endpoint_accepts_only_supported_json_paths(self):
        expected = "https://images.example.invalid/v1/nai/generate-image"
        for base in (
            "https://images.example.invalid",
            "https://images.example.invalid/",
            "https://images.example.invalid/v1",
            "https://images.example.invalid/v1/",
            expected,
            expected + "/",
        ):
            with self.subTest(base=base):
                self.adapter.base_url = base

                self.assertEqual(self.adapter._endpoint_url(), expected)

    async def test_native_zip_and_malformed_endpoints_are_rejected_before_http(self):
        for base in (
            "https://images.example.invalid/native",
            "https://images.example.invalid/native/ai/generate-image",
            "not-a-url",
            "https://images.example.invalid?token=secret",
            "https://images.example.invalid#fragment",
        ):
            with self.subTest(base=base):
                self.adapter.base_url = base
                self.session.post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.session.post.assert_not_called()

    async def test_all_returned_images_are_validated_before_success(self):
        corrupted_png = bytearray(self.image_bytes)
        corrupted_png[corrupted_png.index(b"IDAT") + 4] ^= 1
        invalid_images = (
            {},
            {"images": []},
            {"images": "not-a-list"},
            {"images": ["invalid base64!"]},
            {"images": [base64.b64encode(b"not-an-image").decode()]},
            {"images": [self.image_base64, "invalid base64!"]},
            {"images": [self.image_base64, None]},
            {"images": [base64.b64encode(self.image_bytes[:40]).decode()]},
            {"images": [base64.b64encode(corrupted_png).decode()]},
            [self.image_base64],
        )
        for body in invalid_images:
            with self.subTest(body_type=type(body).__name__, body=body):
                self._set_response(body)
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.generation_post.assert_called_once()

    async def test_failed_json_responses_do_not_return_images_or_retry(self):
        for body in (
            {"error": {"message": "generation failed"}, "images": [self.image_base64]},
            {"job": {"status": "failed"}, "images": [self.image_base64]},
        ):
            with self.subTest(body=body):
                self._set_response(body)
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.generation_post.assert_called_once()

    async def test_completed_job_status_variants_return_valid_image(self):
        jobs = (
            None,
            {},
            {"status": None},
            {"status": "succeeded"},
            {"status": "charged"},
            {"status": " SUCCEEDED "},
            {"status": "  ChArGeD\t"},
        )
        for job in jobs:
            with self.subTest(job=job):
                body = {"images": [self.image_base64]}
                if job is not None:
                    body["job"] = job
                self._set_response(body)
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.error)
                self.assertEqual(result.images, [self.image_bytes])
                self.generation_post.assert_called_once()

    async def test_noncompleted_or_invalid_job_status_fails_without_retry(self):
        statuses = (
            "failed",
            " FAILED ",
            "pending",
            "running",
            "unknown",
            "",
            "   ",
            0,
            1,
            True,
            False,
            [],
            ["charged"],
            {},
            {"status": "charged"},
        )
        for status in statuses:
            with self.subTest(status=status):
                self._set_response(
                    {"job": {"status": status}, "images": [self.image_base64]}
                )
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.generation_post.assert_called_once()

    async def test_charged_job_still_requires_no_error_and_valid_image(self):
        bodies = (
            {"error": {"message": "generation failed"}, "images": [self.image_base64]},
            {"images": ["invalid base64!"]},
            {"images": [base64.b64encode(b"not-an-image").decode("ascii")]},
        )
        for body in bodies:
            with self.subTest(body=body):
                self._set_response({**body, "job": {"status": "charged"}})
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.generation_post.assert_called_once()

    async def test_invalid_json_is_not_retried_after_http_success(self):
        self.response.json.side_effect = json.JSONDecodeError("Invalid", "x", 0)
        self.response.read.return_value = b"not json"
        self.response.text.return_value = "not json"

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.generation_post.assert_called_once()

    async def test_valid_crc_with_corrupt_png_pixels_does_not_succeed_or_retry(self):
        corrupted_png = bytearray(self.image_bytes)
        idat_offset = corrupted_png.index(b"IDAT")
        idat_length = int.from_bytes(
            corrupted_png[idat_offset - 4 : idat_offset], "big"
        )
        corrupted_png[idat_offset + 4] ^= 1
        crc_offset = idat_offset + 4 + idat_length
        crc = binascii.crc32(corrupted_png[idat_offset:crc_offset])
        corrupted_png[crc_offset : crc_offset + 4] = crc.to_bytes(4, "big")
        # PNG structural verification accepts this CRC, but pixel decoding fails.
        with Image.open(io.BytesIO(corrupted_png)) as image:
            image.verify()
        with Image.open(io.BytesIO(corrupted_png)) as image:
            with self.assertRaises(OSError):
                image.load()
        self._set_response(
            {"images": [base64.b64encode(corrupted_png).decode("ascii")]}
        )

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.generation_post.assert_called_once()

    async def test_local_decoder_failure_does_not_resubmit_completed_generation(self):
        with patch.object(
            self.adapter, "_extract_images", side_effect=RuntimeError("decoder failed")
        ):
            result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.generation_post.assert_called_once()

    async def test_http_errors_are_not_automatically_resubmitted(self):
        for status in (400, 401, 409, 429, 500, 503):
            with self.subTest(status=status):
                self._set_response(
                    {"error": {"message": "request rejected"}}, status=status
                )
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertIn(str(status), result.error)
                self.generation_post.assert_called_once()

    async def test_interrupted_http_error_body_does_not_trigger_network_retry(self):
        for status in (400, 503):
            for exception in (
                aiohttp.ClientPayloadError("response body interrupted"),
                asyncio.TimeoutError(),
            ):
                with self.subTest(status=status, exception=type(exception).__name__):
                    self.response.status = status
                    self.response.read.side_effect = exception
                    self.generation_post.reset_mock()

                    result = await self.adapter.generate(
                        self.GenerationRequest(prompt="1girl")
                    )

                    self.assertIsNone(result.images)
                    self.assertIn(str(status), result.error)
                    self.generation_post.assert_called_once()

    async def test_network_retry_keeps_seed_api_key_and_idempotency_key(self):
        successful_context = self.generation_post.return_value
        self.generation_post.side_effect = [
            aiohttp.ClientConnectionError("connection interrupted"),
            successful_context,
        ]
        request = self.GenerationRequest(prompt="1girl", task_id="task-test")
        with patch.object(self.adapter, "_rotate_api_key") as rotate:
            result = await self.adapter.generate(request)

        self.assertIsNone(result.error)
        self.assertEqual(result.images, [self.image_bytes])
        self.assertEqual(self.generation_post.call_count, 2)
        first, second = self.generation_post.call_args_list
        self.assertEqual(first.kwargs["json"], second.kwargs["json"])
        self.assertEqual(first.kwargs["headers"], second.kwargs["headers"])
        self.assertEqual(
            first.kwargs["headers"]["Authorization"], "Bearer first-test-key"
        )
        self.assertTrue(first.kwargs["headers"]["Idempotency-Key"])
        self.assertGreaterEqual(first.kwargs["json"]["parameters"]["seed"], 0)
        rotate.assert_not_called()

    async def test_separate_generations_receive_distinct_idempotency_keys(self):
        request = self.GenerationRequest(prompt="1girl", task_id="task-test")

        first_result = await self.adapter.generate(request)
        second_result = await self.adapter.generate(request)

        self.assertIsNone(first_result.error)
        self.assertIsNone(second_result.error)
        first, second = self.generation_post.call_args_list
        self.assertNotEqual(
            first.kwargs["headers"]["Idempotency-Key"],
            second.kwargs["headers"]["Idempotency-Key"],
        )

    async def test_network_retries_stop_at_configured_attempt_count(self):
        self.adapter.max_retry_attempts = 2
        self.generation_post.side_effect = asyncio.TimeoutError

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.images)
        self.assertTrue(result.error)
        self.assertEqual(self.generation_post.call_count, 2)
        first, second = self.generation_post.call_args_list
        self.assertEqual(first.kwargs["json"], second.kwargs["json"])
        self.assertEqual(first.kwargs["headers"], second.kwargs["headers"])

    async def test_http_cancellation_propagates_without_retry(self):
        self.generation_post.side_effect = asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.generation_post.assert_called_once()

    async def test_response_read_cancellation_propagates_without_retry(self):
        self.response.read.side_effect = asyncio.CancelledError
        self.response.json.side_effect = asyncio.CancelledError
        self.response.text.side_effect = asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.generation_post.assert_called_once()

    async def test_expensive_manual_portrait_is_rejected_before_generation(self):
        self.adapter.config.extra["max_cost_gems"] = 5
        self._set_response({"total_gems": 80, "balance_gems": 120}, quote=True)

        result = await self.adapter.generate(
            self.GenerationRequest(
                prompt="1girl, cat ears", aspect_ratio="3:4", image_size=(896, 1152)
            )
        )

        self.assertIsNone(result.images)
        self.assertIn("80", result.error)
        self.assertIn("5", result.error)
        self.quote_post.assert_called_once()
        self.generation_post.assert_not_called()
        parameters = self.quote_post.call_args.kwargs["json"]["parameters"]
        self.assertEqual((parameters["width"], parameters["height"]), (896, 1152))
        self.assertEqual(parameters["steps"], 28)
        self.assertEqual(parameters["n_samples"], 1)

    async def test_paid_quotes_at_or_below_configured_limit_are_allowed(self):
        for limit in (2, 3, 5, 2.5):
            for quote in (0, limit / 2, limit):
                with self.subTest(limit=limit, quote=quote):
                    self.adapter.config.extra["max_cost_gems"] = limit
                    self._set_response({"total_gems": quote}, quote=True)
                    self.quote_post.reset_mock()
                    self.generation_post.reset_mock()

                    result = await self.adapter.generate(
                        self.GenerationRequest(prompt="1girl")
                    )

                    self.assertIsNone(result.error)
                    self.assertEqual(result.images, [self.image_bytes])
                    self.quote_post.assert_called_once()
                    self.generation_post.assert_called_once()

    async def test_zero_limit_only_allows_zero_cost_quotes(self):
        self.adapter.config.extra["max_cost_gems"] = 0
        for quote in (0, 0.01):
            with self.subTest(quote=quote):
                self._set_response({"total_gems": quote}, quote=True)
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                if quote == 0:
                    self.assertIsNone(result.error)
                    self.generation_post.assert_called_once()
                else:
                    self.assertTrue(result.error)
                    self.generation_post.assert_not_called()

    async def test_missing_limit_defaults_to_five_gems(self):
        self.assertNotIn("max_cost_gems", self.adapter.config.extra)
        self._set_response({"total_gems": 5.01}, quote=True)

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertTrue(result.error)
        self.generation_post.assert_not_called()

    async def test_fractional_quote_above_limit_is_not_rounded_down(self):
        self.adapter.config.extra["max_cost_gems"] = 2
        self.quote_response.read.return_value = b'{"total_gems":2.0000000000000001}'

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertIsNone(result.images)
        self.assertIn("2.0000000000000001", result.error)
        self.quote_post.assert_called_once()
        self.generation_post.assert_not_called()

    async def test_invalid_cost_limits_are_rejected_without_http(self):
        for limit in (-1, True, False, None, "", "invalid", float("nan"), float("inf")):
            with self.subTest(limit=limit):
                self.adapter.config.extra["max_cost_gems"] = limit
                self.session.post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertTrue(result.error)
                self.session.post.assert_not_called()

    async def test_invalid_quote_costs_never_fall_back_to_generation(self):
        invalid_bodies = (
            {},
            [],
            None,
            {"total_gems": None},
            {"total_gems": True},
            {"total_gems": False},
            {"total_gems": "0"},
            {"total_gems": -1},
            {"total_gems": float("nan")},
            {"total_gems": float("inf")},
            {"total_gems": []},
            {"total_gems": 0, "error": {"message": "quotation failed"}},
        )
        for body in invalid_bodies:
            with self.subTest(body=body):
                self._set_response(body, quote=True)
                self.quote_post.reset_mock()
                self.generation_post.reset_mock()

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertIsNone(result.images)
                self.assertTrue(result.error)
                self.quote_post.assert_called_once()
                self.generation_post.assert_not_called()

    async def test_invalid_quote_json_does_not_generate(self):
        self.quote_response.read.return_value = b"not json"
        self.quote_response.json.side_effect = json.JSONDecodeError("Invalid", "x", 0)

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertTrue(result.error)
        self.quote_post.assert_called_once()
        self.generation_post.assert_not_called()

    async def test_quote_http_errors_do_not_generate_or_rotate_credentials(self):
        for status in (401, 404, 500):
            with self.subTest(status=status):
                self._set_response({"error": "quote failed"}, status=status, quote=True)
                self.quote_post.reset_mock()
                self.generation_post.reset_mock()
                with patch.object(self.adapter, "_rotate_api_key") as rotate:
                    result = await self.adapter.generate(
                        self.GenerationRequest(prompt="1girl")
                    )

                self.assertTrue(result.error)
                self.quote_post.assert_called_once()
                self.generation_post.assert_not_called()
                rotate.assert_not_called()

    async def test_quote_transport_errors_do_not_generate(self):
        for exception in (
            aiohttp.ClientConnectionError("offline"),
            asyncio.TimeoutError(),
        ):
            with self.subTest(exception=type(exception).__name__):
                self.quote_post.reset_mock()
                self.quote_post.side_effect = exception

                result = await self.adapter.generate(
                    self.GenerationRequest(prompt="1girl")
                )

                self.assertTrue(result.error)
                self.quote_post.assert_called_once()
                self.generation_post.assert_not_called()

    async def test_quote_read_errors_do_not_generate(self):
        self.quote_response.read.side_effect = aiohttp.ClientPayloadError("truncated")

        result = await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.assertTrue(result.error)
        self.quote_post.assert_called_once()
        self.generation_post.assert_not_called()

    async def test_quote_and_generation_share_payload_seed_and_api_key(self):
        result = await self.adapter.generate(
            self.GenerationRequest(prompt="1girl", aspect_ratio="4:3")
        )

        self.assertIsNone(result.error)
        self.assertEqual(self.session.post.call_count, 2)
        quote, generation = self.session.post.call_args_list
        self.assertTrue(quote.args[0].endswith("/ai/generate-image/quote"))
        self.assertTrue(generation.args[0].endswith("/v1/nai/generate-image"))
        self.assertEqual(quote.kwargs["json"], generation.kwargs["json"])
        self.assertEqual(
            quote.kwargs["headers"]["Authorization"],
            generation.kwargs["headers"]["Authorization"],
        )
        self.assertEqual(
            quote.kwargs["headers"]["Authorization"], "Bearer first-test-key"
        )
        self.assertNotIn("Idempotency-Key", quote.kwargs["headers"])
        self.assertTrue(generation.kwargs["headers"]["Idempotency-Key"])
        self.assertGreaterEqual(quote.kwargs["json"]["parameters"]["seed"], 0)

    async def test_image_editing_uses_same_cost_limit(self):
        self.adapter.config.extra["max_cost_gems"] = 3
        self._set_response({"total_gems": 4}, quote=True)

        result = await self.adapter.generate(
            self.GenerationRequest(
                prompt="change hair color",
                images=[self.ImageData(data=self.image_bytes, mime_type="image/png")],
            )
        )

        self.assertTrue(result.error)
        self.generation_post.assert_not_called()
        payload = self.quote_post.call_args.kwargs["json"]
        self.assertEqual(payload["action"], "img2img")
        self.assertTrue(payload["parameters"]["image"])

    async def test_quote_url_preserves_service_prefix_for_supported_base_urls(self):
        for prefix in ("", "/proxy"):
            root = "https://images.example.invalid" + prefix
            expected_quote_url = root + "/ai/generate-image/quote"
            for suffix in ("", "/", "/v1", "/v1/", "/v1/nai/generate-image"):
                with self.subTest(prefix=prefix, suffix=suffix):
                    self.adapter.base_url = root + suffix
                    self.quote_post.reset_mock()

                    result = await self.adapter.generate(
                        self.GenerationRequest(prompt="1girl")
                    )

                    self.assertIsNone(result.error)
                    self.assertEqual(
                        self.quote_post.call_args.args, (expected_quote_url,)
                    )

    async def test_quote_cancellation_propagates_without_generation(self):
        self.quote_response.read.side_effect = asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await self.adapter.generate(self.GenerationRequest(prompt="1girl"))

        self.quote_post.assert_called_once()
        self.generation_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
