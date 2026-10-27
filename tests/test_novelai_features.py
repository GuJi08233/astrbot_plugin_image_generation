from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import logging
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from PIL import Image


class NovelAIFeaturesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Create isolated adapter modules and forbid real HTTP sessions."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_novelai_features_test_plugin"
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
            adapter_module = importlib.import_module(
                f"{namespace}.adapter.novelai_adapter"
            )
            self.shared = importlib.import_module(f"{namespace}.core.shared.types")
            options_module = importlib.import_module(
                f"{namespace}.core.generation.novelai_options"
            )
        self.validate = options_module.validate_novelai_options
        self.adapter = adapter_module.NovelAIAdapter(
            self.shared.AdapterConfig(
                type=self.shared.AdapterType.NOVELAI,
                api_keys=["test-key"],
                model="nai-diffusion-4-5-full",
                capability_options={"text_to_image": True, "image_to_image": True},
                extra={"seed": 123, "negative_prompt": "default negative", "steps": 28},
            )
        )
        image = io.BytesIO()
        Image.new("RGB", (64, 32), "red").save(image, format="PNG")
        self.image = self.shared.ImageData(image.getvalue(), "image/png")
        mask = Image.new("L", (64, 32), 0)
        mask.paste(255, (0, 0, 32, 32))
        image = io.BytesIO()
        mask.save(image, format="PNG")
        self.mask = self.shared.ImageData(image.getvalue(), "image/png")
        self.encoding = base64.b64encode(b"encoded-vibe-features").decode()
        no_http = patch.object(
            aiohttp,
            "ClientSession",
            side_effect=AssertionError("Real HTTP is forbidden"),
        )
        no_http.start()
        self.addCleanup(no_http.stop)

    def test_sampling_overrides_are_request_scoped_and_manual_sizes_are_exact(self):
        original = deepcopy(self.adapter.config.extra)
        options = {
            "width": 192,
            "height": 320,
            "steps": 42,
            "sampler": "k_dpmpp_2m",
            "noise_schedule": "exponential",
            "scale": 6.25,
            "cfg_rescale": 0.8,
            "seed": 0,
            "sm": True,
            "sm_dyn": True,
            "variety_boost": True,
        }
        request = self.shared.GenerationRequest(prompt="a cat", novelai=options)
        payload = self.adapter._build_payload(request)
        parameters = payload["parameters"]
        for key in options.keys() - {"variety_boost"}:
            self.assertEqual(parameters[key], options[key])
        self.assertEqual(parameters["skip_cfg_above_sigma"], 58)
        self.assertEqual(parameters["n_samples"], 1)
        self.assertEqual(self.adapter.config.extra, original)
        self.assertEqual(request.novelai, options)
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.adapter._build_payload(
                self.shared.GenerationRequest(
                    prompt="cat",
                    image_size=(512, 512),
                    novelai=options,
                )
            )
        payload = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="cat",
                image_size=(192, 320),
                novelai=options,
            )
        )
        self.assertEqual(payload["parameters"]["width"], 192)

    def test_character_prompts_preserve_positive_negative_pairing_and_positions(self):
        payload = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="two people",
                negative_prompt="base negative",
                novelai={
                    "characters": [
                        {
                            "prompt": "red hair",
                            "negative_prompt": "hat",
                            "position": "A1",
                        },
                        {"prompt": "blue hair", "position": [0.58, 0.96]},
                        {"prompt": "disabled", "enabled": False},
                    ]
                },
            )
        )
        parameters = payload["parameters"]
        self.assertTrue(parameters["use_coords"])
        self.assertTrue(parameters["v4_prompt"]["use_coords"])
        positive = parameters["v4_prompt"]["caption"]["char_captions"]
        negative = parameters["v4_negative_prompt"]["caption"]["char_captions"]
        self.assertEqual(
            [item["char_caption"] for item in positive], ["red hair", "blue hair"]
        )
        self.assertEqual([item["char_caption"] for item in negative], ["hat", ""])
        self.assertEqual(positive[0]["centers"], [{"x": 0.1, "y": 0.1}])
        self.assertEqual(positive[1]["centers"], [{"x": 0.5, "y": 0.9}])
        self.assertEqual(len(parameters["characterPrompts"]), 3)
        self.assertFalse(parameters["characterPrompts"][-1]["enabled"])

    def test_automatic_character_positions_and_v5_coordinates(self):
        options = {
            "characters": [{"prompt": "one", "position": None}, {"prompt": "two"}]
        }
        parameters = self.adapter._build_payload(
            self.shared.GenerationRequest(prompt="two people", novelai=options)
        )["parameters"]
        self.assertFalse(parameters["use_coords"])
        self.adapter.model = "nai-diffusion-5-full"
        options["characters"][0]["position"] = [0.22, 0.37]
        parameters = self.adapter._build_payload(
            self.shared.GenerationRequest(prompt="two people", novelai=options)
        )["parameters"]
        self.assertEqual(parameters["params_version"], 4)
        self.assertEqual(
            parameters["characterPrompts"][0]["center"], {"x": 0.22, "y": 0.37}
        )

    def test_inpainting_uses_aligned_mask_and_infill_model(self):
        payload = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="replace left half",
                images=[self.image],
                novelai={
                    "width": 128,
                    "height": 128,
                    "inpaint": {"mask": self.mask, "strength": 0.9, "seed": 12},
                },
            )
        )
        self.assertEqual(payload["action"], "infill")
        self.assertEqual(payload["model"], "nai-diffusion-4-5-full-inpainting")
        parameters = payload["parameters"]
        self.assertEqual(parameters["inpaintImg2ImgStrength"], 0.9)
        self.assertEqual(parameters["extra_noise_seed"], 12)
        self.assertFalse(parameters["add_original_image"])
        with Image.open(io.BytesIO(base64.b64decode(parameters["image"]))) as source:
            self.assertEqual(source.size, (128, 128))
            self.assertEqual(source.getpixel((64, 0)), (255, 255, 255))
            self.assertEqual(source.getpixel((64, 64)), (255, 0, 0))
        with Image.open(io.BytesIO(base64.b64decode(parameters["mask"]))) as mask:
            self.assertEqual(mask.size, (128, 128))
            self.assertEqual(mask.getpixel((16, 64)), (255, 255, 255, 255))
            self.assertEqual(mask.getpixel((96, 64)), (0, 0, 0, 255))
            self.assertEqual(mask.getpixel((16, 0)), (0, 0, 0, 255))
        bad_mask = io.BytesIO()
        Image.new("L", (32, 64), 255).save(bad_mask, format="PNG")
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            self.adapter._build_payload(
                self.shared.GenerationRequest(
                    prompt="cat",
                    images=[self.image],
                    novelai={
                        "inpaint": {
                            "mask": self.shared.ImageData(
                                bad_mask.getvalue(), "image/png"
                            ),
                        }
                    },
                )
            )

    def test_img2img_can_use_explicit_source_and_per_request_noise_parameters(self):
        payload = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="a cat",
                novelai={
                    "i2i": {
                        "image": self.image,
                        "strength": 0.4,
                        "noise": 0.2,
                        "seed": -1,
                    }
                },
            )
        )
        self.assertEqual(payload["action"], "img2img")
        self.assertEqual(payload["parameters"]["strength"], 0.4)
        self.assertEqual(payload["parameters"]["noise"], 0.2)
        self.assertGreaterEqual(payload["parameters"]["extra_noise_seed"], 0)
        self.assertLessEqual(payload["parameters"]["extra_noise_seed"], 2**32 - 1)

    def test_disabled_image_to_image_rejects_nested_and_collected_sources(self):
        requests = [
            self.shared.GenerationRequest(
                prompt="a cat", novelai={"i2i": {"image": self.image}}
            ),
            self.shared.GenerationRequest(
                prompt="a cat",
                novelai={"inpaint": {"image": self.image, "mask": self.mask}},
            ),
            self.shared.GenerationRequest(prompt="a cat", images=[self.image]),
        ]
        # Re-check current capabilities even for tasks created before a config change.
        self.adapter.config.capability_options["image_to_image"] = False
        for request in requests:
            with (
                self.subTest(options=request.novelai),
                self.assertRaisesRegex(
                    ValueError, "image-to-image capability is disabled"
                ),
            ):
                self.adapter._build_payload(request)
        self.assertEqual(
            self.adapter._build_payload(self.shared.GenerationRequest(prompt="a cat"))[
                "action"
            ],
            "generate",
        )

    def test_rgba_mask_keeps_black_and_transparent_pixels_unmodified(self):
        mask = Image.new("RGBA", (64, 32), (0, 0, 0, 255))
        mask.paste((255, 255, 255, 255), (21, 0, 43, 32))
        mask.paste((255, 255, 255, 0), (43, 0, 64, 32))
        output = io.BytesIO()
        mask.save(output, format="PNG")
        parameters = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="replace white area",
                novelai={
                    "width": 64,
                    "height": 64,
                    "inpaint": {
                        "image": self.image,
                        "mask": self.shared.ImageData(output.getvalue(), "image/png"),
                    },
                },
            )
        )["parameters"]
        with Image.open(io.BytesIO(base64.b64decode(parameters["mask"]))) as prepared:
            self.assertEqual(prepared.getpixel((10, 32)), (0, 0, 0, 255))
            self.assertEqual(prepared.getpixel((30, 32)), (255, 255, 255, 255))
            self.assertEqual(prepared.getpixel((52, 32)), (0, 0, 0, 255))

    def test_precise_references_are_distinct_from_img2img_and_invert_fidelity(self):
        payload = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="a person",
                novelai={
                    "character_references": [
                        {
                            "image": self.image,
                            "type": "character",
                            "fidelity": 0.8,
                            "strength": 0.7,
                        }
                    ]
                },
            )
        )
        self.assertEqual(payload["action"], "generate")
        parameters = payload["parameters"]
        self.assertNotIn("image", parameters)
        self.assertEqual(
            parameters["director_reference_secondary_strength_values"], [0.2]
        )
        self.assertEqual(parameters["director_reference_strength_values"], [0.7])
        self.assertEqual(parameters["director_reference_information_extracted"], [1.0])
        self.assertEqual(
            parameters["director_reference_descriptions"][0]["caption"]["base_caption"],
            "character",
        )
        with Image.open(
            io.BytesIO(base64.b64decode(parameters["director_reference_images"][0]))
        ) as image:
            self.assertEqual(image.size, (1024, 1536))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_vibe_uses_preencoded_features_without_any_encoding_request(self):
        payload = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="a cat",
                novelai={
                    "vibe_references": [
                        {
                            "encoding": self.encoding,
                            "strength": 0.6,
                            "information_extracted": 0.8,
                        }
                    ]
                },
            )
        )
        parameters = payload["parameters"]
        self.assertEqual(parameters["reference_image_multiple"], [self.encoding])
        self.assertEqual(parameters["reference_strength_multiple"], [0.6])
        self.assertEqual(parameters["reference_information_extracted_multiple"], [0.8])
        with self.assertRaisesRegex(ValueError, "pre-encoded"):
            self.adapter._build_payload(
                self.shared.GenerationRequest(
                    prompt="a cat",
                    novelai={"vibe_references": [{"image": self.image}]},
                )
            )
        self.adapter.model = "nai-diffusion-3"
        parameters = self.adapter._build_payload(
            self.shared.GenerationRequest(
                prompt="a cat",
                novelai={"vibe_references": [{"image": self.image}]},
            )
        )["parameters"]
        self.assertTrue(
            base64.b64decode(parameters["reference_image_multiple"][0]).startswith(
                b"\x89PNG"
            )
        )

    def test_invalid_schema_values_and_unsupported_combinations_are_rejected(self):
        invalid = [
            {"unknown": True},
            {"n_samples": 8},
            {"steps": True},
            {"steps": 51},
            {"scale": float("nan")},
            {"cfg_rescale": 2},
            {"seed": 2**32},
            {"sampler": "not-a-sampler"},
            {"noise_schedule": "unknown"},
            {"width": 832},
            {"width": 2048, "height": 2048},
            {"i2i": {}, "inpaint": {"mask": self.mask}},
            {"inpaint": {}},
            {"action": "generate", "i2i": {}},
            {"action": "infill"},
            {"characters": [{"prompt": "cat", "position": [1.1, 0.5]}]},
            {"characters": [{"prompt": "cat", "position": "F6"}]},
            {"character_references": [{"image": self.image, "fidelity": -1}]},
            {
                "vibe_references": [
                    {"encoding": self.encoding, "information_extracted": 0}
                ]
            },
            {
                "vibe_references": [
                    {"encoding": base64.b64encode(self.image.data).decode()}
                ]
            },
            {"vibe_references": [{"encoding": "not base64"}]},
            {"vibe_references": [{"encoding": self.encoding}] * 17},
            {
                "inpaint": {"mask": self.mask},
                "vibe_references": [{"encoding": self.encoding}],
            },
            {
                "character_references": [{"image": self.image}],
                "vibe_references": [{"encoding": self.encoding}],
            },
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.validate(options)

    def test_v5_free_render_and_other_model_restrictions_are_local(self):
        self.adapter.model = "nai-diffusion-5-full"
        for options in (
            {"steps": 29},
            {"width": 128, "height": 128},
            {"sm": True},
            {"character_references": [{"image": self.image}]},
            {"vibe_references": [{"encoding": self.encoding}]},
            {"i2i": {"image": self.image}},
            {"inpaint": {"image": self.image, "mask": self.mask}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.adapter._build_payload(
                    self.shared.GenerationRequest(prompt="a cat", novelai=options)
                )
        self.adapter.model = "nai-diffusion-4-full"
        with self.assertRaisesRegex(ValueError, "NAI4.5"):
            self.adapter._build_payload(
                self.shared.GenerationRequest(
                    prompt="a cat",
                    novelai={"character_references": [{"image": self.image}]},
                )
            )
        self.adapter.model = "nai-diffusion-3"
        with self.assertRaisesRegex(ValueError, "Character prompts"):
            self.adapter._build_payload(
                self.shared.GenerationRequest(
                    prompt="a cat", novelai={"characters": [{"prompt": "cat"}]}
                )
            )

        self.adapter.model = "nai-diffusion-4-5-full-inpainting"
        with self.assertRaisesRegex(ValueError, "require action infill"):
            self.adapter._build_payload(self.shared.GenerationRequest(prompt="a cat"))

    def test_unresolved_or_oversized_sources_are_rejected_before_network(self):
        for image in (
            "https://example.invalid/image.png",
            self.shared.ImageData(b"invalid", "image/png"),
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                self.adapter._build_payload(
                    self.shared.GenerationRequest(
                        prompt="a cat", novelai={"i2i": {"image": image}}
                    )
                )
        oversized = io.BytesIO()
        Image.new("RGB", (4097, 1)).save(oversized, format="PNG")
        with self.assertRaisesRegex(ValueError, "4096"):
            self.adapter._build_payload(
                self.shared.GenerationRequest(
                    prompt="a cat",
                    novelai={
                        "i2i": {
                            "image": self.shared.ImageData(
                                oversized.getvalue(), "image/png"
                            )
                        }
                    },
                )
            )

    async def test_concurrent_features_use_identical_quote_and_generation_payloads(
        self,
    ):
        session = MagicMock()
        response_data = {
            "job": {"status": "charged"},
            "images": [base64.b64encode(self.image.data).decode()],
        }

        def fake_post(url, **kwargs):
            response = MagicMock()
            response.status = 200
            response.headers = {"Content-Type": "application/json"}
            data = {"total_gems": 0} if url.endswith("/quote") else response_data
            response.read = AsyncMock(return_value=json.dumps(data).encode())
            response.json = AsyncMock(return_value=data)
            manager = MagicMock()
            manager.__aenter__.return_value = response
            return manager

        session.post.side_effect = fake_post
        before = deepcopy(self.adapter.config.extra)
        requests = [
            self.shared.GenerationRequest(
                prompt=f"cat-{index}",
                novelai={
                    "steps": index + 1,
                    "seed": -1,
                    "i2i": {
                        "image": self.image,
                        "strength": index / 10,
                        "seed": -1,
                    },
                },
            )
            for index in range(3)
        ]
        with patch.object(self.adapter, "_get_session", return_value=session):
            results = await asyncio.gather(
                *(self.adapter.generate(request) for request in requests)
            )
        self.assertTrue(all(result.images and not result.error for result in results))
        calls = session.post.call_args_list
        self.assertEqual(len(calls), 6)
        for request in requests:
            matching = [
                call.kwargs["json"]
                for call in calls
                if call.kwargs["json"]["input"] == request.prompt
            ]
            self.assertEqual(len(matching), 2)
            self.assertEqual(matching[0], matching[1])
            self.assertEqual(
                matching[0]["parameters"]["steps"], request.novelai["steps"]
            )
        self.assertEqual(self.adapter.config.extra, before)
        self.assertFalse(any("encode" in call.args[0] for call in calls))


if __name__ == "__main__":
    unittest.main()
