from __future__ import annotations

import base64
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import test_page_negative_prompt as negative_page_fixture


class NovelAIPageTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Reuse the isolated page fixture and create disposable image uploads."""
        negative_page_fixture.PageNegativePromptTest.setUp(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.plugin.page_upload_dir = Path(temporary.name)
        self.png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
        )
        for name in ("source.png", "mask.png", "character.png"):
            (self.plugin.page_upload_dir / name).write_bytes(self.png)
        provider = self.AdapterConfig(
            type=self.AdapterType.NOVELAI,
            name="NovelAI",
            model="nai-diffusion-4-5-full",
            available_models=["nai-diffusion-4-5-full"],
            capability_options={"image_to_image": True},
        )
        self.plugin.config_manager.adapter_config = provider
        self.plugin.config_manager._all_provider_configs = [provider]
        self.payload = {
            "prompt": "white cat",
            "model": "NovelAI/nai-diffusion-4-5-full",
            "reference_tokens": ["source.png"],
        }

    async def test_empty_options_preserve_defaults_and_ordinary_source(self):
        """Leave omitted or empty NovelAI options unset in the public API."""
        for options in (None, {}):
            with self.subTest(options=options):
                self.request.json.return_value = {**self.payload, "novelai": options}
                result = await self.page.page_generate()
                self.assertTrue(result["ok"])
                submitted = (
                    self.plugin.public_api.submit_generation_task.call_args.kwargs
                )
                self.assertIsNone(submitted["novelai"])
                self.assertEqual(
                    submitted["reference_image_sources"],
                    [str(self.plugin.page_upload_dir / "source.png")],
                )

    async def test_mask_and_ordered_precise_references_use_dedicated_assets(self):
        """Resolve role tokens while retaining duplicate references and metadata."""
        options = {
            "steps": 30,
            "characters": [{"prompt": "cat", "position": "B3"}],
            "character_references": [{"image": "https://example.invalid/first.png"}],
            "inpaint": {"strength": 0.8},
        }
        before = copy.deepcopy(options)
        self.request.json.return_value = {
            **self.payload,
            "novelai": options,
            "negative_prompt": "watermark",
            "novelai_mask_token": "mask.png",
            "novelai_character_references": [
                {"token": "character.png", "strength": 0.4, "fidelity": 0.9},
                {"token": "character.png", "type": "style", "strength": 0.7},
            ],
        }
        result = await self.page.page_generate()
        self.assertTrue(result["ok"])
        submitted = self.plugin.public_api.submit_generation_task.call_args.kwargs
        self.assertEqual(submitted["negative_prompt"], "watermark")
        self.assertEqual(len(submitted["reference_image_sources"]), 1)
        resolved = submitted["novelai"]
        self.assertEqual(
            resolved["inpaint"]["mask"], str(self.plugin.page_upload_dir / "mask.png")
        )
        references = resolved["character_references"]
        self.assertEqual(len(references), 3)
        self.assertEqual(references[1]["image"], references[2]["image"])
        self.assertEqual(references[1]["strength"], 0.4)
        self.assertEqual(references[2]["strength"], 0.7)
        self.assertEqual(references[2]["type"], "style")
        self.assertEqual(options, before)

    async def test_invalid_options_fail_before_any_model_change(self):
        """Reject malformed grouped options without changing runtime settings."""
        for options in (
            [],
            "{}",
            {"i2i": None},
            {"inpaint": []},
            {"characters": [None]},
            {"characters": [{}] * 7},
            {"action": []},
            {"action": "unknown"},
            {"action": "generate", "i2i": {}},
            {"i2i": {}, "inpaint": {"mask": "mask.png"}},
            {"inpaint": {}},
            {"scale": float("nan")},
            {"steps": 0},
            {"width": 832},
            {"unknown_option": True},
            {"character_references": [{"image": "x"}], "vibe_references": [{}]},
        ):
            with self.subTest(options=options):
                self.request.json.return_value = {
                    **self.payload,
                    "model": "Other/model",
                    "novelai": options,
                }
                result = await self.page.page_generate()
                self.assertEqual(result["status_code"], 400)
        self.plugin.config_manager.save_model_setting.assert_not_called()
        self.plugin.public_api.submit_generation_task.assert_not_awaited()

    async def test_invalid_source_tokens_cannot_downgrade_novelai_to_text_generation(
        self,
    ):
        """Reject missing source images instead of dropping their upload tokens."""
        for tokens in (["missing.png"], ["../source.png"], [""], "source.png", [1]):
            with self.subTest(tokens=tokens):
                self.request.json.return_value = {
                    **self.payload,
                    "reference_tokens": tokens,
                }
                result = await self.page.page_generate()
                self.assertEqual(result["status_code"], 400)
        self.plugin.public_api.submit_generation_task.assert_not_awaited()

    async def test_invalid_role_tokens_and_metadata_are_never_silently_skipped(self):
        """Reject missing role assets, traversal, invalid types, and bad weights."""
        for fields in (
            {"novelai_mask_token": "missing.png"},
            {"novelai_mask_token": "../mask.png"},
            {"novelai_mask_token": ""},
            {"novelai_mask_token": 1},
            {"novelai_character_references": {}},
            {"novelai_character_references": [{}]},
            {"novelai_character_references": [{"token": "missing.png"}]},
            {"novelai_character_references": [{"token": "character.png", "type": []}]},
            {
                "novelai_character_references": [
                    {"token": "character.png", "strength": True}
                ]
            },
            {
                "novelai_character_references": [
                    {"token": "character.png", "fidelity": 1.1}
                ]
            },
            {"novelai_character_references": [{"token": "character.png"}] * 17},
        ):
            with self.subTest(fields=fields):
                self.request.json.return_value = {**self.payload, **fields}
                result = await self.page.page_generate()
                self.assertEqual(result["status_code"], 400)
        self.plugin.public_api.submit_generation_task.assert_not_awaited()

    async def test_explicit_text_mode_does_not_submit_uploaded_source_images(self):
        """Keep text generation explicit even when the page retains a source."""
        self.request.json.return_value = {
            **self.payload,
            "novelai": {"action": "generate", "steps": 25},
        }
        result = await self.page.page_generate()
        self.assertTrue(result["ok"])
        submitted = self.plugin.public_api.submit_generation_task.call_args.kwargs
        self.assertEqual(submitted["reference_image_sources"], [])
        self.assertEqual(submitted["novelai"]["action"], "generate")

    async def test_selected_non_novelai_model_rejects_options_before_switching(self):
        """Use the selected model's metadata without activating that model first."""
        other = self.AdapterConfig(
            type=self.AdapterType.OPENAI,
            name="Other",
            available_models=["image"],
        )
        self.plugin.config_manager._all_provider_configs.append(other)
        self.request.json.return_value = {
            **self.payload,
            "model": "Other/image",
            "novelai": {"steps": 25},
        }
        result = await self.page.page_generate()
        self.assertEqual(result["status_code"], 400)
        self.plugin.config_manager.save_model_setting.assert_not_called()

    async def test_existing_upload_endpoint_accepts_mask_images_without_generation(
        self,
    ):
        """Reuse image validation and upload tokens without invoking paid tools."""
        self.request.files = AsyncMock(
            return_value={
                "file": SimpleNamespace(
                    read=AsyncMock(return_value=self.png), filename="mask.png"
                )
            }
        )
        self.plugin.config_manager.usage_settings = SimpleNamespace(max_image_size_mb=2)
        self.plugin.image_processor = SimpleNamespace(
            validate_image_data=MagicMock(
                return_value=SimpleNamespace(mime_type="image/png")
            )
        )
        result = await self.page.page_upload_reference()
        self.assertTrue(result["token"].startswith("upload_"))
        self.assertEqual(
            (self.plugin.page_upload_dir / result["token"]).read_bytes(), self.png
        )
        self.plugin.public_api.submit_generation_task.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
