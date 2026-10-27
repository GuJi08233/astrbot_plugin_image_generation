from __future__ import annotations

import ast
import asyncio
import importlib
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class NegativePromptPipelineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load the real pipeline while isolating AstrBot runtime boundaries."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_negative_prompt_pipeline_test_plugin"
        modules = {}
        for suffix in (
            "",
            "core",
            "core.adapters",
            "core.api",
            "core.config",
            "core.generation",
            "core.shared",
            "core.tasks",
        ):
            name = f"{namespace}.{suffix}" if suffix else namespace
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.split(".")))]
            modules[name] = module
        api = types.ModuleType("astrbot.api")
        api.logger = logging.getLogger(__name__)
        modules[api.__name__] = api
        event = types.ModuleType("astrbot.api.event")
        event.MessageChain = MagicMock
        modules[event.__name__] = event
        star = types.ModuleType("astrbot.api.star")
        star.Context = object
        modules[star.__name__] = star
        for suffix, symbol in (
            ("core.audit.safety", "SafetyAuditor"),
            ("core.config.manager", "ConfigManager"),
            ("core.generation.image_processor", "ImageProcessor"),
            ("core.tasks.manager", "TaskManager"),
            ("core.tasks.usage", "UsageManager"),
        ):
            module = types.ModuleType(f"{namespace}.{suffix}")
            setattr(module, symbol, object)
            modules[module.__name__] = module
        formatter = types.ModuleType(f"{namespace}.core.formatting.result")
        formatter.build_result_info_message = MagicMock(return_value="")
        formatter.format_generation_failure_message = str
        modules[formatter.__name__] = formatter
        with patch.dict(sys.modules, modules):
            self.public_module = importlib.import_module(f"{namespace}.core.api.public")
            self.executor_module = importlib.import_module(
                f"{namespace}.core.generation.executor"
            )
            self.generator_module = importlib.import_module(
                f"{namespace}.core.adapters.generator"
            )
            self.shared = importlib.import_module(f"{namespace}.core.shared.types")

        main_tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
        create_method = next(
            method
            for node in main_tree.body
            if isinstance(node, ast.ClassDef)
            for method in node.body
            if isinstance(method, ast.FunctionDef)
            and method.name == "create_generation_task"
        )
        isolated_method = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                create_method,
            ],
            type_ignores=[],
        )
        method_globals = {
            "format_template_summary": self.public_module.format_template_summary,
            "GenerationTaskCreationError": self.public_module.GenerationTaskCreationError,
        }
        exec(
            compile(
                ast.fix_missing_locations(isolated_method),
                str(root / "main.py"),
                "exec",
            ),
            method_globals,
        )
        self.create_method = method_globals["create_generation_task"]

        self.plugin = MagicMock()
        self.plugin.generator.adapter.get_capabilities.return_value = (
            self.shared.ImageCapability.TEXT_TO_IMAGE
            | self.shared.ImageCapability.NEGATIVE_PROMPT
            | self.shared.ImageCapability.NOVELAI_OPTIONS
        )
        self.plugin.has_required_api_key.return_value = True
        self.plugin.normalize_image_count.side_effect = int
        self.plugin.config_manager.default_image_count = 1
        self.plugin.config_manager.default_aspect_ratio = "1:1"
        self.plugin.config_manager.default_resolution = "1K"
        self.plugin.config_manager.match_templates_in_prompt_body = False
        self.plugin.safety_auditor.audit_prompt = AsyncMock(return_value=(True, ""))
        self.plugin.usage_manager.check_rate_limit.return_value = None
        self.plugin.task_manager.get_generation_queue_rejection.return_value = None
        self.plugin.create_generation_task.return_value.task_id = "test-task"
        self.public_api = self.public_module.ImageGenerationPublicAPI(self.plugin)

    async def test_public_entry_points_preserve_inherit_override_and_clear(self):
        """Exercise both public calls through actual submission and task creation."""
        completed = object()
        self.public_api.wait_generation_result = AsyncMock(return_value=completed)
        for method in (
            self.public_api.submit_generation_task,
            self.public_api.generate_image_files,
        ):
            for arguments, expected in (
                ({}, None),
                ({"negative_prompt": None}, None),
                ({"negative_prompt": "blurry, watermark"}, "blurry, watermark"),
                ({"negative_prompt": ""}, ""),
            ):
                with self.subTest(method=method.__name__, arguments=arguments):
                    self.plugin.create_generation_task.reset_mock()
                    result = await method(
                        prompt="a white cat",
                        unified_msg_origin="test-session",
                        **arguments,
                    )
                    self.plugin.create_generation_task.assert_called_once()
                    self.assertEqual(
                        self.plugin.create_generation_task.call_args.kwargs[
                            "negative_prompt"
                        ],
                        expected,
                    )
                    if method.__name__ == "submit_generation_task":
                        self.assertTrue(result.ok)
                    else:
                        self.assertIs(result, completed)

    async def test_public_rejections_precede_audit_quota_and_task_creation(self):
        """Reject invalid values and unsupported overrides without side effects."""
        for method in (
            self.public_api.submit_generation_task,
            self.public_api.generate_image_files,
        ):
            for value in ([], {}, 1, False):
                with self.subTest(method=method.__name__, invalid=value):
                    result = await method(
                        prompt="a white cat",
                        unified_msg_origin="test-session",
                        negative_prompt=value,
                    )
                    self.assertFalse(result.ok)
                    self.assertEqual(result.code, "rejected")
            self.plugin.generator.adapter.get_capabilities.return_value = (
                self.shared.ImageCapability.TEXT_TO_IMAGE
            )
            for value in ("watermark", ""):
                with self.subTest(method=method.__name__, unsupported=value):
                    result = await method(
                        prompt="a white cat",
                        unified_msg_origin="test-session",
                        negative_prompt=value,
                    )
                    self.assertFalse(result.ok)
                    self.assertEqual(result.code, "rejected")
        self.plugin.safety_auditor.audit_prompt.assert_not_awaited()
        self.plugin.usage_manager.check_rate_limit.assert_not_called()
        self.plugin.create_generation_task.assert_not_called()

    async def test_real_queue_factory_preserves_each_tasks_override(self):
        """Execute the actual main method's deferred factories independently."""
        self.plugin.generation_executor.generate_and_send_image_async = AsyncMock()
        factories = []
        for index, negative_prompt in enumerate((None, "watermark", "")):
            self.create_method(
                self.plugin,
                task_id=f"queued-{index}",
                source="test",
                prompt="a white cat",
                negative_prompt=negative_prompt,
                novelai={"steps": 20 + index},
                images_data=[],
                unified_msg_origin="test-session",
                aspect_ratio="1:1",
                resolution="1K",
                image_count=2,
                is_usage_limit_admin=False,
            )
            factories.append(
                self.plugin.task_manager.create_generation_task.call_args.args[0]
            )
        await asyncio.gather(*(factory() for factory in factories))
        calls = self.plugin.generation_executor.generate_and_send_image_async.await_args_list
        self.assertEqual(
            {call.kwargs["task_id"]: call.kwargs["negative_prompt"] for call in calls},
            {"queued-0": None, "queued-1": "watermark", "queued-2": ""},
        )
        self.assertEqual(
            {call.kwargs["task_id"]: call.kwargs["novelai"] for call in calls},
            {f"queued-{index}": {"steps": 20 + index} for index in range(3)},
        )

    async def test_executor_and_generator_keep_overrides_across_concurrent_images(self):
        """Run the real executor and request copier for concurrent multi-image tasks."""
        capabilities = (
            self.shared.ImageCapability.TEXT_TO_IMAGE
            | self.shared.ImageCapability.IMAGE_TO_IMAGE
            | self.shared.ImageCapability.ASPECT_RATIO
            | self.shared.ImageCapability.RESOLUTION
            | self.shared.ImageCapability.NEGATIVE_PROMPT
            | self.shared.ImageCapability.NOVELAI_OPTIONS
        )
        adapter = MagicMock()
        adapter.get_capabilities.return_value = capabilities
        adapter.generate = AsyncMock(
            return_value=self.shared.GenerationResult(images=[b"mock-image"])
        )
        generator = object.__new__(self.generator_module.ImageGenerator)
        generator.adapter = adapter
        generator.convert_reference_images = AsyncMock(
            side_effect=lambda images: images
        )
        config = types.SimpleNamespace(
            default_image_count=1,
            max_image_count=10,
            max_concurrent_tasks=2,
        )
        task_manager = MagicMock()
        task_manager.get_generation_task.return_value = None
        image_processor = MagicMock()
        image_processor.save_generated_image.side_effect = lambda task_id, _image: (
            f"{task_id}.png"
        )
        auditor = types.SimpleNamespace(
            audit_generated_images=AsyncMock(return_value=(True, "", []))
        )
        executor = self.executor_module.GenerationExecutor(
            context=MagicMock(),
            config_manager=config,
            image_processor=image_processor,
            task_manager=task_manager,
            usage_manager=MagicMock(),
            safety_auditor=auditor,
        )
        executor.update_generator(generator)
        executor.refresh_request_semaphore()
        image = self.shared.ImageData(data=b"reference", mime_type="image/png")
        await asyncio.gather(
            *(
                executor.generate_and_send_image_async(
                    prompt="a white cat",
                    negative_prompt=negative_prompt,
                    novelai={"steps": 20 + index},
                    unified_msg_origin="test-session",
                    images_data=[image],
                    task_id=f"batch-{index}",
                    image_count=3,
                    auto_send=False,
                )
                for index, negative_prompt in enumerate((None, "watermark", ""))
            )
        )
        self.assertEqual(adapter.generate.await_count, 9)
        for index, expected in enumerate((None, "watermark", "")):
            requests = [
                call.args[0]
                for call in adapter.generate.await_args_list
                if call.args[0].task_id == f"batch-{index}"
            ]
            self.assertEqual(len(requests), 3)
            self.assertEqual({request.batch_index for request in requests}, {1, 2, 3})
            for request in requests:
                self.assertEqual(request.negative_prompt, expected)
                self.assertEqual(request.novelai, {"steps": 20 + index})
                self.assertEqual(request.images, [image])
                self.assertEqual(request.batch_count, 3)
        self.assertEqual(task_manager.mark_generation_task_succeeded.call_count, 3)
        task_manager.mark_generation_task_failed.assert_not_called()

    async def test_public_novelai_controls_reach_both_entry_points_and_audit(self):
        """Keep structured controls independent of the scene and negative prompt."""
        options = {
            "steps": 24,
            "characters": [
                {"prompt": "a white cat", "position": "A3"},
                {"prompt": "a black dog", "enabled": False},
            ],
        }
        self.public_api.wait_generation_result = AsyncMock(return_value=object())
        for method in (
            self.public_api.submit_generation_task,
            self.public_api.generate_image_files,
        ):
            with self.subTest(method=method.__name__):
                await method(prompt="garden", negative_prompt="blur", novelai=options)
                arguments = self.plugin.create_generation_task.call_args.kwargs
                self.assertEqual(arguments["prompt"], "garden")
                self.assertEqual(arguments["negative_prompt"], "blur")
                self.assertEqual(arguments["novelai"]["steps"], 24)
                self.assertEqual(
                    arguments["novelai"]["characters"][0]["prompt"], "a white cat"
                )
                audited = self.plugin.safety_auditor.audit_prompt.await_args.args[0]
                self.assertIn("a white cat", audited)
                self.assertNotIn("a black dog", audited)
        self.assertEqual(options["characters"][0]["position"], "A3")

    async def test_invalid_novelai_controls_are_rejected_before_side_effects(self):
        """Do not audit, reserve quota, or read assets for invalid options."""
        for value in ([], "text", {"steps": 0}, {"unknown_field": True}):
            with self.subTest(value=value):
                result = await self.public_api.submit_generation_task(
                    prompt="a cat", novelai=value, unified_msg_origin="test-session"
                )
                self.assertFalse(result.ok)
                self.assertEqual(result.code, "rejected")
        self.plugin.generator.adapter.get_capabilities.return_value = (
            self.shared.ImageCapability.TEXT_TO_IMAGE
        )
        result = await self.public_api.submit_generation_task(
            prompt="a cat", novelai={"steps": 24}, unified_msg_origin="test-session"
        )
        self.assertFalse(result.ok)
        self.plugin.safety_auditor.audit_prompt.assert_not_awaited()
        self.plugin.usage_manager.check_rate_limit.assert_not_called()
        self.plugin.create_generation_task.assert_not_called()

    async def test_novelai_assets_preserve_duplicate_roles_and_fail_closed(self):
        """Validate each named asset without flattening or deduplicating roles."""
        processor = MagicMock()
        processor._max_image_size_mb = 8
        image = self.shared.ImageData(data=b"validated-image", mime_type="image/png")
        processor.download_image = AsyncMock(return_value=image)
        processor.validate_image_data.return_value = image
        options = {
            "inpaint": {"mask": "https://example.invalid/mask.png"},
            "character_references": [
                {"image": "https://example.invalid/character.png", "strength": 0.4},
                {"image": "https://example.invalid/character.png", "strength": 0.8},
            ],
        }
        prepared = await self.public_module.prepare_novelai_assets(
            options, image_processor=processor, unified_msg_origin="test-session"
        )
        self.assertEqual(processor.download_image.await_count, 3)
        self.assertIs(prepared["inpaint"]["mask"], image)
        self.assertEqual(len(prepared["character_references"]), 2)
        self.assertEqual(prepared["character_references"][1]["strength"], 0.8)
        self.assertIsInstance(options["inpaint"]["mask"], str)
        processor.download_image.return_value = None
        with self.assertRaises(ValueError):
            await self.public_module.prepare_novelai_assets(
                options, image_processor=processor, unified_msg_origin="test-session"
            )

    async def test_generator_rejects_options_after_an_unsupported_adapter_switch(self):
        """Do not silently discard a queued task's options after a model switch."""
        adapter = MagicMock()
        adapter.get_capabilities.return_value = (
            self.shared.ImageCapability.TEXT_TO_IMAGE
        )
        adapter.generate = AsyncMock()
        generator = object.__new__(self.generator_module.ImageGenerator)
        generator.adapter = adapter
        for arguments in ({"novelai": {"steps": 24}}, {"negative_prompt": "blur"}):
            with self.subTest(arguments=arguments):
                result = await generator.generate_preconverted(
                    self.shared.GenerationRequest(prompt="a cat", **arguments),
                    images=[],
                )
                self.assertIsNotNone(result.error)
        adapter.generate.assert_not_awaited()
