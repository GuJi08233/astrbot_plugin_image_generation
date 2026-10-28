from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import jsonschema


class _FunctionTool:
    """Minimal generic tool base for isolated plugin tests."""

    def __class_getitem__(cls, _context):
        return cls


class NovelAIToolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load actual tool and validation code with mocked AstrBot boundaries."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_novelai_tool_test_plugin"
        modules = {}
        for suffix in (
            "",
            "core",
            "core.config",
            "core.generation",
            "core.llm",
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
        context = types.ModuleType("astrbot.core.agent.run_context")
        context.ContextWrapper = _FunctionTool
        modules[context.__name__] = context
        tool_api = types.ModuleType("astrbot.core.agent.tool")
        tool_api.FunctionTool = _FunctionTool
        tool_api.ToolExecResult = str
        modules[tool_api.__name__] = tool_api
        agent_context = types.ModuleType("astrbot.core.astr_agent_context")
        agent_context.AstrAgentContext = object
        modules[agent_context.__name__] = agent_context
        message_tools = types.ModuleType("astrbot.core.tools.message_tools")
        message_tools.SendMessageToUserTool = _FunctionTool
        modules[message_tools.__name__] = message_tools
        with patch.dict(sys.modules, modules):
            self.module = importlib.import_module(f"{namespace}.core.llm.tools")
            self.shared = importlib.import_module(f"{namespace}.core.shared.types")

        self.capabilities = self.shared.ImageCapability
        self.plugin = MagicMock()
        self.plugin.generator.adapter.get_capabilities.return_value = (
            self.capabilities.TEXT_TO_IMAGE
            | self.capabilities.NEGATIVE_PROMPT
            | self.capabilities.NOVELAI_OPTIONS
        )
        self.plugin.config_manager.default_aspect_ratio = "1:1"
        self.plugin.config_manager.default_resolution = "1K"
        self.plugin.config_manager.default_image_count = 1
        self.plugin.config_manager.match_templates_in_prompt_body = False
        self.plugin.normalize_image_count.return_value = 1
        self.plugin.is_usage_limit_admin.return_value = False
        self.plugin.has_required_api_key.return_value = True
        self.plugin.usage_manager.check_rate_limit.return_value = None
        self.plugin.task_manager.get_generation_queue_rejection.return_value = None
        self.plugin.safety_auditor.audit_prompt = AsyncMock(return_value=(True, ""))
        self.plugin.llm_result_handler.format_tool_start_result.return_value = (
            "accepted"
        )
        self.tool = self.module.ImageGenerationTool(plugin=self.plugin)
        self.event = types.SimpleNamespace(unified_msg_origin="test-session")
        collector = patch.object(
            self.module, "collect_tool_reference_images", AsyncMock(return_value=[])
        )
        self.collector = collector.start()
        self.addCleanup(collector.stop)

    async def test_omitted_null_and_sampling_overrides_reach_task_creation(self):
        """Preserve per-task settings without changing caller-owned input."""
        for arguments in (
            {},
            {"novelai": None},
            {"novelai": {}},
            {
                "novelai": {
                    "action": "generate",
                    "steps": 28,
                    "sampler": "k_euler_ancestral",
                    "scale": 5,
                    "seed": 42,
                    "width": 832,
                    "height": 1216,
                }
            },
        ):
            with self.subTest(arguments=arguments):
                original = deepcopy(arguments)
                result = await self.tool.call(
                    {"event": self.event}, prompt="a white cat", **arguments
                )
                self.assertEqual(result, "accepted")
                self.assertEqual(arguments, original)
                self.assertEqual(
                    self.plugin.create_generation_task.call_args.kwargs["novelai"],
                    arguments.get("novelai"),
                )

    async def test_invalid_options_are_rejected_before_audit_and_quota(self):
        """Run actual option validation before any usage-related side effect."""
        for invalid in (
            [],
            "steps=28",
            {"steps": True},
            {"width": 832},
            {"unknown": 1},
        ):
            with self.subTest(invalid=invalid):
                self.plugin.safety_auditor.audit_prompt.reset_mock()
                self.plugin.usage_manager.check_rate_limit.reset_mock()
                self.plugin.create_generation_task.reset_mock()
                result = await self.tool.call(
                    {"event": self.event}, prompt="a white cat", novelai=invalid
                )
                self.assertIn("NovelAI 参数错误", result)
                self.plugin.safety_auditor.audit_prompt.assert_not_awaited()
                self.plugin.usage_manager.check_rate_limit.assert_not_called()
                self.plugin.create_generation_task.assert_not_called()

    async def test_unsupported_adapter_rejects_even_empty_options(self):
        """Avoid silently dropping explicit NovelAI settings on another adapter."""
        self.plugin.generator.adapter.get_capabilities.return_value = (
            self.capabilities.TEXT_TO_IMAGE
        )
        for options in ({}, {"steps": 28}):
            with self.subTest(options=options):
                result = await self.tool.call(
                    {"event": self.event}, prompt="a white cat", novelai=options
                )
                self.assertIn("不支持 NovelAI 专属参数", result)
                self.plugin.safety_auditor.audit_prompt.assert_not_awaited()
                self.plugin.usage_manager.check_rate_limit.assert_not_called()
                self.plugin.create_generation_task.assert_not_called()

    async def test_enabled_character_prompts_are_audited_without_changing_base(self):
        """Audit every enabled character while preserving the submitted base prompt."""
        options = {
            "characters": [
                {"prompt": "red-haired mage", "position": "A1"},
                {"prompt": "blue-haired knight", "enabled": True},
                {"prompt": "disabled description", "enabled": False},
            ]
        }
        result = await self.tool.call(
            {"event": self.event}, prompt="two people in a forest", novelai=options
        )
        self.assertEqual(result, "accepted")
        audited_prompt = self.plugin.safety_auditor.audit_prompt.await_args.args[0]
        self.assertIn("two people in a forest", audited_prompt)
        self.assertIn("red-haired mage", audited_prompt)
        self.assertIn("blue-haired knight", audited_prompt)
        self.assertNotIn("disabled description", audited_prompt)
        self.assertEqual(
            self.plugin.create_generation_task.call_args.kwargs["prompt"],
            "two people in a forest",
        )

    async def test_character_only_prompt_is_allowed_and_audited(self):
        """Allow an empty base prompt when enabled characters supply the content."""
        result = await self.tool.call(
            {"event": self.event},
            novelai={"characters": [{"prompt": "red-haired mage"}]},
        )
        self.assertEqual(result, "accepted")
        self.assertEqual(
            self.plugin.create_generation_task.call_args.kwargs["prompt"], ""
        )
        self.assertIn(
            "red-haired mage",
            self.plugin.safety_auditor.audit_prompt.await_args.args[0],
        )

    async def test_blocked_character_prompt_does_not_create_or_reserve_task(self):
        """Stop a rejected character description before reserving generation usage."""
        self.plugin.safety_auditor.audit_prompt.return_value = (
            False,
            "blocked character",
        )
        result = await self.tool.call(
            {"event": self.event},
            prompt="two people",
            novelai={"characters": [{"prompt": "character to review"}]},
        )
        self.assertIn("提示词审核未通过", result)
        self.plugin.create_generation_task.assert_not_called()
        # Only the read-only pre-check ran; quota is reserved by task creation.
        self.assertEqual(self.plugin.usage_manager.check_rate_limit.call_count, 1)
        self.plugin.usage_manager.reserve_usage.assert_not_called()

    async def test_nested_assets_are_prepared_and_keep_their_roles(self):
        """Forward role-specific prepared images rather than generic references."""
        options = {"inpaint": {"mask": "https://images.example.invalid/mask.png"}}
        mask = self.shared.ImageData(data=b"mask", mime_type="image/png")
        prepared = {"inpaint": {"mask": mask}}
        with patch.object(
            self.module, "prepare_novelai_assets", AsyncMock(return_value=prepared)
        ) as prepare:
            result = await self.tool.call(
                {"event": self.event}, prompt="a white cat", novelai=options
            )
        self.assertEqual(result, "accepted")
        prepare.assert_awaited_once_with(
            options,
            image_processor=self.plugin.image_processor,
            unified_msg_origin="test-session",
        )
        submitted = self.plugin.create_generation_task.call_args.kwargs
        self.assertIs(submitted["novelai"], prepared)
        self.assertEqual(submitted["images_data"], [])
        self.assertEqual(
            options["inpaint"]["mask"], "https://images.example.invalid/mask.png"
        )

    async def test_asset_failure_or_cancellation_creates_no_task(self):
        """Stop before task creation, which is where quota gets reserved."""
        # No task record exists yet, so there is no reservation to release.
        self.plugin.task_manager.get_generation_task.return_value = None
        for failure in (ValueError("invalid mask"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                self.plugin.usage_manager.release_reserved_usage.reset_mock()
                with patch.object(
                    self.module,
                    "prepare_novelai_assets",
                    AsyncMock(side_effect=failure),
                ):
                    if isinstance(failure, asyncio.CancelledError):
                        with self.assertRaises(asyncio.CancelledError):
                            await self.tool.call(
                                {"event": self.event}, prompt="a white cat", novelai={}
                            )
                    else:
                        result = await self.tool.call(
                            {"event": self.event}, prompt="a white cat", novelai={}
                        )
                        self.assertIn("invalid mask", result)
                self.plugin.usage_manager.release_reserved_usage.assert_not_called()
                self.plugin.create_generation_task.assert_not_called()

    def test_schema_is_valid_and_hidden_for_other_adapters(self):
        """Publish the structured options only for a compatible image adapter."""
        jsonschema.Draft202012Validator.check_schema(self.tool.parameters)
        jsonschema.validate(
            {
                "prompt": "a mage",
                "novelai": {
                    "characters": [{"prompt": "red hair", "position": "A1"}],
                    "vibe_references": [{"encoding": "YWJj", "strength": 0.6}],
                },
            },
            self.tool.parameters,
        )
        self.module.adjust_tool_parameters(
            self.tool,
            self.capabilities.TEXT_TO_IMAGE | self.capabilities.NOVELAI_OPTIONS,
        )
        self.assertIn("novelai", self.tool.parameters["properties"])
        self.module.adjust_tool_parameters(self.tool, self.capabilities.TEXT_TO_IMAGE)
        self.assertNotIn("novelai", self.tool.parameters["properties"])
