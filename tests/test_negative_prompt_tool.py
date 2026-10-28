from __future__ import annotations

import importlib
import logging
import sys
import types
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class _FunctionTool:
    """Minimal generic tool base for isolated plugin tests."""

    def __class_getitem__(cls, _context):
        return cls


class NegativePromptToolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Load real tool code with mocked AstrBot boundaries and no network."""
        root = Path(__file__).resolve().parents[1]
        namespace = "_negative_prompt_tool_test_plugin"
        modules = {}
        for suffix in (
            "",
            "adapter",
            "core",
            "core.adapters",
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
        modules["astrbot.api"] = api
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
            self.openai_adapters = (
                (
                    importlib.import_module(
                        f"{namespace}.adapter.openai_adapter"
                    ).OpenAIAdapter,
                    self.shared.AdapterType.OPENAI,
                    "gpt-image-1",
                ),
                (
                    importlib.import_module(
                        f"{namespace}.adapter.openai_chat_adapter"
                    ).OpenAIChatAdapter,
                    self.shared.AdapterType.OPENAI_CHAT,
                    "image-chat-model",
                ),
            )

        self.capabilities = self.module.ImageCapability
        self.plugin = MagicMock()
        self.plugin.generator.adapter.get_capabilities.return_value = (
            self.capabilities.TEXT_TO_IMAGE | self.capabilities.NEGATIVE_PROMPT
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
        collector.start()
        self.addCleanup(collector.stop)

    async def test_omitted_null_override_and_empty_reach_task_creation(self):
        """Preserve all three override states through the actual tool call."""
        for arguments, expected in (
            ({}, None),
            ({"negative_prompt": None}, None),
            ({"negative_prompt": "watermark, blurry"}, "watermark, blurry"),
            ({"negative_prompt": ""}, ""),
        ):
            with self.subTest(arguments=arguments):
                self.plugin.create_generation_task.reset_mock()
                result = await self.tool.call(
                    {"event": self.event}, prompt="a white cat", **arguments
                )
                self.assertEqual(result, "accepted")
                self.plugin.create_generation_task.assert_called_once()
                self.assertEqual(
                    self.plugin.create_generation_task.call_args.kwargs[
                        "negative_prompt"
                    ],
                    expected,
                )

    async def test_invalid_negative_prompt_does_not_reserve_quota(self):
        """Reject structured or scalar non-string input before side effects."""
        for invalid in ([], {}, 1, False):
            with self.subTest(invalid=invalid):
                result = await self.tool.call(
                    {"event": self.event},
                    prompt="a white cat",
                    negative_prompt=invalid,
                )
                self.assertIn("必须为字符串", result)
                self.plugin.usage_manager.check_rate_limit.assert_not_called()
                self.plugin.create_generation_task.assert_not_called()

    async def test_unsupported_adapter_rejects_override_and_clear(self):
        """Avoid accepting an override that the selected adapter cannot use."""
        self.plugin.generator.adapter.get_capabilities.return_value = (
            self.capabilities.TEXT_TO_IMAGE
        )
        for negative_prompt in ("watermark", ""):
            with self.subTest(negative_prompt=negative_prompt):
                result = await self.tool.call(
                    {"event": self.event},
                    prompt="a white cat",
                    negative_prompt=negative_prompt,
                )
                self.assertIn("不支持反向提示词输入", result)
                self.plugin.usage_manager.check_rate_limit.assert_not_called()
                self.plugin.create_generation_task.assert_not_called()

    async def test_prompt_prose_is_not_parsed_as_negative_parameter(self):
        """Leave natural-language avoidance instructions in the positive prompt."""
        prompt = "a white cat. negative prompt: watermark"
        result = await self.tool.call({"event": self.event}, prompt=prompt)
        self.assertEqual(result, "accepted")
        arguments = self.plugin.create_generation_task.call_args.kwargs
        self.assertEqual(arguments["prompt"], prompt)
        self.assertIsNone(arguments["negative_prompt"])

    async def test_openai_tools_submit_positive_and_negative_as_one_request(self):
        """Keep both inputs for adapter-specific handling in the same task."""
        positive = "a white cat sitting near a window"
        for (adapter_class, adapter_type, model), mode in product(
            self.openai_adapters, ("prompt", "parameter")
        ):
            adapter = adapter_class(
                self.shared.AdapterConfig(
                    type=adapter_type,
                    model=model,
                    capability_options={"text_to_image": True},
                    extra={
                        "negative_prompt_mode": mode,
                        "negative_prompt": "provider default negative",
                    },
                )
            )
            self.plugin.generator.adapter = adapter
            tool = self.module.ImageGenerationTool(plugin=self.plugin)
            self.module.adjust_tool_parameters(tool, adapter.get_capabilities())
            self.assertIn("prompt", tool.parameters["properties"])
            self.assertIn("negative_prompt", tool.parameters["properties"])
            for arguments, expected in (
                ({"negative_prompt": "watermark, blurry"}, "watermark, blurry"),
                ({}, None),
                ({"negative_prompt": ""}, ""),
            ):
                with self.subTest(
                    adapter=adapter_type.value, mode=mode, arguments=arguments
                ):
                    self.plugin.create_generation_task.reset_mock()
                    result = await tool.call(
                        {"event": self.event}, prompt=positive, **arguments
                    )
                    self.assertEqual(result, "accepted")
                    self.plugin.create_generation_task.assert_called_once()
                    submitted = self.plugin.create_generation_task.call_args.kwargs
                    self.assertEqual(submitted["prompt"], positive)
                    self.assertEqual(submitted["negative_prompt"], expected)

    def test_negative_parameter_is_only_advertised_for_supported_adapters(self):
        """Expose the parameter only while the adapter advertises support."""
        self.module.adjust_tool_parameters(
            self.tool,
            self.capabilities.TEXT_TO_IMAGE | self.capabilities.NEGATIVE_PROMPT,
        )
        self.assertIn("negative_prompt", self.tool.parameters["properties"])
        self.assertNotIn("negative_prompt", self.tool.parameters["required"])
        self.module.adjust_tool_parameters(self.tool, self.capabilities.TEXT_TO_IMAGE)
        self.assertNotIn("negative_prompt", self.tool.parameters["properties"])
