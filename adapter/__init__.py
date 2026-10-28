"""Adapter module for the image generation plugin."""

from .agnes_ai_adapter import AgnesAIAdapter
from .codex_responses_adapter import CodexResponsesAdapter
from .comfyui_adapter import ComfyUIAdapter
from .custom_http_adapter import CustomHTTPAdapter
from .gemini_adapter import GeminiAdapter
from .gitee_ai_adapter import GiteeAIAdapter
from .grok_adapter import GrokAdapter
from .modelscope_adapter import ModelScopeAdapter
from .novelai_adapter import NovelAIAdapter
from .openai_adapter import OpenAIAdapter
from .openai_chat_adapter import OpenAIChatAdapter
from .siliconflow_adapter import SiliconFlowAdapter
from .volcengine_ark_adapter import VolcengineArkAdapter

__all__ = [
    "AgnesAIAdapter",
    "CodexResponsesAdapter",
    "ComfyUIAdapter",
    "CustomHTTPAdapter",
    "GeminiAdapter",
    "GiteeAIAdapter",
    "GrokAdapter",
    "ModelScopeAdapter",
    "NovelAIAdapter",
    "OpenAIAdapter",
    "OpenAIChatAdapter",
    "SiliconFlowAdapter",
    "VolcengineArkAdapter",
]
