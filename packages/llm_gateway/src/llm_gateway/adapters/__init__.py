"""adapters — translators to the LLM provider APIs (registry, instance cache, entry points).

`OpenAICompatibleAdapter` covers any `/chat/completions` dialect; OpenAI, Groq, Cerebras and
Mistral are settings of it. Google has its own translator. A third-party package brings its own
through the `llm_gateway.adapters` entry point.
"""

from llm_gateway.adapters.base import (
    ADAPTERS_ENTRY_POINT,
    BaseAdapter,
    ProviderClientError,
    ProviderError,
    ProviderParseError,
    ProviderServerError,
    close_all_adapters,
    get_adapter,
    register_adapter,
)
from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter

__all__ = [
    "ADAPTERS_ENTRY_POINT",
    "BaseAdapter",
    "OpenAICompatibleAdapter",
    "ProviderClientError",
    "ProviderError",
    "ProviderParseError",
    "ProviderServerError",
    "close_all_adapters",
    "get_adapter",
    "register_adapter",
]
