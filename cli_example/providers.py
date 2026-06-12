from __future__ import annotations
from agent_template.core.providers.registry import ProviderRegistry
# from agent_template.core.providers.anthropic import AnthropicProvider
from agent_template.core.providers.openai_compat import OpenAICompatProvider
# from agent_template.core.providers.google import GoogleProvider
from agent_template.env import env


def build_providers() -> ProviderRegistry:
    registry = ProviderRegistry()

    # registry.register(AnthropicProvider(api_key=env.get("ANTHROPIC_API_KEY", ""), base_url=env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")))
    # registry.register(OpenAICompatProvider(name="openai", api_key=env.get("OPENAI_API_KEY", ""), base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    # registry.register(OpenAICompatProvider(name="deepseek", api_key=env.get("DEEPSEEK_API_KEY", ""), base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")))
    # registry.register(OpenAICompatProvider(name="groq", api_key=env.get("GROQ_API_KEY", ""), base_url=env.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")))
    # registry.register(OpenAICompatProvider(name="openrouter", api_key=env.get("OPENROUTER_API_KEY", ""), base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")))
    # registry.register(OpenAICompatProvider(name="xai", api_key=env.get("XAI_API_KEY", ""), base_url=env.get("XAI_BASE_URL", "https://api.x.ai/v1")))
    # registry.register(GoogleProvider(api_key=env.get("GEMINI_API_KEY", ""), base_url=env.get("GEMINI_BASE_URL", "")))
    registry.register(OpenAICompatProvider(name="ollama", api_key="ollama", base_url="http://localhost:11434/v1"))

    return registry