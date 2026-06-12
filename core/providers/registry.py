from __future__ import annotations
from agent_template.core.providers.base import LLMProvider


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)

    def has(self, name: str) -> bool:
        return name in self._providers

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())