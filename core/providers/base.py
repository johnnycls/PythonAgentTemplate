from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_template.core.runner import LLMResponse


@dataclass
class LLMProvider(ABC):
    name: str
    _api_key: str
    _base_url: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> "LLMResponse":
        pass