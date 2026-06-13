from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from agent_template.core.agent import LLMResponse, StreamChunk


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
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LLMResponse":
        pass

    @abstractmethod
    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Generator["StreamChunk", None, "LLMResponse"]:
        pass

    @abstractmethod
    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
    ) -> int:
        pass
