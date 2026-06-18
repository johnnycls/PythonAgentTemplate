from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from agent_template.core.memory.types import Content


class Memory(ABC):
    @abstractmethod
    def add_user_message(self, content: Content) -> None:
        pass

    @abstractmethod
    def add_assistant_message(self, content: Content, tool_calls: list[dict[str, Any]] | None = None) -> None:
        pass

    @abstractmethod
    def add_tool_results(self, results: list[Any]) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_summary(self) -> str | None:
        pass

    @abstractmethod
    def compact(self) -> None:
        pass

    def clear(self) -> None:
        pass
