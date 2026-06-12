from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# Standard content block formats
# Text: {"type": "text", "text": "..."}
# Image: {"type": "image", "data": "base64...", "mime_type": "image/png"}
#        or {"type": "image", "url": "https://..."}
# Audio: {"type": "audio", "data": "base64...", "mime_type": "audio/wav"}
# File:  {"type": "file", "data": "base64...", "mime_type": "application/pdf", "name": "doc.pdf"}
Content = list[dict[str, Any]]


@dataclass
class Message:
    role: str
    content: Content
    tool_call_id: str | None = None
    name: str | None = None


class Memory(ABC):
    @abstractmethod
    def add_user_message(self, content: Content) -> None:
        pass

    @abstractmethod
    def add_assistant_message(self, content: Content) -> None:
        pass

    @abstractmethod
    def add_tool_results(self, results: list[Any]) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def compact(self) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass