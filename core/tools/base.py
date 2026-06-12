from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    concurrency_safe: bool = True

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        pass