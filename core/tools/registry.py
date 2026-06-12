from __future__ import annotations
from typing import Any
from agent_template.core.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.schema["name"]] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())