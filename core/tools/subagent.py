from __future__ import annotations
from typing import Any
from agent_template.core.agent import Agent
from agent_template.core.tools.base import Tool
from agent_template.core.tools.registry import ToolRegistry


def register_subagents_as_tools(
    agents: list[Agent],
    tool_registry: ToolRegistry,
) -> None:
    for agent in agents:
        tool = _AgentTool(agent)
        tool_registry.register(tool)


class _AgentTool(Tool):
    concurrency_safe = False

    def __init__(self, agent: Agent):
        self._agent = agent

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self._agent.config.name,
            "description": self._agent.config.description,
            "input_schema": self._agent.config.input_schema,
        }

    def execute(self, **kwargs: Any) -> Any:
        if "content" not in kwargs:
            schema = self._agent.config.input_schema
            raise ValueError(
                f"Tool '{self._agent.config.name}' requires 'content' parameter. "
                f"Expected format: {schema}"
            )
        return self._agent.run(kwargs["content"])