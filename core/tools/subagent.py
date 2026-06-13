from __future__ import annotations
from typing import Any
from agent_template.core.agent import Agent
from agent_template.core.tools.base import Tool


def agents_to_tools(agents: list[Agent]) -> list[Tool]:
    return [_AgentTool(agent) for agent in agents]


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
