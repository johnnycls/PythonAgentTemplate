from __future__ import annotations
from typing import Any
from agent_template.core.agent import Agent
from agent_template.core.tools.base import Tool


class SubagentTool(Tool):
    concurrency_safe = False

    def __init__(self, agent: Agent, system_prompt: str = ""):
        self._agent = agent
        self._system_prompt = system_prompt

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self._agent.config.name,
            "description": self._agent.config.description,
            "input_schema": self._agent.config.input_schema,
        }

    def execute(self, **kwargs: Any) -> Any:
        return self._agent.run(kwargs["content"], system_prompt=self._system_prompt)
