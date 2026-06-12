from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from agent_template.core.runner import Runner
from agent_template.core.memory import Memory, Content
from agent_template.core.tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    model: str
    provider: str
    system_prompt: str
    name: str
    description: str
    input_schema: dict[str, Any]

    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)

    max_turns: int = 40
    compact_threshold: float = 0.6
    context_window: int = 200000


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        runner: Runner,
        memory: Memory,
        tools: ToolRegistry,
    ):
        self.config = config
        self.runner = runner
        self.memory = memory
        self.tools = tools

    def run(self, input: list[dict[str, Any]]) -> Content:
        self.memory.add_user_message(input)

        for turn in range(self.config.max_turns):
            response = self.runner.run(
                messages=self.memory.get_messages(),
                system_prompt=self.config.system_prompt,
                tools=self._get_tools(),
                model=self.config.model,
                **self.config.provider_kwargs,
            )

            self.memory.add_assistant_message(response.content)

            if response.tool_calls:
                tool_results = self.runner.execute_tools(
                    response.tool_calls,
                    self.tools,
                )
                self.memory.add_tool_results(tool_results)

                if self._should_compact():
                    self.memory.compact()
            else:
                if self._should_compact():
                    self.memory.compact()
                return response.content

        return [{"type": "text", "text": "Max turns reached"}]

    def _get_tools(self) -> list[dict[str, Any]]:
        if not self.config.allowed_tools:
            return self.tools.get_schemas()
        return [t for t in self.tools.get_schemas() if t["name"] in self.config.allowed_tools]

    def _should_compact(self) -> bool:
        return self.runner.get_last_input_tokens() > self.config.compact_threshold * self.config.context_window