from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from agent_template.core.providers.base import LLMProvider
from agent_template.core.memory import Memory, Content
from agent_template.core.tools.base import Tool


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    content: Content
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentConfig:
    model: str
    provider: LLMProvider | None = None
    system_prompt: str = ""
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    max_turns: int = 40
    compact_threshold: float = 0.6
    context_window: int = 200000


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        memory: Memory,
        tools: list[Tool] | None = None,
    ):
        self.config = config
        self.memory = memory
        self.tools = tools or []
        self._last_input_tokens = 0

    def run(self, input: list[dict[str, Any]]) -> Content:
        self.memory.add_user_message(input)

        for turn in range(self.config.max_turns):
            response = self.config.provider.complete(
                messages=self.memory.get_messages(),
                system_prompt=self.config.system_prompt,
                tools=[t.schema for t in self.tools],
                model=self.config.model,
                output_schema=self.config.output_schema,
                **self.config.provider_kwargs,
            )
            self._last_input_tokens = response.input_tokens

            self.memory.add_assistant_message(response.content)

            if response.tool_calls:
                tool_results = self._execute_tools(response.tool_calls)
                self.memory.add_tool_results(tool_results)

                if self._should_compact():
                    self.memory.compact()
            else:
                if self._should_compact():
                    self.memory.compact()
                return response.content

        return [{"type": "text", "text": "Max turns reached"}]

    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        if not tool_calls:
            return []

        tool_map = {t.schema["name"]: t for t in self.tools}
        results: list[ToolResult] = []
        safe = []
        unsafe = []
        for call in tool_calls:
            tool = tool_map.get(call.name)
            if tool is None:
                results.append(ToolResult(tool_call_id=call.id, content=f"Unknown tool: {call.name}", is_error=True))
            elif tool.concurrency_safe:
                safe.append((call, tool))
            else:
                unsafe.append((call, tool))

        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(self._exec_one, call, tool): call
                for call, tool in safe
            }
            for future in as_completed(futures):
                results.append(future.result())

        for call, tool in unsafe:
            results.append(self._exec_one(call, tool))

        return results

    def _exec_one(self, call: ToolCall, tool: Tool) -> ToolResult:
        try:
            result = tool.execute(**call.arguments)
            return ToolResult(tool_call_id=call.id, content=str(result))
        except Exception as e:
            return ToolResult(tool_call_id=call.id, content=str(e), is_error=True)

    def _should_compact(self) -> bool:
        return self._last_input_tokens > self.config.compact_threshold * self.config.context_window
