from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generator

from agent_template.core.agent.config import AgentConfig
from agent_template.core.agent.types import ToolCall, ToolResult, StreamChunk
from agent_template.core.memory import Memory, Content
from agent_template.core.tools.base import Tool


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

    def run(self, input: list[dict[str, Any]], system_prompt: str = "") -> Content:
        self.memory.add_user_message(input)

        for turn in range(self.config.max_turns):
            self._compact_if_needed(system_prompt)

            response = self.config.provider.complete(
                messages=self.memory.get_messages(),
                system_prompt=system_prompt,
                tools=[t.schema for t in self.tools],
                model=self.config.model,
                output_schema=self.config.output_schema,
                **self.config.provider_kwargs,
            )

            self.memory.add_assistant_message(response.content)

            if response.tool_calls:
                tool_results = self._execute_tools(response.tool_calls)
                self.memory.add_tool_results(tool_results)
            else:
                return response.content

        return [{"type": "text", "text": "Max turns reached"}]

    def run_stream(self, input: list[dict[str, Any]], system_prompt: str = "") -> Generator[StreamChunk, None, Content]:
        self.memory.add_user_message(input)

        for turn in range(self.config.max_turns):
            self._compact_if_needed(system_prompt)

            tool_calls: list[ToolCall] = []
            content: Content = []

            for chunk in self.config.provider.complete_stream(
                messages=self.memory.get_messages(),
                system_prompt=system_prompt,
                tools=[t.schema for t in self.tools],
                model=self.config.model,
                output_schema=self.config.output_schema,
                **self.config.provider_kwargs,
            ):
                if chunk.delta:
                    content.append({"type": "text", "text": chunk.delta})
                    yield chunk
                elif chunk.tool_name:
                    yield chunk
                elif chunk.tool_call:
                    tool_calls.append(chunk.tool_call)

            if content:
                self.memory.add_assistant_message(content)

            if tool_calls:
                tool_results = self._execute_tools(tool_calls)
                self.memory.add_tool_results(tool_results)
            else:
                return content

        return [{"type": "text", "text": "Max turns reached"}]

    def _compact_if_needed(self, system_prompt: str) -> None:
        token_count = self.config.provider.count_tokens(
            messages=self.memory.get_messages(),
            system_prompt=system_prompt,
            tools=[t.schema for t in self.tools],
            model=self.config.model,
        )
        if token_count > self.config.compact_threshold * self.config.context_window:
            self.memory.compact()

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
