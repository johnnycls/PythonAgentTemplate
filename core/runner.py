from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from agent_template.core.providers.registry import ProviderRegistry
from agent_template.core.memory import Content

if TYPE_CHECKING:
    from agent_template.core.tools.registry import ToolRegistry


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

    def get_text(self) -> str:
        parts = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)


class Runner:
    def __init__(
        self,
        providers: ProviderRegistry | None,
        provider_name: str,
    ):
        self.providers = providers or ProviderRegistry()
        self.provider_name = provider_name
        self._last_input_tokens = 0

    @property
    def provider(self):
        p = self.providers.get(self.provider_name)
        if not p:
            raise ValueError(f"Provider '{self.provider_name}' not registered")
        return p

    def run(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        response = self.provider.complete(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
            **kwargs,
        )
        self._last_input_tokens = response.input_tokens
        return response

    def execute_tools(
        self,
        tool_calls: list[ToolCall],
        tools: "ToolRegistry",
    ) -> list[ToolResult]:
        if not tool_calls:
            return []

        results: list[ToolResult] = []
        safe = []
        unsafe = []
        for call in tool_calls:
            tool = tools.get(call.name)
            if tool is None:
                results.append(ToolResult(tool_call_id=call.id, content=f"Unknown tool: {call.name}", is_error=True))
            elif tool.concurrency_safe:
                safe.append((call, tool))
            else:
                unsafe.append(call)

        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(self._exec_one, call, tool): call
                for call, tool in safe
            }
            for future in as_completed(futures):
                results.append(future.result())

        for call in unsafe:
            tool = tools.get(call.name)
            results.append(self._exec_one(call, tool))

        return results

    def _exec_one(self, call: ToolCall, tool: Any) -> ToolResult:
        try:
            result = tool.execute(**call.arguments)
            return ToolResult(tool_call_id=call.id, content=str(result))
        except Exception as e:
            return ToolResult(tool_call_id=call.id, content=str(e), is_error=True)

    def get_last_input_tokens(self) -> int:
        return self._last_input_tokens