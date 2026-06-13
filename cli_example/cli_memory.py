from __future__ import annotations
from typing import Any
from collections import deque

from agent_template.core.memory import Memory, Message, Content
from agent_template.core.providers.base import LLMProvider


class CLIMemory(Memory):
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        provider_kwargs: dict[str, Any] | None = None,
        max_working_messages: int = 50,
    ):
        self._provider = provider
        self._model = model
        self._provider_kwargs = provider_kwargs or {}
        self._messages: deque[Message] = deque(maxlen=max_working_messages)
        self._compacted_summary: Content | None = None

    def add_user_message(self, content: Content) -> None:
        self._messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: Content) -> None:
        self._messages.append(Message(role="assistant", content=content))

    def add_tool_results(self, results: list[Any]) -> None:
        for result in results:
            self._messages.append(Message(
                role="tool",
                content=result.content,
                tool_call_id=result.tool_call_id,
            ))

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []
        if self._compacted_summary:
            messages.append({"role": "system", "content": f"Conversation summary: {self._compacted_summary}"})

        for msg in self._messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.name:
                m["name"] = msg.name
            messages.append(m)
        return messages

    def compact(self) -> None:
        compact_prompt = "Summarize the conversation history concisely, preserving key facts and decisions."
        summary_response = self._provider.complete(
            messages=[{"role": "user", "content": compact_prompt}],
            system_prompt="",
            tools=[],
            model=self._model,
            **self._provider_kwargs,
        )
        last_messages = list(self._messages)[-10:]
        self._messages.clear()
        self._compacted_summary = summary_response.content
        for msg in last_messages:
            self._messages.append(msg)

    def clear(self) -> None:
        self._messages.clear()
        self._compacted_summary = None
