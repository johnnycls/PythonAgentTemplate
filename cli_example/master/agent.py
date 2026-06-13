from __future__ import annotations
from typing import Any, Generator

from agent_template.core.agent import Agent, AgentConfig, StreamChunk
from agent_template.core.memory import Content
from agent_template.cli_example.config import ollama_provider
from agent_template.cli_example.master.memory import MasterMemory
from agent_template.cli_example.master.config import MODEL_NAME, MAX_TOKENS, TEMPERATURE
from agent_template.cli_example.master.prompts import SYSTEM_PROMPT
from agent_template.cli_example.master.schemas import INPUT_SCHEMA, OUTPUT_SCHEMA
from agent_template.cli_example.subagents.tools import RESEARCHER_TOOL
from agent_template.cli_example.tools import Echo


class MasterAgent(Agent):
    def __init__(self) -> None:
        config = AgentConfig(
            name="master",
            description="A helpful assistant.",
            provider=ollama_provider,
            model=MODEL_NAME,
            provider_kwargs={"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
        )
        memory = MasterMemory(provider=ollama_provider, model=MODEL_NAME, provider_kwargs=config.provider_kwargs)
        tools = [Echo(), RESEARCHER_TOOL]
        super().__init__(config=config, memory=memory, tools=tools)

    def run(self, input: list[dict[str, Any]], system_prompt: str = "") -> Content:
        system_prompt = self._build_system_prompt()
        return super().run(input, system_prompt=system_prompt)

    def run_stream(self, input: list[dict[str, Any]], system_prompt: str = "") -> Generator[StreamChunk, None, Content]:
        system_prompt = self._build_system_prompt()
        yield from super().run_stream(input, system_prompt=system_prompt)

    def _build_system_prompt(self) -> str:
        summary = self.memory.get_summary()
        if summary:
            return f"{SYSTEM_PROMPT}\n\nConversation summary:\n{summary}"
        return SYSTEM_PROMPT
