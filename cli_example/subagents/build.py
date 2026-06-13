from __future__ import annotations
from agent_template.core.agent import Agent, AgentConfig
from agent_template.cli_example.cli_memory import CLIMemory
from agent_template.core.providers.base import LLMProvider


def build_subagents(
    configs: list[AgentConfig],
    memory_provider: LLMProvider,
    memory_model: str,
    memory_provider_kwargs: dict | None = None,
) -> list[Agent]:
    agents = []
    for config in configs:
        memory = CLIMemory(provider=memory_provider, model=memory_model, provider_kwargs=memory_provider_kwargs)
        agent = Agent(config=config, memory=memory)
        agents.append(agent)
    return agents
