from __future__ import annotations
from agent_template.cli_example.subagents.researcher import RESEARCHER
from agent_template.core.agent import Agent, AgentConfig
from agent_template.core.runner import Runner
from agent_template.cli_example.cli_memory import CLIMemory
from agent_template.core.tools.registry import ToolRegistry
from agent_template.core.providers.registry import ProviderRegistry


def build_subagents(
    providers: ProviderRegistry,
    tool_registry: ToolRegistry,
) -> list[Agent]:
    configs = [RESEARCHER]
    agents = []
    for config in configs:
        runner = Runner(providers=providers, provider_name=config.provider)
        memory = CLIMemory(runner=runner, model=config.model, provider_kwargs=config.provider_kwargs)
        agent = Agent(config=config, runner=runner, memory=memory, tools=tool_registry)
        agents.append(agent)
    return agents