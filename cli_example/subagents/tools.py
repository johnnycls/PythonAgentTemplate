from __future__ import annotations
from agent_template.core.agent import Agent, AgentConfig
from agent_template.core.tools.subagent import agents_to_tools
from agent_template.cli_example.subagents.memory import SubagentMemory
from agent_template.cli_example.subagents.researcher import RESEARCHER, RESEARCHER_SYSTEM_PROMPT
from agent_template.cli_example.config import ollama_provider

memory_provider = ollama_provider
memory_model = "llama3"
memory_provider_kwargs: dict = {}


def _build_agent(config: AgentConfig) -> Agent:
    memory = SubagentMemory(provider=memory_provider, model=memory_model, provider_kwargs=memory_provider_kwargs)
    return Agent(config=config, memory=memory)


researcher_agent = _build_agent(RESEARCHER)
RESEARCHER_TOOL = agents_to_tools([researcher_agent], system_prompts=[RESEARCHER_SYSTEM_PROMPT])[0]
