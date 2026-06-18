from agent_template.core.agent import Agent, AgentConfig
from agent_template.core.tools.subagent import SubagentTool
from agent_template.cli_example.subagents.memory import SubagentMemory
from agent_template.cli_example.config import ollama_provider
from agent_template.cli_example.tools import Echo

RESEARCHER_SYSTEM_PROMPT = "You are a research assistant. Find and analyze information to answer questions accurately."

RESEARCHER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "array",
            "description": "Content blocks with the research question. Example: [{'type': 'text', 'text': 'What is quantum computing?'}]",
        },
    },
    "required": ["content"],
}

_config = AgentConfig(
    name="researcher",
    description="Research complex topics thoroughly.",
    model="nemotron-3-nano:4b",
    provider=ollama_provider,
    input_schema=RESEARCHER_INPUT_SCHEMA,
)

_agent = Agent(
    config=_config,
    memory=SubagentMemory(provider=ollama_provider, model="nemotron-3-nano:4b"),
    tools=[Echo()],
)

RESEARCHER_TOOL = SubagentTool(agent=_agent, system_prompt=RESEARCHER_SYSTEM_PROMPT)
