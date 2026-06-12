from agent_template.env import env
from agent_template.core.agent import Agent, AgentConfig
from agent_template.core.runner import Runner
from agent_template.cli_example.cli_memory import CLIMemory
from agent_template.core.tools.registry import ToolRegistry
from agent_template.core.tools.subagent import register_subagents_as_tools
from agent_template.cli_example.providers import build_providers
from agent_template.cli_example.subagents import build_subagents
from agent_template.cli_example.tools import Echo


def main():
    providers = build_providers()
    subagent_agents = build_subagents(providers=providers, tool_registry=ToolRegistry())

    config = AgentConfig(
        name="main",
        description="A helpful assistant.",
        system_prompt="You are a helpful assistant.",
        provider="ollama",
        model="llama3",
        provider_kwargs={"max_tokens": 4096, "temperature": 0.7},
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "array",
                    "description": "Content blocks in standard format (e.g., [{type: 'text', text: '...'}])",
                },
            },
            "required": ["content"],
        },
    )
    runner = Runner(providers=providers, provider_name=config.provider)
    memory = CLIMemory(runner=runner, model=config.model, provider_kwargs=config.provider_kwargs)
    tools = ToolRegistry()
    tools.register(Echo())
    register_subagents_as_tools(subagent_agents, tools)

    agent = Agent(config=config, runner=runner, memory=memory, tools=tools)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        response = agent.run([{"type": "text", "text": user_input}])
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()