from agent_template.env import env
from agent_template.core.agent import Agent, AgentConfig
from agent_template.core.providers.openai_compat import OpenAICompatProvider
from agent_template.cli_example.cli_memory import CLIMemory
from agent_template.core.tools.subagent import agents_to_tools
from agent_template.cli_example.subagents import RESEARCHER
from agent_template.cli_example.subagents.build import build_subagents
from agent_template.cli_example.tools import Echo


SYSTEM_PROMPT = """You are a helpful assistant.
- "result" contains your full answer.
- "options" is a list of strings: answer choices if you are asking a question,
  or suggested follow-up questions the user might ask. Use an empty list if none."""

memory_provider = OpenAICompatProvider(name="ollama", api_key="ollama", base_url="http://localhost:11434/v1")
memory_model = "llama3"
memory_provider_kwargs: dict = {}


def main():
    subagent_agents = build_subagents(
        configs=[RESEARCHER],
        memory_provider=memory_provider,
        memory_model=memory_model,
        memory_provider_kwargs=memory_provider_kwargs,
    )

    config = AgentConfig(
        name="main",
        description="A helpful assistant.",
        system_prompt=SYSTEM_PROMPT,
        provider=memory_provider,
        model=memory_model,
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
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["result", "options"],
        },
    )
    memory = CLIMemory(provider=memory_provider, model=memory_model, provider_kwargs=config.provider_kwargs)

    tools = [Echo()] + agents_to_tools(subagent_agents)

    agent = Agent(config=config, memory=memory, tools=tools)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        response = agent.run([{"type": "text", "text": user_input}])
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
