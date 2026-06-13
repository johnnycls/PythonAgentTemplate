from agent_template.core.agent import AgentConfig
from agent_template.core.providers.openai_compat import OpenAICompatProvider


RESEARCHER = AgentConfig(
    name="researcher",
    description="Research complex topics thoroughly.",
    system_prompt="You are a research assistant. Find and analyze information to answer questions accurately.",
    model="llama3",
    provider=OpenAICompatProvider(name="ollama", api_key="ollama", base_url="http://localhost:11434/v1"),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "array",
                "description": "Content blocks with the research question. Example: [{'type': 'text', 'text': 'What is quantum computing?'}]",
            },
        },
        "required": ["content"],
    },
)
