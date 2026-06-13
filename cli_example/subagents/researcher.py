from agent_template.core.agent import AgentConfig
from agent_template.cli_example.config import ollama_provider

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

RESEARCHER = AgentConfig(
    name="researcher",
    description="Research complex topics thoroughly.",
    model="llama3",
    provider=ollama_provider,
    input_schema=RESEARCHER_INPUT_SCHEMA,
)
