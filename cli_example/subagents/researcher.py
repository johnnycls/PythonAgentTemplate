from agent_template.core.agent import AgentConfig


RESEARCHER = AgentConfig(
    name="researcher",
    description="Research complex topics thoroughly.",
    system_prompt="You are a research assistant. Find and analyze information to answer questions accurately.",
    model="llama3",
    provider="ollama",
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
    allowed_tools=["echo"],
)