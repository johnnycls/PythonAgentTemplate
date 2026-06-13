# agent_template

A minimal, reusable agent framework. No hardcoded prompts, no opinionated defaults — just the building blocks.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill API keys for your providers
```

## Quick Start

```bash
python -m agent_template.cli_example.main
```

## Architecture

```
agent_template/
├── core/                        # Generic framework — no hardcoded logic
│   ├── agent/                   # Agent, AgentConfig, types
│   ├── memory/                  # Memory ABC, Content, Message
│   ├── providers/               # LLM providers (openai_compat, anthropic, google)
│   └── tools/                   # Tool ABC, subagent-as-tool
├── cli_example/                 # Reference implementation
│   ├── main.py                  # CLI entry point
│   ├── config.py                # Global config (provider, base URL, API key)
│   ├── master/                  # MasterAgent + config/prompts/schemas/memory
│   ├── subagents/               # Subagent definitions and tools
│   └── tools/                   # Example tools (Echo)
└── env.py                       # Loads .env via dotenv
```

## Key Design Decisions

- **No registries** — Tools and providers passed as lists, not registered
- **No Runner** — Agent calls provider directly
- **System prompt is app-level** — Not in AgentConfig. Passed to `agent.run(system_prompt=...)` or built by subclass
- **Memory builds summary** — `memory.get_summary()` returns compacted summary; caller adds to system prompt
- **Structured output** — Each provider handles `output_schema` natively

## Adding a Provider

Implement `LLMProvider` ABC in `core/providers/base.py`. Handle `output_schema` natively:
- OpenAI: `response_format` with `json_schema`
- Anthropic: `output_config.format`
- Google: `response_mime_type` + `response_schema`

## Adding a Tool

```python
from agent_template.core.tools.base import Tool

class MyTool(Tool):
    @property
    def schema(self):
        return {"name": "my_tool", "description": "...", "input_schema": {...}}

    def execute(self, **kwargs):
        return {"result": "done"}
```

## Environment Variables

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=
GEMINI_API_KEY=
```
