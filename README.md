# agent_template

A minimal, reusable agent framework. No hardcoded prompts, no opinionated defaults — just the building blocks.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill API keys for your providers
```

## Quick Start

```python
from agent_template.core import Agent, AgentConfig, Runner, ToolRegistry
from agent_template.core.providers.registry import ProviderRegistry
from agent_template.core.providers.openai_compat import OpenAICompatProvider
from agent_template.cli_example.cli_memory import CLIMemory

# 1. Register providers
providers = ProviderRegistry()
providers.register(OpenAICompatProvider(name="ollama", api_key="ollama", base_url="http://localhost:11434/v1"))

# 2. Configure agent
config = AgentConfig(
    name="my_agent",
    description="A helpful assistant.",
    system_prompt="You are a helpful assistant.",
    provider="ollama",
    model="llama3",
    input_schema={"type": "object", "properties": {"content": {"type": "array"}}, "required": ["content"]},
)

# 3. Wire it up
runner = Runner(providers=providers, provider_name=config.provider)
memory = CLIMemory(runner=runner, model=config.model, provider_kwargs=config.provider_kwargs)
tools = ToolRegistry()
agent = Agent(config=config, runner=runner, memory=memory, tools=tools)

# 4. Run
response = agent.run([{"type": "text", "text": "Hello"}])
```

Or run the CLI example directly:

```bash
python -m agent_template.cli_example.main
```

## Architecture

```
agent_template/
├── core/                    # Framework — no hardcoded logic
│   ├── agent.py             # Agent + AgentConfig
│   ├── runner.py            # LLM call loop + tool execution
│   ├── memory.py            # Memory ABC + Message + Content type
│   ├── providers/           # LLM providers (Anthropic, OpenAI-compat, Google)
│   └── tools/               # Tool ABC, registry, subagent-as-tool
├── cli_example/             # Reference implementation
│   ├── main.py              # CLI entry point
│   ├── cli_memory.py        # CLIMemory (deque-based, with compaction)
│   ├── providers.py         # build_providers()
│   ├── subagents/           # Example subagent configs
│   └── tools/               # Example tools (Echo)
└── env.py                   # Loads .env via dotenv
```

## Key Concepts

**Unified content format** — All I/O uses `list[dict[str, Any]]` (content blocks):

```python
[{"type": "text", "text": "Hello"}]                          # text
[{"type": "image", "url": "https://..."}]                    # image from URL
[{"type": "image", "data": "base64...", "mime_type": "image/png"}]  # inline image
```

**Provider kwargs** — Each provider translates params internally:
- Anthropic/OpenAI: `max_tokens`, `temperature`
- Google: `max_output_tokens`, `temperature`

**Tools** — Return a schema dict directly, no wrapper class:

```python
from agent_template.core.tools.base import Tool

class MyTool(Tool):
    @property
    def schema(self):
        return {"name": "my_tool", "description": "...", "input_schema": {...}}

    def execute(self, **kwargs):
        return {"result": "done"}
```

**Subagents as tools** — Wrap an `Agent` as a tool, auto-registered:

```python
from agent_template.core.tools.subagent import register_subagents_as_tools

subagents = [agent1, agent2]
register_subagents_as_tools(subagents, tool_registry)
# Now "agent1" and "agent2" appear as callable tools
```

**Compaction** — Memory owns the compact logic. Triggered when `input_tokens > context_window * compact_threshold`.

## Supported Providers

| Provider | Class | Notes |
|----------|-------|-------|
| Anthropic | `AnthropicProvider` | Native API |
| OpenAI-compatible | `OpenAICompatProvider` | Works with OpenAI, Ollama, DeepSeek, Groq, OpenRouter, xAI |
| Google | `GoogleProvider` | Gemini / Vertex AI |

## Environment Variables

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
```
