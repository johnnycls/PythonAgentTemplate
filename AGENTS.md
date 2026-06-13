# Agent Template

## Architecture

- `core/` — Generic framework, no hardcoded logic
  - `core/agent/` — Agent, AgentConfig, types (ToolCall, ToolResult, LLMResponse)
  - `core/memory/` — Memory ABC, Content type, Message dataclass
  - `core/providers/` — LLM providers (openai_compat, anthropic, google)
  - `core/tools/` — Tool ABC, subagent-as-tool
- `cli_example/` — Reference implementation
  - `cli_example/master/` — MasterAgent with its own config/prompts/schemas/memory
  - `cli_example/subagents/` — Subagent definitions and tools
  - `cli_example/tools/` — Example tools (Echo)

## Key Design Decisions

- **No registries** — Tools and providers passed as lists, not registered
- **No Runner** — Agent calls provider directly via `self.config.provider.complete()`
- **System prompt is app-level** — Not in AgentConfig. Passed to `agent.run(system_prompt=...)` or built by subclass (see MasterAgent)
- **Memory builds summary** — `memory.get_summary()` returns compacted summary; caller adds to system prompt
- **Structured output** — Each provider handles `output_schema` natively (OpenAI `response_format`, Anthropic `output_config.format`, Google `response_mime_type`)

## Running

```bash
python -m agent_template.cli_example.main
```

## Conventions

- Config files: `config.py` for values, `prompts.py` for prompts, `schemas.py` for schemas
- Memory classes: `MasterMemory` (master), `SubagentMemory` (subagents)
- Subagent tools exported from `cli_example/subagents/tools.py`
- Core files use `from __future__ import annotations`
