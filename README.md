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
