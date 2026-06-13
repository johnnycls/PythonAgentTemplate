from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from agent_template.core.providers.base import LLMProvider


@dataclass
class AgentConfig:
    model: str
    provider: LLMProvider | None = None
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    max_turns: int = 40
    compact_threshold: float = 0.6
    context_window: int = 200000
