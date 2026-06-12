from __future__ import annotations
from typing import Any
from agent_template.core.providers.base import LLMProvider
from agent_template.core.runner import LLMResponse, ToolCall


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str):
        super().__init__(name="anthropic", _api_key=api_key, _base_url=base_url)

    def _get_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key, base_url=self._base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        normalized = [self._normalize_message(m) for m in messages]
        params: dict[str, Any] = {
            "model": model,
            "messages": normalized,
        }
        if system_prompt:
            params["system"] = system_prompt
        if tools:
            params["tools"] = [self._convert_tool(t) for t in tools]

        params.update(kwargs)

        response = client.messages.create(**params)

        content: list[dict[str, Any]] = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _normalize_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool":
            return {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": content}],
            }

        if isinstance(content, list):
            normalized_blocks = []
            for block in content:
                normalized_blocks.append(self._normalize_block(block))
            return {"role": role, "content": normalized_blocks}

        return {"role": role, "content": content}

    def _normalize_block(self, block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type", "")

        if block_type == "image":
            if "url" in block:
                return {
                    "type": "image",
                    "source": {"type": "url", "url": block["url"]},
                }
            return block

        return block

    def _convert_tool(self, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "input_schema": schema.get("input_schema", {"type": "object", "properties": {}}),
        }