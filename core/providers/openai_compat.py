from __future__ import annotations
from typing import Any
from agent_template.core.providers.base import LLMProvider
from agent_template.core.agent import LLMResponse, ToolCall


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name=name, _api_key=api_key, _base_url=base_url)

    def _get_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        formatted = self._format_messages(messages, system_prompt)

        params: dict[str, Any] = {
            "model": model,
            "messages": formatted,
        }
        if tools:
            params["tools"] = [self._convert_tool(t) for t in tools]
        if output_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": output_schema,
                },
            }

        params.update(kwargs)

        response = client.chat.completions.create(**params)
        choice = response.choices[0]

        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                import json
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            content=[{"type": "text", "text": content}],
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def _format_messages(self, messages: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content,
                })
            elif role in ("user", "assistant"):
                formatted.append({"role": role, "content": self._normalize_content(content)})
        return formatted

    def _normalize_content(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for block in content:
            normalized.append(self._normalize_block(block))
        return normalized

    def _normalize_block(self, block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type", "")

        if block_type == "image":
            if "url" in block:
                return {"type": "image_url", "image_url": {"url": block["url"]}}
            return block

        if block_type == "audio":
            if "data" in block:
                return {
                    "type": "input_audio",
                    "input_audio": {"data": block["data"], "format": block.get("format", "wav")},
                }
            return block

        return block

    def _convert_tool(self, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("input_schema", {"type": "object", "properties": {}}),
            },
        }