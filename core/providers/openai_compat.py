from __future__ import annotations
from typing import Any, Generator
from agent_template.core.providers.base import LLMProvider
from agent_template.core.agent import LLMResponse, ToolCall, StreamChunk


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name=name, _api_key=api_key, _base_url=base_url)

    def _get_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _parse_tool_calls(self, tool_calls_data: list[dict[str, Any]]) -> list[ToolCall]:
        import json
        return [
            ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"],
            )
            for tc in tool_calls_data
        ]

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
    ) -> int:
        client = self._get_client()
        input_items = self._format_messages_for_responses(messages, system_prompt)
        response = client.responses.input_tokens.count(
            model=model,
            input=input_items,
        )
        return response.input_tokens

    def _format_messages_for_responses(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        input_items = []
        if system_prompt:
            input_items.append({"role": "developer", "content": system_prompt})
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                input_items.append({"role": role, "content": content})
        return input_items

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
        return params

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
        params = self._build_params(messages, system_prompt, tools, model, output_schema, **kwargs)

        response = client.chat.completions.create(**params)
        choice = response.choices[0]

        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = self._parse_tool_calls([
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ])

        return LLMResponse(
            content=[{"type": "text", "text": content}],
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, LLMResponse]:
        import json
        client = self._get_client()
        params = self._build_params(messages, system_prompt, tools, model, output_schema, **kwargs)

        stream = client.chat.completions.create(**params, stream=True)

        content_text = ""
        tool_calls_map: dict[int, dict[str, Any]] = {}
        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                content_text += delta.content
                yield StreamChunk(delta=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_calls_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_map[idx]["name"] = tc_delta.function.name
                            yield StreamChunk(tool_name=tc_delta.function.name)
                        if tc_delta.function.arguments:
                            tool_calls_map[idx]["arguments"] += tc_delta.function.arguments

            if choice.finish_reason == "tool_calls":
                for idx in sorted(tool_calls_map.keys()):
                    tc = tool_calls_map[idx]
                    yield StreamChunk(tool_call=self._parse_tool_calls([tc])[0])

        tool_calls = self._parse_tool_calls([
            tool_calls_map[idx] for idx in sorted(tool_calls_map.keys())
        ])

        return LLMResponse(
            content=[{"type": "text", "text": content_text}],
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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