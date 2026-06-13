from __future__ import annotations
from typing import Any, Generator
from agent_template.core.providers.base import LLMProvider
from agent_template.core.agent import LLMResponse, ToolCall, StreamChunk


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str):
        super().__init__(name="anthropic", _api_key=api_key, _base_url=base_url)

    def _get_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key, base_url=self._base_url)

    def _parse_content_blocks(self, blocks: list[Any]) -> tuple[list[dict[str, Any]], list[ToolCall]]:
        content: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        for block in blocks:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
        return content, tool_calls

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
    ) -> int:
        client = self._get_client()
        params = self._build_params(messages, system_prompt, tools, model)
        response = client.messages.count_tokens(**params)
        return response.input_tokens

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized = [self._normalize_message(m) for m in messages]
        params: dict[str, Any] = {
            "model": model,
            "messages": normalized,
        }
        if system_prompt:
            params["system"] = system_prompt
        if tools:
            params["tools"] = [self._convert_tool(t) for t in tools]
        if output_schema:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
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

        response = client.messages.create(**params)

        content, tool_calls = self._parse_content_blocks(response.content)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
        client = self._get_client()
        params = self._build_params(messages, system_prompt, tools, model, output_schema, **kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []
        tool_calls_by_id: dict[str, dict[str, Any]] = {}
        input_tokens = 0
        output_tokens = 0

        with client.messages.stream(**params) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls_by_id[block.id] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                        }
                        yield StreamChunk(tool_name=block.name)
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        content_text += delta.text
                        yield StreamChunk(delta=delta.text)
                    elif delta.type == "input_json_delta":
                        for tc_id, tc_data in tool_calls_by_id.items():
                            if tc_id in str(delta):
                                tc_data["arguments"] += delta.partial_json
                                break

            final_message = stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens

            _, tool_calls = self._parse_content_blocks(final_message.content)

        for tc in tool_calls:
            yield StreamChunk(tool_call=tc)

        return LLMResponse(
            content=[{"type": "text", "text": content_text}],
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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