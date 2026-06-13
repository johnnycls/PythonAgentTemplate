from __future__ import annotations
import base64
import urllib.request
from typing import Any, Generator
from agent_template.core.providers.base import LLMProvider
from agent_template.core.agent import LLMResponse, ToolCall, StreamChunk


class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self, api_key: str, base_url: str):
        super().__init__(name="google", _api_key=api_key, _base_url=base_url)

    def _get_client(self):
        from google import genai
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["vertexai"] = True
            kwargs["http_options"] = {"base_url": self._base_url}
        return genai.Client(**kwargs)

    def _parse_tool_calls(self, function_calls: list[Any]) -> list[ToolCall]:
        return [
            ToolCall(
                id=fc.id or "",
                name=fc.name,
                arguments=dict(fc.args),
            )
            for fc in function_calls
        ]

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        model: str,
    ) -> int:
        client = self._get_client()
        contents = self._format_messages(messages)
        config = self._build_config(system_prompt, tools)
        response = client.models.count_tokens(model=model, contents=contents, config=config)
        return response.total_tokens

    def _build_config(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if system_prompt:
            config["system_instruction"] = system_prompt
        if tools:
            config["tools"] = [self._convert_tool(t) for t in tools]
        if output_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = output_schema
        config.update(kwargs)
        return config

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
        contents = self._format_messages(messages)
        config = self._build_config(system_prompt, tools, output_schema, **kwargs)

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        content: list[dict[str, Any]] = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content.append({"type": "text", "text": part.text})
                elif part.inline_data:
                    content.append({
                        "type": "image" if part.inline_data.mime_type.startswith("image/") else "data",
                        "data": part.inline_data.data,
                        "mime_type": part.inline_data.mime_type,
                    })

        tool_calls = []
        if response.function_calls:
            tool_calls = self._parse_tool_calls(response.function_calls)

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            content=content if content else [{"type": "text", "text": response.text or ""}],
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        contents = self._format_messages(messages)
        config = self._build_config(system_prompt, tools, output_schema, **kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        content_text += part.text
                        yield StreamChunk(delta=part.text)

            if chunk.function_calls:
                parsed = self._parse_tool_calls(chunk.function_calls)
                for tc in parsed:
                    yield StreamChunk(tool_name=tc.name)
                tool_calls.extend(parsed)

            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count or 0
                output_tokens = chunk.usage_metadata.candidates_token_count or 0

        for tc in tool_calls:
            yield StreamChunk(tool_call=tc)

        return LLMResponse(
            content=[{"type": "text", "text": content_text}],
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _format_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        from google.genai import types
        contents = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts = []
            if isinstance(content, list):
                for block in content:
                    parts.append(self._normalize_part(block))
            if role == "user":
                contents.append(types.Content(role="user", parts=parts))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(types.Content(role="user", parts=parts))
        return contents

    def _normalize_part(self, block: dict[str, Any]) -> Any:
        from google.genai import types
        block_type = block.get("type", "")

        if block_type == "text":
            return types.Part.from_text(text=block.get("text", ""))

        if block_type == "image":
            if "url" in block:
                data = urllib.request.urlopen(block["url"]).read()
                return types.Part.from_bytes(data=data, mime_type=block.get("mime_type", "image/png"))
            if "data" in block:
                data = base64.b64decode(block["data"])
                return types.Part.from_bytes(data=data, mime_type=block.get("mime_type", "image/png"))
            return types.Part.from_text(text="[image]")

        if block_type == "audio":
            if "data" in block:
                data = base64.b64decode(block["data"])
                return types.Part.from_bytes(data=data, mime_type=block.get("mime_type", "audio/wav"))
            return types.Part.from_text(text="[audio]")

        if block_type == "video":
            if "data" in block:
                data = base64.b64decode(block["data"])
                return types.Part.from_bytes(data=data, mime_type=block.get("mime_type", "video/mp4"))
            return types.Part.from_text(text="[video]")

        if block_type == "file":
            if "data" in block:
                data = base64.b64decode(block["data"])
                return types.Part.from_bytes(data=data, mime_type=block.get("mime_type", "application/octet-stream"))
            return types.Part.from_text(text="[file]")

        return types.Part.from_text(text=str(block))

    def _convert_tool(self, schema: dict[str, Any]) -> Any:
        from google.genai import types
        input_schema = schema.get("input_schema", {"type": "object", "properties": {}})
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    parameters=input_schema,
                )
            ]
        )