from typing import Any
from agent_template.core.tools.base import Tool


class Echo(Tool):
    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "echo",
            "description": "Echo the input string back to the user.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The string to echo.",
                    },
                },
                "required": ["input"],
            },
        }

    def execute(self, **kwargs: Any) -> str:
        return kwargs.get("input", "")