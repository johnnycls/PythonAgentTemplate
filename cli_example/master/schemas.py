INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "array",
            "description": "Content blocks in standard format (e.g., [{type: 'text', text: '...'}])",
        },
    },
    "required": ["content"],
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["result", "options"],
}
