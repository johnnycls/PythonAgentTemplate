from agent_template.core.providers.openai_compat import OpenAICompatProvider

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"

ollama_provider = OpenAICompatProvider(name="ollama", api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)
