# Ollama provider — thin alias over OpenAI-compatible for local runtime
from app.models.providers.openai import OpenAICompatibleProvider as OllamaProvider  # noqa: F401
__all__ = [\"OllamaProvider\"]