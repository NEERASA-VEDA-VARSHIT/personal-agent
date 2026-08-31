from __future__ import annotations

from .config import ModelSettings, get_model_settings
from personal_agent.inference.providers.openai import OpenAICompatibleProvider


def build_provider(settings: ModelSettings | None = None):
    resolved_settings = settings or get_model_settings()

    if resolved_settings.provider in {"ollama", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(
            base_url=resolved_settings.base_url,
            api_key=resolved_settings.api_key,
            default_model=resolved_settings.model,
        )

    raise ValueError(f"Unsupported model provider: {resolved_settings.provider}")


class ModelGateway:
    def __init__(self, provider=None, settings: ModelSettings | None = None):
        self.provider = provider
        if self.provider is None:
            self.provider = build_provider(settings)

    def generate(self, messages, model=None, temperature=0.2, **kwargs):
        return self.provider.generate(messages=messages, model=model, temperature=temperature, **kwargs)

    def stream(self, messages, model=None, temperature=0.2, **kwargs):
        return self.provider.stream(messages=messages, model=model, temperature=temperature, **kwargs)

    def embed(self, texts, model=None, **kwargs):
        return self.provider.embed(texts=texts, model=model, **kwargs)


def get_default_gateway() -> ModelGateway:
    return ModelGateway()
