import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    provider: str = os.getenv("MODEL_PROVIDER", "ollama")
    model: str = os.getenv("MODEL_NAME", "llama3.2")
    base_url: str = os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1")
    api_key: str = os.getenv("MODEL_API_KEY", "ollama")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def get_model_settings() -> ModelSettings:
    return ModelSettings(
        provider=os.getenv("MODEL_PROVIDER", "ollama"),
        model=os.getenv("MODEL_NAME", "llama3.2"),
        base_url=os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("MODEL_API_KEY", "ollama"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )
