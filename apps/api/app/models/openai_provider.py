from __future__ import annotations

from typing import Any

from openai import OpenAI


class OpenAICompatibleProvider:
    def __init__(self, *, base_url: str, api_key: str, default_model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = default_model

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        return self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=model or self.default_model,
            input=texts,
            **kwargs,
        )
        return [item.embedding for item in response.data]
