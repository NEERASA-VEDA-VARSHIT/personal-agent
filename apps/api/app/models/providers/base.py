from __future__ import annotations

from typing import Any, Protocol


class ModelProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        ...

    def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        ...

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        ...
