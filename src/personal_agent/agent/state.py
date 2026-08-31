"""Agent state — conversation and provenance tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    user_id: int
    conversation_id: int | None = None
    turn: int = 0
    provenance: list[dict] = field(default_factory=list)

    def trace(self, stage: str, data: dict) -> None:
        self.provenance.append({"stage": stage, "data": data})
