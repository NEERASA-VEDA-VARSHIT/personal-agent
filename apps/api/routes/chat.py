"""POST /chat — Agent orchestration.

Flow:
    POST /chat -> Agent -> QuestionPolicy -> RetrievalPipeline (single path) -> DecisionEngine -> Gateway -> Response
"""

from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from personal_agent.persistence.session import get_session
from personal_agent.agent.agent import Agent

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: int
    message: str
    conversation_id: int | None = None
    allow_sensitive: bool = False


class ChatResponse(BaseModel):
    response: str
    action: str
    question_to_ask: str | None = None
    memories_used: list[dict]
    assessment: dict | None = None
    provenance: list[dict]


def get_agent() -> Agent:
    return Agent()


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_session), agent: Agent = Depends(get_agent)):
    result = agent.chat(db, user_id=req.user_id, message=req.message, conversation_id=req.conversation_id, allow_sensitive=req.allow_sensitive)
    return ChatResponse(
        response=result.response,
        action=result.action,
        question_to_ask=result.question_to_ask,
        memories_used=result.memories_used,
        assessment=result.assessment,
        provenance=result.provenance,
    )
