"""Baseline A: cosine similarity only (EmbeddingService.retrieve_similar_memories)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from personal_agent.inference.embeddings import EmbeddingService


def retrieve_vector_only(db: Session, user_id: int, query: str, embedding_service: EmbeddingService, k: int = 5) -> list[int]:
    results = embedding_service.retrieve_similar_memories(db, user_id=user_id, query_text=query, top_k=k)
    return [mem.id for mem, _ in results]
