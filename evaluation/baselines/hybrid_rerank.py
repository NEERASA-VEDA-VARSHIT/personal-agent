"""Baseline C: hybrid + reranking + policy (full RetrievalPipeline)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.retrieval import RetrievalPipeline


def retrieve_hybrid_rerank(db: Session, user_id: int, query: str, pipeline: RetrievalPipeline, k: int = 5) -> list[int]:
    result = pipeline.run(db, user_id, query, top_k=k, allow_sensitive=False)
    return [r.memory.id for r in result.ranked[:k]]
