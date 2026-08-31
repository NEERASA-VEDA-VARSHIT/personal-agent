"""Baseline B: hybrid semantic + lexical (CandidateRetriever, no reranking/policy)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.retrieval import CandidateRetriever, QueryAnalyzer


def retrieve_hybrid(db: Session, user_id: int, query: str, retriever: CandidateRetriever, k: int = 5) -> list[int]:
    analyzer = QueryAnalyzer()
    emb = retriever.embedding_service.embed_text(query)
    understanding = analyzer.analyze(query, embedding=emb)
    candidates = retriever.retrieve(db, user_id, query, understanding=understanding)
    # rank by hybrid score used inside retriever (semantic 0.7 + lexical 0.3), already sorted
    return [c.memory.id for c in candidates[:k]]
