"""Embeddings and vector similarity operations."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from personal_agent.persistence.models import Memory
from personal_agent.inference.gateway import ModelGateway


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


class EmbeddingService:
    """Service for generating and managing embeddings."""

    def __init__(self, gateway: ModelGateway | None = None):
        """Initialize with a ModelGateway for embedding generation."""
        from personal_agent.inference.gateway import get_default_gateway

        self.gateway = gateway or get_default_gateway()

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = self.gateway.embed([text])
        return result[0] if result else []

    def embed_memory(self, db: Session, memory: Memory, model: str | None = None) -> None:
        """Generate and store embedding for a memory."""
        if not memory.content:
            return

        embedding_vector = self.embed_text(memory.content)
        if embedding_vector:
            memory.embedding = json.dumps(embedding_vector)
            db.add(memory)
            db.commit()

    def retrieve_similar_memories(
        self,
        db: Session,
        user_id: int,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        memory_types: Optional[list[str]] = None,
    ) -> list[tuple[Memory, float]]:
        """
        Retrieve similar memories for a query.

        Args:
            db: Database session
            user_id: User ID to scope memories
            query_text: Text to find similar memories for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            memory_types: Optional filter by memory types (e.g., ['explicit', 'candidate'])

        Returns:
            List of (memory, similarity_score) tuples, sorted by similarity descending
        """
        # Generate query embedding
        query_embedding = self.embed_text(query_text)
        if not query_embedding:
            return []

        # Fetch all active memories for this user
        query = db.query(Memory).filter(and_(Memory.user_id == user_id, Memory.is_active == True))

        if memory_types:
            query = query.filter(Memory.memory_type.in_(memory_types))

        memories = query.all()

        # Compute similarities
        scored_memories: list[tuple[Memory, float]] = []
        for memory in memories:
            if not memory.embedding:
                continue

            try:
                memory_vector = json.loads(memory.embedding)
                similarity = cosine_similarity(query_embedding, memory_vector)

                if similarity >= min_similarity:
                    scored_memories.append((memory, similarity))
            except (json.JSONDecodeError, ValueError):
                # Skip memories with invalid embeddings
                continue

        # Sort by similarity descending and return top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[:top_k]
