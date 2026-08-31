"""Retrieval-Augmented Generation: memory-informed LLM responses."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Memory
from app.embeddings import EmbeddingService
from app.models import ModelGateway


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Combines user memories with LLM inference to produce personalized responses.
    """

    def __init__(self, gateway: ModelGateway | None = None, embedding_service: EmbeddingService | None = None):
        """Initialize RAG service with model gateway and embedding service."""
        from app.models import get_default_gateway

        self.gateway = gateway or get_default_gateway()
        self.embedding_service = embedding_service or EmbeddingService(gateway=self.gateway)

    def _build_memory_context(self, memories: list[tuple[Memory, float]]) -> str:
        """Format retrieved memories into a context string."""
        if not memories:
            return ""

        context_parts = ["## User's Relevant Memories:"]
        for memory, score in memories:
            confidence_marker = "✓" if memory.confidence >= 0.9 else "~" if memory.confidence >= 0.7 else "?"
            context_parts.append(f"- [{confidence_marker}] {memory.content}")

        return "\n".join(context_parts)

    def _build_augmented_prompt(self, user_question: str, memory_context: str) -> list[dict[str, str]]:
        """
        Build a messages list with system context and user question.

        Args:
            user_question: The user's actual question
            memory_context: Formatted memory context from retrieval

        Returns:
            Messages list for LLM consumption
        """
        system_prompt = """You are a thoughtful personal AI assistant.

You have access to the user's personal memories and preferences.
Use this context to give personalized, relevant advice.
Be honest if you don't have enough context to answer well.
Acknowledge what you know about them when relevant.

Always be conversational and helpful."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": memory_context,
                }
            )

        messages.append({"role": "user", "content": user_question})

        return messages

    def generate_response(
        self,
        db: Session,
        user_id: int,
        user_question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_memory_citations: bool = False,
    ) -> dict:
        """
        Generate a personalized response using memory-augmented generation.

        Args:
            db: Database session
            user_id: User ID to retrieve memories for
            user_question: The user's question
            top_k: Number of relevant memories to retrieve
            temperature: LLM temperature (creativity)
            include_memory_citations: Whether to include memory IDs in response

        Returns:
            Dict with keys: response, memories_used, system_prompt
        """
        # Retrieve relevant memories
        memories = self.embedding_service.retrieve_similar_memories(
            db,
            user_id=user_id,
            query_text=user_question,
            top_k=top_k,
        )

        # Build context
        memory_context = self._build_memory_context(memories)

        # Build augmented prompt
        messages = self._build_augmented_prompt(user_question, memory_context)

        # Call LLM
        response_text = self.gateway.generate(messages, temperature=temperature)

        # If requested, append memory citations
        if include_memory_citations and memories:
            citation_text = "\n\n---\n*This response was informed by your memories:*\n"
            for memory, score in memories:
                citation_text += f"- Memory #{memory.id}: {memory.content[:80]}...\n"
            response_text += citation_text

        return {
            "response": response_text,
            "memories_used": [
                {
                    "id": memory.id,
                    "content": memory.content,
                    "confidence": memory.confidence,
                    "similarity": score,
                }
                for memory, score in memories
            ],
            "system_prompt_used": memory_context,
        }
