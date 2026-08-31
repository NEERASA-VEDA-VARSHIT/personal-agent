"""Retrieval-Augmented Generation: memory-informed LLM responses.

M6.10.1 — Single retrieval path:

    ❌  embedding_service.retrieve_similar_memories()  (bypasses policy)
    ✅  RetrievalPipeline -> policy filter -> reranker -> context builder

RAG is now a thin orchestration layer over RetrievalPipeline, not a competing
retrieval implementation. Privacy, temporal validity and source quality are
enforced centrally in the pipeline, not per-caller.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from personal_agent.persistence.models import Memory
from personal_agent.inference.embeddings import EmbeddingService
from personal_agent.inference.gateway import ModelGateway
from personal_agent.memory.retrieval import RetrievalPipeline


class RAGService:
    """Thin RAG orchestration over the single RetrievalPipeline."""

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        embedding_service: EmbeddingService | None = None,
        pipeline: RetrievalPipeline | None = None,
    ):
        from personal_agent.inference.gateway import get_default_gateway

        self.gateway = gateway or get_default_gateway()
        # Single path: pipeline owns embedding_service + policy + reranker
        if pipeline is not None:
            self.pipeline = pipeline
            # keep reference for backward-compat callers that pass embedding_service
            self.embedding_service = embedding_service or pipeline.retriever.embedding_service
        else:
            self.embedding_service = embedding_service or EmbeddingService(gateway=self.gateway)
            self.pipeline = RetrievalPipeline(
                retriever=None,  # will create default with our embedding_service
            )
            # Inject our embedding_service into the pipeline's retriever
            # (RetrievalPipeline creates its own retriever if None; override)
            from personal_agent.memory.retrieval import CandidateRetriever

            self.pipeline.retriever = CandidateRetriever(embedding_service=self.embedding_service)

    def _build_memory_context(self, ranked) -> str:
        """Build context from RankedMemory list (new path) or legacy tuple list."""
        if not ranked:
            return ""
        parts = ["## User's Relevant Memories:"]
        for item in ranked:
            # Support both RankedMemory and legacy (Memory, score) tuple for backward-compat tests
            if isinstance(item, tuple) and len(item) == 2:
                mem, score = item
                marker = "✓" if getattr(mem, "confidence", 0.5) >= 0.9 else "~" if getattr(mem, "confidence", 0.5) >= 0.7 else "?"
                parts.append(f"- [{marker}] {mem.content}")
            else:
                r = item
                mem = r.memory
                marker = "✓" if mem.confidence >= 0.9 else "~" if mem.confidence >= 0.7 else "?"
                src = mem.sources[0].source_type if getattr(mem, "sources", None) and mem.sources else "unknown"
                parts.append(f"- [{marker}][{mem.type or mem.memory_type}][{src}] {mem.content} (score={r.final_score:.2f})")
        return "\n".join(parts)

    def _build_augmented_prompt(self, user_question: str, memory_context: str) -> list[dict[str, str]]:
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
            messages.append({"role": "system", "content": memory_context})
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
        allow_sensitive: bool = False,
    ) -> dict:
        """Generate response via single RetrievalPipeline path."""

        # Single retrieval path — all policy (status, validity, sensitivity, source quality) enforced here
        result = self.pipeline.run(
            db, user_id, user_question, top_k=top_k, allow_sensitive=allow_sensitive
        )
        ranked = result.ranked

        memory_context = self._build_memory_context(ranked)
        messages = self._build_augmented_prompt(user_question, memory_context)
        response_text = self.gateway.generate(messages, temperature=temperature)

        if include_memory_citations and ranked:
            citation_text = "\n\n---\n*This response was informed by your memories:*\n"
            for r in ranked:
                mem = r.memory
                citation_text += f"- Memory #{mem.id}: {mem.content[:80]}... (score={r.final_score:.2f})\n"
            response_text += citation_text

        return {
            "response": response_text,
            "memories_used": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "confidence": r.memory.confidence,
                    "similarity": r.semantic_score,
                    "final_score": r.final_score,
                    "source_type": r.memory.sources[0].source_type if r.memory.sources else None,
                    "status": r.memory.status,
                    "sensitivity": r.memory.sensitivity,
                    "debug": r.debug,
                }
                for r in ranked
            ],
            "system_prompt_used": memory_context,
            # Provenance trace for M6.10.3
            "retrieval_debug": {
                "query": result.understanding.raw_query,
                "temporal_intent": result.understanding.temporal_intent,
                "candidates": len(result.candidates),
                "filtered_out": len(result.filtered.filtered_out),
                "ranked": len(ranked),
            },
        }

    # Backward-compat shim for callers that passed (prompt, user_id, context_window) incorrectly
    # e.g. old DecisionRecommender. Prefer gateway.generate directly for decision reasoning.
    def generate_with_prompt(self, prompt: str, **kwargs) -> str:
        return self.gateway.generate([{"role": "user", "content": prompt}], **kwargs)
