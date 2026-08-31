"""Extract candidate memories from conversations using LLM."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Conversation, Memory, MemoryEvent, MemorySource, MemoryAudit
from app.memory_policy import MemoryCandidate, MemoryPolicy, MemoryType, SourceType, MemoryStatus
from app.models import ModelGateway


class MemoryExtractionService:
    """
    Extract candidate memories from conversations using LLM.

    Workflow:
    1. Analyze conversation with LLM
    2. LLM proposes candidate memories as structured JSON
    3. Apply memory policy validation
    4. Store approved memories with audit trail
    """

    def __init__(self, gateway: ModelGateway | None = None, policy: MemoryPolicy | None = None):
        """Initialize with model gateway and memory policy."""
        from app.models import get_default_gateway

        self.gateway = gateway or get_default_gateway()
        self.policy = policy or MemoryPolicy()

    def _build_extraction_prompt(self, conversation_text: str) -> list[dict[str, str]]:
        """Build prompt for LLM to extract memories from conversation."""
        system_prompt = """You are a memory extraction specialist for a personal AI agent.

Your job is to read a conversation and identify facts, preferences, goals, and insights worth remembering.

Return a JSON object with this structure:
{
  "memories": [
    {
      "type": "fact|preference|goal|episode|decision|relationship|hypothesis",
      "content": "What to remember",
      "confidence": 0.0-1.0,
      "reason": "Why this matters",
      "source_markers": ["quoted text 1", "quoted text 2"]
    }
  ]
}

Guidelines:

- FACT: Verifiable information (e.g., "User has 5 years of Python experience")
  Confidence: 0.95-1.0 when from direct statement

- PREFERENCE: User's stated likes, dislikes, or preferences
  Confidence: 0.90-1.0 when directly stated

- GOAL: User's stated objectives or aspirations
  Confidence: 0.90-1.0 when directly stated

- EPISODE: Significant event or story from user's experience
  Confidence: 0.85-1.0 when detailed

- DECISION: Important choice or commitment
  Confidence: 0.85-1.0 when clearly stated

- RELATIONSHIP: Connection between concepts or people
  Confidence: 0.70-0.95 based on evidence

- HYPOTHESIS: Speculative inference about user
  Confidence: 0.50-0.80 with multiple supporting quotes
  Only include if you have 2+ supporting statements

Do NOT:
- Create vague memories
- Mix multiple concepts into one memory
- Include "probably" or "might" without evidence for hypotheses
- Store obvious facts that don't reflect preferences/goals

Only return valid JSON."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract memories from this conversation:\n\n{conversation_text}"},
        ]

    def extract_from_conversation(
        self,
        db: Session,
        user_id: int,
        conversation_id: int,
        apply_policy: bool = True,
    ) -> dict:
        """
        Extract candidate memories from a conversation.

        Args:
            db: Database session
            user_id: User ID
            conversation_id: Conversation ID to analyze
            apply_policy: Whether to apply policy validation before returning

        Returns:
            Dict with keys: candidates, approved, rejected, needs_review
        """
        # Fetch conversation with messages
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation or conversation.user_id != user_id:
            raise ValueError(f"Conversation {conversation_id} not found for user {user_id}")

        # Build conversation text
        conversation_text = self._format_conversation(conversation)

        # Call LLM to extract memories
        messages = self._build_extraction_prompt(conversation_text)
        response_text = self.gateway.generate(messages, temperature=0.3)

        # Parse response
        candidates = self._parse_extraction_response(response_text)

        # Apply policy
        result = {
            "candidates": candidates,
            "approved": [],
            "rejected": [],
            "needs_review": [],
        }

        if apply_policy:
            for candidate in candidates:
                validation = self.policy.validate(candidate)
                policy_result = {
                    "candidate": candidate,
                    "validation": validation,
                }

                if validation["should_ask"]:
                    result["needs_review"].append(policy_result)
                elif validation["should_store"]:
                    result["approved"].append(policy_result)
                else:
                    result["rejected"].append(policy_result)

        return result

    def store_approved_memories(
        self,
        db: Session,
        user_id: int,
        conversation_id: int,
        approved_candidates: list[dict],
    ) -> list[Memory]:
        """
        Store approved memory candidates to database.

        Args:
            db: Database session
            user_id: User ID
            conversation_id: Conversation ID (source)
            approved_candidates: List of approved policy results

        Returns:
            List of created Memory objects
        """
        stored_memories = []

        for approved in approved_candidates:
            candidate: MemoryCandidate = approved["candidate"]

            # Create memory with v2 schema
            memory = Memory(
                user_id=user_id,
                source_conversation_id=conversation_id,
                type=candidate.memory_type.value,  # FACT, PREFERENCE, GOAL, etc.
                memory_type=candidate.memory_type.value,  # Backward compatibility
                content=candidate.content,
                confidence=candidate.confidence,
                status=MemoryStatus.ACTIVE.value,  # Direct store (policy already validated)
                is_active=True,
            )
            db.add(memory)
            db.flush()

            # Add source citation with MODEL_EXTRACTED origin
            source = MemorySource(
                memory_id=memory.id,
                source_type=SourceType.MODEL_EXTRACTED.value,
                source_ref=f"conversation_{conversation_id}",
                confidence=candidate.confidence,
            )
            db.add(source)

            # Add creation audit record
            audit = MemoryAudit(
                memory_id=memory.id,
                action="created",
                reason=f"Extracted from conversation {conversation_id}",
                actor="extraction_service",
            )
            db.add(audit)

            stored_memories.append(memory)

        db.commit()
        return stored_memories

    def _format_conversation(self, conversation: Conversation) -> str:
        """Format conversation messages into readable text."""
        lines = [f"Conversation: {conversation.title}\n"]
        for message in conversation.messages:
            role = "User" if message.role == "user" else "Assistant"
            lines.append(f"{role}: {message.content}\n")
        return "".join(lines)

    def _parse_extraction_response(self, response_text: str) -> list[MemoryCandidate]:
        """Parse LLM response into MemoryCandidate objects."""
        candidates = []

        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start < 0 or json_end <= json_start:
                return candidates

            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            for mem_data in data.get("memories", []):
                try:
                    candidate = MemoryCandidate(
                        memory_type=MemoryType(mem_data["type"]),
                        content=mem_data["content"],
                        confidence=float(mem_data["confidence"]),
                        reason=mem_data.get("reason", ""),
                        source_markers=mem_data.get("source_markers", []),
                    )
                    candidates.append(candidate)
                except (KeyError, ValueError, TypeError):
                    # Skip malformed candidates
                    continue

        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            pass

        return candidates
