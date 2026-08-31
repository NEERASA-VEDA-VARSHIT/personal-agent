"""Extract candidate memories from conversations using LLM."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Conversation, Memory, MemoryEvent, MemorySource
from app.memory_policy import MemoryCandidate, MemoryPolicy, MemoryType
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
      "type": "explicit|candidate|hypothesis",
      "content": "What to remember",
      "confidence": 0.0-1.0,
      "reason": "Why this matters",
      "source_markers": ["quoted text 1", "quoted text 2"]
    }
  ]
}

Guidelines:

- EXPLICIT: User directly stated "Remember that..." or clearly stated a goal/preference
  Confidence: 0.95-1.0

- CANDIDATE: Inferred from conversation but not explicitly stated
  Confidence: 0.60-0.95
  Examples: "They seem to enjoy X", "They mentioned wanting to Y"

- HYPOTHESIS: Speculative inference with supporting evidence
  Confidence: 0.40-0.80
  Only include if you have multiple supporting quotes
  Examples: "User might be interested in distributed systems" (with 2+ supporting statements)

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

            # Create memory
            memory = Memory(
                user_id=user_id,
                source_conversation_id=conversation_id,
                memory_type=candidate.memory_type.value,
                content=candidate.content,
                confidence=candidate.confidence,
                is_active=True,
            )
            db.add(memory)
            db.flush()

            # Add source citation
            source = MemorySource(
                memory_id=memory.id,
                source_type="conversation",
                source_ref=f"conversation_{conversation_id}",
                confidence=candidate.confidence,
            )
            db.add(source)

            # Add creation event
            event = MemoryEvent(
                memory_id=memory.id,
                event_type="created",
                reason=f"Extracted from conversation {conversation_id}",
            )
            db.add(event)

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
