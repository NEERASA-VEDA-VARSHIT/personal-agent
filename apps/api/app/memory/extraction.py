"""Extract candidate memories from conversations using LLM."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Conversation, Memory, MemoryEvent, MemorySource, MemoryAudit
from app.memory.policy import (
    EvidenceStrength,
    MemoryCandidate,
    MemoryPolicy,
    MemoryStatus,
    MemoryType,
    SourceType,
    Stability,
)
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
      "reason": "Why this matters",
      "source_markers": ["quoted text 1", "quoted text 2"],
      "source_type": "user_stated|model_extracted|model_inferred",
      "evidence_strength": "high|moderate|low",
      "stability": "stable|volatile|unknown",
      "sensitivity": "public|private|confidential"
    }
  ]
}

Guidelines:

- FACT: Verifiable information (e.g., "User has 5 years of Python experience")
  evidence_strength: high when directly stated by user

- PREFERENCE: User's stated likes, dislikes, or preferences
  evidence_strength: high when directly stated

- GOAL: User's stated objectives or aspirations
  evidence_strength: high when directly stated

- EPISODE: Significant event or story from user's experience
  evidence_strength: high when detailed

- DECISION: Important choice or commitment
  evidence_strength: high when clearly stated

- RELATIONSHIP: Connection between concepts or people
  evidence_strength: high only with explicit evidence

- HYPOTHESIS: Speculative inference about user
  evidence_strength: low or moderate, never high without direct confirmation
  Only include if you have 2+ supporting statements

Rules for the new fields:
- source_type: user_stated if the user said it verbatim, model_inferred if you inferred it
- evidence_strength: high = verbatim + repeated, moderate = single clear statement, low = weak hint
- stability: stable = long-term trait, volatile = could change soon, unknown = unclear
- Do NOT invent a numeric confidence — use evidence_strength instead

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
                    # Back-compat: old prompts still send "confidence"
                    raw_conf = mem_data.get("confidence", 0.5)
                    ev = mem_data.get("evidence_strength", "moderate")
                    st = mem_data.get("stability", "unknown")
                    src = mem_data.get("source_type", "model_extracted")
                    try:
                        source_type = SourceType(src)
                    except ValueError:
                        source_type = SourceType.MODEL_EXTRACTED
                    candidate = MemoryCandidate(
                        memory_type=MemoryType(mem_data["type"]),
                        content=mem_data["content"],
                        reason=mem_data.get("reason", ""),
                        source_markers=mem_data.get("source_markers", []),
                        source_type=source_type,
                        evidence_strength=ev,
                        stability=st,
                        sensitivity=mem_data.get("sensitivity", "private"),
                        confidence=float(raw_conf),
                        model_confidence=float(raw_conf) if raw_conf is not None else None,
                        requires_confirmation=bool(mem_data.get("requires_confirmation", False)),
                    )
                    candidates.append(candidate)
                except (KeyError, ValueError, TypeError):
                    # Skip malformed candidates
                    continue

        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            pass

        return candidates
