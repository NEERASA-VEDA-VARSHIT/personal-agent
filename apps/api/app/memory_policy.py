"""Memory policy: rules for what gets stored as memories."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class MemoryType(str, Enum):
    """Classification of memory types."""

    EXPLICIT = "explicit"  # User directly stated
    CANDIDATE = "candidate"  # Inferred, needs review
    INFERENCE = "inference"  # Agent inference (not stored directly)
    HYPOTHESIS = "hypothesis"  # Speculative, evidence-backed


@dataclass
class MemoryCandidate:
    """Proposed memory before validation."""

    memory_type: MemoryType
    content: str
    confidence: float  # 0.0-1.0
    reason: str  # Why the agent thinks this is important
    source_markers: list[str]  # Evidence from conversation


class MemoryPolicy:
    """
    Validates proposed memories against policy rules.

    Policy decisions:
    - EXPLICIT: Always store (user said it)
    - CANDIDATE: Store if confidence >= threshold, else flag for review
    - INFERENCE: Never store directly (log only)
    - HYPOTHESIS: Store with evidence linkage only
    """

    def __init__(
        self,
        candidate_confidence_threshold: float = 0.75,
        hypothesis_confidence_threshold: float = 0.60,
    ):
        self.candidate_threshold = candidate_confidence_threshold
        self.hypothesis_threshold = hypothesis_confidence_threshold

    def should_store(self, candidate: MemoryCandidate) -> bool:
        """Determine if a memory candidate should be stored."""
        if candidate.memory_type == MemoryType.EXPLICIT:
            # Always store explicit memories
            return True

        if candidate.memory_type == MemoryType.CANDIDATE:
            # Store if confidence meets threshold
            return candidate.confidence >= self.candidate_threshold

        if candidate.memory_type == MemoryType.INFERENCE:
            # Never store inferences directly
            return False

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            # Store hypotheses only if high confidence and well-evidenced
            has_evidence = len(candidate.source_markers) > 0
            meets_confidence = candidate.confidence >= self.hypothesis_threshold
            return has_evidence and meets_confidence

        return False

    def should_ask_user(self, candidate: MemoryCandidate) -> bool:
        """Determine if memory candidate needs user confirmation."""
        if candidate.memory_type == MemoryType.EXPLICIT:
            # Don't ask for explicit memories
            return False

        if candidate.memory_type == MemoryType.CANDIDATE:
            # Ask if confidence is borderline
            threshold_min = self.candidate_threshold * 0.7
            threshold_max = self.candidate_threshold
            return threshold_min <= candidate.confidence < threshold_max

        # All other types don't require user interaction per policy
        return False

    def validate(self, candidate: MemoryCandidate) -> dict:
        """
        Validate a memory candidate against policy.

        Returns:
            Dict with keys: should_store, should_ask, reason
        """
        return {
            "should_store": self.should_store(candidate),
            "should_ask": self.should_ask_user(candidate),
            "reason": self._get_validation_reason(candidate),
        }

    def _get_validation_reason(self, candidate: MemoryCandidate) -> str:
        """Explain why a memory is being accepted, rejected, or queued."""
        if candidate.memory_type == MemoryType.EXPLICIT:
            return "Explicit statement — storing immediately"

        if candidate.memory_type == MemoryType.INFERENCE:
            return "Agent inference — storing only as log, not as fact"

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            if not candidate.source_markers:
                return "Hypothesis without evidence — not storing"
            if candidate.confidence < self.hypothesis_threshold:
                return f"Hypothesis confidence {candidate.confidence:.2f} below threshold {self.hypothesis_threshold:.2f}"
            return "Well-evidenced hypothesis — storing"

        # CANDIDATE
        if candidate.confidence >= self.candidate_threshold:
            return f"Candidate confidence {candidate.confidence:.2f} meets threshold — storing"
        elif candidate.confidence >= (self.candidate_threshold * 0.7):
            return f"Candidate confidence {candidate.confidence:.2f} is borderline — asking user"
        else:
            return f"Candidate confidence {candidate.confidence:.2f} below threshold — discarding"
