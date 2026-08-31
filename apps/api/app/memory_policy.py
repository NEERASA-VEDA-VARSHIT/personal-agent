"""Memory policy: rules for what gets stored as memories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryType(str, Enum):
    """What the memory represents — not how it was produced."""

    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    EPISODE = "episode"
    DECISION = "decision"
    RELATIONSHIP = "relationship"
    HYPOTHESIS = "hypothesis"


class SourceType(str, Enum):
    """How the memory was produced."""

    USER_STATED = "user_stated"
    MODEL_EXTRACTED = "model_extracted"
    MODEL_INFERRED = "model_inferred"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    FORGOTTEN = "forgotten"


@dataclass
class MemoryCandidate:
    """Proposed memory before validation."""

    memory_type: MemoryType
    content: str
    confidence: float  # 0.0-1.0 — evidence strength, not truth probability
    reason: str  # Why the agent thinks this is important
    source_markers: list[str]  # Evidence from conversation


class MemoryPolicy:
    """
    Validates proposed memories against policy rules.

    Policy decisions:
    - FACT/PREFERENCE/GOAL/EPISODE/DECISION from user: Store immediately
    - RELATIONSHIP: Store with evidence linkage
    - HYPOTHESIS: Store with evidence linkage only
    - MODEL_INFERRED: Never store directly (log only)
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
        if candidate.memory_type in (MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.GOAL, MemoryType.EPISODE, MemoryType.DECISION):
            return True

        if candidate.memory_type == MemoryType.RELATIONSHIP:
            return len(candidate.source_markers) > 0

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            has_evidence = len(candidate.source_markers) > 0
            meets_confidence = candidate.confidence >= self.hypothesis_threshold
            return has_evidence and meets_confidence

        if candidate.memory_type == MemoryType.HYPOTHESIS and candidate.confidence < self.hypothesis_threshold:
            return False

        return False

    def should_ask_user(self, candidate: MemoryCandidate) -> bool:
        """Determine if memory candidate needs user confirmation."""
        if candidate.memory_type in (MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.GOAL, MemoryType.EPISODE, MemoryType.DECISION):
            return False

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            threshold_min = self.hypothesis_threshold * 0.7
            threshold_max = self.hypothesis_threshold
            return threshold_min <= candidate.confidence < threshold_max

        if candidate.memory_type == MemoryType.RELATIONSHIP:
            threshold_min = self.candidate_threshold * 0.7
            threshold_max = self.candidate_threshold
            return threshold_min <= candidate.confidence < threshold_max

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
        if candidate.memory_type in (MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.GOAL, MemoryType.EPISODE, MemoryType.DECISION):
            return f"{candidate.memory_type.value.title()} — storing immediately"

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            if not candidate.source_markers:
                return "Hypothesis without evidence — not storing"
            if candidate.confidence < self.hypothesis_threshold:
                return f"Hypothesis confidence {candidate.confidence:.2f} below threshold {self.hypothesis_threshold:.2f}"
            return "Well-evidenced hypothesis — storing"

        if candidate.memory_type == MemoryType.RELATIONSHIP:
            if not candidate.source_markers:
                return "Relationship without evidence — not storing"
            return "Relationship with evidence — storing"

        return f"Candidate confidence {candidate.confidence:.2f} — review needed"


def get_source_type_label(source_type: str) -> str:
    """Return a human-readable label for a source type."""
    labels = {
        SourceType.USER_STATED.value: "User stated",
        SourceType.MODEL_EXTRACTED.value: "Model extracted",
        SourceType.MODEL_INFERRED.value: "Model inferred",
    }
    return labels.get(source_type, source_type)


def get_status_label(status: str) -> str:
    """Return a human-readable label for a memory status."""
    labels = {
        MemoryStatus.CANDIDATE.value: "Candidate",
        MemoryStatus.ACTIVE.value: "Active",
        MemoryStatus.SUPERSEDED.value: "Superseded",
        MemoryStatus.REJECTED.value: "Rejected",
        MemoryStatus.FORGOTTEN.value: "Forgotten",
    }
    return labels.get(status, status)