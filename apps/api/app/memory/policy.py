"""Memory policy: rules for what gets stored as memories.

Confidence here is *evidence strength*, not truth probability.
A model returning 0.92 does not mean the fact is 92% true — it means the
evidence for extracting that memory is judged to be strong/moderate/weak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


class EvidenceStrength(str, Enum):
    """Qualitative evidence strength — replaces raw LLM probability."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class Stability(str, Enum):
    """Expected stability of the memory over time."""

    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Candidate model — explicit about what each field means
# ---------------------------------------------------------------------------


@dataclass
class MemoryCandidate:
    """Proposed memory before validation.

    Field semantics (important for the research story):
    - memory_type:      WHAT the memory is (fact, preference, hypothesis …)
    - source_type:      HOW it was produced (user stated vs model inferred)
    - source_markers:   verbatim evidence spans from the conversation
    - evidence_strength: qualitative judgment of evidence quality
    - stability:        expected persistence of the fact
    - sensitivity:      privacy tier (public / private / confidential)
    - confidence:       deprecated raw LLM number — kept only for backward
                        compatibility; policy MUST NOT branch on it alone
    - model_confidence: optional raw 0–1 number for observability
    - requires_confirmation: hint that human confirmation would help
    """

    memory_type: MemoryType
    content: str
    reason: str
    source_markers: list[str] = field(default_factory=list)

    # Provenance
    source_type: SourceType = SourceType.MODEL_EXTRACTED

    # Qualitative evidence model (preferred over raw confidence)
    evidence_strength: EvidenceStrength | str = EvidenceStrength.MODERATE
    stability: Stability | str = Stability.UNKNOWN
    sensitivity: str = "private"

    # Deprecated / observability — do not use as store-or-not signal
    confidence: float = 0.5
    model_confidence: float | None = None

    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        # Normalize string inputs to enums where possible
        if isinstance(self.evidence_strength, str):
            try:
                self.evidence_strength = EvidenceStrength(self.evidence_strength.lower())
            except ValueError:
                self.evidence_strength = EvidenceStrength.MODERATE
        if isinstance(self.stability, str):
            try:
                self.stability = Stability(self.stability.lower())
            except ValueError:
                self.stability = Stability.UNKNOWN

        # Backward compatibility: if caller only supplied legacy `confidence`
        # and left evidence_strength at default, infer qualitative strength.
        # This branch is intentionally conservative and will be removed once
        # callers migrate to evidence_strength.
        if self.model_confidence is None and self.confidence != 0.5:
            self.model_confidence = self.confidence
        # If evidence_strength was left at MODERATE but confidence is extreme,
        # infer — but do not treat confidence as truth probability.
        # We only do this when the caller clearly used the old API.
        # New callers should set evidence_strength explicitly.


def _evidence_is_strong(candidate: MemoryCandidate) -> bool:
    return candidate.evidence_strength == EvidenceStrength.HIGH


def _evidence_is_moderate(candidate: MemoryCandidate) -> bool:
    return candidate.evidence_strength == EvidenceStrength.MODERATE


def _has_direct_evidence(candidate: MemoryCandidate) -> bool:
    return len(candidate.source_markers) > 0


# ---------------------------------------------------------------------------
# Policy — interpretable rules, not LLM-probability thresholds
# ---------------------------------------------------------------------------


class MemoryPolicy:
    """
    Validates proposed memories against interpretable policy rules.

    Core principle: do NOT treat LLM confidence as truth probability.
      "I've been enjoying backend lately" + model_confidence 0.92
        does NOT mean a 92% chance the user has a stable backend preference.

    Instead, decide on:
      evidence_strength  (high / moderate / low)
      source_type        (user_stated vs inferred)
      stability          (stable / volatile / unknown)
      sensitivity        (public / private / confidential)
      requires_confirmation
    """

    def __init__(
        self,
        candidate_confidence_threshold: float = 0.75,
        hypothesis_confidence_threshold: float = 0.60,
    ):
        # Kept for backward-compatible callers that still pass thresholds.
        # New code should not rely on numeric thresholds.
        self.candidate_threshold = candidate_confidence_threshold
        self.hypothesis_threshold = hypothesis_confidence_threshold

    # -- store decision -----------------------------------------------------

    def should_store(self, candidate: MemoryCandidate) -> bool:
        """Return True if the candidate should be stored (ACTIVE/CANDIDATE)."""

        # USER_STATED + direct evidence => store. This is the highest signal.
        if candidate.source_type == SourceType.USER_STATED and _has_direct_evidence(candidate):
            return True

        # Stable facts/preferences/goals/episodes/decisions with at least
        # moderate evidence => store.
        if candidate.memory_type in (
            MemoryType.FACT,
            MemoryType.PREFERENCE,
            MemoryType.GOAL,
            MemoryType.EPISODE,
            MemoryType.DECISION,
        ):
            if _evidence_is_strong(candidate) or _evidence_is_moderate(candidate):
                return True
            # Low evidence + volatile => do not store without confirmation
            return False

        if candidate.memory_type == MemoryType.RELATIONSHIP:
            # Relationships need evidence and at least moderate strength
            return _has_direct_evidence(candidate) and _evidence_is_strong(candidate) or (
                _has_direct_evidence(candidate) and _evidence_is_moderate(candidate) and candidate.stability != Stability.VOLATILE
            )

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            # Hypotheses require strong evidence and explicit markers
            return _has_direct_evidence(candidate) and _evidence_is_strong(candidate)

        return False

    def should_ask_user(self, candidate: MemoryCandidate) -> bool:
        """Return True if the candidate should be queued for confirmation."""

        # Never ask for directly stated stable facts — they are already good.
        if candidate.source_type == SourceType.USER_STATED and _evidence_is_strong(candidate):
            return False

        # Moderate-evidence hypotheses / relationships benefit from confirmation
        if candidate.memory_type == MemoryType.HYPOTHESIS:
            return _has_direct_evidence(candidate) and _evidence_is_moderate(candidate)

        if candidate.memory_type == MemoryType.RELATIONSHIP:
            return _has_direct_evidence(candidate) and _evidence_is_moderate(candidate)

        # Volatile preferences with moderate evidence => confirm
        if candidate.memory_type == MemoryType.PREFERENCE and candidate.stability == Stability.VOLATILE:
            return _evidence_is_moderate(candidate)

        # Explicit requires_confirmation hint
        if candidate.requires_confirmation:
            return True

        return False

    def validate(self, candidate: MemoryCandidate) -> dict:
        """Validate a candidate; return {should_store, should_ask, reason}."""
        return {
            "should_store": self.should_store(candidate),
            "should_ask": self.should_ask_user(candidate),
            "reason": self._get_validation_reason(candidate),
        }

    def _get_validation_reason(self, candidate: MemoryCandidate) -> str:
        """Human-readable reason — must reference evidence/source, not just a number."""
        if candidate.source_type == SourceType.USER_STATED and _has_direct_evidence(candidate):
            return f"{candidate.memory_type.value.title()} directly stated with evidence — storing"

        if candidate.memory_type in (
            MemoryType.FACT,
            MemoryType.PREFERENCE,
            MemoryType.GOAL,
            MemoryType.EPISODE,
            MemoryType.DECISION,
        ):
            if _evidence_is_strong(candidate):
                return f"{candidate.memory_type.value.title()} with strong evidence ({candidate.evidence_strength.value}) — storing"
            if _evidence_is_moderate(candidate):
                return f"{candidate.memory_type.value.title()} with moderate evidence — storing"
            return f"{candidate.memory_type.value.title()} with weak evidence — not storing without confirmation"

        if candidate.memory_type == MemoryType.HYPOTHESIS:
            if not _has_direct_evidence(candidate):
                return "Hypothesis without evidence — not storing"
            if _evidence_is_strong(candidate):
                return "Well-evidenced hypothesis (strong) — storing"
            if _evidence_is_moderate(candidate):
                return "Hypothesis with moderate evidence — needs confirmation"
            return "Hypothesis with weak evidence — not storing"

        if candidate.memory_type == MemoryType.RELATIONSHIP:
            if not _has_direct_evidence(candidate):
                return "Relationship without evidence — not storing"
            if _evidence_is_strong(candidate):
                return "Relationship with strong evidence — storing"
            if _evidence_is_moderate(candidate):
                return "Relationship with moderate evidence — needs confirmation"
            return "Relationship with weak evidence — not storing"

        return f"{candidate.memory_type.value} with {candidate.evidence_strength.value} evidence — review needed"


def get_source_type_label(source_type: str) -> str:
    labels = {
        SourceType.USER_STATED.value: "User stated",
        SourceType.MODEL_EXTRACTED.value: "Model extracted",
        SourceType.MODEL_INFERRED.value: "Model inferred",
    }
    return labels.get(source_type, source_type)


def get_status_label(status: str) -> str:
    labels = {
        MemoryStatus.CANDIDATE.value: "Candidate",
        MemoryStatus.ACTIVE.value: "Active",
        MemoryStatus.SUPERSEDED.value: "Superseded",
        MemoryStatus.REJECTED.value: "Rejected",
        MemoryStatus.FORGOTTEN.value: "Forgotten",
    }
    return labels.get(status, status)
