"""Decision v2 models — assessment-based reasoning (no numeric confidence)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvidenceStrength(str, Enum):
    """Qualitative evidence quality — not a probability."""

    LIMITED = "limited"
    MODERATE = "moderate"
    STRONG = "strong"


class Assessment(str, Enum):
    """Overall stance — replaces 78% confidence."""

    STRONGLY_POSITIVE = "strongly_positive"
    POSITIVE = "positive"
    CAUTIOUSLY_POSITIVE = "cautiously_positive"
    NEUTRAL = "neutral"
    CAUTIOUSLY_NEGATIVE = "cautiously_negative"
    NEGATIVE = "negative"
    STRONGLY_NEGATIVE = "strongly_negative"


class RecommendationStrength(str, Enum):
    """How strongly we recommend an option — distinct from evidence strength."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


@dataclass
class DecisionAssessment:
    """Final assessment — validated application output.

    LLM proposes via JSON, application validates before returning.
    """

    decision_statement: str
    recommendation: str  # human-readable recommendation text
    assessment: Assessment | str
    evidence_strength: EvidenceStrength | str
    key_factors: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    reasons_for: list[str] = field(default_factory=list)
    reasons_against: list[str] = field(default_factory=list)
    what_would_change: list[str] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.assessment, str):
            try:
                self.assessment = Assessment(self.assessment)
            except ValueError:
                self.assessment = Assessment.CAUTIOUSLY_POSITIVE
        if isinstance(self.evidence_strength, str):
            try:
                self.evidence_strength = EvidenceStrength(self.evidence_strength)
            except ValueError:
                self.evidence_strength = EvidenceStrength.MODERATE


@dataclass
class EvidenceBundle:
    """Built from retrieved personal context + stakes + unknowns."""

    decision_statement: str
    known_facts: list[str] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    for_factors: list[str] = field(default_factory=list)
    against_factors: list[str] = field(default_factory=list)
