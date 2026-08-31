"""
M6: Decision Support Engine

Autonomous decision-making with stakes assessment, evidence analysis, and recommendations.
Uses extracted memories + RAG pipeline + policy framework to evaluate and recommend decisions.

Architecture:
1. StakesAssessment: Evaluates decision impact and reversibility
2. EvidenceAnalyzer: Weighs pro/con arguments with confidence scoring
3. DecisionRecommender: Synthesizes recommendation with reasoning
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.models.gateway import ModelGateway
from app.rag import RAGService


class ImpactLevel(Enum):
    """Decision impact severity."""

    LOW = "low"  # Easily reversible, minimal impact
    MEDIUM = "medium"  # Partially reversible, moderate impact
    HIGH = "high"  # Hard to reverse, major impact
    CRITICAL = "critical"  # Irreversible, life-changing impact


class Reversibility(Enum):
    """How easily a decision can be undone."""

    FULLY_REVERSIBLE = "fully_reversible"  # Can be completely undone
    PARTIALLY_REVERSIBLE = "partially_reversible"  # Some aspects can be undone
    IRREVERSIBLE = "irreversible"  # Cannot be undone


class RecommendationType(Enum):
    """Decision recommendation type."""

    STRONGLY_RECOMMEND = "strongly_recommend"  # High confidence positive
    RECOMMEND = "recommend"  # Moderate confidence positive
    NEUTRAL = "neutral"  # Equal pros/cons or insufficient evidence
    CAUTION = "caution"  # Some concerns, needs careful consideration
    STRONGLY_CAUTION = "strongly_caution"  # High confidence negative


@dataclass
class Stake:
    """A single stake (benefit or risk) in a decision."""

    description: str  # e.g., "Career growth opportunity"
    impact_level: ImpactLevel  # Severity of this stake
    probability: float  # 0.0-1.0 likelihood this will occur
    confidence: float  # 0.0-1.0 confidence in the assessment

    def weighted_impact(self) -> float:
        """Calculate weighted impact (probability × confidence)."""
        impact_scores = {
            ImpactLevel.LOW: 1.0,
            ImpactLevel.MEDIUM: 2.5,
            ImpactLevel.HIGH: 5.0,
            ImpactLevel.CRITICAL: 10.0,
        }
        return impact_scores[self.impact_level] * self.probability * self.confidence


@dataclass
class StakesAssessment:
    """Assessment of a decision's stakes and reversibility."""

    decision_statement: str  # The decision being evaluated
    benefits: list[Stake]  # Positive outcomes
    risks: list[Stake]  # Negative outcomes
    reversibility: Reversibility  # How easily can it be undone
    reversibility_explanation: str  # Why this reversibility level

    def net_impact_score(self) -> float:
        """Calculate net impact (benefits - risks)."""
        benefit_score = sum(stake.weighted_impact() for stake in self.benefits)
        risk_score = sum(stake.weighted_impact() for stake in self.risks)
        return benefit_score - risk_score

    def is_low_risk(self) -> bool:
        """Is this a low-risk decision?"""
        return (
            self.reversibility
            in (Reversibility.FULLY_REVERSIBLE, Reversibility.PARTIALLY_REVERSIBLE)
            and self.net_impact_score() >= 0
        )


@dataclass
class Evidence:
    """A single piece of evidence for or against a decision."""

    argument: str  # The claim or observation
    supports_decision: bool  # True if pro, False if con
    confidence: float  # 0.0-1.0 confidence in this evidence
    source: str  # Where this came from (memory, conversation, reasoning)


@dataclass
class EvidenceAnalysis:
    """Analysis of pro/con evidence for a decision."""

    decision_statement: str
    pro_evidence: list[Evidence]  # Arguments in favor
    con_evidence: list[Evidence]  # Arguments against
    uncertainty_factors: list[str]  # Unknowns that affect the decision

    def pro_score(self) -> float:
        """Calculate weighted pro score."""
        return sum(e.confidence for e in self.pro_evidence)

    def con_score(self) -> float:
        """Calculate weighted con score."""
        return sum(e.confidence for e in self.con_evidence)

    def evidence_strength(self) -> float:
        """Confidence in having enough evidence (0.0-1.0)."""
        total_evidence = len(self.pro_evidence) + len(self.con_evidence)
        if total_evidence == 0:
            return 0.0
        if total_evidence >= 5:
            return 1.0
        return total_evidence / 5.0

    def evidence_ratio(self) -> float:
        """Ratio of pro to con evidence (-1.0 to 1.0)."""
        pro = self.pro_score()
        con = self.con_score()
        total = pro + con
        if total == 0:
            return 0.0
        return (pro - con) / total


@dataclass
class DecisionRecommendation:
    """A structured recommendation for a decision."""

    decision_statement: str
    recommendation_type: RecommendationType
    confidence: float  # 0.0-1.0 overall confidence
    summary: str  # Brief summary of reasoning
    key_considerations: list[str]  # Main factors driving recommendation
    next_steps: list[str]  # Actions if decision is made
    monitoring_plan: list[str]  # How to track if decision was good
    reversibility_note: str  # Note on how to undo if needed

    def is_actionable(self) -> bool:
        """Should the user act on this recommendation?"""
        actionable_types = {RecommendationType.STRONGLY_RECOMMEND, RecommendationType.RECOMMEND}
        return self.recommendation_type in actionable_types


class DecisionRecommender:
    """Synthesizes decision recommendations using stakes and evidence."""

    def __init__(self, gateway: ModelGateway, rag_service: RAGService):
        """Initialize recommender.

        Args:
            gateway: Model gateway for LLM generation
            rag_service: RAG service for augmented reasoning with memories
        """
        self.gateway = gateway
        self.rag_service = rag_service

    def recommend(
        self,
        decision_statement: str,
        stakes_assessment: StakesAssessment,
        evidence_analysis: EvidenceAnalysis,
        user_id: int,
        user_preferences: Optional[dict] = None,
    ) -> DecisionRecommendation:
        """Generate a recommendation for a decision.

        Args:
            decision_statement: The decision to recommend on
            stakes_assessment: Assessment of decision stakes
            evidence_analysis: Analysis of pro/con evidence
            user_id: User ID for memory context
            user_preferences: Optional user preferences (e.g., risk tolerance)

        Returns:
            DecisionRecommendation with structured reasoning
        """
        # Build reasoning context
        reversibility_text = stakes_assessment.reversibility.value.replace("_", " ").title()
        net_impact = stakes_assessment.net_impact_score()

        benefits_text = "\n".join(
            f"• {s.description} (impact: {s.impact_level.value}, prob: {s.probability:.1%})"
            for s in stakes_assessment.benefits
        )

        risks_text = "\n".join(
            f"• {s.description} (impact: {s.impact_level.value}, prob: {s.probability:.1%})"
            for s in stakes_assessment.risks
        )

        pro_text = "\n".join(f"• {e.argument} (confidence: {e.confidence:.1%})" for e in evidence_analysis.pro_evidence)

        con_text = "\n".join(f"• {e.argument} (confidence: {e.confidence:.1%})" for e in evidence_analysis.con_evidence)

        uncertainties_text = "\n".join(f"• {u}" for u in evidence_analysis.uncertainty_factors)

        # Create prompt for LLM-based reasoning
        prompt = f"""Given the following decision analysis, provide a structured recommendation:

DECISION: {decision_statement}

STAKES ASSESSMENT:
Net Impact Score: {net_impact:.2f} (positive means benefits > risks)
Reversibility: {reversibility_text} - {stakes_assessment.reversibility_explanation}

BENEFITS:
{benefits_text if benefits_text else "None identified"}

RISKS:
{risks_text if risks_text else "None identified"}

EVIDENCE FOR:
{pro_text if pro_text else "None provided"}

EVIDENCE AGAINST:
{con_text if con_text else "None provided"}

UNCERTAINTIES:
{uncertainties_text if uncertainties_text else "None identified"}

Based on this analysis, provide your recommendation in JSON format:
{{
  "recommendation_type": "strongly_recommend|recommend|neutral|caution|strongly_caution",
  "confidence": 0.0-1.0,
  "summary": "2-3 sentence summary of reasoning",
  "key_considerations": ["important factor 1", "important factor 2"],
  "next_steps": ["action 1 if decided yes", "action 2"],
  "monitoring_plan": ["how to track outcome 1", "how to track outcome 2"],
  "reversibility_note": "How to undo if this doesn't work out"
}}"""

        # Use RAG for augmented reasoning with user memories
        recommendation_json = self.rag_service.generate_response(
            prompt, user_id=user_id, context_window=3, memory_citations=False
        )

        # Parse JSON response (simplified - in production would have error handling)
        import json

        try:
            rec_data = json.loads(recommendation_json)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            rec_data = self._fallback_recommendation(stakes_assessment, evidence_analysis)

        return DecisionRecommendation(
            decision_statement=decision_statement,
            recommendation_type=RecommendationType(rec_data.get("recommendation_type", "neutral")),
            confidence=float(rec_data.get("confidence", 0.5)),
            summary=rec_data.get("summary", "Insufficient evidence for strong recommendation"),
            key_considerations=rec_data.get("key_considerations", []),
            next_steps=rec_data.get("next_steps", []),
            monitoring_plan=rec_data.get("monitoring_plan", []),
            reversibility_note=rec_data.get("reversibility_note", "Review reversibility assessment"),
        )

    def _fallback_recommendation(
        self, stakes_assessment: StakesAssessment, evidence_analysis: EvidenceAnalysis
    ) -> dict:
        """Generate a fallback recommendation using heuristics."""
        net_impact = stakes_assessment.net_impact_score()
        evidence_ratio = evidence_analysis.evidence_ratio()
        evidence_strength = evidence_analysis.evidence_strength()

        # Simple heuristic: combine impact, evidence, and reversibility
        if net_impact > 3 and evidence_ratio > 0.3 and stakes_assessment.is_low_risk():
            rec_type = "strongly_recommend"
            confidence = min(0.95, 0.7 + evidence_strength * 0.25)
        elif net_impact > 0 and evidence_ratio > -0.2:
            rec_type = "recommend"
            confidence = min(0.85, 0.5 + evidence_strength * 0.35)
        elif net_impact < -3 and evidence_ratio < -0.3:
            rec_type = "strongly_caution"
            confidence = min(0.95, 0.7 + evidence_strength * 0.25)
        elif net_impact < 0 and evidence_ratio < 0.2:
            rec_type = "caution"
            confidence = min(0.85, 0.5 + evidence_strength * 0.35)
        else:
            rec_type = "neutral"
            confidence = 0.5 + evidence_strength * 0.3

        return {
            "recommendation_type": rec_type,
            "confidence": confidence,
            "summary": f"Based on net impact score of {net_impact:.2f} and evidence analysis.",
            "key_considerations": [
                "Review stakes assessment carefully",
                "Consider reversibility before committing",
            ],
            "next_steps": ["Discuss with trusted advisor", "Set clear success metrics"],
            "monitoring_plan": ["Track outcomes weekly", "Reassess if circumstances change"],
            "reversibility_note": f"Reversibility level: {stakes_assessment.reversibility.value}",
        }
