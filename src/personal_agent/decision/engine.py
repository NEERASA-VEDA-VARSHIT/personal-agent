"""
M6: Decision Support Engine (legacy) + M6.9 Assessment Engine (v2).

M6.9 replaces numeric confidence (e.g. 78%) with qualitative assessment:

    Evidence strength: Moderate
    Assessment: cautiously_positive
    Main uncertainty: mentorship quality
    What would change: if mentorship is poor, option B preferable

Flow for v2 (LLM proposes → application validates):
    User decision
        ↓ QuestionPolicy → enough info?
        ↓ Memory Retrieval → EvidenceBundle (known facts / retrieved context / unknowns / options / tradeoffs)
        ↓ Decision Analysis (for / against)
        ↓ LLM structured output
        ↓ Validation layer
        ↓ Final DecisionAssessment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from personal_agent.decision.models import Assessment, DecisionAssessment, EvidenceBundle, EvidenceStrength, RecommendationStrength
from personal_agent.inference.gateway import ModelGateway
from app.memory.rag import RAGService


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

        # Legacy path: use RAG service if mocked (tests), otherwise use gateway directly.
        # Single retrieval path is enforced at Agent level; decision itself does not do retrieval.
        try:
            rag_result = self.rag_service.generate_response(
                prompt, user_id=user_id, context_window=3, memory_citations=False
            )
            # rag_service mock may return MagicMock if not configured; fallback to gateway
            if isinstance(rag_result, str):
                recommendation_json = rag_result
            elif isinstance(rag_result, dict) and "response" in rag_result:
                recommendation_json = rag_result["response"]
            else:
                raise ValueError("unexpected rag result type")
        except Exception:
            recommendation_json = self.gateway.generate(
                [{"role": "user", "content": prompt}], temperature=0.2
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


# ---------------------------------------------------------------------------
# M6.9 — Assessment Engine v2 (LLM proposes → application validates)
# ---------------------------------------------------------------------------

_ALLOWED_ASSESSMENTS = {a.value for a in Assessment}
_ALLOWED_EVIDENCE = {e.value for e in EvidenceStrength}


def _validate_assessment(raw: dict, fallback_statement: str) -> DecisionAssessment:
    """Validate LLM JSON before it becomes application state."""
    assessment = raw.get("assessment", "neutral")
    if assessment not in _ALLOWED_ASSESSMENTS:
        assessment = "cautiously_positive" if raw.get("evidence_strength") == "moderate" else "neutral"
    evidence = raw.get("evidence_strength", "moderate")
    if evidence not in _ALLOWED_EVIDENCE:
        evidence = "moderate"

    def _list(key: str) -> list[str]:
        v = raw.get(key, [])
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    return DecisionAssessment(
        decision_statement=raw.get("decision_statement", fallback_statement),
        recommendation=raw.get("recommendation", raw.get("summary", "")) or "No recommendation",
        assessment=assessment,
        evidence_strength=evidence,
        key_factors=_list("key_factors"),
        uncertainties=_list("uncertainties"),
        assumptions=_list("assumptions"),
        alternatives=_list("alternatives"),
        reasons_for=_list("reasons_for"),
        reasons_against=_list("reasons_against"),
        what_would_change=_list("what_would_change") or _list("what_would_change_my_recommendation"),
        summary=raw.get("summary", raw.get("recommendation", "")) or "",
    )


def _fallback_assessment(
    statement: str,
    stakes: StakesAssessment | None,
    evidence: EvidenceAnalysis | None,
    bundle: EvidenceBundle | None,
) -> DecisionAssessment:
    """Heuristic fallback when LLM JSON fails — still qualitative, never numeric confidence."""
    # evidence strength from bundle / evidence analysis
    ev_strength = EvidenceStrength.MODERATE
    if evidence is not None:
        es = evidence.evidence_strength()
        if es >= 0.7:
            ev_strength = EvidenceStrength.STRONG
        elif es <= 0.3:
            ev_strength = EvidenceStrength.LIMITED
    elif bundle and len(bundle.for_factors) + len(bundle.against_factors) >= 4:
        ev_strength = EvidenceStrength.STRONG
    elif bundle and not bundle.for_factors and not bundle.against_factors:
        ev_strength = EvidenceStrength.LIMITED

    # assessment from stakes + evidence ratio
    assessment = Assessment.CAUTIOUSLY_POSITIVE
    if evidence is not None:
        ratio = evidence.evidence_ratio()
        if ratio > 0.5:
            assessment = Assessment.POSITIVE if ev_strength == EvidenceStrength.STRONG else Assessment.CAUTIOUSLY_POSITIVE
        elif ratio < -0.3:
            assessment = Assessment.CAUTIOUSLY_NEGATIVE
        elif abs(ratio) < 0.2:
            assessment = Assessment.NEUTRAL

    return DecisionAssessment(
        decision_statement=statement,
        recommendation=f"Based on {len(bundle.for_factors) if bundle else 0} supporting and {len(bundle.against_factors) if bundle else 0} opposing factors.",
        assessment=assessment,
        evidence_strength=ev_strength,
        key_factors=(bundle.for_factors[:2] if bundle else []) if bundle else [],
        uncertainties=evidence.uncertainty_factors if evidence else (bundle.unknowns if bundle else []),
        assumptions=bundle.assumptions if bundle and hasattr(bundle, "assumptions") else [],
        alternatives=bundle.options if bundle else [],
        reasons_for=bundle.for_factors if bundle else [],
        reasons_against=bundle.against_factors if bundle else [],
        what_would_change=["If key uncertainty resolves differently, reassess"] if bundle and bundle.unknowns else [],
        summary=f"Evidence {ev_strength.value}; assessment {assessment.value}",
    )


class AssessmentEngine:
    """Assessment-based decision support — does NOT emit numeric confidence.

    LLM proposes structured JSON, this class validates and decides what is safe.
    """

    def __init__(self, gateway: ModelGateway | None = None):
        from personal_agent.inference.gateway import get_default_gateway

        self.gateway = gateway or get_default_gateway()

    def build_evidence_bundle(
        self,
        decision_statement: str,
        stakes: StakesAssessment | None = None,
        evidence: EvidenceAnalysis | None = None,
        retrieved_memories: list | None = None,
        unknowns: list[str] | None = None,
        options: list[str] | None = None,
    ) -> EvidenceBundle:
        retrieved_context = []
        if retrieved_memories:
            for m in retrieved_memories:
                # Support both Memory objects and (Memory, score) tuples
                mem = m[0] if isinstance(m, (list, tuple)) else m
                if hasattr(mem, "content"):
                    retrieved_context.append(mem.content)
                elif isinstance(mem, str):
                    retrieved_context.append(mem)
        return EvidenceBundle(
            decision_statement=decision_statement,
            known_facts=[s.description for s in (stakes.benefits + stakes.risks) if stakes] if stakes else [],
            retrieved_context=retrieved_context,
            unknowns=unknowns or (evidence.uncertainty_factors if evidence else []),
            options=options or [],
            tradeoffs=[],
            for_factors=[e.argument for e in (evidence.pro_evidence if evidence else [])],
            against_factors=[e.argument for e in (evidence.con_evidence if evidence else [])],
        )

    def assess(
        self,
        decision_statement: str,
        stakes: StakesAssessment | None = None,
        evidence: EvidenceAnalysis | None = None,
        bundle: EvidenceBundle | None = None,
        retrieved_memories: list | None = None,
    ) -> DecisionAssessment:
        bundle = bundle or self.build_evidence_bundle(
            decision_statement, stakes=stakes, evidence=evidence, retrieved_memories=retrieved_memories
        )

        # Prompt for LLM — asks for assessment, not confidence
        stakes_text = ""
        if stakes:
            stakes_text = f"Stakes: {stakes.reversibility.value} ({stakes.reversibility_explanation}); net impact {stakes.net_impact_score():.1f}"

        prompt = f"""You are a decision-support assistant. Do NOT output numeric confidence.

Decision: {decision_statement}
{stakes_text}

Known facts: {bundle.known_facts or 'none'}
Retrieved personal context: {bundle.retrieved_context or 'none'}
Unknowns: {bundle.unknowns or 'none'}
Options: {bundle.options or 'none'}

Evidence FOR: {bundle.for_factors or 'none'}
Evidence AGAINST: {bundle.against_factors or 'none'}

Return ONLY JSON with this schema:
{{
  "decision_statement": "...",
  "recommendation": "1-2 sentence recommendation",
  "assessment": "strongly_positive|positive|cautiously_positive|neutral|cautiously_negative|negative|strongly_negative",
  "evidence_strength": "limited|moderate|strong",
  "key_factors": ["factor 1"],
  "uncertainties": ["uncertainty 1"],
  "assumptions": ["assumption 1"],
  "alternatives": ["alternative option"],
  "reasons_for": ["reason for"],
  "reasons_against": ["reason against"],
  "what_would_change": ["what would flip the assessment"],
  "summary": "brief summary"
}}
Rules: assessment must reflect evidence_strength, not overstate; list main uncertainty honestly.
"""

        try:
            raw_text = self.gateway.generate(
                [{"role": "user", "content": prompt}], temperature=0.2
            )
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("no json")
            data = json.loads(raw_text[start:end])
            return _validate_assessment(data, decision_statement)
        except Exception:
            return _fallback_assessment(decision_statement, stakes, evidence, bundle)

    def assess_from_bundle(self, bundle: EvidenceBundle) -> DecisionAssessment:
        return self.assess(bundle.decision_statement, bundle=bundle)
