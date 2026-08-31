"""Agent — composes QuestionPolicy, RetrievalPipeline, DecisionEngine, Inference.

Flow:
    POST /chat
        ↓ Agent
        ↓ QuestionPolicy (structured observations → deterministic ASK/ANSWER)
        ↓ Memory Retrieval (single path: RetrievalPipeline)
        ↓ Decision Engine (when needed, assessment-based)
        ↓ Inference Gateway
        ↓ Response (with provenance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from personal_agent.decision.question_policy import QuestionAnalyzer, QuestionPolicy, PolicyAction
from personal_agent.decision.engine import AssessmentEngine
from personal_agent.memory.retrieval import RetrievalPipeline
from personal_agent.memory.rag import RAGService
from personal_agent.inference.gateway import ModelGateway
from personal_agent.agent.state import AgentState
from personal_agent.agent.prompts import build_chat_prompt


def _is_decision_query(text: str) -> bool:
    low = text.lower()
    triggers = ["should i", "should we", "decision", "choose between", "quit", "accept", "offer", "which language", "what should i focus"]
    return any(t in low for t in triggers)


@dataclass
class AgentResponse:
    response: str
    action: str  # ANSWER, ASK, CLARIFY
    memories_used: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    question_to_ask: Optional[str] = None
    assumptions: list[str] = field(default_factory=list)
    assessment: Optional[dict] = None
    retrieval_debug: Optional[dict] = None


class Agent:
    def __init__(
        self,
        gateway: ModelGateway | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        question_analyzer: QuestionAnalyzer | None = None,
        question_policy: QuestionPolicy | None = None,
        assessment_engine: AssessmentEngine | None = None,
        rag_service: RAGService | None = None,
    ):
        from personal_agent.inference.gateway import get_default_gateway

        self.gateway = gateway or get_default_gateway()
        self.retrieval = retrieval_pipeline or RetrievalPipeline()
        # RAGService now wraps the same single pipeline
        self.rag = rag_service or RAGService(gateway=self.gateway, pipeline=self.retrieval)
        self.q_analyzer = question_analyzer or QuestionAnalyzer(gateway=self.gateway)
        self.q_policy = question_policy or QuestionPolicy()
        self.assessment_engine = assessment_engine or AssessmentEngine(gateway=self.gateway)

    def chat(
        self,
        db: Session,
        user_id: int,
        message: str,
        conversation_id: int | None = None,
        allow_sensitive: bool = False,
    ) -> AgentResponse:
        state = AgentState(user_id=user_id, conversation_id=conversation_id)

        # 1) Question Policy — LLM proposes observations, policy decides
        # First, do a lightweight retrieval to give analyzer context about what's available
        # (so it can mark available_in_memory correctly)
        prelim = self.retrieval.run(db, user_id, message, top_k=3, allow_sensitive=allow_sensitive)
        prelim_contents = [r.memory.content for r in prelim.ranked]
        state.trace("retrieval_prelim", {"candidates": len(prelim.candidates), "ranked": len(prelim.ranked)})

        try:
            analysis = self.q_analyzer.analyze_with_llm(message, memories=[type("M", (), {"content": c})() for c in prelim_contents])
        except Exception:
            # Fallback heuristic analysis for offline/tests
            from personal_agent.decision.question_policy import QuestionAnalysis, StakesLevel, AmbiguityLevel

            analysis = QuestionAnalysis(query=message, goal=message, stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW, missing_information=[])

        decision = self.q_policy.decide(analysis)
        state.trace("question_policy", {"action": decision.action.value, "reason": decision.reason, "stakes": analysis.stakes.value, "ambiguity": analysis.ambiguity.value})

        if decision.action in (PolicyAction.ASK, PolicyAction.CLARIFY):
            return AgentResponse(
                response=decision.question_to_ask or "Could you share more detail?",
                action=decision.action.value,
                memories_used=[],
                provenance=state.provenance,
                question_to_ask=decision.question_to_ask,
                assumptions=decision.assumptions,
                assessment=None,
                retrieval_debug={"prelim": prelim.ranked[0].debug if prelim.ranked else {}},
            )

        # 2) Memory Retrieval — single path (already have prelim, but re-run with full top_k for answer)
        retrieval = self.retrieval.run(db, user_id, message, top_k=5, allow_sensitive=allow_sensitive)
        state.trace("retrieval", {"ranked": len(retrieval.ranked), "filtered_out": len(retrieval.filtered.filtered_out), "temporal_intent": retrieval.understanding.temporal_intent})
        memories_used = [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "type": r.memory.type,
                "status": r.memory.status,
                "sensitivity": r.memory.sensitivity,
                "source_type": r.memory.sources[0].source_type if r.memory.sources else None,
                "final_score": r.final_score,
                "debug": r.debug,
            }
            for r in retrieval.ranked
        ]

        # 3) Decision Engine (when needed)
        assessment_dict = None
        assessment_context = ""
        if _is_decision_query(message):
            bundle = self.assessment_engine.build_evidence_bundle(
                decision_statement=message,
                retrieved_memories=[r.memory for r in retrieval.ranked],
                unknowns=[m.description for m in analysis.missing_information],
                options=[],
            )
            assessment = self.assessment_engine.assess(message, bundle=bundle)
            assessment_dict = {
                "recommendation": assessment.recommendation,
                "assessment": assessment.assessment.value if hasattr(assessment.assessment, "value") else str(assessment.assessment),
                "evidence_strength": assessment.evidence_strength.value if hasattr(assessment.evidence_strength, "value") else str(assessment.evidence_strength),
                "key_factors": assessment.key_factors,
                "uncertainties": assessment.uncertainties,
                "assumptions": assessment.assumptions,
                "alternatives": assessment.alternatives,
                "reasons_for": assessment.reasons_for,
                "reasons_against": assessment.reasons_against,
                "what_would_change": assessment.what_would_change,
                "summary": assessment.summary,
            }
            assessment_context = f"Decision assessment: {assessment.assessment} (evidence {assessment.evidence_strength}); Key factors: {assessment.key_factors}; Uncertainties: {assessment.uncertainties}; What would change: {assessment.what_would_change}"
            state.trace("decision", assessment_dict)

        # 4) Inference — with retrieved context + assessment
        memory_context = self.retrieval.builder.build(retrieval.ranked, top_k=5) if hasattr(self.retrieval, "builder") else "\n".join([f"- {r.memory.content}" for r in retrieval.ranked])
        # Use RAGService's gateway path but with our single retrieval context
        prompt_messages = build_chat_prompt(message, memory_context, assessment_context)
        response_text = self.gateway.generate(prompt_messages, temperature=0.7)
        state.trace("inference", {"model": getattr(self.gateway, "provider", None) and getattr(self.gateway.provider, "default_model", "unknown")})

        return AgentResponse(
            response=response_text,
            action=PolicyAction.ANSWER.value,
            memories_used=memories_used,
            provenance=state.provenance,
            assumptions=decision.assumptions,
            assessment=assessment_dict,
            retrieval_debug={
                "candidates": len(retrieval.candidates),
                "filtered_out": len(retrieval.filtered.filtered_out),
                "ranked": len(retrieval.ranked),
                "temporal_intent": retrieval.understanding.temporal_intent,
            },
        )
