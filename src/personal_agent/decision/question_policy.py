"""M6.8 — Question Policy: Answer vs Ask

Core principle — same as memory:

    LLM proposes structured observations
        ↓
    deterministic policy
        ↓
    action (ANSWER / ASK / CLARIFY)

The LLM NEVER directly decides whether to ask.

Flow::

                    User request
                         ↓
                  What is the goal?
                         ↓
                 How high are stakes?
                         ↓
               Is information missing?
                         ↓
        Would missing information change
              the recommendation?
                  /             \\
                NO              YES
                ↓                ↓
             ANSWER       Is asking worth
                           the user effort?
                              /     \\
                            NO       YES
                            ↓         ↓
                         ANSWER     ASK
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StakesLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AmbiguityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InformationImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyAction(str, Enum):
    ANSWER = "ANSWER"
    ASK = "ASK"
    CLARIFY = "CLARIFY"


@dataclass
class MissingInformation:
    field: str
    description: str
    impact: InformationImpact | str = InformationImpact.MEDIUM
    effort: UserEffort | str = UserEffort.LOW
    available_in_memory: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.impact, str):
            try:
                self.impact = InformationImpact(self.impact.lower())
            except ValueError:
                self.impact = InformationImpact.MEDIUM
        if isinstance(self.effort, str):
            try:
                self.effort = UserEffort(self.effort.lower())
            except ValueError:
                self.effort = UserEffort.LOW


@dataclass
class QuestionAnalysis:
    """Structured observations produced by LLM (or test harness)."""

    query: str
    goal: str = ""
    stakes: StakesLevel | str = StakesLevel.LOW
    ambiguity: AmbiguityLevel | str = AmbiguityLevel.LOW
    missing_information: list[MissingInformation] = field(default_factory=list)
    reversibility: str = "unknown"  # fully_reversible / partially_reversible / irreversible
    assumptions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.stakes, str):
            try:
                self.stakes = StakesLevel(self.stakes.lower())
            except ValueError:
                self.stakes = StakesLevel.LOW
        if isinstance(self.ambiguity, str):
            try:
                self.ambiguity = AmbiguityLevel(self.ambiguity.lower())
            except ValueError:
                self.ambiguity = AmbiguityLevel.LOW


@dataclass
class PolicyDecision:
    action: PolicyAction
    reason: str
    question_to_ask: Optional[str] = None
    assumptions: list[str] = field(default_factory=list)
    relevant_missing: list[MissingInformation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic policy engine
# ---------------------------------------------------------------------------

_STAKES_WEIGHT = {
    StakesLevel.LOW: 1.0,
    StakesLevel.MEDIUM: 2.5,
    StakesLevel.HIGH: 5.0,
    StakesLevel.CRITICAL: 10.0,
}

_IMPACT_WEIGHT = {
    InformationImpact.LOW: 1.0,
    InformationImpact.MEDIUM: 2.5,
    InformationImpact.HIGH: 5.0,
}

_EFFORT_WEIGHT = {
    UserEffort.LOW: 0.5,
    UserEffort.MEDIUM: 1.0,
    UserEffort.HIGH: 2.0,
}


class QuestionPolicy:
    """Deterministic Answer vs Ask policy."""

    def decide(self, analysis: QuestionAnalysis) -> PolicyDecision:
        # Filter to truly missing (not already in memory)
        relevant = [mi for mi in analysis.missing_information if not mi.available_in_memory]

        # No relevant missing -> ANSWER (with assumptions if ambiguous)
        if not relevant:
            if analysis.ambiguity == AmbiguityLevel.HIGH:
                return PolicyDecision(
                    action=PolicyAction.ANSWER,
                    reason="No material missing information; answering with explicit assumptions",
                    assumptions=analysis.assumptions or ["Assuming most common interpretation"],
                    relevant_missing=[],
                )
            return PolicyDecision(
                action=PolicyAction.ANSWER,
                reason="No missing information would change recommendation; answering directly",
                relevant_missing=[],
            )

        # Find highest-impact missing
        # Sort by impact weight descending
        relevant_sorted = sorted(relevant, key=lambda mi: _IMPACT_WEIGHT[mi.impact], reverse=True)
        top = relevant_sorted[0]
        stakes_w = _STAKES_WEIGHT[analysis.stakes]
        impact_w = _IMPACT_WEIGHT[top.impact]
        effort_w = _EFFORT_WEIGHT[top.effort]

        # Expected value of information (heuristic)
        evoi = stakes_w * impact_w - effort_w * 1.5
        # Adjust for reversibility: irreversible amplifies stakes
        if analysis.reversibility == "irreversible":
            evoi *= 1.5

        # Decision tree
        # 1. Would missing info change recommendation? (impact HIGH/MEDIUM)
        if top.impact == InformationImpact.LOW:
            # Low impact even if ambiguous -> answer with assumptions
            return PolicyDecision(
                action=PolicyAction.ANSWER,
                reason=f"Missing '{top.field}' is low impact; answering with assumptions",
                assumptions=[f"Assuming {top.description} is not decisive"],
                relevant_missing=relevant,
            )

        # 2. Is asking worth user effort? (EVOI)
        # Threshold tuned so:
        # - LOW stakes + LOW impact => ANSWER
        # - HIGH stakes + HIGH impact + LOW effort => ASK
        # - MEDIUM stakes + MEDIUM impact + MEDIUM effort => borderline, prefer ASK if irreversible
        if evoi >= 4.0:
            return PolicyDecision(
                action=PolicyAction.ASK,
                reason=f"High stakes ({analysis.stakes.value}) and missing '{top.field}' (impact={top.impact.value}) would change recommendation; asking is worth effort ({top.effort.value})",
                question_to_ask=self._form_question(top),
                relevant_missing=relevant,
            )

        # Low EVOI => answer with caveats
        if top.impact == InformationImpact.MEDIUM and stakes_w < 3:
            return PolicyDecision(
                action=PolicyAction.ANSWER,
                reason="Missing information is medium impact but stakes/effort tradeoff favors answering with caveats",
                assumptions=[f"Assuming {top.description}"],
                relevant_missing=relevant,
            )

        # Default: if ambiguous query (many interpretations) but not high impact, clarify
        if analysis.ambiguity == AmbiguityLevel.HIGH and top.impact == InformationImpact.MEDIUM:
            return PolicyDecision(
                action=PolicyAction.CLARIFY,
                reason=f"Query is ambiguous; clarifying '{top.field}' would help",
                question_to_ask=self._form_question(top),
                relevant_missing=relevant,
            )

        # Otherwise ask if EVOI moderate
        if evoi >= 2.0:
            return PolicyDecision(
                action=PolicyAction.ASK,
                reason=f"Missing '{top.field}' could change recommendation; worth asking",
                question_to_ask=self._form_question(top),
                relevant_missing=relevant,
            )

        return PolicyDecision(
            action=PolicyAction.ANSWER,
            reason="Missing information would not materially change recommendation; answering",
            relevant_missing=relevant,
        )

    @staticmethod
    def _form_question(mi: MissingInformation) -> str:
        # Produce smallest useful question
        return f"Could you share more about {mi.field}? ({mi.description})"


# ---------------------------------------------------------------------------
# LLM analyzer (structured observations only — does NOT decide)
# ---------------------------------------------------------------------------

class QuestionAnalyzer:
    """Calls LLM to produce QuestionAnalysis; policy decides after."""

    def __init__(self, gateway=None):
        self.gateway = gateway

    def analyze_with_llm(self, query: str, memories: list | None = None, stakes_hint: str | None = None) -> QuestionAnalysis:
        if self.gateway is None:
            from personal_agent.inference.gateway import get_default_gateway

            self.gateway = get_default_gateway()

        mem_context = ""
        if memories:
            mem_context = "\n".join(f"- {m.content}" for m in memories[:3])

        prompt = f"""You are a decision-analysis assistant. Given a user request, produce structured observations ONLY — do NOT decide to answer or ask.

User request: {query}

Relevant memories:
{mem_context or "(none)"}

Return JSON with:
{{
  "goal": "what user wants to achieve",
  "stakes": "low|medium|high|critical",
  "ambiguity": "low|medium|high",
  "reversibility": "fully_reversible|partially_reversible|irreversible",
  "assumptions": ["assumption 1"],
  "missing_information": [
    {{"field": "financial_dependency", "description": "whether user depends on internship income", "impact": "high|medium|low", "effort": "low|medium|high", "available_in_memory": false}}
  ]
}}

Rules:
- impact = would answer change if we knew this? high = would flip recommendation
- effort = how hard for user to answer (low = one sentence)
- available_in_memory = true if memories already contain this
Only return valid JSON."""

        messages = [
            {"role": "system", "content": "You produce structured observations for a policy engine. Never decide to answer or ask."},
            {"role": "user", "content": prompt},
        ]
        raw = self.gateway.generate(messages, temperature=0.2)
        return self._parse(raw, query)

    @staticmethod
    def _parse(raw: str, query: str) -> QuestionAnalysis:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("no json")
            data = json.loads(raw[start:end])
            missing = []
            for mi in data.get("missing_information", []):
                missing.append(
                    MissingInformation(
                        field=mi.get("field", "unknown"),
                        description=mi.get("description", ""),
                        impact=mi.get("impact", "medium"),
                        effort=mi.get("effort", "low"),
                        available_in_memory=bool(mi.get("available_in_memory", False)),
                    )
                )
            return QuestionAnalysis(
                query=query,
                goal=data.get("goal", ""),
                stakes=data.get("stakes", "low"),
                ambiguity=data.get("ambiguity", "low"),
                missing_information=missing,
                reversibility=data.get("reversibility", "unknown"),
                assumptions=data.get("assumptions", []),
            )
        except Exception:
            # Fallback: treat as low stakes ambiguous
            return QuestionAnalysis(query=query, stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW)
