"""M6.9 — Decision engine v2: assessment-based, not numeric confidence."""

import unittest
from unittest.mock import MagicMock

from personal_agent.decision.engine import AssessmentEngine, Evidence, EvidenceAnalysis, ImpactLevel, Reversibility, StakesAssessment, Stake
from personal_agent.decision.models import Assessment, DecisionAssessment, EvidenceStrength


class TestAssessmentEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_gw = MagicMock()
        self.engine = AssessmentEngine(gateway=self.mock_gw)

    def _stakes(self, reversible=True):
        return StakesAssessment(
            decision_statement="Should I accept internship?",
            benefits=[Stake("Learning", ImpactLevel.MEDIUM, 0.8, 0.9)],
            risks=[Stake("Low pay", ImpactLevel.LOW, 0.5, 0.7)],
            reversibility=Reversibility.PARTIALLY_REVERSIBLE if reversible else Reversibility.IRREVERSIBLE,
            reversibility_explanation="can leave",
        )

    def _evidence(self, pro=2, con=1):
        return EvidenceAnalysis(
            decision_statement="Should I accept internship?",
            pro_evidence=[Evidence(f"pro {i}", True, 0.9, "memory") for i in range(pro)],
            con_evidence=[Evidence(f"con {i}", False, 0.8, "memory") for i in range(con)],
            uncertainty_factors=["mentorship quality"],
        )

    def test_assessment_no_numeric_confidence(self) -> None:
        """LLM proposes assessment JSON, engine validates — no confidence field."""
        self.mock_gw.generate.return_value = """{
          "decision_statement": "Should I accept internship?",
          "recommendation": "Accept if mentorship is solid",
          "assessment": "cautiously_positive",
          "evidence_strength": "moderate",
          "key_factors": ["learning opportunity", "backend alignment"],
          "uncertainties": ["mentorship quality"],
          "assumptions": ["mentorship is adequate"],
          "alternatives": ["Stay at current internship"],
          "reasons_for": ["learning"],
          "reasons_against": ["low pay"],
          "what_would_change": ["If mentorship is poor, option B preferable"],
          "summary": "3 factors support, 1 main uncertainty"
        }"""
        stakes = self._stakes()
        ev = self._evidence()
        result = self.engine.assess("Should I accept internship?", stakes=stakes, evidence=ev)
        self.assertIsInstance(result, DecisionAssessment)
        self.assertEqual(result.assessment, Assessment.CAUTIOUSLY_POSITIVE)
        self.assertEqual(result.evidence_strength, EvidenceStrength.MODERATE)
        self.assertIn("mentorship quality", result.uncertainties)
        self.assertIn("If mentorship is poor", result.what_would_change[0])
        # must not have numeric confidence
        self.assertFalse(hasattr(result, "confidence"))
        self.assertNotIn("confidence", result.__dict__)

    def test_evidence_strength_separate_from_assessment(self) -> None:
        """Evidence strength and assessment are distinct: moderate evidence can still be cautiously_positive."""
        self.mock_gw.generate.return_value = """{
          "recommendation": "Cautiously positive",
          "assessment": "cautiously_positive",
          "evidence_strength": "moderate",
          "key_factors": ["3 factors support"],
          "uncertainties": ["Whether opportunity provides meaningful mentorship"],
          "assumptions": [],
          "alternatives": [],
          "reasons_for": ["a"],
          "reasons_against": ["b"],
          "what_would_change": ["If mentorship is poor, option B becomes preferable"],
          "summary": "moderate evidence, cautiously positive"
        }"""
        result = self.engine.assess("Should I accept?", stakes=self._stakes(), evidence=self._evidence())
        self.assertEqual(result.evidence_strength, EvidenceStrength.MODERATE)
        self.assertEqual(result.assessment, Assessment.CAUTIOUSLY_POSITIVE)
        # they are not conflated
        self.assertNotEqual(result.evidence_strength.value, result.assessment.value)

    def test_validation_rejects_numeric_confidence(self) -> None:
        """If LLM tries to return confidence 0.82, validator ignores it and uses assessment."""
        self.mock_gw.generate.return_value = """{
          "recommendation": "Go",
          "assessment": "positive",
          "evidence_strength": "strong",
          "confidence": 0.82,
          "key_factors": [],
          "uncertainties": [],
          "assumptions": [],
          "alternatives": [],
          "reasons_for": [],
          "reasons_against": [],
          "what_would_change": [],
          "summary": ""
        }"""
        result = self.engine.assess("Test", evidence=self._evidence(pro=3, con=0))
        # confidence should be ignored
        self.assertFalse(hasattr(result, "confidence"))
        self.assertEqual(result.assessment, Assessment.POSITIVE)

    def test_fallback_is_still_qualitative(self) -> None:
        """When LLM JSON fails, fallback is qualitative."""
        self.mock_gw.generate.return_value = "not json"
        result = self.engine.assess("Fallback test", stakes=self._stakes(), evidence=self._evidence())
        self.assertIsInstance(result.evidence_strength, EvidenceStrength)
        self.assertIsInstance(result.assessment, Assessment)
        self.assertFalse(hasattr(result, "confidence"))

    def test_evidence_bundle_integration(self) -> None:
        """Evidence bundle from retrieved memories flows into assessment."""
        bundle_stakes = self._stakes()
        # simulate retrieved personal context
        retrieved = [MagicMock(content="Prefers backend development"), MagicMock(content="Values mentorship")]
        # Build bundle via helper
        bundle = self.engine.build_evidence_bundle(
            "Should I accept backend internship?",
            stakes=bundle_stakes,
            evidence=self._evidence(),
            retrieved_memories=retrieved,
            unknowns=["mentorship quality"],
            options=["Accept", "Decline"],
        )
        self.assertIn("Prefers backend", bundle.retrieved_context[0])
        self.mock_gw.generate.return_value = """{
          "recommendation": "Accept backend role",
          "assessment": "positive",
          "evidence_strength": "moderate",
          "key_factors": ["backend alignment"],
          "uncertainties": ["mentorship quality"],
          "assumptions": ["mentorship adequate"],
          "alternatives": ["Decline"],
          "reasons_for": ["backend fit"],
          "reasons_against": ["low pay"],
          "what_would_change": ["Poor mentorship flips to decline"],
          "summary": ""
        }"""
        result = self.engine.assess("Should I accept backend internship?", bundle=bundle)
        self.assertEqual(result.assessment, Assessment.POSITIVE)
        self.assertIn("backend", result.reasons_for[0].lower())

    def test_legacy_engine_still_works(self) -> None:
        """Old DecisionRecommender still returns confidence for backward compat."""
        from personal_agent.decision.engine import DecisionRecommender
        from personal_agent.memory.rag import RAGService

        mock_rag = MagicMock()
        mock_rag.generate_response.return_value = """{
          "recommendation_type": "recommend",
          "confidence": 0.78,
          "summary": "old",
          "key_considerations": [],
          "next_steps": [],
          "monitoring_plan": [],
          "reversibility_note": ""
        }"""
        rec = DecisionRecommender(gateway=self.mock_gw, rag_service=mock_rag)
        stakes = self._stakes()
        ev = self._evidence()
        old = rec.recommend("Should I accept?", stakes, ev, user_id=1)
        self.assertAlmostEqual(old.confidence, 0.78)  # legacy still has numeric
        # new engine does not


if __name__ == "__main__":
    unittest.main()
