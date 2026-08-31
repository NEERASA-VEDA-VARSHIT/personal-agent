"""Tests for M6 decision support engine."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from personal_agent.persistence.models import Base, Conversation, Message, User
from personal_agent.decision.engine import (
    DecisionRecommender,
    EvidenceAnalysis,
    Evidence,
    ImpactLevel,
    RecommendationType,
    Reversibility,
    Stake,
    StakesAssessment,
)
from personal_agent.memory.rag import RAGService


class TestStakesAssessment(unittest.TestCase):
    """Test stakes assessment logic."""

    def test_benefit_with_weighted_impact(self) -> None:
        """Test benefit stake calculation."""
        stake = Stake(
            description="Career advancement",
            impact_level=ImpactLevel.HIGH,
            probability=0.8,
            confidence=0.9,
        )
        # HIGH = 5.0, so: 5.0 * 0.8 * 0.9 = 3.6
        self.assertAlmostEqual(stake.weighted_impact(), 3.6)

    def test_risk_with_weighted_impact(self) -> None:
        """Test risk stake calculation."""
        stake = Stake(
            description="Time investment",
            impact_level=ImpactLevel.MEDIUM,
            probability=0.5,
            confidence=0.7,
        )
        # MEDIUM = 2.5, so: 2.5 * 0.5 * 0.7 = 0.875
        self.assertAlmostEqual(stake.weighted_impact(), 0.875)

    def test_critical_impact(self) -> None:
        """Test critical impact scoring."""
        stake = Stake(
            description="Life-changing decision",
            impact_level=ImpactLevel.CRITICAL,
            probability=0.9,
            confidence=0.95,
        )
        # CRITICAL = 10.0, so: 10.0 * 0.9 * 0.95 = 8.55
        self.assertAlmostEqual(stake.weighted_impact(), 8.55)

    def test_net_impact_positive(self) -> None:
        """Test positive net impact calculation."""
        assessment = StakesAssessment(
            decision_statement="Accept new job offer",
            benefits=[
                Stake(
                    description="Higher salary",
                    impact_level=ImpactLevel.HIGH,
                    probability=1.0,
                    confidence=1.0,
                ),  # 5.0
                Stake(
                    description="Career growth",
                    impact_level=ImpactLevel.HIGH,
                    probability=0.8,
                    confidence=0.9,
                ),  # 3.6
            ],
            risks=[
                Stake(
                    description="Relocation stress",
                    impact_level=ImpactLevel.LOW,
                    probability=0.5,
                    confidence=0.7,
                ),  # 0.35
            ],
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            reversibility_explanation="Can return to similar role if unhappy",
        )
        # Net: (5.0 + 3.6) - 0.35 = 8.25
        self.assertAlmostEqual(assessment.net_impact_score(), 8.25)

    def test_net_impact_negative(self) -> None:
        """Test negative net impact calculation."""
        assessment = StakesAssessment(
            decision_statement="Make risky investment",
            benefits=[
                Stake(
                    description="Potential wealth gain",
                    impact_level=ImpactLevel.MEDIUM,
                    probability=0.3,
                    confidence=0.6,
                ),  # 0.45
            ],
            risks=[
                Stake(
                    description="Total loss of investment",
                    impact_level=ImpactLevel.CRITICAL,
                    probability=0.6,
                    confidence=0.8,
                ),  # 4.8
            ],
            reversibility=Reversibility.IRREVERSIBLE,
            reversibility_explanation="Cannot recover lost funds",
        )
        # Net: 0.45 - 4.8 = -4.35
        self.assertAlmostEqual(assessment.net_impact_score(), -4.35)

    def test_is_low_risk_true(self) -> None:
        """Test low-risk assessment."""
        assessment = StakesAssessment(
            decision_statement="Try new learning path",
            benefits=[
                Stake(
                    description="New skills",
                    impact_level=ImpactLevel.MEDIUM,
                    probability=0.9,
                    confidence=0.85,
                )
            ],
            risks=[],
            reversibility=Reversibility.FULLY_REVERSIBLE,
            reversibility_explanation="Can stop anytime",
        )
        self.assertTrue(assessment.is_low_risk())

    def test_is_low_risk_false_irreversible(self) -> None:
        """Test that irreversible decisions are not low-risk."""
        assessment = StakesAssessment(
            decision_statement="Major surgery",
            benefits=[
                Stake(
                    description="Health improvement",
                    impact_level=ImpactLevel.HIGH,
                    probability=0.7,
                    confidence=0.8,
                )
            ],
            risks=[],
            reversibility=Reversibility.IRREVERSIBLE,
            reversibility_explanation="Cannot undo surgical change",
        )
        self.assertFalse(assessment.is_low_risk())

    def test_is_low_risk_false_negative_impact(self) -> None:
        """Test that negative net impact is not low-risk."""
        assessment = StakesAssessment(
            decision_statement="Bad deal",
            benefits=[
                Stake(
                    description="Small gain",
                    impact_level=ImpactLevel.LOW,
                    probability=0.5,
                    confidence=0.5,
                )
            ],
            risks=[
                Stake(
                    description="Major loss",
                    impact_level=ImpactLevel.CRITICAL,
                    probability=0.8,
                    confidence=0.9,
                )
            ],
            reversibility=Reversibility.FULLY_REVERSIBLE,
            reversibility_explanation="Somehow reversible",
        )
        self.assertFalse(assessment.is_low_risk())


class TestEvidenceAnalysis(unittest.TestCase):
    """Test evidence analysis logic."""

    def test_pro_score(self) -> None:
        """Test pro evidence scoring."""
        analysis = EvidenceAnalysis(
            decision_statement="Learn Rust",
            pro_evidence=[
                Evidence(
                    argument="Systems programming skill valuable",
                    supports_decision=True,
                    confidence=0.85,
                    source="market_research",
                ),
                Evidence(
                    argument="Matches career goals",
                    supports_decision=True,
                    confidence=0.9,
                    source="personal_memory",
                ),
            ],
            con_evidence=[],
            uncertainty_factors=[],
        )
        self.assertAlmostEqual(analysis.pro_score(), 1.75)

    def test_con_score(self) -> None:
        """Test con evidence scoring."""
        analysis = EvidenceAnalysis(
            decision_statement="Major career change",
            pro_evidence=[],
            con_evidence=[
                Evidence(
                    argument="Risk of job loss",
                    supports_decision=False,
                    confidence=0.7,
                    source="risk_assessment",
                ),
                Evidence(
                    argument="Significant time investment",
                    supports_decision=False,
                    confidence=0.6,
                    source="personal_reflection",
                ),
            ],
            uncertainty_factors=[],
        )
        self.assertAlmostEqual(analysis.con_score(), 1.3)

    def test_evidence_strength(self) -> None:
        """Test evidence strength calculation."""
        # Few pieces of evidence
        analysis1 = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[Evidence("arg1", True, 0.8, "src1")],
            con_evidence=[],
            uncertainty_factors=[],
        )
        self.assertAlmostEqual(analysis1.evidence_strength(), 0.2)

        # Moderate evidence
        analysis2 = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[
                Evidence("arg1", True, 0.8, "src1"),
                Evidence("arg2", True, 0.7, "src2"),
                Evidence("arg3", True, 0.9, "src3"),
            ],
            con_evidence=[Evidence("arg4", False, 0.6, "src4")],
            uncertainty_factors=[],
        )
        self.assertAlmostEqual(analysis2.evidence_strength(), 0.8)

        # Strong evidence
        analysis3 = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[
                Evidence(f"arg{i}", True, 0.8, f"src{i}") for i in range(4)
            ],
            con_evidence=[Evidence(f"arg{i}", False, 0.6, f"src{i}") for i in range(2)],
            uncertainty_factors=[],
        )
        self.assertEqual(analysis3.evidence_strength(), 1.0)

    def test_evidence_ratio_all_pro(self) -> None:
        """Test evidence ratio with only pro evidence."""
        analysis = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[
                Evidence("pro1", True, 0.8, "src"),
                Evidence("pro2", True, 0.7, "src"),
            ],
            con_evidence=[],
            uncertainty_factors=[],
        )
        # Ratio: (1.5 - 0) / 1.5 = 1.0
        self.assertAlmostEqual(analysis.evidence_ratio(), 1.0)

    def test_evidence_ratio_all_con(self) -> None:
        """Test evidence ratio with only con evidence."""
        analysis = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[],
            con_evidence=[
                Evidence("con1", False, 0.8, "src"),
                Evidence("con2", False, 0.6, "src"),
            ],
            uncertainty_factors=[],
        )
        # Ratio: (0 - 1.4) / 1.4 = -1.0
        self.assertAlmostEqual(analysis.evidence_ratio(), -1.0)

    def test_evidence_ratio_balanced(self) -> None:
        """Test evidence ratio with balanced pro/con."""
        analysis = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[
                Evidence("pro1", True, 0.8, "src"),
                Evidence("pro2", True, 0.7, "src"),
            ],
            con_evidence=[
                Evidence("con1", False, 0.9, "src"),
            ],
            uncertainty_factors=[],
        )
        # Ratio: (1.5 - 0.9) / (1.5 + 0.9) = 0.6 / 2.4 ≈ 0.25
        self.assertAlmostEqual(analysis.evidence_ratio(), 0.25)

    def test_evidence_ratio_no_evidence(self) -> None:
        """Test evidence ratio with no evidence."""
        analysis = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[],
            con_evidence=[],
            uncertainty_factors=[],
        )
        self.assertEqual(analysis.evidence_ratio(), 0.0)


class TestDecisionRecommender(unittest.TestCase):
    """Test decision recommendation generation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create in-memory SQLite database for testing."""
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db_url = f"sqlite:///{cls.db_path}"
        cls.engine = create_engine(cls.db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test database."""
        cls.engine.dispose()
        os.close(cls.db_fd)
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    _user_counter = 0

    def setUp(self) -> None:
        """Create test session and mocks."""
        self.session: Session = self.SessionLocal()

        # Create user for tests
        user = User(username=f"test_user_{self._get_unique_id()}")
        self.session.add(user)
        self.session.flush()
        self.user_id = user.id

        # Create mock gateway and RAG service
        self.mock_gateway = MagicMock()
        self.mock_rag = MagicMock()
        self.recommender = DecisionRecommender(
            gateway=self.mock_gateway, rag_service=self.mock_rag
        )

    def _get_unique_id(self) -> int:
        """Get unique ID for test data."""
        TestDecisionRecommender._user_counter += 1
        return TestDecisionRecommender._user_counter

    def tearDown(self) -> None:
        """Clean up session."""
        self.session.rollback()
        self.session.close()

    def test_recommend_strongly_positive(self) -> None:
        """Test strongly positive recommendation."""
        stakes = StakesAssessment(
            decision_statement="Accept excellent job offer",
            benefits=[
                Stake(
                    "Significant salary increase",
                    ImpactLevel.HIGH,
                    1.0,
                    0.99,
                ),
                Stake("Career advancement", ImpactLevel.HIGH, 0.95, 0.9),
                Stake("Better work-life balance", ImpactLevel.MEDIUM, 0.8, 0.85),
            ],
            risks=[
                Stake("Relocation", ImpactLevel.LOW, 0.5, 0.7),
            ],
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            reversibility_explanation="Can find another job if needed",
        )

        evidence = EvidenceAnalysis(
            decision_statement="Accept excellent job offer",
            pro_evidence=[
                Evidence("Company is industry leader", True, 0.95, "research"),
                Evidence("Team reputation excellent", True, 0.9, "interviews"),
                Evidence("Aligns with career goals", True, 0.92, "memory"),
            ],
            con_evidence=[],
            uncertainty_factors=[],
        )

        # Mock RAG response
        self.mock_rag.generate_response.return_value = """{
            "recommendation_type": "strongly_recommend",
            "confidence": 0.95,
            "summary": "This is an exceptional opportunity with clear benefits and manageable reversibility.",
            "key_considerations": ["Excellent company culture", "Strong career growth"],
            "next_steps": ["Accept offer", "Plan relocation"],
            "monitoring_plan": ["Track job satisfaction", "Review after 6 months"],
            "reversibility_note": "Can transition to another role if needed"
        }"""

        rec = self.recommender.recommend(
            decision_statement="Accept excellent job offer",
            stakes_assessment=stakes,
            evidence_analysis=evidence,
            user_id=self.user_id,
        )

        self.assertEqual(rec.recommendation_type, RecommendationType.STRONGLY_RECOMMEND)
        self.assertGreater(rec.confidence, 0.9)
        self.assertTrue(rec.is_actionable())

    def test_recommend_caution(self) -> None:
        """Test cautionary recommendation."""
        stakes = StakesAssessment(
            decision_statement="Start risky business",
            benefits=[
                Stake("Potential independence", ImpactLevel.HIGH, 0.3, 0.6),
            ],
            risks=[
                Stake("Financial loss", ImpactLevel.CRITICAL, 0.6, 0.9),
                Stake("Time investment", ImpactLevel.HIGH, 0.9, 0.8),
            ],
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            reversibility_explanation="Can go back to employment but timing matters",
        )

        evidence = EvidenceAnalysis(
            decision_statement="Start risky business",
            pro_evidence=[
                Evidence("Market opportunity exists", True, 0.7, "research"),
            ],
            con_evidence=[
                Evidence("High failure rate for startups", False, 0.95, "statistics"),
                Evidence("Insufficient savings", False, 0.85, "finances"),
                Evidence("No business experience", False, 0.8, "reflection"),
            ],
            uncertainty_factors=["Market timing", "Personal resilience"],
        )

        # Mock RAG response
        self.mock_rag.generate_response.return_value = """{
            "recommendation_type": "caution",
            "confidence": 0.85,
            "summary": "Significant risks outweigh benefits. Needs careful planning.",
            "key_considerations": ["High failure risk", "Financial vulnerability"],
            "next_steps": ["Build more savings", "Get business mentorship"],
            "monitoring_plan": ["Develop detailed business plan", "Validate market assumptions"],
            "reversibility_note": "Limited reversibility - ensure safety net first"
        }"""

        rec = self.recommender.recommend(
            decision_statement="Start risky business",
            stakes_assessment=stakes,
            evidence_analysis=evidence,
            user_id=self.user_id,
        )

        self.assertEqual(rec.recommendation_type, RecommendationType.CAUTION)
        self.assertFalse(rec.is_actionable())

    def test_recommend_neutral(self) -> None:
        """Test neutral recommendation with mixed evidence."""
        stakes = StakesAssessment(
            decision_statement="Change roles internally",
            benefits=[
                Stake("New experience", ImpactLevel.MEDIUM, 0.7, 0.8),
            ],
            risks=[
                Stake("Starting over", ImpactLevel.MEDIUM, 0.6, 0.7),
            ],
            reversibility=Reversibility.FULLY_REVERSIBLE,
            reversibility_explanation="Can return to previous role",
        )

        evidence = EvidenceAnalysis(
            decision_statement="Change roles internally",
            pro_evidence=[
                Evidence("Growth opportunity", True, 0.8, "career"),
            ],
            con_evidence=[
                Evidence("Unknown team dynamics", False, 0.6, "uncertainty"),
            ],
            uncertainty_factors=["New manager quality", "Team fit"],
        )

        # Mock RAG response
        self.mock_rag.generate_response.return_value = """{
            "recommendation_type": "neutral",
            "confidence": 0.6,
            "summary": "Balanced pros and cons. Decision depends on personal goals.",
            "key_considerations": ["Talk to new team", "Understand role expectations"],
            "next_steps": ["Schedule conversations", "Research team"],
            "monitoring_plan": ["Set clear success metrics"],
            "reversibility_note": "Fully reversible within company"
        }"""

        rec = self.recommender.recommend(
            decision_statement="Change roles internally",
            stakes_assessment=stakes,
            evidence_analysis=evidence,
            user_id=self.user_id,
        )

        self.assertEqual(rec.recommendation_type, RecommendationType.NEUTRAL)
        self.assertFalse(rec.is_actionable())

    def test_fallback_recommendation_positive(self) -> None:
        """Test fallback recommendation with positive heuristics."""
        stakes = StakesAssessment(
            decision_statement="Test",
            benefits=[
                Stake("Benefit", ImpactLevel.HIGH, 1.0, 1.0),
                Stake("Benefit 2", ImpactLevel.MEDIUM, 0.8, 0.8),
            ],
            risks=[],
            reversibility=Reversibility.FULLY_REVERSIBLE,
            reversibility_explanation="Can undo",
        )

        evidence = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[
                Evidence("Pro 1", True, 0.9, "src"),
                Evidence("Pro 2", True, 0.8, "src"),
            ],
            con_evidence=[],
            uncertainty_factors=[],
        )

        # Mock RAG failing - should use fallback
        self.mock_rag.generate_response.return_value = "invalid json"

        rec = self.recommender.recommend(
            decision_statement="Test decision",
            stakes_assessment=stakes,
            evidence_analysis=evidence,
            user_id=self.user_id,
        )

        # Fallback should recommend positively
        self.assertIn(
            rec.recommendation_type,
            {RecommendationType.STRONGLY_RECOMMEND, RecommendationType.RECOMMEND},
        )

    def test_fallback_recommendation_negative(self) -> None:
        """Test fallback recommendation with negative heuristics."""
        stakes = StakesAssessment(
            decision_statement="Test",
            benefits=[],
            risks=[
                Stake("Risk", ImpactLevel.CRITICAL, 0.9, 0.9),
                Stake("Risk 2", ImpactLevel.HIGH, 0.8, 0.8),
            ],
            reversibility=Reversibility.IRREVERSIBLE,
            reversibility_explanation="Cannot undo",
        )

        evidence = EvidenceAnalysis(
            decision_statement="Test",
            pro_evidence=[],
            con_evidence=[
                Evidence("Con 1", False, 0.95, "src"),
                Evidence("Con 2", False, 0.85, "src"),
            ],
            uncertainty_factors=[],
        )

        # Mock RAG failing - should use fallback
        self.mock_rag.generate_response.return_value = "invalid json"

        rec = self.recommender.recommend(
            decision_statement="Test decision",
            stakes_assessment=stakes,
            evidence_analysis=evidence,
            user_id=self.user_id,
        )

        # Fallback should caution negatively
        self.assertIn(
            rec.recommendation_type,
            {RecommendationType.CAUTION, RecommendationType.STRONGLY_CAUTION},
        )


if __name__ == "__main__":
    unittest.main()
