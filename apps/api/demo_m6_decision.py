#!/usr/bin/env python
"""End-to-end M6 demonstration: Decision support engine with stakes and evidence analysis."""

import sys
import tempfile
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup path for imports
sys.path.insert(0, "apps/api")

from app.db.models import Base, User
from app.decision_support import (
    DecisionRecommender,
    EvidenceAnalysis,
    Evidence,
    ImpactLevel,
    Reversibility,
    Stake,
    StakesAssessment,
)


def setup_demo_database():
    """Create an in-memory database for demonstration."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal, engine


def create_demo_user(session):
    """Create a user for the decision scenario."""
    user = User(username="jordan")
    session.add(user)
    session.commit()
    return user.id


def run_m6_demo():
    """Run the M6 decision support demonstration."""
    print("\n" + "=" * 90)
    print("M6: DECISION SUPPORT ENGINE - AUTONOMOUS DECISION ANALYSIS")
    print("=" * 90)

    # Setup
    SessionLocal, engine = setup_demo_database()
    session = SessionLocal()

    try:
        # Create demo user
        print("\n👤 Setting up scenario for career decision...")
        user_id = create_demo_user(session)
        print(f"   ✓ User created (ID: {user_id})")

        # Scenario: Should I accept a job offer from a startup?
        print("\n" + "=" * 90)
        print("SCENARIO: Accept Job Offer from Promising Startup?")
        print("=" * 90)

        # 1. Build stakes assessment
        print("\n📊 STEP 1: STAKES ASSESSMENT")
        print("─" * 90)

        stakes_assessment = StakesAssessment(
            decision_statement="Accept job offer from Series B startup",
            benefits=[
                Stake(
                    description="Equity opportunity (potential 5-10x return)",
                    impact_level=ImpactLevel.HIGH,
                    probability=0.3,  # 30% chance of success
                    confidence=0.75,
                ),
                Stake(
                    description="Early employee role (significant learning opportunity)",
                    impact_level=ImpactLevel.HIGH,
                    probability=0.95,
                    confidence=0.9,
                ),
                Stake(
                    description="Mission-driven work (combating climate change)",
                    impact_level=ImpactLevel.MEDIUM,
                    probability=0.9,
                    confidence=0.85,
                ),
                Stake(
                    description="Lead engineer role (career advancement)",
                    impact_level=ImpactLevel.MEDIUM,
                    probability=0.85,
                    confidence=0.8,
                ),
            ],
            risks=[
                Stake(
                    description="Startup failure (company goes under)",
                    impact_level=ImpactLevel.CRITICAL,
                    probability=0.4,  # 40% chance startups fail in first 5 years
                    confidence=0.85,
                ),
                Stake(
                    description="Lower salary vs FAANG alternative",
                    impact_level=ImpactLevel.MEDIUM,
                    probability=1.0,
                    confidence=0.95,
                ),
                Stake(
                    description="Early-stage chaos (long hours, uncertain direction)",
                    impact_level=ImpactLevel.HIGH,
                    probability=0.85,
                    confidence=0.9,
                ),
                Stake(
                    description="Limited resources (not enough budget/tools)",
                    impact_level=ImpactLevel.LOW,
                    probability=0.7,
                    confidence=0.8,
                ),
            ],
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            reversibility_explanation="Can return to tech industry, but may lose equity if company succeeds",
        )

        # Display stakes
        print("\n✓ BENEFITS:")
        for stake in stakes_assessment.benefits:
            impact_score = stake.weighted_impact()
            print(
                f"  • {stake.description}"
            )
            print(
                f"    Impact: {stake.impact_level.value.title()} | "
                f"Probability: {stake.probability:.0%} | "
                f"Confidence: {stake.confidence:.0%} | "
                f"Weighted: {impact_score:.2f}"
            )

        print("\n✗ RISKS:")
        for stake in stakes_assessment.risks:
            impact_score = stake.weighted_impact()
            print(
                f"  • {stake.description}"
            )
            print(
                f"    Impact: {stake.impact_level.value.title()} | "
                f"Probability: {stake.probability:.0%} | "
                f"Confidence: {stake.confidence:.0%} | "
                f"Weighted: {impact_score:.2f}"
            )

        net_impact = stakes_assessment.net_impact_score()
        print(f"\n📈 Net Impact Score: {net_impact:.2f}")
        print(f"   (Positive = Benefits > Risks | Negative = Risks > Benefits)")
        print(f"\n🔄 Reversibility: {stakes_assessment.reversibility.value.title()}")
        print(f"   → {stakes_assessment.reversibility_explanation}")
        print(f"\n⚖️ Low-Risk Decision? {'✓ YES' if stakes_assessment.is_low_risk() else '✗ NO'}")

        # 2. Build evidence analysis
        print("\n" + "─" * 90)
        print("\n📚 STEP 2: EVIDENCE ANALYSIS")
        print("─" * 90)

        evidence_analysis = EvidenceAnalysis(
            decision_statement="Accept job offer from Series B startup",
            pro_evidence=[
                Evidence(
                    argument="Team includes founder from successful exit (2 unicorns)",
                    supports_decision=True,
                    confidence=0.95,
                    source="conversation_with_team",
                ),
                Evidence(
                    argument="Market problem is large ($50B+ TAM in climate tech)",
                    supports_decision=True,
                    confidence=0.9,
                    source="market_research_memory",
                ),
                Evidence(
                    argument="Personal career goal: lead technical teams at growth-stage",
                    supports_decision=True,
                    confidence=0.92,
                    source="extracted_memory",
                ),
                Evidence(
                    argument="Values learning and mission-driven work highly",
                    supports_decision=True,
                    confidence=0.88,
                    source="personal_preference_memory",
                ),
                Evidence(
                    argument="Runway of 24+ months reduces near-term failure risk",
                    supports_decision=True,
                    confidence=0.85,
                    source="financial_disclosure",
                ),
                Evidence(
                    argument="Located in thriving tech hub (easy to find next role)",
                    supports_decision=True,
                    confidence=0.9,
                    source="location_analysis",
                ),
            ],
            con_evidence=[
                Evidence(
                    argument="Startups have 70-90% failure rate in first 5 years",
                    supports_decision=False,
                    confidence=0.92,
                    source="startup_statistics",
                ),
                Evidence(
                    argument="Salary is 30% below FAANG alternatives",
                    supports_decision=False,
                    confidence=0.95,
                    source="market_data",
                ),
                Evidence(
                    argument="Early-stage companies often have culture challenges",
                    supports_decision=False,
                    confidence=0.75,
                    source="industry_observation",
                ),
                Evidence(
                    argument="Equity compensation heavily diluted (Series B already happened)",
                    supports_decision=False,
                    confidence=0.88,
                    source="cap_table_analysis",
                ),
                Evidence(
                    argument="Previous startup experience was stressful",
                    supports_decision=False,
                    confidence=0.8,
                    source="personal_memory",
                ),
            ],
            uncertainty_factors=[
                "Product-market fit status still being validated",
                "Future funding rounds not guaranteed",
                "New regulatory changes in climate tech unknown",
                "Personal stress tolerance may change",
                "Family circumstances may shift",
            ],
        )

        # Display evidence
        print("\n✓ SUPPORTING EVIDENCE:")
        for evidence in evidence_analysis.pro_evidence:
            print(f"  • {evidence.argument}")
            print(f"    Source: {evidence.source} | Confidence: {evidence.confidence:.0%}")

        print("\n✗ AGAINST EVIDENCE:")
        for evidence in evidence_analysis.con_evidence:
            print(f"  • {evidence.argument}")
            print(f"    Source: {evidence.source} | Confidence: {evidence.confidence:.0%}")

        print("\n❓ UNCERTAINTIES:")
        for uncertainty in evidence_analysis.uncertainty_factors:
            print(f"  • {uncertainty}")

        pro_score = evidence_analysis.pro_score()
        con_score = evidence_analysis.con_score()
        ratio = evidence_analysis.evidence_ratio()
        strength = evidence_analysis.evidence_strength()

        print(f"\n📊 Evidence Summary:")
        print(f"  Pro Score: {pro_score:.2f} | Con Score: {con_score:.2f}")
        print(f"  Evidence Ratio: {ratio:.2f} (1.0=all pro, -1.0=all con)")
        print(f"  Evidence Strength: {strength:.0%} (confidence in decision quality)")

        # 3. Generate recommendation
        print("\n" + "─" * 90)
        print("\n💡 STEP 3: SYNTHESIZED RECOMMENDATION")
        print("─" * 90)

        # Create mock gateway and RAG service
        mock_gateway = MagicMock()
        mock_rag = MagicMock()

        # Mock a realistic LLM response
        mock_rag.generate_response.return_value = """{
            "recommendation_type": "recommend",
            "confidence": 0.78,
            "summary": "This is a well-considered opportunity that aligns with career goals. The downside is manageable with proper risk mitigation. Decision hinges on personal risk tolerance and family financial stability.",
            "key_considerations": [
                "Strong team and market opportunity reduce typical startup risk",
                "Mission alignment with personal values is significant motivator",
                "Financial runway provides 2-year window to prove concept",
                "Downside protection: can return to tech jobs if startup fails",
                "Equity upside could be substantial if exit occurs"
            ],
            "next_steps": [
                "Negotiate equity package to clarify total compensation",
                "Request detailed product roadmap and customer validation data",
                "Meet with technical team to assess engineering culture",
                "Confirm financial runway and funding strategy",
                "Discuss relocation support and benefits package",
                "Make final family/financial stability assessment"
            ],
            "monitoring_plan": [
                "Check product-market fit indicators monthly (customer growth, engagement)",
                "Monitor team dynamics and culture (1:1 feedback from team members)",
                "Track runway burn rate and next funding milestones",
                "Review personal stress levels and work-life balance quarterly",
                "Reassess career growth trajectory at 6-month mark"
            ],
            "reversibility_note": "Job market is strong for experienced engineers. Can return to larger companies or find new startup. Main loss would be equity upside if company succeeds—mitigate by negotiating best possible terms now."
        }"""

        recommender = DecisionRecommender(
            gateway=mock_gateway, rag_service=mock_rag
        )

        recommendation = recommender.recommend(
            decision_statement="Accept job offer from Series B startup",
            stakes_assessment=stakes_assessment,
            evidence_analysis=evidence_analysis,
            user_id=user_id,
        )

        # Display recommendation
        print(f"\n🎯 RECOMMENDATION: {recommendation.recommendation_type.value.upper().replace('_', ' ')}")
        print(f"   Confidence: {recommendation.confidence:.0%}")
        print(f"   Actionable: {'✓ YES - Should act' if recommendation.is_actionable() else '✗ NO - Needs more consideration'}")

        print(f"\n📋 Summary:")
        print(f"   {recommendation.summary}")

        print(f"\n🔑 Key Considerations:")
        for i, consideration in enumerate(recommendation.key_considerations, 1):
            print(f"   {i}. {consideration}")

        print(f"\n✅ Next Steps (if you decide to accept):")
        for i, step in enumerate(recommendation.next_steps, 1):
            print(f"   {i}. {step}")

        print(f"\n📈 Monitoring Plan (to track if decision is working):")
        for i, monitor in enumerate(recommendation.monitoring_plan, 1):
            print(f"   {i}. {monitor}")

        print(f"\n🔄 Reversibility Note:")
        print(f"   {recommendation.reversibility_note}")

        # Summary
        print("\n" + "=" * 90)
        print("M6 COMPLETE: Autonomous decision analysis working!")
        print("=" * 90)
        print("\n✓ Decision analysis pipeline:")
        print("  1. Stakes Assessment: Evaluated benefits (4), risks (4), reversibility")
        print("  2. Evidence Analysis: Collected pro (6) and con (5) evidence")
        print("  3. Synthesized Recommendation: Generated actionable recommendation")
        print(f"\n✓ Outcome: {recommendation.recommendation_type.value.upper().replace('_', ' ')}")
        print(f"   User can now act with confidence: {recommendation.confidence:.0%}")
        print("=" * 90)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    run_m6_demo()
