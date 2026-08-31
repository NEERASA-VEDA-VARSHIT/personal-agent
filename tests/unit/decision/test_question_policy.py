"""M6.8 — Question Policy tests: Answer vs Ask"""

import unittest

from personal_agent.decision.question_policy import (
    AmbiguityLevel,
    InformationImpact,
    MissingInformation,
    PolicyAction,
    QuestionAnalysis,
    QuestionPolicy,
    StakesLevel,
    UserEffort,
)


class TestQuestionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = QuestionPolicy()

    def test_low_stakes_low_ambiguity_answers(self) -> None:
        """low stakes + low ambiguity -> ANSWER"""
        analysis = QuestionAnalysis(
            query="Should I learn Docker or Kubernetes first?",
            goal="choose learning order",
            stakes=StakesLevel.LOW,
            ambiguity=AmbiguityLevel.LOW,
            missing_information=[],
        )
        decision = self.policy.decide(analysis)
        self.assertEqual(decision.action, PolicyAction.ANSWER)

    def test_high_stakes_missing_critical_asks(self) -> None:
        """high stakes + missing critical information -> ASK"""
        analysis = QuestionAnalysis(
            query="Should I quit my internship?",
            goal="decide on quitting",
            stakes=StakesLevel.HIGH,
            ambiguity=AmbiguityLevel.HIGH,
            reversibility="partially_reversible",
            missing_information=[
                MissingInformation(
                    field="financial_dependency",
                    description="whether user depends on internship income",
                    impact=InformationImpact.HIGH,
                    effort=UserEffort.LOW,
                ),
                MissingInformation(
                    field="alternative_offer",
                    description="whether user has another offer",
                    impact=InformationImpact.HIGH,
                    effort=UserEffort.LOW,
                ),
            ],
        )
        decision = self.policy.decide(analysis)
        self.assertEqual(decision.action, PolicyAction.ASK)
        self.assertIsNotNone(decision.question_to_ask)
        self.assertIn("financial_dependency", decision.question_to_ask.lower())

    def test_high_stakes_info_already_in_memory_answers(self) -> None:
        """high stakes + information already available in memory -> ANSWER"""
        analysis = QuestionAnalysis(
            query="Should I quit my internship? I have no financial dependency and have another offer",
            goal="decide on quitting",
            stakes=StakesLevel.HIGH,
            ambiguity=AmbiguityLevel.LOW,
            missing_information=[
                MissingInformation(
                    field="financial_dependency",
                    description="depends on income",
                    impact=InformationImpact.HIGH,
                    effort=UserEffort.LOW,
                    available_in_memory=True,
                )
            ],
        )
        decision = self.policy.decide(analysis)
        self.assertEqual(decision.action, PolicyAction.ANSWER)

    def test_ambiguous_but_low_impact_answers_with_assumptions(self) -> None:
        """ambiguous but low impact -> ANSWER with assumptions"""
        analysis = QuestionAnalysis(
            query="Should I use Python or Go for a small script?",
            goal="choose language for script",
            stakes=StakesLevel.LOW,
            ambiguity=AmbiguityLevel.HIGH,
            missing_information=[
                MissingInformation(
                    field="script_size",
                    description="exact lines of code",
                    impact=InformationImpact.LOW,
                    effort=UserEffort.LOW,
                )
            ],
            assumptions=["Assuming script is <500 lines"],
        )
        decision = self.policy.decide(analysis)
        self.assertEqual(decision.action, PolicyAction.ANSWER)
        self.assertTrue(len(decision.assumptions) > 0)

    def test_single_variable_dramatically_changes_outcome_asks_specifically(self) -> None:
        """one missing variable with HIGH impact -> ASK specifically about that variable"""
        analysis = QuestionAnalysis(
            query="Should I accept the startup offer?",
            goal="decide on offer",
            stakes=StakesLevel.HIGH,
            ambiguity=AmbiguityLevel.MEDIUM,
            reversibility="irreversible",
            missing_information=[
                MissingInformation(
                    field="equity_details",
                    description="equity compensation and vesting",
                    impact=InformationImpact.HIGH,
                    effort=UserEffort.LOW,
                )
            ],
        )
        decision = self.policy.decide(analysis)
        self.assertEqual(decision.action, PolicyAction.ASK)
        self.assertIsNotNone(decision.question_to_ask)
        self.assertIn("equity", decision.question_to_ask.lower())

    def test_irreversible_high_stakes_lowers_ask_threshold(self) -> None:
        """irreversible + high stakes should ask even with medium impact"""
        analysis = QuestionAnalysis(
            query="Should I drop out of college?",
            goal="education decision",
            stakes=StakesLevel.CRITICAL,
            ambiguity=AmbiguityLevel.MEDIUM,
            reversibility="irreversible",
            missing_information=[
                MissingInformation(
                    field="alternative_plan",
                    description="what user will do after dropping out",
                    impact=InformationImpact.MEDIUM,
                    effort=UserEffort.MEDIUM,
                )
            ],
        )
        decision = self.policy.decide(analysis)
        # EVOI amplified by irreversible, should ASK or CLARIFY, not plain ANSWER
        self.assertIn(decision.action, (PolicyAction.ASK, PolicyAction.CLARIFY))

    def test_llm_does_not_directly_decide(self) -> None:
        """Policy must decide; LLM output is just observations"""
        # Simulate LLM producing observations with high stakes but low impact
        analysis = QuestionAnalysis(
            query="Should I learn Docker first?",
            goal="learning order",
            stakes=StakesLevel.MEDIUM,
            ambiguity=AmbiguityLevel.LOW,
            missing_information=[
                MissingInformation(
                    field="current_docker_knowledge",
                    description="whether user knows containers",
                    impact=InformationImpact.LOW,
                    effort=UserEffort.LOW,
                )
            ],
        )
        decision = self.policy.decide(analysis)
        # Even though LLM said missing info, policy says impact LOW -> ANSWER
        self.assertEqual(decision.action, PolicyAction.ANSWER)

    def test_question_is_smallest_useful(self) -> None:
        """Should ask the smallest useful question (top impact first)"""
        analysis = QuestionAnalysis(
            query="Should I quit?",
            stakes=StakesLevel.HIGH,
            ambiguity=AmbiguityLevel.HIGH,
            missing_information=[
                MissingInformation(field="low_field", description="minor detail", impact=InformationImpact.LOW, effort=UserEffort.LOW),
                MissingInformation(field="critical_field", description="whether user has savings", impact=InformationImpact.HIGH, effort=UserEffort.LOW),
            ],
        )
        decision = self.policy.decide(analysis)
        self.assertEqual(decision.action, PolicyAction.ASK)
        self.assertIn("critical_field", decision.question_to_ask)


if __name__ == "__main__":
    unittest.main()
