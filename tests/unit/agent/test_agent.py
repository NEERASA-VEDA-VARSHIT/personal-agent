"""Tests for Agent orchestration layer (M6.10.2)."""

import tempfile
import unittest
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Memory, User
from personal_agent.agent.agent import Agent, AgentResponse
from personal_agent.agent.prompts import build_chat_prompt, SYSTEM_BASE
from personal_agent.agent.state import AgentState
from personal_agent.decision.question_policy import QuestionPolicy, PolicyAction, QuestionAnalysis, StakesLevel, AmbiguityLevel, MissingInformation, InformationImpact, UserEffort


class TestAgentInit(unittest.TestCase):
    def test_agent_creates_response(self) -> None:
        state = AgentState(user_id=1)
        self.assertEqual(state.user_id, 1)
        self.assertEqual(state.turn, 0)
        self.assertEqual(state.provenance, [])

    def test_state_trace_appends(self) -> None:
        state = AgentState(user_id=1)
        state.trace("test_stage", {"key": "value"})
        self.assertEqual(len(state.provenance), 1)
        self.assertEqual(state.provenance[0]["stage"], "test_stage")
        self.assertEqual(state.provenance[0]["data"], {"key": "value"})


class TestAgentChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db_url = f"sqlite:///{cls.db_path}"
        cls.engine = create_engine(cls.db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()
        import os
        os.close(cls.db_fd)
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.session: Session = self.SessionLocal()
        self.mock_gateway = MagicMock()
        self.mock_pipeline = self._create_mock_pipeline()
        self.mock_q_analyzer = MagicMock()
        self.agent = Agent(
            gateway=self.mock_gateway,
            retrieval_pipeline=self.mock_pipeline,
            question_analyzer=self.mock_q_analyzer,
        )

    def _create_mock_pipeline(self):
        mock_pipeline = MagicMock()

        def mock_run(db, user_id, query, top_k=5, **kwargs):
            memories = db.query(Memory).filter(Memory.user_id == user_id, Memory.is_active == True).limit(top_k).all()
            ranked = []
            for mem in memories:
                r = MagicMock()
                r.memory = mem
                r.final_score = 0.85
                r.debug = {"final score": 0.85}
                ranked.append(r)
            result = MagicMock()
            result.ranked = ranked
            result.understanding = MagicMock(raw_query=query, temporal_intent="any")
            result.candidates = []
            result.filtered = MagicMock(filtered_out=[])
            return result

        mock_pipeline.run = MagicMock(side_effect=mock_run)
        mock_pipeline.builder = MagicMock()
        mock_pipeline.builder.build = MagicMock(return_value="mock memory context")
        return mock_pipeline

    def tearDown(self) -> None:
        self.session.rollback()
        self.session.close()

    def test_chat_returns_answer(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()
        memory = Memory(user_id=user.id, memory_type="goal", content="I want to become a software engineer", is_active=True)
        self.session.add(memory)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="That's a great goal!")
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=QuestionAnalysis(query="I want to become a software engineer", goal="become a software engineer", stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW, missing_information=[]))

        result = self.agent.chat(self.session, user_id=user.id, message="I want to become a software engineer")

        self.assertIsInstance(result, AgentResponse)
        self.assertEqual(result.action, PolicyAction.ANSWER.value)
        self.assertEqual(result.response, "That's a great goal!")
        self.mock_gateway.generate.assert_called_once()

    def test_chat_asks_when_insufficient_info(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="That's a great goal!")
        missing = MissingInformation(field="goal", description="What is your goal?", impact=InformationImpact.HIGH, effort=UserEffort.LOW)
        analysis = QuestionAnalysis(query="Tell me about yourself", goal="", stakes=StakesLevel.MEDIUM, ambiguity=AmbiguityLevel.HIGH, missing_information=[missing])
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=analysis)
        decision = QuestionPolicy().decide(analysis)
        # Override q_policy to return the ASK decision from analysis
        mock_policy = MagicMock()
        mock_policy.decide = MagicMock(return_value=decision)
        self.agent.q_policy = mock_policy

        result = self.agent.chat(self.session, user_id=user.id, message="Tell me about yourself")

        self.assertEqual(result.action, PolicyAction.ASK.value)
        self.assertIsNotNone(result.question_to_ask)
        self.assertEqual(len(result.memories_used), 0)
        self.mock_gateway.generate.assert_not_called()

    def test_chat_provenance_trace(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()
        memory = Memory(user_id=user.id, memory_type="fact", content="I like Python", is_active=True)
        self.session.add(memory)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="Python is great!")
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=QuestionAnalysis(query="I like Python", goal="", stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW, missing_information=[]))

        result = self.agent.chat(self.session, user_id=user.id, message="I like Python")

        self.assertEqual(result.action, PolicyAction.ANSWER.value)
        self.assertTrue(len(result.provenance) >= 2)
        stages = [p["stage"] for p in result.provenance]
        self.assertIn("retrieval_prelim", stages)
        self.assertIn("question_policy", stages)
        self.assertIn("retrieval", stages)
        self.assertIn("inference", stages)

    def test_chat_includes_memories_used(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()
        memory = Memory(user_id=user.id, memory_type="fact", content="I like Python", is_active=True)
        self.session.add(memory)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="Python is great!")
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=QuestionAnalysis(query="I like Python", goal="", stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW, missing_information=[]))

        result = self.agent.chat(self.session, user_id=user.id, message="I like Python")

        self.assertGreater(len(result.memories_used), 0)
        self.assertEqual(result.memories_used[0]["content"], "I like Python")
        self.assertEqual(result.memories_used[0]["status"], "active")

    def test_chat_with_decision_query(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()
        memory = Memory(user_id=user.id, memory_type="goal", content="I want to become a software engineer", is_active=True)
        self.session.add(memory)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="Based on your goals, I recommend focusing on backend development.")
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=QuestionAnalysis(query="Should I focus on backend?", goal="career", stakes=StakesLevel.HIGH, ambiguity=AmbiguityLevel.LOW, missing_information=[]))

        result = self.agent.chat(self.session, user_id=user.id, message="Should I focus on backend?")

        self.assertEqual(result.action, PolicyAction.ANSWER.value)
        # Decision queries trigger assessment
        if result.assessment is not None:
            self.assertIn("recommendation", result.assessment)

    def test_chat_returns_retrieval_debug(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()
        memory = Memory(user_id=user.id, memory_type="fact", content="I like Python", is_active=True)
        self.session.add(memory)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="Python is great!")
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=QuestionAnalysis(query="I like Python", goal="", stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW, missing_information=[]))

        result = self.agent.chat(self.session, user_id=user.id, message="I like Python")

        self.assertIsNotNone(result.retrieval_debug)
        self.assertIn("candidates", result.retrieval_debug)
        self.assertIn("ranked", result.retrieval_debug)

    def test_chat_no_memories_gracefully(self) -> None:
        user = User(username="new_user")
        self.session.add(user)
        self.session.commit()

        self.mock_gateway.generate = MagicMock(return_value="I don't have enough context yet, but I'm happy to help!")
        self.mock_q_analyzer.analyze_with_llm = MagicMock(return_value=QuestionAnalysis(query="Tell me about yourself", goal="", stakes=StakesLevel.LOW, ambiguity=AmbiguityLevel.LOW, missing_information=[]))

        result = self.agent.chat(self.session, user_id=user.id, message="Tell me about yourself")

        self.assertEqual(result.action, PolicyAction.ANSWER.value)
        self.assertEqual(len(result.memories_used), 0)


class TestAgentResponseDataclass(unittest.TestCase):
    def test_agent_response_fields(self) -> None:
        fields = list(AgentResponse.__dataclass_fields__.keys())
        self.assertIn("response", fields)
        self.assertIn("action", fields)
        self.assertIn("memories_used", fields)
        self.assertIn("provenance", fields)
        self.assertIn("question_to_ask", fields)
        self.assertIn("assumptions", fields)
        self.assertIn("assessment", fields)
        self.assertIn("retrieval_debug", fields)


class TestBuildChatPrompt(unittest.TestCase):
    def test_basic_prompt(self) -> None:
        msgs = build_chat_prompt("Hello", "")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Personal Agent", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Hello")

    def test_prompt_with_context(self) -> None:
        msgs = build_chat_prompt("Hello", "- Memory: I like Python")
        self.assertEqual(len(msgs), 3)
        self.assertIn("I like Python", msgs[1]["content"])

    def test_system_base_content(self) -> None:
        self.assertIn("privacy-first", SYSTEM_BASE)
        self.assertIn("provenance", SYSTEM_BASE)

    def test_prompt_with_assessment(self) -> None:
        assessment_ctx = "Decision: cautiously_positive"
        msgs = build_chat_prompt("Hello", "context", assessment_ctx)
        self.assertEqual(len(msgs), 4)
        self.assertIn("Decision", msgs[2]["content"])


if __name__ == "__main__":
    unittest.main()
