"""Test memory extraction and policy validation."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Conversation, Memory, Message, User
from app.memory.extraction import MemoryExtractionService
from app.memory.policy import MemoryCandidate, MemoryPolicy, MemoryType, SourceType, MemoryStatus


class TestMemoryPolicy(unittest.TestCase):
    """Test memory policy validation rules."""

    def setUp(self) -> None:
        self.policy = MemoryPolicy(
            candidate_confidence_threshold=0.75,
            hypothesis_confidence_threshold=0.60,
        )

    def test_fact_memories_always_stored(self) -> None:
        """FACT memories should always be stored."""
        candidate = MemoryCandidate(
            memory_type=MemoryType.FACT,
            content="I have 5 years of Python experience",
            confidence=0.99,
            reason="User stated directly",
            source_markers=["I've been using Python for the last 5 years"],
        )

        self.assertTrue(self.policy.should_store(candidate))
        self.assertFalse(self.policy.should_ask_user(candidate))

    def test_preference_memories_always_stored(self) -> None:
        """PREFERENCE memories should always be stored."""
        candidate = MemoryCandidate(
            memory_type=MemoryType.PREFERENCE,
            content="Prefers Python over Java",
            confidence=0.95,
            reason="User stated preference",
            source_markers=["I prefer Python"],
        )

        self.assertTrue(self.policy.should_store(candidate))
        self.assertFalse(self.policy.should_ask_user(candidate))

    def test_goal_memories_always_stored(self) -> None:
        """GOAL memories should always be stored."""
        candidate = MemoryCandidate(
            memory_type=MemoryType.GOAL,
            content="Become a staff engineer in 5 years",
            confidence=0.99,
            reason="User stated goal directly",
            source_markers=["I want to become a staff engineer in 5 years"],
        )

        self.assertTrue(self.policy.should_store(candidate))
        self.assertFalse(self.policy.should_ask_user(candidate))

    def test_hypothesis_with_evidence_stored(self) -> None:
        """Hypothesis with strong evidence should be stored (high evidence_strength)."""
        candidate = MemoryCandidate(
            memory_type=MemoryType.HYPOTHESIS,
            content="User is interested in distributed systems",
            reason="Multiple mentions across conversation",
            source_markers=["scaling is important", "microservices architecture", "consensus algorithms"],
            evidence_strength="high",
            source_type=SourceType.MODEL_EXTRACTED,
        )

        self.assertTrue(self.policy.should_store(candidate))

    def test_hypothesis_without_evidence_rejected(self) -> None:
        """Hypothesis without source markers should be rejected."""
        candidate = MemoryCandidate(
            memory_type=MemoryType.HYPOTHESIS,
            content="User is secretly interested in quantum computing",
            confidence=0.70,
            reason="No supporting evidence",
            source_markers=[],
        )

        self.assertFalse(self.policy.should_store(candidate))

    def test_validate_method_returns_dict(self) -> None:
        """Validate method should return detailed decision dict."""
        candidate = MemoryCandidate(
            memory_type=MemoryType.FACT,
            content="Goal: become a staff engineer",
            confidence=0.98,
            reason="User stated goal",
            source_markers=["My goal is to become a staff engineer"],
        )

        result = self.policy.validate(candidate)

        self.assertIn("should_store", result)
        self.assertIn("should_ask", result)
        self.assertIn("reason", result)
        self.assertTrue(result["should_store"])
        self.assertIsInstance(result["reason"], str)


class TestMemoryExtraction(unittest.TestCase):
    """Test memory extraction from conversations."""

    _user_counter = 0  # Class variable to track unique usernames

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

    def setUp(self) -> None:
        """Create a session for each test."""
        self.session: Session = self.SessionLocal()

        # Create mock gateway
        self.mock_gateway = MagicMock()
        self.extraction_service = MemoryExtractionService(gateway=self.mock_gateway)

    def _get_unique_username(self) -> str:
        """Generate a unique username for test isolation."""
        TestMemoryExtraction._user_counter += 1
        return f"test_user_{TestMemoryExtraction._user_counter}"

    def tearDown(self) -> None:
        """Clean up session."""
        self.session.rollback()
        self.session.close()

    def test_parse_valid_extraction_response(self) -> None:
        """Test parsing valid LLM extraction response."""
        response = """
        {
          "memories": [
            {
              "type": "goal",
              "content": "I want to specialize in distributed systems",
              "confidence": 0.99,
              "reason": "User stated goal",
              "source_markers": ["I want to specialize in distributed systems"]
            },
            {
              "type": "preference",
              "content": "Prefers backend work",
              "confidence": 0.82,
              "reason": "Multiple mentions",
              "source_markers": ["I love backend", "frontend is not my thing"]
            }
          ]
        }
        """

        candidates = self.extraction_service._parse_extraction_response(response)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].memory_type, MemoryType.GOAL)
        self.assertEqual(candidates[0].confidence, 0.99)
        self.assertEqual(candidates[1].memory_type, MemoryType.PREFERENCE)

    def test_parse_invalid_json_returns_empty(self) -> None:
        """Test parsing invalid JSON returns empty list."""
        response = "This is not valid JSON at all"

        candidates = self.extraction_service._parse_extraction_response(response)

        self.assertEqual(len(candidates), 0)

    def test_format_conversation(self) -> None:
        """Test conversation formatting."""
        user = User(username=self._get_unique_username())
        self.session.add(user)
        self.session.flush()

        conversation = Conversation(user_id=user.id, title="Test Conversation")
        self.session.add(conversation)
        self.session.flush()

        msg1 = Message(conversation_id=conversation.id, role="user", content="Hello")
        msg2 = Message(conversation_id=conversation.id, role="assistant", content="Hi there")
        self.session.add_all([msg1, msg2])
        self.session.commit()

        formatted = self.extraction_service._format_conversation(conversation)

        self.assertIn("Test Conversation", formatted)
        self.assertIn("User: Hello", formatted)
        self.assertIn("Assistant: Hi there", formatted)

    def test_extract_from_conversation_with_policy(self) -> None:
        """Test full extraction workflow with policy validation."""
        # Create test data
        user = User(username=self._get_unique_username())
        self.session.add(user)
        self.session.flush()

        conversation = Conversation(user_id=user.id, title="Career Planning")
        self.session.add(conversation)
        self.session.flush()

        msg = Message(conversation_id=conversation.id, role="user", content="I want to learn Rust")
        self.session.add(msg)
        self.session.commit()

        # Mock LLM response with one approved and one needs-review
        mock_response = """
        {
          "memories": [
            {
              "type": "goal",
              "content": "Wants to learn Rust",
              "confidence": 0.99,
              "reason": "Directly stated",
              "source_markers": ["I want to learn Rust"]
            },
            {
              "type": "hypothesis",
              "content": "Interested in systems programming",
              "confidence": 0.55,
              "reason": "Learning Rust suggests interest (borderline)",
              "source_markers": ["I want to learn Rust", "systems work is important"]
            }
          ]
        }
        """
        self.mock_gateway.generate = MagicMock(return_value=mock_response)

        # Extract
        result = self.extraction_service.extract_from_conversation(
            self.session,
            user_id=user.id,
            conversation_id=conversation.id,
            apply_policy=True,
        )

        # Verify results
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(len(result["approved"]), 1)  # GOAL memory (always approved)
        self.assertEqual(len(result["needs_review"]), 1)  # HYPOTHESIS at borderline confidence

    def test_store_approved_memories(self) -> None:
        """Test storing approved memories to database."""
        user = User(username=self._get_unique_username())
        self.session.add(user)
        self.session.flush()

        conversation = Conversation(user_id=user.id, title="Test")
        self.session.add(conversation)
        self.session.flush()

        # Create approved candidates
        candidate = MemoryCandidate(
            memory_type=MemoryType.FACT,
            content="I love Python",
            confidence=0.99,
            reason="User stated",
            source_markers=["I love Python"],
        )

        approved = [
            {
                "candidate": candidate,
                "validation": {"should_store": True, "should_ask": False},
            }
        ]

        # Store
        stored = self.extraction_service.store_approved_memories(
            self.session,
            user_id=user.id,
            conversation_id=conversation.id,
            approved_candidates=approved,
        )

        # Verify
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].content, "I love Python")
        self.assertEqual(stored[0].type, "fact")  # New v2 field
        self.assertEqual(stored[0].status, MemoryStatus.ACTIVE.value)

        # Verify database persistence
        retrieved = self.session.query(Memory).filter(Memory.id == stored[0].id).first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.type, "fact")
        self.assertEqual(retrieved.status, MemoryStatus.ACTIVE.value)
        self.assertEqual(retrieved.content, "I love Python")


if __name__ == "__main__":
    unittest.main()
