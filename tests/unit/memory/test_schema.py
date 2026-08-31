"""Test memory schema and basic persistence."""

import os
import tempfile
import unittest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from personal_agent.persistence.models import Base, Conversation, Memory, Message, MemorySource, User


class TestMemorySchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Create in-memory SQLite database for testing."""
        # Use a temporary file for SQLite to simulate real conditions
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
            pass  # File may be locked on Windows

    def setUp(self) -> None:
        """Create a session for each test."""
        self.session: Session = self.SessionLocal()

    def tearDown(self) -> None:
        """Clean up session and rollback any changes."""
        self.session.rollback()
        self.session.close()

    def test_create_user(self) -> None:
        """Test creating a user."""
        user = User(username="alice")
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        self.assertIsNotNone(user.id)
        self.assertEqual(user.username, "alice")

    def test_create_conversation_with_messages(self) -> None:
        """Test creating a conversation and messages."""
        user = User(username="bob")
        self.session.add(user)
        self.session.flush()

        conversation = Conversation(user_id=user.id, title="Career decision")
        self.session.add(conversation)
        self.session.flush()

        message1 = Message(conversation_id=conversation.id, role="user", content="Should I focus on backend?")
        message2 = Message(conversation_id=conversation.id, role="assistant", content="That depends on your goals.")
        self.session.add_all([message1, message2])
        self.session.commit()

        # Verify relationships work
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.messages[0].role, "user")

    def test_create_memory_with_source(self) -> None:
        """Test creating a memory and its source citation."""
        user = User(username="charlie")
        self.session.add(user)
        self.session.flush()

        conversation = Conversation(user_id=user.id, title="Test conversation")
        self.session.add(conversation)
        self.session.flush()

        # Create explicit memory
        memory = Memory(
            user_id=user.id,
            source_conversation_id=conversation.id,
            memory_type="explicit",
            content="Wants to become a strong software engineer",
            confidence=0.95,
        )
        self.session.add(memory)
        self.session.flush()

        # Add source citation
        source = MemorySource(
            memory_id=memory.id,
            source_type="conversation",
            source_ref=f"conversation_{conversation.id}",
            confidence=0.95,
        )
        self.session.add(source)
        self.session.commit()

        # Verify relationships
        self.assertEqual(len(memory.sources), 1)
        self.assertEqual(memory.sources[0].source_type, "conversation")

    def test_retrieve_active_memories(self) -> None:
        """Test retrieving active memories for a user."""
        user = User(username="diana")
        self.session.add(user)
        self.session.flush()

        # Create multiple memories
        for i in range(3):
            memory = Memory(
                user_id=user.id,
                memory_type="explicit",
                content=f"Goal {i+1}",
                confidence=0.9 + (i * 0.01),
                is_active=True,
            )
            self.session.add(memory)
        self.session.commit()

        # Retrieve active memories
        active_memories = self.session.query(Memory).filter(Memory.user_id == user.id, Memory.is_active == True).all()
        self.assertEqual(len(active_memories), 3)

    def test_memory_confidence_tracking(self) -> None:
        """Test that memory confidence values are tracked."""
        user = User(username="eve")
        self.session.add(user)
        self.session.flush()

        memory = Memory(
            user_id=user.id,
            memory_type="hypothesis",
            content="May be interested in distributed systems",
            confidence=0.45,
        )
        self.session.add(memory)
        self.session.commit()

        retrieved = self.session.query(Memory).filter(Memory.id == memory.id).first()
        self.assertEqual(retrieved.confidence, 0.45)
        self.assertEqual(retrieved.memory_type, "hypothesis")


if __name__ == "__main__":
    unittest.main()
