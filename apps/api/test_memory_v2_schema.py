"""Comprehensive tests for Memory v2 schema and lifecycle."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    Base,
    Conversation,
    Memory,
    MemoryAudit,
    MemoryRelation,
    MemorySource,
    Message,
    User,
)
from app.memory_policy import MemoryStatus, SourceType


class TestMemoryV2Schema(unittest.TestCase):
    """Test Memory v2 schema structure and relationships."""

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

    _counter = 0

    def setUp(self) -> None:
        """Create session and test user."""
        self.session: Session = self.SessionLocal()
        user = User(username=f"test_user_{self._get_id()}")
        self.session.add(user)
        self.session.flush()
        self.user_id = user.id
        self.user = user

    def tearDown(self) -> None:
        """Clean up session."""
        self.session.rollback()
        self.session.close()

    def _get_id(self) -> int:
        """Get unique ID."""
        TestMemoryV2Schema._counter += 1
        return TestMemoryV2Schema._counter

    def test_memory_core_fields(self) -> None:
        """Test Memory v2 core fields exist and are accessible."""
        memory = Memory(
            user_id=self.user_id,
            type="fact",
            content="User prefers Python",
            summary="Strong Python preference",
            sensitivity="public",
            confidence=0.95,
            status=MemoryStatus.ACTIVE.value,
            model_version="v1.0",
            embedding="[1.0, 0.5, 0.2]",
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertEqual(retrieved.type, "fact")
        self.assertEqual(retrieved.content, "User prefers Python")
        self.assertEqual(retrieved.summary, "Strong Python preference")
        self.assertEqual(retrieved.sensitivity, "public")
        self.assertEqual(retrieved.confidence, 0.95)
        self.assertEqual(retrieved.status, MemoryStatus.ACTIVE.value)
        self.assertEqual(retrieved.model_version, "v1.0")
        self.assertTrue(retrieved.is_active)

    def test_memory_temporal_validity_range(self) -> None:
        """Test Memory temporal validity fields (valid_from, valid_until)."""
        now = datetime.utcnow()
        future = now + timedelta(days=30)

        memory = Memory(
            user_id=self.user_id,
            type="preference",
            content="Temporary preference during project",
            status=MemoryStatus.ACTIVE.value,
            valid_from=now,
            valid_until=future,
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertIsNotNone(retrieved.valid_from)
        self.assertIsNotNone(retrieved.valid_until)
        self.assertEqual(retrieved.valid_from, now)
        self.assertEqual(retrieved.valid_until, future)

    def test_memory_valid_until_nullable(self) -> None:
        """Test that valid_until can be NULL for permanent memories."""
        memory = Memory(
            user_id=self.user_id,
            type="fact",
            content="Permanent fact",
            status=MemoryStatus.ACTIVE.value,
            valid_from=datetime.utcnow(),
            valid_until=None,  # Explicitly nullable
            is_active=True,
        )
        self.session.add(memory)
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertIsNotNone(retrieved.valid_from)
        self.assertIsNone(retrieved.valid_until)

    def test_memory_source_creation(self) -> None:
        """Test creating and storing memory sources."""
        memory = Memory(
            user_id=self.user_id,
            type="fact",
            content="User said this directly",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()

        source = MemorySource(
            memory_id=memory.id,
            source_type=SourceType.USER_STATED.value,
            source_ref="conversation_123",
            confidence=0.99,
        )
        self.session.add(source)
        self.session.commit()

        # Verify relationship
        retrieved_memory = self.session.query(Memory).get(memory.id)
        self.assertEqual(len(retrieved_memory.sources), 1)
        self.assertEqual(retrieved_memory.sources[0].source_type, SourceType.USER_STATED.value)
        self.assertEqual(retrieved_memory.sources[0].confidence, 0.99)

    def test_memory_source_types(self) -> None:
        """Test all three source types."""
        memory = Memory(
            user_id=self.user_id,
            type="goal",
            content="Career goal",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()

        sources = [
            MemorySource(
                memory_id=memory.id,
                source_type=SourceType.USER_STATED.value,
                source_ref="direct statement",
                confidence=1.0,
            ),
            MemorySource(
                memory_id=memory.id,
                source_type=SourceType.MODEL_EXTRACTED.value,
                source_ref="extracted from conversation",
                confidence=0.85,
            ),
            MemorySource(
                memory_id=memory.id,
                source_type=SourceType.MODEL_INFERRED.value,
                source_ref="inferred pattern",
                confidence=0.65,
            ),
        ]
        self.session.add_all(sources)
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertEqual(len(retrieved.sources), 3)
        source_types = [s.source_type for s in retrieved.sources]
        self.assertIn(SourceType.USER_STATED.value, source_types)
        self.assertIn(SourceType.MODEL_EXTRACTED.value, source_types)
        self.assertIn(SourceType.MODEL_INFERRED.value, source_types)

    def test_memory_relation_supports(self) -> None:
        """Test SUPPORTS relationship between memories."""
        mem1 = Memory(
            user_id=self.user_id,
            type="fact",
            content="User has Python experience",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        mem2 = Memory(
            user_id=self.user_id,
            type="goal",
            content="Wants to lead Python team",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add_all([mem1, mem2])
        self.session.flush()

        relation = MemoryRelation(
            from_memory_id=mem1.id,
            to_memory_id=mem2.id,
            relation_type="supports",
            confidence=0.95,
        )
        self.session.add(relation)
        self.session.commit()

        # Check forward relationship
        retrieved1 = self.session.query(Memory).get(mem1.id)
        self.assertEqual(len(retrieved1.relations_from), 1)
        self.assertEqual(retrieved1.relations_from[0].relation_type, "supports")

        # Check backward relationship
        retrieved2 = self.session.query(Memory).get(mem2.id)
        self.assertEqual(len(retrieved2.relations_to), 1)

    def test_memory_relation_contradicts(self) -> None:
        """Test CONTRADICTS relationship between memories."""
        mem1 = Memory(
            user_id=self.user_id,
            type="preference",
            content="Prefers remote work",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        mem2 = Memory(
            user_id=self.user_id,
            type="preference",
            content="Prefers office work",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add_all([mem1, mem2])
        self.session.flush()

        relation = MemoryRelation(
            from_memory_id=mem1.id,
            to_memory_id=mem2.id,
            relation_type="contradicts",
            confidence=0.85,
        )
        self.session.add(relation)
        self.session.commit()

        retrieved1 = self.session.query(Memory).get(mem1.id)
        self.assertEqual(len(retrieved1.relations_from), 1)
        self.assertEqual(retrieved1.relations_from[0].relation_type, "contradicts")

    def test_memory_relation_supersedes(self) -> None:
        """Test SUPERSEDES relationship for memory lifecycle."""
        mem_old = Memory(
            user_id=self.user_id,
            type="preference",
            content="Likes Java",
            status=MemoryStatus.SUPERSEDED.value,
            is_active=False,
        )
        mem_new = Memory(
            user_id=self.user_id,
            type="preference",
            content="Prefers Python now",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add_all([mem_old, mem_new])
        self.session.flush()

        relation = MemoryRelation(
            from_memory_id=mem_new.id,
            to_memory_id=mem_old.id,
            relation_type="supersedes",
            confidence=1.0,
        )
        self.session.add(relation)
        self.session.commit()

        # Verify supersession
        retrieved_old = self.session.query(Memory).get(mem_old.id)
        retrieved_new = self.session.query(Memory).get(mem_new.id)
        self.assertEqual(retrieved_old.status, MemoryStatus.SUPERSEDED.value)
        self.assertFalse(retrieved_old.is_active)
        self.assertEqual(retrieved_new.status, MemoryStatus.ACTIVE.value)
        self.assertTrue(retrieved_new.is_active)

    def test_memory_relation_related_to(self) -> None:
        """Test RELATED_TO relationship for loose associations."""
        mem1 = Memory(
            user_id=self.user_id,
            type="episode",
            content="Worked on distributed systems project",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        mem2 = Memory(
            user_id=self.user_id,
            type="goal",
            content="Become systems architect",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add_all([mem1, mem2])
        self.session.flush()

        relation = MemoryRelation(
            from_memory_id=mem1.id,
            to_memory_id=mem2.id,
            relation_type="related_to",
            confidence=0.70,
        )
        self.session.add(relation)
        self.session.commit()

        retrieved = self.session.query(Memory).get(mem1.id)
        self.assertEqual(len(retrieved.relations_from), 1)
        self.assertEqual(retrieved.relations_from[0].relation_type, "related_to")

    def test_memory_audit_created_action(self) -> None:
        """Test CREATED audit action."""
        memory = Memory(
            user_id=self.user_id,
            type="fact",
            content="Test memory",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()

        audit = MemoryAudit(
            memory_id=memory.id,
            action="created",
            reason="Extracted from conversation",
            actor="extraction_service",
        )
        self.session.add(audit)
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertEqual(len(retrieved.audits), 1)
        self.assertEqual(retrieved.audits[0].action, "created")
        self.assertEqual(retrieved.audits[0].reason, "Extracted from conversation")

    def test_memory_audit_lifecycle_actions(self) -> None:
        """Test all lifecycle audit actions."""
        memory = Memory(
            user_id=self.user_id,
            type="fact",
            content="Test memory",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()

        actions = ["created", "updated", "confirmed", "rejected", "forgotten"]
        for action in actions:
            audit = MemoryAudit(
                memory_id=memory.id,
                action=action,
                reason=f"Test {action} action",
                actor="test_actor",
            )
            self.session.add(audit)

        self.session.commit()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertEqual(len(retrieved.audits), 5)
        audit_actions = [a.action for a in retrieved.audits]
        for action in actions:
            self.assertIn(action, audit_actions)

    def test_memory_status_transitions(self) -> None:
        """Test memory status lifecycle transitions."""
        # Create in CANDIDATE status
        memory = Memory(
            user_id=self.user_id,
            type="hypothesis",
            content="User might like Rust",
            status=MemoryStatus.CANDIDATE.value,
            is_active=False,
        )
        self.session.add(memory)
        self.session.flush()
        memory_id = memory.id

        # Transition to ACTIVE
        memory.status = MemoryStatus.ACTIVE.value
        memory.is_active = True
        self.session.add(
            MemoryAudit(
                memory_id=memory_id,
                action="updated",
                reason="User confirmed hypothesis",
                actor="user",
            )
        )
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory_id)
        self.assertEqual(retrieved.status, MemoryStatus.ACTIVE.value)
        self.assertTrue(retrieved.is_active)

        # Transition to SUPERSEDED
        memory.status = MemoryStatus.SUPERSEDED.value
        memory.is_active = False
        self.session.add(
            MemoryAudit(
                memory_id=memory_id,
                action="updated",
                reason="Superseded by newer memory",
                actor="system",
            )
        )
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory_id)
        self.assertEqual(retrieved.status, MemoryStatus.SUPERSEDED.value)
        self.assertFalse(retrieved.is_active)

        # Transition to FORGOTTEN
        memory.status = MemoryStatus.FORGOTTEN.value
        self.session.add(
            MemoryAudit(
                memory_id=memory_id,
                action="deleted",
                reason="User requested memory be forgotten",
                actor="user",
            )
        )
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory_id)
        self.assertEqual(retrieved.status, MemoryStatus.FORGOTTEN.value)

    def test_memory_sensitivity_levels(self) -> None:
        """Test memory sensitivity tracking."""
        sensitivities = ["public", "private", "confidential"]
        memories = []

        for sensitivity in sensitivities:
            memory = Memory(
                user_id=self.user_id,
                type="fact",
                content=f"Memory marked as {sensitivity}",
                sensitivity=sensitivity,
                status=MemoryStatus.ACTIVE.value,
                is_active=True,
            )
            self.session.add(memory)
            memories.append(memory)

        self.session.flush()
        memory_ids = [m.id for m in memories]

        # Verify all sensitivities stored
        retrieved = self.session.query(Memory).filter(Memory.id.in_(memory_ids)).all()
        retrieved_sensitivities = [m.sensitivity for m in retrieved]
        for sensitivity in sensitivities:
            self.assertIn(sensitivity, retrieved_sensitivities)

    def test_memory_to_conversation_relationship(self) -> None:
        """Test Memory back-reference to source conversation."""
        conversation = Conversation(user_id=self.user_id, title="Test Conversation")
        self.session.add(conversation)
        self.session.flush()

        memory = Memory(
            user_id=self.user_id,
            source_conversation_id=conversation.id,
            type="fact",
            content="From conversation",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.commit()

        # Check forward
        retrieved_memory = self.session.query(Memory).get(memory.id)
        self.assertIsNotNone(retrieved_memory.source_conversation)
        self.assertEqual(retrieved_memory.source_conversation.id, conversation.id)

        # Check backward
        retrieved_conv = self.session.query(Conversation).get(conversation.id)
        self.assertGreater(len(retrieved_conv.memories), 0)

    def test_memory_embedding_storage(self) -> None:
        """Test storing vector embeddings in memory."""
        embedding_json = "[0.1, 0.2, 0.3, 0.4, 0.5]"
        memory = Memory(
            user_id=self.user_id,
            type="fact",
            content="Embedding test",
            embedding=embedding_json,
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertEqual(retrieved.embedding, embedding_json)

    def test_memory_backward_compatibility_memory_type(self) -> None:
        """Test backward compatibility of memory_type field."""
        memory = Memory(
            user_id=self.user_id,
            type="fact",  # New v2 field
            memory_type="fact",  # Backward compatible field
            content="Backward compatible",
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
        )
        self.session.add(memory)
        self.session.commit()

        retrieved = self.session.query(Memory).get(memory.id)
        self.assertEqual(retrieved.type, "fact")
        self.assertEqual(retrieved.memory_type, "fact")


if __name__ == "__main__":
    unittest.main()
