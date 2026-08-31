"""M6.6 — Memory Lifecycle tests."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from personal_agent.persistence.models import Base, Memory, MemoryAudit, MemoryRelation, User
from personal_agent.memory.lifecycle import MemoryLifecycle, LifecycleError
from personal_agent.memory.policy import EvidenceStrength, MemoryStatus, SourceType, Stability


class TestMemoryLifecycle(unittest.TestCase):
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
        os.close(cls.db_fd)
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    _counter = 0

    def setUp(self) -> None:
        self.session: Session = self.SessionLocal()
        TestMemoryLifecycle._counter += 1
        user = User(username=f"lc_user_{TestMemoryLifecycle._counter}")
        self.session.add(user)
        self.session.flush()
        self.user_id = user.id

    def tearDown(self) -> None:
        self.session.rollback()
        self.session.close()

    # -- candidate -> active ---------------------------------------------

    def test_candidate_to_active(self) -> None:
        mem = MemoryLifecycle.create_candidate(
            self.session, user_id=self.user_id, type="fact", content="candidate fact"
        )
        self.session.commit()
        self.assertEqual(mem.status, MemoryStatus.CANDIDATE.value)
        self.assertFalse(mem.is_active)

        MemoryLifecycle.approve(self.session, mem, actor="user", reason="confirmed")
        self.session.commit()
        self.assertEqual(mem.status, MemoryStatus.ACTIVE.value)
        self.assertTrue(mem.is_active)
        # audit trail exists
        audits = self.session.query(MemoryAudit).filter(MemoryAudit.memory_id == mem.id).all()
        actions = [a.action for a in audits]
        self.assertIn("confirmed", actions)

    # -- candidate -> rejected -------------------------------------------

    def test_candidate_to_rejected(self) -> None:
        mem = MemoryLifecycle.create_candidate(
            self.session, user_id=self.user_id, type="hypothesis", content="weak hypothesis"
        )
        self.session.commit()
        MemoryLifecycle.reject(self.session, mem, actor="system", reason="weak evidence")
        self.session.commit()
        self.assertEqual(mem.status, MemoryStatus.REJECTED.value)
        self.assertFalse(mem.is_active)

    # -- active -> superseded (temporal) --------------------------------

    def test_active_to_superseded_with_temporal_validity(self) -> None:
        old = MemoryLifecycle.create_active(
            self.session,
            user_id=self.user_id,
            type="preference",
            content="I prefer Python",
            valid_from=datetime(2026, 1, 1),
            valid_until=None,
        )
        new = MemoryLifecycle.create_active(
            self.session,
            user_id=self.user_id,
            type="preference",
            content="I now prefer TypeScript",
            valid_from=datetime(2026, 8, 31),
        )
        self.session.commit()

        rel = MemoryLifecycle.supersede(self.session, old, new, actor="user")
        self.session.commit()

        self.assertEqual(old.status, MemoryStatus.SUPERSEDED.value)
        self.assertFalse(old.is_active)
        self.assertEqual(old.valid_until, datetime(2026, 8, 31))
        self.assertEqual(new.valid_from, datetime(2026, 8, 31))
        self.assertEqual(new.status, MemoryStatus.ACTIVE.value)
        self.assertTrue(new.is_active)
        self.assertEqual(rel.relation_type, "supersedes")
        self.assertEqual(rel.from_memory_id, new.id)
        self.assertEqual(rel.to_memory_id, old.id)

        # query "what do I currently prefer?" returns new
        cur = MemoryLifecycle.get_current_preference(self.session, self.user_id, "prefer")
        self.assertIsNotNone(cur)
        self.assertEqual(cur.id, new.id)

    # -- active -> forgotten ---------------------------------------------

    def test_active_to_forgotten(self) -> None:
        mem = MemoryLifecycle.create_active(
            self.session, user_id=self.user_id, type="fact", content="forget me"
        )
        self.session.commit()
        MemoryLifecycle.forget(self.session, mem, actor="user", reason="user requested")
        self.session.commit()
        self.assertEqual(mem.status, MemoryStatus.FORGOTTEN.value)
        self.assertFalse(mem.is_active)
        audits = self.session.query(MemoryAudit).filter(MemoryAudit.memory_id == mem.id).all()
        self.assertTrue(any(a.action == "forgotten" for a in audits))

    # -- contradiction relation ------------------------------------------

    def test_contradiction_relation(self) -> None:
        m1 = MemoryLifecycle.create_active(self.session, user_id=self.user_id, type="preference", content="Prefers remote")
        m2 = MemoryLifecycle.create_active(self.session, user_id=self.user_id, type="preference", content="Prefers office")
        self.session.commit()
        rel = MemoryLifecycle.contradict(self.session, m1, m2, confidence=0.9)
        self.session.commit()
        self.assertEqual(rel.relation_type, "contradicts")
        # persisted via relation table
        fetched = self.session.query(MemoryRelation).filter(MemoryRelation.from_memory_id == m1.id).first()
        self.assertIsNotNone(fetched)

    # -- temporal validity ------------------------------------------------

    def test_temporal_validity_nullable_valid_until(self) -> None:
        mem = MemoryLifecycle.create_active(
            self.session, user_id=self.user_id, type="fact", content="permanent fact", valid_until=None
        )
        self.session.commit()
        self.assertIsNone(mem.valid_until)

    # -- provenance + audit trail ----------------------------------------

    def test_provenance_and_audit_trail(self) -> None:
        mem = MemoryLifecycle.create_active(
            self.session,
            user_id=self.user_id,
            type="fact",
            content="provenance test",
            sensitivity="private",
            source_conversation_id=None,
        )
        self.session.commit()
        self.assertEqual(mem.sensitivity, "private")
        audits = self.session.query(MemoryAudit).filter(MemoryAudit.memory_id == mem.id).all()
        self.assertGreaterEqual(len(audits), 1)

    # -- invalid transition ----------------------------------------------

    def test_invalid_transition_raises(self) -> None:
        mem = MemoryLifecycle.create_active(self.session, user_id=self.user_id, type="fact", content="x")
        self.session.commit()
        # ACTIVE -> REJECTED is invalid per VALID_TRANSITIONS
        with self.assertRaises(LifecycleError):
            MemoryLifecycle.reject(self.session, mem)

    def test_supersede_requires_active_old(self) -> None:
        cand = MemoryLifecycle.create_candidate(self.session, user_id=self.user_id, type="fact", content="cand")
        new = MemoryLifecycle.create_active(self.session, user_id=self.user_id, type="fact", content="new active")
        self.session.commit()
        with self.assertRaises(LifecycleError):
            MemoryLifecycle.supersede(self.session, cand, new)


if __name__ == "__main__":
    unittest.main()
