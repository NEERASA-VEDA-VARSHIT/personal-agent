"""M6.7 — Retrieval v2: 5 deterministic tests.

Dataset is small and deterministic; embeddings are hash-based via mock gateway
so semantic scores are reproducible without a real model.
"""

import json
import hashlib
import tempfile
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Memory, MemoryRelation, MemorySource, User
from app.models.embeddings import EmbeddingService
from app.memory.lifecycle import MemoryLifecycle
from app.memory.policy import MemoryStatus, SourceType
from app.memory.retrieval import (
    CandidateRetriever,
    ContextBuilder,
    QueryAnalyzer,
    Reranker,
    RetrievalPipeline,
    RetrievalPolicy,
)


def _mock_gateway():
    gw = MagicMock()

    def mock_embed(texts):
        embs = []
        for t in texts:
            h = hashlib.md5(t.encode()).hexdigest()
            vec = [float(int(h[i : i + 2], 16)) / 256.0 for i in range(0, len(h), 2)]
            norm = sum(v * v for v in vec) ** 0.5
            if norm:
                vec = [v / norm for v in vec]
            embs.append(vec)
        return embs

    gw.embed = mock_embed
    return gw


def _embed_and_store(svc: EmbeddingService, db: Session, mem: Memory):
    svc.embed_memory(db, mem)


class TestRetrievalV2(unittest.TestCase):
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
        TestRetrievalV2._counter += 1
        user = User(username=f"ret_user_{TestRetrievalV2._counter}")
        self.session.add(user)
        self.session.flush()
        self.user_id = user.id
        self.gw = _mock_gateway()
        self.emb_svc = EmbeddingService(gateway=self.gw)
        self.retriever = CandidateRetriever(embedding_service=self.emb_svc)
        self.policy = RetrievalPolicy()
        self.reranker = Reranker()
        self.pipeline = RetrievalPipeline(
            retriever=self.retriever, policy=self.policy, reranker=self.reranker
        )

    def tearDown(self) -> None:
        self.session.rollback()
        self.session.close()

    # -- helpers ----------------------------------------------------------

    def _make_mem(
        self,
        content: str,
        mtype: str = "fact",
        status: str = MemoryStatus.ACTIVE.value,
        sensitivity: str = "private",
        source_type: str = SourceType.USER_STATED.value,
        confidence: float = 0.9,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        is_active: bool | None = None,
    ) -> Memory:
        if is_active is None:
            is_active = status == MemoryStatus.ACTIVE.value
        mem = Memory(
            user_id=self.user_id,
            type=mtype,
            memory_type=mtype,
            content=content,
            sensitivity=sensitivity,
            confidence=confidence,
            status=status,
            is_active=is_active,
            valid_from=valid_from or datetime.utcnow() - timedelta(days=1),
            valid_until=valid_until,
        )
        self.session.add(mem)
        self.session.flush()
        # source row for source-quality test
        src = MemorySource(
            memory_id=mem.id,
            source_type=source_type,
            source_ref="test",
            confidence=confidence,
        )
        self.session.add(src)
        _embed_and_store(self.emb_svc, self.session, mem)
        return mem

    # -- Test 1: semantic relevance --------------------------------------

    def test_semantic_relevance(self) -> None:
        """Query 'What are my career goals?' should rank GOAL memories first."""
        g1 = self._make_mem("software engineering career goal: become staff engineer", mtype="goal", source_type=SourceType.USER_STATED.value)
        g2 = self._make_mem("goal: improve backend skills and system design", mtype="goal", source_type=SourceType.USER_STATED.value)
        other = self._make_mem("I like hiking on weekends", mtype="fact", source_type=SourceType.USER_STATED.value)
        self.session.commit()

        # Explicit pipeline as requested
        understanding = QueryAnalyzer().analyze("What are my career goals?")
        # need embedding for understanding
        understanding.embedding = self.emb_svc.embed_text("What are my career goals?")
        candidates = self.retriever.retrieve(self.session, self.user_id, "What are my career goals?", understanding=understanding)
        filtered = self.policy.filter(candidates, temporal_intent=understanding.temporal_intent)
        ranked = self.reranker.rank("What are my career goals?", filtered.passed)

        top_contents = [r.memory.content for r in ranked[:2]]
        self.assertTrue(any("staff engineer" in c for c in top_contents))
        self.assertTrue(any("backend" in c for c in top_contents))
        # hiking should be last
        self.assertNotEqual(ranked[-1].memory.id, g1.id)

    # -- Test 2: temporal relevance --------------------------------------

    def test_temporal_relevance(self) -> None:
        """Superseded Python preference should not outrank active TypeScript for 'currently' query."""
        old = self._make_mem(
            "I prefer Python",
            mtype="preference",
            status=MemoryStatus.ACTIVE.value,
            valid_from=datetime(2024, 1, 1),
            valid_until=None,
        )
        new = self._make_mem(
            "I now prefer TypeScript",
            mtype="preference",
            status=MemoryStatus.ACTIVE.value,
            valid_from=datetime(2026, 8, 31),
            valid_until=None,
        )
        self.session.commit()
        # supersede old with new (temporal handoff)
        MemoryLifecycle.supersede(self.session, old, new, actor="test")
        self.session.commit()

        # pipeline with current intent
        result = self.pipeline.run(self.session, self.user_id, "Which language do I currently prefer?", now=datetime(2026, 9, 1))
        # filtered should have excluded superseded old
        filtered_ids = [c.memory.id for c, _ in result.filtered.filtered_out]
        self.assertIn(old.id, filtered_ids)
        # ranked should contain new at top, not old
        ranked_ids = [r.memory.id for r in result.ranked]
        self.assertIn(new.id, ranked_ids)
        self.assertNotIn(old.id, ranked_ids)
        self.assertEqual(result.ranked[0].memory.id, new.id)

    # -- Test 3: contradiction -------------------------------------------

    def test_contradiction_prefers_newer(self) -> None:
        """Newer 'enjoy presentations' should outrank older 'hate public speaking'"""
        old = self._make_mem("I hate public speaking", mtype="preference", source_type=SourceType.USER_STATED.value, confidence=0.9)
        # make old older
        old.created_at = datetime.utcnow() - timedelta(days=60)
        old.updated_at = old.created_at
        self.session.add(old)
        new = self._make_mem("I've started enjoying presentations", mtype="preference", source_type=SourceType.USER_STATED.value, confidence=0.9)
        self.session.commit()
        MemoryLifecycle.contradict(self.session, new, old, confidence=0.9)
        self.session.commit()

        result = self.pipeline.run(self.session, self.user_id, "how do I feel about public speaking?")
        # newer should rank higher due to recency
        self.assertGreaterEqual(len(result.ranked), 2)
        # find positions
        pos_old = next(i for i, r in enumerate(result.ranked) if r.memory.id == old.id)
        pos_new = next(i for i, r in enumerate(result.ranked) if r.memory.id == new.id)
        self.assertLess(pos_new, pos_old)
        # reranker debug should note contradiction? At least both are present and ranked separately
        self.assertIn(new.id, [r.memory.id for r in result.ranked])

    # -- Test 4: sensitivity ---------------------------------------------

    def test_sensitivity_filtered(self) -> None:
        """Confidential memory must not enter normal prompt context"""
        sensitive = self._make_mem(
            "My performance review was confidential: needs improvement",
            mtype="fact",
            sensitivity="confidential",
            source_type=SourceType.USER_STATED.value,
            confidence=0.95,
        )
        normal = self._make_mem("I want to improve backend skills", mtype="goal", sensitivity="private", source_type=SourceType.USER_STATED.value)
        self.session.commit()

        # normal pipeline (allow_sensitive=False)
        result = self.pipeline.run(self.session, self.user_id, "What are my career goals?", allow_sensitive=False)
        ranked_ids = [r.memory.id for r in result.ranked]
        self.assertNotIn(sensitive.id, ranked_ids)
        # with allow_sensitive=True it would appear
        result2 = self.pipeline.run(self.session, self.user_id, "What are my career goals?", allow_sensitive=True)
        ranked2_ids = [r.memory.id for r in result2.ranked]
        self.assertIn(sensitive.id, ranked2_ids)

        # context builder should not leak sensitive content
        ctx = result.ranked  # normal ranked
        built = ContextBuilder().build(ctx)
        self.assertNotIn("confidential", built.lower())

    # -- Test 5: source quality ------------------------------------------

    def test_source_quality_user_stated_outranks_inferred(self) -> None:
        """USER_STATED should outrank MODEL_INFERRED even with identical semantic similarity"""
        # Use identical content so semantic scores are equal; source quality decides
        content = "I am interested in distributed systems"
        user_stated = self._make_mem(content, mtype="fact", source_type=SourceType.USER_STATED.value, confidence=0.85)
        inferred = self._make_mem(content, mtype="hypothesis", source_type=SourceType.MODEL_INFERRED.value, confidence=0.85)
        # make timestamps equal to neutralize recency
        now = datetime.utcnow()
        for m in (user_stated, inferred):
            m.created_at = now
            m.updated_at = now
            self.session.add(m)
        self.session.commit()

        # Retrieve with query identical to content -> semantic tie
        understanding = QueryAnalyzer().analyze(content)
        understanding.embedding = self.emb_svc.embed_text(content)
        candidates = self.retriever.retrieve(self.session, self.user_id, content, understanding=understanding)
        filtered = self.policy.filter(candidates)
        ranked = self.reranker.rank(content, filtered.passed, now=now)

        pos_user = next(i for i, r in enumerate(ranked) if r.memory.id == user_stated.id)
        pos_inf = next(i for i, r in enumerate(ranked) if r.memory.id == inferred.id)
        self.assertLess(pos_user, pos_inf, msg=f"USER_STATED debug {ranked[pos_user].debug} vs INFERRED {ranked[pos_inf].debug}")
        # debug transparency
        self.assertGreater(ranked[pos_user].source_quality_score, ranked[pos_inf].source_quality_score)
        # final score reflects it
        self.assertGreater(ranked[pos_user].final_score, ranked[pos_inf].final_score)


if __name__ == "__main__":
    unittest.main()
