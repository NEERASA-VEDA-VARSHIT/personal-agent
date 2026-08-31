"""Test retrieval-augmented generation (RAG) pipeline — M6.10.1 single path."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Conversation, Memory, User
from app.memory.rag import RAGService


class TestRAGPipeline(unittest.TestCase):
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

    def setUp(self) -> None:
        self.session: Session = self.SessionLocal()
        self.mock_gateway = MagicMock()
        self.mock_pipeline = self._create_mock_pipeline()
        self.rag_service = RAGService(gateway=self.mock_gateway, pipeline=self.mock_pipeline)

    def _create_mock_pipeline(self):
        mock_pipeline = MagicMock()

        def mock_run(db, user_id, query, top_k=5, **kwargs):
            memories = db.query(Memory).filter(Memory.user_id == user_id, Memory.is_active == True).limit(top_k).all()  # noqa: E712
            ranked = []
            for mem in memories:
                r = MagicMock()
                r.memory = mem
                r.final_score = 0.85
                r.semantic_score = 0.85
                r.debug = {"final score": 0.85}
                ranked.append(r)
            result = MagicMock()
            result.ranked = ranked
            result.understanding = MagicMock(raw_query=query, temporal_intent="any")
            result.candidates = []
            result.filtered = MagicMock(filtered_out=[])
            return result

        mock_pipeline.run = MagicMock(side_effect=mock_run)
        return mock_pipeline

    def tearDown(self) -> None:
        self.session.rollback()
        self.session.close()

    def test_rag_service_retrieves_and_augments(self) -> None:
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()
        conv = Conversation(user_id=user.id, title="Career discussion")
        self.session.add(conv)
        self.session.flush()
        memories_data = [
            ("I'm currently learning machine learning", "explicit", 0.95),
            ("I prefer backend development", "explicit", 0.90),
            ("I value learning opportunities in my work", "explicit", 0.92),
        ]
        for content, mem_type, confidence in memories_data:
            memory = Memory(user_id=user.id, source_conversation_id=conv.id, memory_type=mem_type, content=content, confidence=confidence, is_active=True)
            self.session.add(memory)
        self.session.commit()
        self.mock_gateway.generate = MagicMock(return_value="Based on your background, I think you should focus on ML engineering roles that emphasize backend systems.")
        result = self.rag_service.generate_response(self.session, user_id=user.id, user_question="What career direction should I consider?", top_k=3)
        self.assertIn("response", result)
        self.assertIsInstance(result["response"], str)
        self.assertGreater(len(result["response"]), 0)
        self.assertIn("memories_used", result)
        self.assertEqual(len(result["memories_used"]), 3)
        retrieved_contents = [mem["content"] for mem in result["memories_used"]]
        self.assertTrue(any("machine learning" in c.lower() for c in retrieved_contents))
        self.assertTrue(any("backend" in c.lower() for c in retrieved_contents))
        self.mock_gateway.generate.assert_called_once()
        call_args = self.mock_gateway.generate.call_args
        messages = call_args[0][0]
        self.assertTrue(any("personal AI" in m.get("content", "") for m in messages))
        self.assertTrue(any("career direction" in m.get("content", "") for m in messages))

    def test_rag_builds_correct_memory_context(self) -> None:
        memories_with_scores = [(MagicMock(content="I like Python", confidence=0.95, type="fact", memory_type="fact", sources=[]), 0.88)]
        # legacy tuple path still supported
        context = self.rag_service._build_memory_context(memories_with_scores)
        self.assertIn("User's Relevant Memories", context)
        self.assertIn("I like Python", context)

    def test_rag_handles_no_memories_gracefully(self) -> None:
        user = User(username="new_user")
        self.session.add(user)
        self.session.commit()
        # pipeline returns empty ranked
        def empty_run(db, user_id, query, top_k=5, **kwargs):
            result = MagicMock()
            result.ranked = []
            result.understanding = MagicMock(raw_query=query, temporal_intent="any")
            result.candidates = []
            result.filtered = MagicMock(filtered_out=[])
            return result

        self.mock_pipeline.run = MagicMock(side_effect=empty_run)
        self.mock_gateway.generate = MagicMock(return_value="I don't have enough context about your background yet, but I'm happy to help!")
        result = self.rag_service.generate_response(self.session, user_id=user.id, user_question="Tell me about myself")
        self.assertIn("response", result)
        self.assertEqual(len(result["memories_used"]), 0)
        self.mock_gateway.generate.assert_called_once()

    def test_rag_includes_memory_citations_when_requested(self) -> None:
        user = User(username="citation_test")
        self.session.add(user)
        self.session.flush()
        memory = Memory(user_id=user.id, memory_type="explicit", content="I love building APIs", confidence=0.95, is_active=True)
        self.session.add(memory)
        self.session.commit()

        def single_run(db, user_id, query, top_k=5, **kwargs):
            r = MagicMock()
            r.memory = memory
            r.final_score = 0.90
            r.semantic_score = 0.90
            r.debug = {}
            result = MagicMock()
            result.ranked = [r]
            result.understanding = MagicMock(raw_query=query, temporal_intent="any")
            result.candidates = []
            result.filtered = MagicMock(filtered_out=[])
            return result

        self.mock_pipeline.run = MagicMock(side_effect=single_run)
        base_response = "You should focus on API design and scalability."
        self.mock_gateway.generate = MagicMock(return_value=base_response)
        result = self.rag_service.generate_response(self.session, user_id=user.id, user_question="Where should I focus?", include_memory_citations=True)
        self.assertIn("Memory #", result["response"])
        self.assertIn("building APIs", result["response"])


if __name__ == "__main__":
    unittest.main()
