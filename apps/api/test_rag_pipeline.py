"""Test retrieval-augmented generation (RAG) pipeline."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Conversation, Memory, User
from app.embeddings import EmbeddingService
from app.rag import RAGService


class TestRAGPipeline(unittest.TestCase):
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
        """Set up test fixtures."""
        self.session: Session = self.SessionLocal()

        # Mock gateway and embedding service
        self.mock_gateway = MagicMock()
        self.mock_embedding_service = self._create_mock_embedding_service()
        self.rag_service = RAGService(
            gateway=self.mock_gateway,
            embedding_service=self.mock_embedding_service,
        )

    def _create_mock_embedding_service(self) -> EmbeddingService:
        """Create a mock embedding service with deterministic embeddings."""
        mock_service = MagicMock(spec=EmbeddingService)

        def mock_embed(texts):
            """Generate consistent mock embeddings for testing."""
            import hashlib

            embeddings = []
            for text in texts:
                hash_val = hashlib.md5(text.encode()).hexdigest()
                vector = [float(int(hash_val[i : i + 2], 16)) / 256.0 for i in range(0, len(hash_val), 2)]
                norm = sum(v * v for v in vector) ** 0.5
                if norm > 0:
                    vector = [v / norm for v in vector]
                embeddings.append(vector)
            return embeddings

        # Mock embed method
        mock_service.embed_text = MagicMock(side_effect=lambda text: mock_embed([text])[0])

        # Mock retrieve_similar_memories to return memories we set up
        def mock_retrieve(db, user_id, query_text, top_k=5, **kwargs):
            """Return test memories."""
            memories = db.query(Memory).filter(Memory.user_id == user_id, Memory.is_active == True).limit(top_k).all()
            # Return as (memory, similarity_score) tuples
            return [(mem, 0.85) for mem in memories]

        mock_service.retrieve_similar_memories = MagicMock(side_effect=mock_retrieve)
        return mock_service

    def tearDown(self) -> None:
        """Clean up session."""
        self.session.rollback()
        self.session.close()

    def test_rag_service_retrieves_and_augments(self) -> None:
        """Test that RAG service retrieves memories and augments prompt."""
        # Create a test user
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()

        # Create a conversation context
        conv = Conversation(user_id=user.id, title="Career discussion")
        self.session.add(conv)
        self.session.flush()

        # Add memories about the user
        memories_data = [
            ("I'm currently learning machine learning", "explicit", 0.95),
            ("I prefer backend development", "explicit", 0.90),
            ("I value learning opportunities in my work", "explicit", 0.92),
        ]

        for content, mem_type, confidence in memories_data:
            memory = Memory(
                user_id=user.id,
                source_conversation_id=conv.id,
                memory_type=mem_type,
                content=content,
                confidence=confidence,
                is_active=True,
            )
            self.session.add(memory)
        self.session.commit()

        # Mock the LLM response
        self.mock_gateway.generate = MagicMock(
            return_value="Based on your background, I think you should focus on ML engineering roles that emphasize backend systems."
        )

        # Generate a personalized response
        result = self.rag_service.generate_response(
            self.session,
            user_id=user.id,
            user_question="What career direction should I consider?",
            top_k=3,
        )

        # Verify response was generated
        self.assertIn("response", result)
        self.assertIsInstance(result["response"], str)
        self.assertGreater(len(result["response"]), 0)

        # Verify memories were retrieved
        self.assertIn("memories_used", result)
        self.assertEqual(len(result["memories_used"]), 3)

        # Verify memory content is present in the result
        retrieved_contents = [mem["content"] for mem in result["memories_used"]]
        self.assertTrue(any("machine learning" in c.lower() for c in retrieved_contents))
        self.assertTrue(any("backend" in c.lower() for c in retrieved_contents))

        # Verify the gateway was called with an augmented prompt
        self.mock_gateway.generate.assert_called_once()
        call_args = self.mock_gateway.generate.call_args
        messages = call_args[0][0]

        # Check that system prompt was included
        self.assertTrue(any("personal AI" in m.get("content", "") for m in messages))

        # Check that user question was included
        self.assertTrue(any("career direction" in m.get("content", "") for m in messages))

    def test_rag_builds_correct_memory_context(self) -> None:
        """Test that RAG correctly formats memory context."""
        memories_with_scores = [
            (MagicMock(content="I like Python", confidence=0.95), 0.88),
            (MagicMock(content="I enjoy distributed systems", confidence=0.80), 0.82),
        ]

        context = self.rag_service._build_memory_context(memories_with_scores)

        # Verify format
        self.assertIn("User's Relevant Memories", context)
        self.assertIn("I like Python", context)
        self.assertIn("distributed systems", context)
        self.assertIn("✓", context)  # High confidence marker

    def test_rag_handles_no_memories_gracefully(self) -> None:
        """Test that RAG works even when no memories are retrieved."""
        user = User(username="new_user")
        self.session.add(user)
        self.session.commit()

        # Mock retrieval to return no memories
        self.mock_embedding_service.retrieve_similar_memories = MagicMock(return_value=[])

        # Mock LLM response
        self.mock_gateway.generate = MagicMock(
            return_value="I don't have enough context about your background yet, but I'm happy to help!"
        )

        result = self.rag_service.generate_response(
            self.session,
            user_id=user.id,
            user_question="Tell me about myself",
        )

        # Should still produce a response
        self.assertIn("response", result)
        self.assertEqual(len(result["memories_used"]), 0)

        # Gateway should still be called
        self.mock_gateway.generate.assert_called_once()

    def test_rag_includes_memory_citations_when_requested(self) -> None:
        """Test that memory citations are appended to response when requested."""
        user = User(username="citation_test")
        self.session.add(user)
        self.session.flush()

        memory = Memory(
            user_id=user.id,
            memory_type="explicit",
            content="I love building APIs",
            confidence=0.95,
            is_active=True,
        )
        self.session.add(memory)
        self.session.commit()

        # Mock retrieval
        self.mock_embedding_service.retrieve_similar_memories = MagicMock(
            return_value=[(memory, 0.90)]
        )

        # Mock LLM response
        base_response = "You should focus on API design and scalability."
        self.mock_gateway.generate = MagicMock(return_value=base_response)

        result = self.rag_service.generate_response(
            self.session,
            user_id=user.id,
            user_question="Where should I focus?",
            include_memory_citations=True,
        )

        # Response should include citations
        self.assertIn("Memory #", result["response"])
        self.assertIn("building APIs", result["response"])


if __name__ == "__main__":
    unittest.main()
