"""Test embeddings and retrieval."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Memory, User
from app.models.embeddings import EmbeddingService


class TestEmbeddingsAndRetrieval(unittest.TestCase):
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
        
        # Use mock gateway to avoid dependency on real embedding model
        mock_gateway = MagicMock()
        
        # Create deterministic embeddings based on text content
        def mock_embed(texts):
            """Generate consistent mock embeddings for testing."""
            embeddings = []
            for text in texts:
                # Simple hash-based embedding for deterministic testing
                import hashlib
                hash_val = hashlib.md5(text.encode()).hexdigest()
                # Convert hash to vector of normalized values
                vector = [float(int(hash_val[i:i+2], 16)) / 256.0 for i in range(0, len(hash_val), 2)]
                # Normalize
                norm = sum(v * v for v in vector) ** 0.5
                if norm > 0:
                    vector = [v / norm for v in vector]
                embeddings.append(vector)
            return embeddings
        
        mock_gateway.embed = mock_embed
        self.embedding_service = EmbeddingService(gateway=mock_gateway)

    def tearDown(self) -> None:
        """Clean up session."""
        self.session.rollback()
        self.session.close()

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Test that identical vectors have similarity 1.0."""
        from app.models.embeddings import cosine_similarity

        vec = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec, vec)
        self.assertAlmostEqual(similarity, 1.0)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Test that orthogonal vectors have similarity 0.0."""
        from app.models.embeddings import cosine_similarity

        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, 0.0)

    def test_embed_text(self) -> None:
        """Test that embedding generation returns a vector."""
        text = "I enjoy backend development"
        embedding = self.embedding_service.embed_text(text)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], (int, float))

    def test_embed_memory_stores_embedding(self) -> None:
        """Test that embedding is stored in memory."""
        user = User(username="test_user")
        self.session.add(user)
        self.session.flush()

        memory = Memory(
            user_id=user.id,
            memory_type="explicit",
            content="I want to work on machine learning",
            confidence=0.9,
        )
        self.session.add(memory)
        self.session.flush()

        # Generate and store embedding
        self.embedding_service.embed_memory(self.session, memory)

        # Verify embedding was stored
        self.assertIsNotNone(memory.embedding)
        embedding_data = json.loads(memory.embedding)
        self.assertIsInstance(embedding_data, list)
        self.assertGreater(len(embedding_data), 0)

    def test_retrieve_similar_memories_returns_results(self) -> None:
        """Test that retrieval returns memories when they exist."""
        user = User(username="test_user2")
        self.session.add(user)
        self.session.flush()

        # Create multiple memories
        memories_content = [
            ("I enjoy backend development", "explicit"),
            ("Frontend is interesting but I prefer APIs", "explicit"),
            ("I like hiking and outdoor activities", "explicit"),
            ("Building scalable systems is important to me", "explicit"),
        ]

        for content, memory_type in memories_content:
            memory = Memory(
                user_id=user.id,
                memory_type=memory_type,
                content=content,
                confidence=0.9,
                is_active=True,
            )
            self.session.add(memory)
            self.session.flush()
            self.embedding_service.embed_memory(self.session, memory)

        # Query for something
        results = self.embedding_service.retrieve_similar_memories(
            self.session,
            user_id=user.id,
            query_text="What are your technical interests?",
            top_k=2,
        )

        # Verify we got results
        self.assertEqual(len(results), 2)
        
        # Verify results are tuples of (memory, score)
        for memory, score in results:
            self.assertIsNotNone(memory.id)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_retrieve_similar_memories_respects_memory_types_filter(self) -> None:
        """Test that retrieval respects memory type filters."""
        user = User(username="test_user3")
        self.session.add(user)
        self.session.flush()

        # Create memories of different types
        explicit_memory = Memory(
            user_id=user.id,
            memory_type="explicit",
            content="I love Python",
            confidence=0.95,
            is_active=True,
        )
        self.session.add(explicit_memory)
        self.session.flush()
        self.embedding_service.embed_memory(self.session, explicit_memory)

        hypothesis_memory = Memory(
            user_id=user.id,
            memory_type="hypothesis",
            content="User might be interested in web frameworks",
            confidence=0.5,
            is_active=True,
        )
        self.session.add(hypothesis_memory)
        self.session.flush()
        self.embedding_service.embed_memory(self.session, hypothesis_memory)

        # Query for explicit memories only
        results = self.embedding_service.retrieve_similar_memories(
            self.session,
            user_id=user.id,
            query_text="Python programming",
            memory_types=["explicit"],
        )

        # Should only get explicit memory
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].memory_type, "explicit")

    def test_retrieve_similar_memories_respects_min_similarity(self) -> None:
        """Test that retrieval respects minimum similarity threshold."""
        user = User(username="test_user4")
        self.session.add(user)
        self.session.flush()

        # Create unrelated memory
        memory = Memory(
            user_id=user.id,
            memory_type="explicit",
            content="I enjoy gardening",
            confidence=0.9,
            is_active=True,
        )
        self.session.add(memory)
        self.session.flush()
        self.embedding_service.embed_memory(self.session, memory)

        # Query for something very different
        results = self.embedding_service.retrieve_similar_memories(
            self.session,
            user_id=user.id,
            query_text="quantum computing algorithms",
            min_similarity=0.8,  # Very high threshold
        )

        # Should return no results (gardening is too dissimilar)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
