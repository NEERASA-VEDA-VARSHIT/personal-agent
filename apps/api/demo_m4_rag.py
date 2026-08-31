#!/usr/bin/env python
"""End-to-end M4 demonstration: Personalized responses with memory-augmented generation."""

import sys
import tempfile
from datetime import datetime
from unittest.mock import MagicMock
import hashlib

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup path for imports
sys.path.insert(0, "apps/api")

from app.db.models import Base, Conversation, Memory, User
from app.embeddings import EmbeddingService
from app.models import ModelGateway
from app.rag import RAGService


def create_mock_embedding_service():
    """Create a mock embedding service with deterministic embeddings."""
    mock_service = MagicMock(spec=EmbeddingService)

    def mock_embed(texts):
        """Generate consistent mock embeddings for testing."""
        embeddings = []
        for text in texts:
            hash_val = hashlib.md5(text.encode()).hexdigest()
            vector = [float(int(hash_val[i : i + 2], 16)) / 256.0 for i in range(0, len(hash_val), 2)]
            norm = sum(v * v for v in vector) ** 0.5
            if norm > 0:
                vector = [v / norm for v in vector]
            embeddings.append(vector)
        return embeddings

    mock_service.embed_text = MagicMock(side_effect=lambda text: mock_embed([text])[0])

    def mock_embed_memory(db, memory):
        """Store embedding in memory."""
        import json
        embedding_vector = mock_embed([memory.content])[0]
        memory.embedding = json.dumps(embedding_vector)
        db.add(memory)
        db.commit()

    mock_service.embed_memory = MagicMock(side_effect=mock_embed_memory)

    def mock_retrieve(db, user_id, query_text, top_k=5, **kwargs):
        """Return test memories."""
        from app.db.models import Memory
        memories = db.query(Memory).filter(Memory.user_id == user_id, Memory.is_active == True).limit(top_k).all()
        return [(mem, 0.85) for mem in memories]

    mock_service.retrieve_similar_memories = MagicMock(side_effect=mock_retrieve)
    return mock_service


def setup_demo_database():
    """Create an in-memory database with demo data."""
    # Create temporary database
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal, engine


def populate_demo_memories(session):
    """Populate the database with demo memories."""
    # Create user
    user = User(username="alex")
    session.add(user)
    session.flush()

    # Create a conversation context
    conversation = Conversation(user_id=user.id, title="Career and learning goals")
    session.add(conversation)
    session.flush()

    # Add memories about the user (using mock embeddings)
    demo_memories = [
        ("I'm passionate about machine learning and AI", "explicit", 0.95),
        ("I have 5 years of Python development experience", "explicit", 0.95),
        ("I prefer backend and systems work over frontend", "explicit", 0.92),
        ("I value learning opportunities and mentorship", "explicit", 0.90),
        ("I'm interested in distributed systems and scalability", "explicit", 0.88),
    ]

    embedding_service = create_mock_embedding_service()

    for content, mem_type, confidence in demo_memories:
        memory = Memory(
            user_id=user.id,
            source_conversation_id=conversation.id,
            memory_type=mem_type,
            content=content,
            confidence=confidence,
            is_active=True,
        )
        session.add(memory)
        session.flush()

        # Generate and store embedding (using mock)
        embedding_service.embed_memory(session, memory)

    session.commit()
    return user.id


def run_rag_demo(session, user_id):
    """Run the RAG pipeline with a few example questions."""
    gateway = ModelGateway()
    
    # Use mock embedding service for retrieval (real LLM for responses)
    mock_embedding_service = create_mock_embedding_service()
    rag_service = RAGService(gateway=gateway, embedding_service=mock_embedding_service)

    questions = [
        "What kind of role would be a good fit for me?",
        "Should I focus more on learning new technologies or deepening my current skills?",
    ]

    print("\n" + "=" * 80)
    print("M4: PERSONALIZED RESPONSES WITH MEMORY-AUGMENTED GENERATION")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Question {i}: {question}")
        print("─" * 80)

        # Generate personalized response
        result = rag_service.generate_response(
            session,
            user_id=user_id,
            user_question=question,
            top_k=3,
            temperature=0.7,
            include_memory_citations=False,
        )

        # Display results
        print("\n📚 Memories Retrieved:")
        for mem in result["memories_used"]:
            print(f"  • {mem['content']} (confidence: {mem['confidence']:.2f})")

        print("\n🤖 Response:")
        print(f"  {result['response']}")


if __name__ == "__main__":
    print("Initializing M4 demonstration...")
    print("Setting up database and loading demo memories...\n")

    # Setup
    SessionLocal, engine = setup_demo_database()
    session = SessionLocal()

    try:
        # Populate demo data
        user_id = populate_demo_memories(session)
        print("✓ Demo memories loaded successfully")

        # Run RAG demo with real LLM
        print("\nCalling Ollama on localhost:11434 for personalized responses...\n")
        run_rag_demo(session, user_id)

        print("\n" + "=" * 80)
        print("M4 Complete: Memory-augmented LLM responses working!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        session.close()
        engine.dispose()
