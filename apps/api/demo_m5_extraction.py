#!/usr/bin/env python
"""End-to-end M5 demonstration: Autonomous memory extraction with policy validation."""

import sys
import tempfile
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup path for imports
sys.path.insert(0, "apps/api")

from app.db.models import Base, Conversation, Message, User
from app.memory_extraction import MemoryExtractionService
from app.memory_policy import MemoryPolicy


def setup_demo_database():
    """Create an in-memory database for demonstration."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal, engine


def create_demo_conversation(session):
    """Create a sample conversation to extract memories from."""
    user = User(username="alex")
    session.add(user)
    session.flush()

    conversation = Conversation(user_id=user.id, title="Career and Learning Discussion")
    session.add(conversation)
    session.flush()

    # Add realistic conversation messages
    messages = [
        ("user", "I've been thinking about my career direction. I want to become a staff engineer in 5 years."),
        (
            "assistant",
            "That's a great long-term goal. Staff engineer roles typically require deep expertise and leadership. What areas are you most interested in?",
        ),
        (
            "user",
            "I love working on distributed systems and infrastructure. I've been using Python for the last 5 years, and I think it's the best language for most of my work.",
        ),
        (
            "assistant",
            "Python is excellent for infrastructure work. Are there other areas you're exploring?",
        ),
        ("user", "I've been learning Go lately. It seems promising for systems programming. I also think learning Rust could be valuable."),
        (
            "user",
            "I really value learning opportunities at work. I prefer companies that invest in their engineers' growth. I'm also interested in mentoring junior developers.",
        ),
        (
            "assistant",
            "That's a well-rounded perspective. It sounds like you're building a strong technical foundation while also thinking about growth and mentorship.",
        ),
    ]

    for role, content in messages:
        message = Message(conversation_id=conversation.id, role=role, content=content)
        session.add(message)

    session.commit()
    return user.id, conversation.id


def run_m5_demo():
    """Run the M5 memory extraction demonstration."""
    print("\n" + "=" * 80)
    print("M5: AUTONOMOUS MEMORY EXTRACTION WITH POLICY VALIDATION")
    print("=" * 80)

    # Setup
    SessionLocal, engine = setup_demo_database()
    session = SessionLocal()

    try:
        # Create demo conversation
        print("\n📝 Creating sample conversation...")
        user_id, conversation_id = create_demo_conversation(session)
        print(f"   ✓ Conversation created (ID: {conversation_id})")

        # Display conversation
        conversation = session.query(Conversation).get(conversation_id)
        print("\n📖 Conversation Content:")
        print("─" * 80)
        for message in conversation.messages:
            role = "👤 User" if message.role == "user" else "🤖 Assistant"
            print(f"{role}: {message.content}\n")

        # Create memory extraction service with policy
        print("\n🔍 Extracting candidate memories...")
        policy = MemoryPolicy(candidate_confidence_threshold=0.75, hypothesis_confidence_threshold=0.60)

        # Mock LLM response with realistic extracted memories
        mock_response = """
        {
          "memories": [
            {
              "type": "explicit",
              "content": "Goal: become a staff engineer in 5 years",
              "confidence": 0.99,
              "reason": "User directly stated career goal",
              "source_markers": ["I want to become a staff engineer in 5 years"]
            },
            {
              "type": "explicit",
              "content": "5 years of Python development experience",
              "confidence": 0.98,
              "reason": "User explicitly mentioned tenure",
              "source_markers": ["I've been using Python for the last 5 years"]
            },
            {
              "type": "explicit",
              "content": "Prefers Python as best language for infrastructure work",
              "confidence": 0.95,
              "reason": "User clearly stated preference",
              "source_markers": ["I think it's the best language for most of my work"]
            },
            {
              "type": "candidate",
              "content": "Interested in learning Go and Rust",
              "confidence": 0.88,
              "reason": "User mentioned learning these languages",
              "source_markers": ["I've been learning Go lately", "learning Rust could be valuable"]
            },
            {
              "type": "candidate",
              "content": "Values learning opportunities and professional growth",
              "confidence": 0.82,
              "reason": "User emphasized learning in multiple statements",
              "source_markers": ["I really value learning opportunities", "I prefer companies that invest in their engineers' growth"]
            },
            {
              "type": "hypothesis",
              "content": "Interested in mentorship and team development",
              "confidence": 0.75,
              "reason": "User mentioned mentoring junior developers in context of career growth",
              "source_markers": ["I'm also interested in mentoring junior developers", "long-term career goal"]
            },
            {
              "type": "candidate",
              "content": "Passionate about distributed systems and infrastructure",
              "confidence": 0.70,
              "reason": "Strong focus on these areas, but below confidence threshold",
              "source_markers": ["I love working on distributed systems and infrastructure"]
            }
          ]
        }
        """

        mock_gateway = MagicMock()
        mock_gateway.generate = MagicMock(return_value=mock_response)

        extraction_service = MemoryExtractionService(gateway=mock_gateway, policy=policy)

        # Extract memories with policy validation
        result = extraction_service.extract_from_conversation(
            session, user_id=user_id, conversation_id=conversation_id, apply_policy=True
        )

        # Display extraction results
        print(f"\n✅ Extracted {len(result['candidates'])} candidate memories")
        print(f"   • Approved: {len(result['approved'])}")
        print(f"   • Needs Review: {len(result['needs_review'])}")
        print(f"   • Rejected: {len(result['rejected'])}")

        # Show approved memories
        print("\n✓ APPROVED (Will be stored immediately):")
        print("─" * 80)
        for item in result["approved"]:
            candidate = item["candidate"]
            print(f"  [{candidate.memory_type.value.upper()}] {candidate.content}")
            print(f"    Confidence: {candidate.confidence:.2f} | Reason: {item['validation']['reason']}")
            print()

        # Show memories needing review
        if result["needs_review"]:
            print("❓ NEEDS REVIEW (Ask user for confirmation):")
            print("─" * 80)
            for item in result["needs_review"]:
                candidate = item["candidate"]
                print(f"  [{candidate.memory_type.value.upper()}] {candidate.content}")
                print(f"    Confidence: {candidate.confidence:.2f} | Reason: {item['validation']['reason']}")
                print()

        # Show rejected memories
        if result["rejected"]:
            print("✗ REJECTED (Below policy threshold):")
            print("─" * 80)
            for item in result["rejected"]:
                candidate = item["candidate"]
                print(f"  [{candidate.memory_type.value.upper()}] {candidate.content}")
                print(f"    Confidence: {candidate.confidence:.2f} | Reason: {item['validation']['reason']}")
                print()

        # Store approved memories
        print("\n💾 Storing approved memories...")
        stored = extraction_service.store_approved_memories(
            session, user_id=user_id, conversation_id=conversation_id, approved_candidates=result["approved"]
        )
        print(f"   ✓ Stored {len(stored)} memories to database")

        # Verify storage
        from app.db.models import Memory

        all_memories = session.query(Memory).filter(Memory.user_id == user_id).all()
        print(f"\n📊 Total memories for user: {len(all_memories)}")

        print("\n" + "=" * 80)
        print("M5 COMPLETE: Memory extraction and policy validation working!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    run_m5_demo()
