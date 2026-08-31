"""SQLAlchemy ORM models for memory persistence — Memory v2 schema."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base for all models."""

    pass


class User(Base):
    """User entity."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    conversations: Mapped[list[Conversation]] = relationship("Conversation", back_populates="user")
    memories: Mapped[list[Memory]] = relationship("Memory", back_populates="user")
    decisions: Mapped[list[Decision]] = relationship("Decision", back_populates="user")


class Conversation(Base):
    """Conversation entity."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="conversations")
    messages: Mapped[list[Message]] = relationship("Message", back_populates="conversation")
    memories: Mapped[list[Memory]] = relationship("Memory", back_populates="source_conversation")


class Message(Base):
    """Message entity."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("conversations.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages")


class Memory(Base):
    """Memory entity — v2 schema.

    The ``type`` field represents what the memory *is* (Fact, Preference, Goal, …).
    How it was produced is tracked separately via ``source_type`` on MemorySource.
    ``memory_type`` is kept for backward compatibility with existing data.
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    source_conversation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("conversations.id"), nullable=True)

    # Backward-compatible field — legacy memory classification
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False, default="fact")

    # New v2 memory type — what the memory represents
    type: Mapped[str] = mapped_column(String(50), nullable=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Sensitivity level
    sensitivity: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Confidence — retained but redefined: represents evidence quality, not truth probability
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Lifecycle status
    status: Mapped[str] = mapped_column(String(50), nullable=True, default="active")

    # Temporal validity
    valid_from: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Provenance
    source_message_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("messages.id"), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Integer, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="memories")
    source_conversation: Mapped[Optional[Conversation]] = relationship("Conversation", back_populates="memories")
    sources: Mapped[list[MemorySource]] = relationship("MemorySource", back_populates="memory")
    relations_from: Mapped[list["MemoryRelation"]] = relationship(
        "MemoryRelation", foreign_keys="MemoryRelation.from_memory_id", back_populates="from_memory", cascade="all, delete-orphan"
    )
    relations_to: Mapped[list["MemoryRelation"]] = relationship(
        "MemoryRelation", foreign_keys="MemoryRelation.to_memory_id", back_populates="to_memory", cascade="all, delete-orphan"
    )
    audits: Mapped[list["MemoryAudit"]] = relationship("MemoryAudit", back_populates="memory", cascade="all, delete-orphan")
    events: Mapped[list[MemoryEvent]] = relationship("MemoryEvent", back_populates="memory")


class MemorySource(Base):
    """Source citation for a memory."""

    __tablename__ = "memory_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, default="user_stated")
    source_ref: Mapped[str] = mapped_column(String(500), nullable=False)
    source_message_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("messages.id"), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    memory: Mapped[Memory] = relationship("Memory", back_populates="sources")


class MemoryRelation(Base):
    """Relationship between memories (supports, contradicts, supersedes, related_to)."""

    __tablename__ = "memory_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    to_memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    from_memory: Mapped[Memory] = relationship("Memory", foreign_keys=[from_memory_id], back_populates="relations_from")
    to_memory: Mapped[Memory] = relationship("Memory", foreign_keys=[to_memory_id], back_populates="relations_to")


class MemoryAudit(Base):
    """Lifecycle audit trail for memories."""

    __tablename__ = "memory_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    memory: Mapped[Memory] = relationship("Memory", back_populates="audits")


class MemoryEvent(Base):
    """Legacy audit log for memory lifecycle events — kept for backward compatibility."""

    __tablename__ = "memory_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    memory: Mapped[Memory] = relationship("Memory", back_populates="events")


class Decision(Base):
    """Decision entity for tracking important choices."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=True)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="decisions")