"""SQLAlchemy ORM models for memory persistence."""

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
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages")


class Memory(Base):
    """Memory entity."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    source_conversation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("conversations.id"), nullable=True)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'explicit', 'candidate', 'inference', 'hypothesis'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON-serialized vector
    is_active: Mapped[bool] = mapped_column(Integer, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="memories")
    source_conversation: Mapped[Optional[Conversation]] = relationship("Conversation", back_populates="memories")
    sources: Mapped[list[MemorySource]] = relationship("MemorySource", back_populates="memory")
    events: Mapped[list[MemoryEvent]] = relationship("MemoryEvent", back_populates="memory")


class MemorySource(Base):
    """Source citation for a memory."""

    __tablename__ = "memory_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'conversation', 'user_input', 'inference'
    source_ref: Mapped[str] = mapped_column(String(500), nullable=False)  # e.g., conversation_17, user_stated
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    memory: Mapped[Memory] = relationship("Memory", back_populates="sources")


class Decision(Base):
    """Decision entity for tracking important choices."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=True)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=True)  # 0.0-1.0
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="decisions")


class MemoryEvent(Base):
    """Audit log for memory lifecycle events."""

    __tablename__ = "memory_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'created', 'updated', 'superseded', 'deleted'
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    memory: Mapped[Memory] = relationship("Memory", back_populates="events")
