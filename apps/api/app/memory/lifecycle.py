"""M6.6 — Memory Lifecycle state machine.

State machine::

                 ┌──────────────┐
                 │   CANDIDATE  │
                 └──────┬───────┘
                        │ policy validation
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
         ACTIVE                REJECTED
             │
       ┌─────┴─────┐
       │           │
    superseded  forgotten
       │           │
       ▼           ▼
  SUPERSEDED   FORGOTTEN

Temporal supersession preserves history::

    Memory #12  "I prefer Python"       valid_until=2026-08-31  status=SUPERSEDED
    Memory #47  "I now prefer TypeScript" valid_from=2026-08-31  status=ACTIVE
    MemoryRelation(supersedes): 47 -> 12
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryAudit, MemoryRelation
from app.memory.policy import MemoryStatus


VALID_TRANSITIONS: dict[str, set[str]] = {
    MemoryStatus.CANDIDATE.value: {MemoryStatus.ACTIVE.value, MemoryStatus.REJECTED.value},
    MemoryStatus.ACTIVE.value: {MemoryStatus.SUPERSEDED.value, MemoryStatus.FORGOTTEN.value},
    MemoryStatus.SUPERSEDED.value: {MemoryStatus.FORGOTTEN.value},
    MemoryStatus.REJECTED.value: {MemoryStatus.FORGOTTEN.value},
    MemoryStatus.FORGOTTEN.value: set(),
}


class LifecycleError(ValueError):
    pass


def _audit(db: Session, memory_id: int, action: str, reason: str | None, actor: str | None) -> MemoryAudit:
    entry = MemoryAudit(memory_id=memory_id, action=action, reason=reason, actor=actor)
    db.add(entry)
    return entry


def _transition(db: Session, memory: Memory, to_status: str, *, reason: str | None, actor: str | None) -> Memory:
    cur = memory.status or MemoryStatus.ACTIVE.value
    allowed = VALID_TRANSITIONS.get(cur, set())
    if to_status not in allowed:
        raise LifecycleError(f"Invalid transition {cur!r} -> {to_status!r} for memory {memory.id}")
    memory.status = to_status
    # is_active mirrors ACTIVE only
    memory.is_active = to_status == MemoryStatus.ACTIVE.value
    memory.updated_at = datetime.utcnow()
    _audit(db, memory.id, to_status, reason, actor)
    db.add(memory)
    db.flush()
    return memory


class MemoryLifecycle:
    """Encapsulates all lifecycle operations with audit + relation handling."""

    # -- creation ---------------------------------------------------------

    @staticmethod
    def create_candidate(
        db: Session,
        *,
        user_id: int,
        type: str,
        content: str,
        summary: str | None = None,
        sensitivity: str | None = None,
        confidence: float = 1.0,
        source_conversation_id: int | None = None,
        source_message_id: int | None = None,
        model_version: str | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        actor: str | None = "system",
        reason: str | None = "candidate created",
    ) -> Memory:
        mem = Memory(
            user_id=user_id,
            type=type,
            memory_type=type,
            content=content,
            summary=summary,
            sensitivity=sensitivity,
            confidence=confidence,
            status=MemoryStatus.CANDIDATE.value,
            is_active=False,
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            model_version=model_version,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        db.add(mem)
        db.flush()
        _audit(db, mem.id, "created", reason, actor)
        # also mark candidate state explicitly for audit completeness
        _audit(db, mem.id, MemoryStatus.CANDIDATE.value, reason, actor)
        db.flush()
        return mem

    @staticmethod
    def create_active(
        db: Session,
        *,
        user_id: int,
        type: str,
        content: str,
        summary: str | None = None,
        sensitivity: str | None = None,
        confidence: float = 1.0,
        source_conversation_id: int | None = None,
        source_message_id: int | None = None,
        model_version: str | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        actor: str | None = "system",
        reason: str | None = "active created",
    ) -> Memory:
        mem = Memory(
            user_id=user_id,
            type=type,
            memory_type=type,
            content=content,
            summary=summary,
            sensitivity=sensitivity,
            confidence=confidence,
            status=MemoryStatus.ACTIVE.value,
            is_active=True,
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            model_version=model_version,
            valid_from=valid_from or datetime.utcnow(),
            valid_until=valid_until,
        )
        db.add(mem)
        db.flush()
        _audit(db, mem.id, "created", reason, actor)
        _audit(db, mem.id, MemoryStatus.ACTIVE.value, reason, actor)
        db.flush()
        return mem

    # -- transitions ------------------------------------------------------

    @staticmethod
    def approve(db: Session, memory: Memory, *, actor: str = "user", reason: str = "approved") -> Memory:
        result = _transition(db, memory, MemoryStatus.ACTIVE.value, reason=reason, actor=actor)
        # additional domain audit
        _audit(db, memory.id, "confirmed", reason, actor)
        db.flush()
        return result

    @staticmethod
    def reject(db: Session, memory: Memory, *, actor: str = "system", reason: str = "rejected by policy") -> Memory:
        result = _transition(db, memory, MemoryStatus.REJECTED.value, reason=reason, actor=actor)
        _audit(db, memory.id, "rejected", reason, actor)
        db.flush()
        return result

    @staticmethod
    def forget(db: Session, memory: Memory, *, actor: str = "user", reason: str = "user requested forget") -> Memory:
        # forget is allowed from ACTIVE / SUPERSEDED / REJECTED / CANDIDATE? Keep strict per map.
        # For UX, also allow ACTIVE->FORGOTTEN and CANDIDATE->FORGOTTEN via REJECTED+FORGOTTEN intermediate?
        # Here we allow direct if transition permits; otherwise raise.
        result = _transition(db, memory, MemoryStatus.FORGOTTEN.value, reason=reason, actor=actor)
        _audit(db, memory.id, "forgotten", reason, actor)
        db.flush()
        return result

    @staticmethod
    def supersede(
        db: Session,
        old_memory: Memory,
        new_memory: Memory,
        *,
        actor: str = "system",
        reason: str = "superseded by newer memory",
        valid_from: datetime | None = None,
    ) -> MemoryRelation:
        """Supersede old_memory with new_memory.

        - old must be ACTIVE, new must be ACTIVE
        - sets old.valid_until = new.valid_from (if not already set)
        - creates MemoryRelation(supersedes) new -> old
        - audits both memories
        """
        if old_memory.status != MemoryStatus.ACTIVE.value:
            raise LifecycleError(f"Can only supersede an ACTIVE memory, got {old_memory.status!r} (id={old_memory.id})")
        if new_memory.status != MemoryStatus.ACTIVE.value:
            raise LifecycleError(f"New memory must be ACTIVE to supersede, got {new_memory.status!r} (id={new_memory.id})")
        if old_memory.user_id != new_memory.user_id:
            raise LifecycleError("Cannot supersede memories belonging to different users")
        if old_memory.id == new_memory.id:
            raise LifecycleError("Cannot supersede a memory with itself")

        supersede_from = valid_from or new_memory.valid_from or datetime.utcnow()

        # temporal handoff
        if old_memory.valid_until is None:
            old_memory.valid_until = supersede_from
        new_memory.valid_from = supersede_from
        new_memory.valid_until = None

        # transition old ACTIVE -> SUPERSEDED
        old_memory.status = MemoryStatus.SUPERSEDED.value
        old_memory.is_active = False
        old_memory.updated_at = datetime.utcnow()
        db.add(old_memory)

        relation = MemoryRelation(
            from_memory_id=new_memory.id,
            to_memory_id=old_memory.id,
            relation_type="supersedes",
            confidence=1.0,
        )
        db.add(relation)
        db.add(new_memory)

        _audit(db, old_memory.id, MemoryStatus.SUPERSEDED.value, reason, actor)
        _audit(db, new_memory.id, "created", f"supersedes memory {old_memory.id}", actor)

        db.flush()
        return relation

    @staticmethod
    def contradict(db: Session, from_memory: Memory, to_memory: Memory, *, confidence: float = 0.85) -> MemoryRelation:
        rel = MemoryRelation(
            from_memory_id=from_memory.id,
            to_memory_id=to_memory.id,
            relation_type="contradicts",
            confidence=confidence,
        )
        db.add(rel)
        db.flush()
        return rel

    # -- queries ----------------------------------------------------------

    @staticmethod
    def get_active(db: Session, user_id: int) -> list[Memory]:
        return db.query(Memory).filter(Memory.user_id == user_id, Memory.status == MemoryStatus.ACTIVE.value, Memory.is_active == True).all()  # noqa: E712

    @staticmethod
    def get_current_preference(db: Session, user_id: int, content_like: str) -> Optional[Memory]:
        """Example temporal query: latest ACTIVE memory matching a keyword."""
        return (
            db.query(Memory)
            .filter(
                Memory.user_id == user_id,
                Memory.status == MemoryStatus.ACTIVE.value,
                Memory.is_active == True,  # noqa: E712
                Memory.content.contains(content_like),
            )
            .order_by(Memory.valid_from.desc())
            .first()
        )
