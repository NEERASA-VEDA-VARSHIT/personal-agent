# Memory v2 Schema Migration: Complete

**Status**: ✅ **COMPLETE** — All 66 tests passing
**Date**: August 31, 2026
**Diff Ready**: Yes — Review before M6.6

---

## Executive Summary

The Memory schema has been successfully migrated from v1 to v2. All M0-M6 functionality is preserved, with enhanced provenance tracking, lifecycle management, and memory relationships.

**Key Achievement**: The schema now properly separates:
- **What** the memory is (MemoryType)
- **How** it was produced (SourceType)  
- **Where** it is in its lifecycle (MemoryStatus)

This prevents conflating production method with memory content, fixing the conceptual issue where "confidence" meant different things.

---

## Test Results

### By Component
| Component | Tests | Status |
|-----------|-------|--------|
| M0: Model Gateway | 3 | ✅ PASS |
| M2: Memory Schema | 5 | ✅ PASS |
| M3: Embeddings | 7 | ✅ PASS |
| M4: RAG Pipeline | 4 | ✅ PASS |
| M5: Memory Extraction | 11 | ✅ PASS (updated) |
| M6: Decision Support | 20 | ✅ PASS |
| **v2 Schema Tests** | **16** | **✅ PASS (new)** |
| **TOTAL** | **66** | **✅ PASS** |

### v2 Schema Test Coverage (16 tests)

#### 1. Core Fields (1 test)
- ✅ All v2 fields accessible and persistent

#### 2. Temporal Validity (2 tests)
- ✅ `valid_from` / `valid_until` range support
- ✅ `valid_until` nullable for permanent memories

#### 3. MemorySource Relationships (3 tests)
- ✅ Source creation and relationship persistence
- ✅ All three source types (USER_STATED, MODEL_EXTRACTED, MODEL_INFERRED)
- ✅ Multiple sources per memory

#### 4. MemoryRelation Relationships (4 tests)
- ✅ SUPPORTS relationship (one memory supports another)
- ✅ CONTRADICTS relationship (conflicting memories)
- ✅ SUPERSEDES relationship (memory lifecycle)
- ✅ RELATED_TO relationship (loose associations)

#### 5. MemoryAudit Lifecycle (3 tests)
- ✅ CREATED action tracking
- ✅ All lifecycle actions (created, updated, confirmed, rejected, deleted)
- ✅ Audit trail with reason and actor

#### 6. Status Transitions & Features (2 tests)
- ✅ Full lifecycle: CANDIDATE → ACTIVE → SUPERSEDED → FORGOTTEN
- ✅ Sensitivity levels: public, private, confidential

#### 7. Relationships & Compatibility (1 test)
- ✅ Memory-Conversation relationship
- ✅ Backward compatibility of `memory_type` field

---

## Schema Structure

### Memory Table (Enhanced)

```python
class Memory:
    # Identity
    id: int (PK)
    user_id: int (FK → users)
    
    # Classification
    type: str  # NEW v2: fact|preference|goal|episode|decision|relationship|hypothesis
    memory_type: str  # Backward compatibility
    
    # Content
    content: str
    summary: str (optional)
    
    # Quality
    sensitivity: str (optional)  # public|private|confidential
    confidence: float  # Evidence strength (0.0-1.0)
    
    # Lifecycle
    status: str (optional)  # candidate|active|superseded|rejected|forgotten
    is_active: bool  # Quick filter
    
    # Temporal
    valid_from: datetime (optional)
    valid_until: datetime (nullable)
    
    # Provenance
    source_conversation_id: int (FK → conversations, optional)
    source_message_id: int (FK → messages, optional)
    model_version: str (optional)
    
    # Embedding
    embedding: str  # JSON vector
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Relationships
    sources: [MemorySource]  # How it was produced
    relations_from: [MemoryRelation]  # Outgoing relationships
    relations_to: [MemoryRelation]  # Incoming relationships
    audits: [MemoryAudit]  # Lifecycle audit trail
```

### Supporting Tables

#### MemorySource
```python
class MemorySource:
    id: int (PK)
    memory_id: int (FK → memories)
    source_type: str  # user_stated|model_extracted|model_inferred
    source_ref: str  # Citation reference
    confidence: float  # This source's confidence (0.0-1.0)
    created_at: datetime
```

#### MemoryRelation
```python
class MemoryRelation:
    id: int (PK)
    from_memory_id: int (FK → memories)
    to_memory_id: int (FK → memories)
    relation_type: str  # supports|contradicts|supersedes|related_to
    confidence: float  # Confidence in the relationship
    created_at: datetime
```

#### MemoryAudit
```python
class MemoryAudit:
    id: int (PK)
    memory_id: int (FK → memories)
    action: str  # created|updated|confirmed|rejected|deleted
    reason: str (optional)  # Why this action
    actor: str (optional)  # Who/what performed it
    created_at: datetime
```

---

## Key Design Decisions

### 1. **Type vs. Source Type vs. Status**

| Field | Values | Meaning | Example |
|-------|--------|---------|---------|
| `type` | FACT, PREFERENCE, GOAL, ... | **What** the memory represents | "Prefers Python" |
| `source_type` | USER_STATED, MODEL_EXTRACTED, MODEL_INFERRED | **How** it was produced | "Extracted from conversation" |
| `status` | CANDIDATE, ACTIVE, SUPERSEDED, FORGOTTEN | **Where** in lifecycle | ACTIVE (approved) |

**Why this matters:**
- Old schema: `memory_type = "candidate"` mixed production method with storage decision
- New schema: Clear separation allows policy to say "Store CANDIDATE=true AND status=ACTIVE"

### 2. **Confidence Redefinition**

**Old interpretation** (❌ Wrong):
> "Confidence 0.85" = "85% probability user actually prefers Python"

**New interpretation** (✅ Correct):
> "Confidence 0.85" = "85% confidence in the extraction/observation"

The user's actual preference is either TRUE or FALSE (binary). The confidence tracks how confident the LLM is in the extraction, not the user's commitment level.

### 3. **Temporal Validity**

```python
# Permanent memory
Memory(
    content="User has 10 years of Python experience",
    valid_from=datetime(2020, 1, 1),
    valid_until=None  # Forever
)

# Temporary memory
Memory(
    content="Currently learning Rust",
    valid_from=datetime(2026, 8, 31),
    valid_until=datetime(2026, 12, 31)  # 4-month goal
)
```

This allows the system to auto-expire temporary preferences/goals without explicit deletion.

### 4. **Backward Compatibility**

```python
# Old code still works:
memory = Memory(
    memory_type="fact",  # Accepts old value
    content="...",
)

# New code uses v2:
memory = Memory(
    type="fact",  # New field
    status=MemoryStatus.ACTIVE.value,  # Lifecycle
    source_conversation_id=conv_id,
)
```

Existing data migrations:
- `type` auto-populated from `memory_type`
- `status` defaults to ACTIVE
- `source_type` defaults to USER_STATED

### 5. **No Graph Database Yet**

Decision: **Keep PostgreSQL + MemoryRelation table**

Rationale:
- `MemoryRelation` can handle graph queries (WITH RECURSIVE)
- Sufficient for current needs (supersession, relationships)
- Can add Neo4j/Graphiti later if evaluation shows value
- Avoids premature architecture complexity

---

## Code Changes Summary

### 1. `app/db/models.py`
- ✅ Memory table: Added v2 fields (type, status, sensitivity, valid_from, valid_until)
- ✅ MemorySource: New table for provenance tracking
- ✅ MemoryRelation: New table for memory relationships
- ✅ MemoryAudit: Enhanced from MemoryEvent (dropped old table)
- ✅ Relationships: Full back-references between tables

### 2. `app/memory_policy.py`
- ✅ MemoryType enum: FACT, PREFERENCE, GOAL, EPISODE, DECISION, RELATIONSHIP, HYPOTHESIS
- ✅ SourceType enum: USER_STATED, MODEL_EXTRACTED, MODEL_INFERRED
- ✅ MemoryStatus enum: CANDIDATE, ACTIVE, SUPERSEDED, REJECTED, FORGOTTEN
- ✅ Policy logic: Updated to use new enums
- ✅ Validation reasons: Clearer decision explanations

### 3. `app/memory_extraction.py`
- ✅ Import: Added MemoryAudit, SourceType, MemoryStatus
- ✅ Extraction prompt: Updated to ask for v2 memory types
- ✅ Storage: Stores status=ACTIVE, source_type=MODEL_EXTRACTED
- ✅ Audit trail: Uses MemoryAudit instead of MemoryEvent

### 4. `test_memory_extraction.py`
- ✅ Imports: Added SourceType, MemoryStatus
- ✅ Test data: Updated to use new MemoryType enums
- ✅ Assertions: Check new v2 fields (type, status, source_type)
- ✅ All 11 tests passing

### 5. `test_memory_v2_schema.py` (NEW)
- ✅ 16 comprehensive tests for v2 schema
- ✅ Tests all new tables and relationships
- ✅ Validates lifecycle transitions
- ✅ All tests passing

---

## Migration Checklist

### ✅ Schema Design
- [x] Define v2 schema structure
- [x] Create MemorySource table
- [x] Create MemoryRelation table  
- [x] Create MemoryAudit table
- [x] Add v2 fields to Memory table

### ✅ Policy & Enums
- [x] Define MemoryType (FACT, PREFERENCE, GOAL, …)
- [x] Define SourceType (USER_STATED, MODEL_EXTRACTED, MODEL_INFERRED)
- [x] Define MemoryStatus (CANDIDATE, ACTIVE, …)
- [x] Update policy validation logic

### ✅ Code Integration
- [x] Update memory_extraction.py
- [x] Update memory_policy.py
- [x] Maintain backward compatibility

### ✅ Testing
- [x] Fix extraction tests (11 passing)
- [x] Add v2 schema tests (16 passing)
- [x] Full integration test suite (66 passing)
- [x] No breaking changes to M0-M6

---

## Next: M6.6 Memory Lifecycle

With v2 schema foundation complete, M6.6 will implement:

```
Conversation
    ↓
Extract (LLM)
    ↓
MemoryCandidate (list)
    ↓
Policy.validate() 
    ↓
status = CANDIDATE
    ↓
User approval (if borderline)
    ↓
status = ACTIVE
    ↓
Store to DB (with audit)
    ↓
Retrieve (ACTIVE only)
    ↓
User confirm/update
    ↓
Query MemoryRelation.supersedes()
    ↓
Old memory: status = SUPERSEDED
New memory: status = ACTIVE
    ↓
Audit trail: "User updated preference"
```

---

## Files Ready for Review

**Modified:**
- ✅ [app/db/models.py](../../apps/api/app/db/models.py) — v2 schema
- ✅ [app/memory_policy.py](../../apps/api/app/memory_policy.py) — New enums
- ✅ [app/memory_extraction.py](../../apps/api/app/memory_extraction.py) — v2 integration
- ✅ [test_memory_extraction.py](../../apps/api/test_memory_extraction.py) — Updated tests

**New:**
- ✅ [test_memory_v2_schema.py](../../apps/api/test_memory_v2_schema.py) — 16 new tests

**All tests passing**: 66/66 ✅

---

## Conclusion

The Memory v2 schema is complete, tested, and ready for the memory lifecycle implementation in M6.6. The foundation properly separates production method from memory classification, enabling sophisticated memory management without conceptual confusion.

**Ready to proceed to M6.6: Memory Lifecycle?** ✅
