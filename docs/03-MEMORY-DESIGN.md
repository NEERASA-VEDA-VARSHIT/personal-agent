# 03 — Memory Design

## 1. Core idea

Memory is not one database table and not "everything the user ever said."

We use a layered model.

## 2. Memory types

| Type | Example | Storage | Default |
|---|---|---|---|
| Working | Current conversation | Context | Temporary |
| Episodic | "On Aug 31 I considered internship X" | PostgreSQL + embeddings | Store selectively |
| Semantic | "User prefers backend projects" | PostgreSQL + embeddings | Store if stable |
| Preference | "Prefers concise explanations" | Structured DB | Store |
| Goal | "Wants to become strong backend engineer" | Structured DB | Store |
| Decision | Option/rationale/outcome | Structured DB + embeddings | Store |
| Relationship | Person ↔ event ↔ decision | Relational; graph later | Selective |
| Raw evidence | Conversation/document | PostgreSQL/object store | User controlled |
| Derived inference | "User may be avoiding X" | Separate inference table | Never treat as fact |

## 3. Critical distinction

### Fact

> "User said they are applying for internships."

### Inference

> "User is anxious about their career."

The second must never silently become a permanent fact.

Derived information should carry:

```text
source_ids
confidence
created_at
model_version
status = hypothesis
```

## 4. Proposed memory schema

### memory

```text
id
user_id
type
content
summary
sensitivity
confidence
status
created_at
updated_at
valid_from
valid_until
source_conversation_id
source_message_id
model_version
embedding
```

### memory_relation

```text
id
from_memory_id
to_memory_id
relation_type
confidence
created_at
```

### decision

```text
id
question
options
criteria
evidence
decision
rationale
uncertainties
outcome
created_at
updated_at
```

### memory_audit

```text
id
memory_id
action
reason
actor
timestamp
```

## 5. Memory write policy

Do NOT store every message.

Candidate memory should pass:

```text
Is this useful later?
        ↓
Is it likely to remain relevant?
        ↓
Is it actually stated by the user?
        ↓
Is sensitivity acceptable?
        ↓
Does it duplicate existing memory?
        ↓
Does it conflict with existing memory?
        ↓
Should user approval be required?
```

## 6. Memory tiers

### Tier 0 — transient

Current context.

### Tier 1 — user-approved memory

Explicitly saved.

### Tier 2 — system-extracted memory

Useful candidate facts, but subject to review.

### Tier 3 — derived hypotheses

Never treated as ground truth.

## 7. Conflict handling

Never silently overwrite:

```text
2026-01:
User prefers Python.

2026-08:
User says they now prefer TypeScript.
```

Instead:

```text
old fact:
valid_until = 2026-08

new fact:
valid_from = 2026-08
```

This preserves history.

## 8. Retrieval

Use hybrid retrieval:

```text
semantic similarity
+
keyword/BM25
+
structured filters
+
entity matching
+
recency
+
memory type
+
confidence
```

Then rerank candidates.

## 9. Why PostgreSQL + pgvector first?

It gives us:

- one source of truth;
- relational constraints;
- transactions;
- structured queries;
- vector search;
- easy local development;
- easy Docker deployment.

A dedicated graph database should be justified by an evaluation showing that graph queries materially improve quality.

## 10. Forgetting

Deletion is a first-class feature.

When a user deletes a memory:

1. mark/delete source record;
2. remove vector index entry;
3. invalidate derived relations;
4. invalidate cached retrieval;
5. ensure future evaluation cannot retrieve it.

The system should have automated deletion tests.

## 11. Memory UI

Users should see:

```text
Memory
├── What Personal Agent remembers
├── Why it remembers it
├── Source
├── Confidence
├── When it was true
├── Edit
├── Forget
└── Never remember this type
```
