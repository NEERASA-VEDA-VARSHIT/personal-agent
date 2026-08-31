# ADR-001 — Use PostgreSQL + pgvector as the initial memory foundation

## Status

Accepted for MVP.

## Decision

Use PostgreSQL as the canonical store and pgvector as the semantic retrieval index.

## Alternatives considered

- Vector database only
- Graph database
- Mem0 as the primary abstraction
- Letta-managed memory
- Graphiti/Zep temporal graph

## Rationale

The project is intended to teach us how memory works, not merely hide it behind a library.

PostgreSQL gives us:

- structured schemas;
- transactions;
- provenance;
- temporal validity;
- deletion;
- relational constraints;
- vector search.

A graph layer can be added later if experiments demonstrate meaningful gains for relationship-heavy or temporal queries.

## Consequence

We must implement retrieval and memory policies ourselves initially, which creates more work but makes the project more educational and measurable.
