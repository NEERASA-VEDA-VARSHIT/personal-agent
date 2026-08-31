# 10 — Roadmap

## Phase 0 — Foundations

Learn:

- tokens/context;
- embeddings;
- inference;
- structured outputs;
- RAG;
- tool calling;
- evaluation.

Deliverable:

A local model answering basic questions.

## Phase 1 — Local Chat

Build:

- Next.js UI;
- FastAPI;
- local model;
- streaming;
- conversation persistence.

Deliverable:

Private local chatbot.

## Phase 2 — Memory

Build:

- PostgreSQL;
- pgvector;
- memory extraction;
- retrieval;
- provenance;
- memory UI.

Deliverable:

Assistant remembers selected user context.

## Phase 3 — Decision Support

Build:

- intent classification;
- stakes assessment;
- answer-vs-question policy;
- decision schema;
- evidence-grounded response format;
- anti-sycophancy tests.

Deliverable:

Useful decision-support agent.

## Phase 4 — Privacy

Build:

- strict local mode;
- hybrid router;
- encryption;
- data export/delete;
- privacy tests.

Deliverable:

Demonstrable privacy architecture.

## Phase 5 — Evaluation

Build:

- benchmark dataset;
- retrieval metrics;
- advice rubric;
- question efficiency;
- privacy canary;
- regression suite.

Deliverable:

Research-style evaluation report.

## Phase 6 — Advanced Memory

Only if experiments justify it:

- temporal memory;
- entity linking;
- graph relationships;
- memory consolidation;
- reranking.

## Phase 7 — Tools

Add carefully:

- calendar;
- tasks;
- notes;
- local files.

Every tool requires:

- permission;
- audit trail;
- failure handling.

## Phase 8 — Portfolio

Produce:

- architecture diagram;
- demo;
- benchmark report;
- design decisions;
- failure analysis;
- security model;
- technical blog;
- resume bullets.

## MVP stopping rule

Do not add features until:

1. memory retrieval works;
2. deletion works;
3. privacy mode is testable;
4. agent can distinguish fact vs inference;
5. evaluation can detect regressions.
