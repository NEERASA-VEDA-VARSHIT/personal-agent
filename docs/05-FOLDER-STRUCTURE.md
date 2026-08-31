# 05 — Repository Structure

## Target architecture

This is the target we are migrating toward. Do not create all folders at once — a folder structure is an architecture, not a checklist.

```text
personal-agent/
│
├── apps/
│   ├── api/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   │
│   │   │   ├── api/
│   │   │   │   ├── routes/
│   │   │   │   │   ├── chat.py
│   │   │   │   │   ├── memories.py
│   │   │   │   │   └── decisions.py
│   │   │   │   └── dependencies.py
│   │   │   │
│   │   │   ├── agent/
│   │   │   │   ├── agent.py
│   │   │   │   ├── state.py
│   │   │   │   ├── prompts.py
│   │   │   │   └── orchestration.py
│   │   │   │
│   │   │   ├── models/
│   │   │   │   ├── gateway.py
│   │   │   │   ├── providers/
│   │   │   │   │   ├── openai.py
│   │   │   │   │   ├── ollama.py
│   │   │   │   │   └── base.py
│   │   │   │   └── embeddings.py
│   │   │   │
│   │   │   ├── memory/
│   │   │   │   ├── models.py
│   │   │   │   ├── repository.py
│   │   │   │   ├── extraction.py
│   │   │   │   ├── lifecycle.py
│   │   │   │   ├── retrieval.py
│   │   │   │   ├── reranking.py
│   │   │   │   ├── policy.py
│   │   │   │   └── provenance.py
│   │   │   │
│   │   │   ├── decision/
│   │   │   │   ├── models.py
│   │   │   │   ├── question_policy.py
│   │   │   │   ├── analyzer.py
│   │   │   │   ├── engine.py
│   │   │   │   └── evaluator.py
│   │   │   │
│   │   │   ├── db/
│   │   │   │   ├── models.py
│   │   │   │   ├── session.py
│   │   │   │   └── migrations/
│   │   │   │
│   │   │   ├── privacy/
│   │   │   │   ├── classifier.py
│   │   │   │   ├── policy.py
│   │   │   │   └── router.py
│   │   │   │
│   │   │   ├── evaluation/
│   │   │   │   ├── datasets/
│   │   │   │   ├── metrics.py
│   │   │   │   └── runner.py
│   │   │   │
│   │   │   └── config.py
│   │   │
│   │   └── tests/
│   │       ├── memory/
│   │       ├── retrieval/
│   │       ├── decision/
│   │       ├── privacy/
│   │       └── integration/
│   │
│   └── web/
│       ├── app/
│       ├── components/
│       ├── lib/
│       └── tests/
│
├── packages/
│   └── shared/
│       ├── schemas/
│       └── types/
│
├── infra/
│   ├── docker/
│   ├── postgres/
│   └── ollama/
│
├── scripts/
│   ├── seed.py
│   ├── evaluate.py
│   └── dev.py
│
├── docs/
│   ├── 01-PRD.md
│   ├── 02-ARCHITECTURE.md
│   ├── 03-MEMORY-DESIGN.md
│   ├── 04-TECH-STACK.md
│   ├── 05-FOLDER-STRUCTURE.md
│   ├── 06-UI-FLOW.md
│   ├── 07-PRIVACY.md
│   ├── 08-EVALUATION.md
│   ├── 09-RESEARCH.md
│   ├── adr/
│   │   ├── 001-postgres-pgvector.md
│   │   ├── 002-provider-abstraction.md
│   │   └── 003-memory-lifecycle.md
│   └── milestones/
│       ├── M0-M6-COMPLETION.md
│       └── M6.5-M6.9.md
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

## Current stage (what to actually create now)

For our current stage, keep the tree smaller and migrate incrementally:

```text
apps/api/app/
├── agent/
├── models/
├── memory/        # domain: what/extraction/retrieval/lifecycle/provenance
├── decision/      # domain: question_policy/analyzer/engine
├── db/            # infra: SQLAlchemy models, sessions, migrations
└── config.py

apps/api/tests/
├── memory/
├── retrieval/
└── decision/

docs/
├── ...
└── milestones/
```

Create `privacy/`, `web/`, `evaluation/`, etc. **when those components actually exist**.

## Key separation

### `db/` vs `memory/`

Different responsibilities:

- **`db/` — Infrastructure**: SQLAlchemy models, sessions, migrations, transactions.

- **`memory/` — Domain logic**: What constitutes a memory? How is it extracted? Retrieved? How does it evolve? How is it forgotten?

Dependency direction:

```text
memory/
      ↓
repository
      ↓
db/
      ↓
PostgreSQL
```

Not all memory logic inside `db/`.

### `decision/` as its own domain

```text
decision/
├── question_policy.py  → "Should I ask?"
├── analyzer.py         → "What information is missing?"
├── engine.py           → "What are options/tradeoffs?"
└── evaluator.py        → "How good was the decision support?"
```

- **Memory** provides context.
- **Decision** provides reasoning.
- **Agent** provides orchestration.
- **Model gateway** provides intelligence.
- **Database** provides persistence.

```text
                    AGENT
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       MEMORY      DECISION    PRIVACY
          │           │
          └─────┬─────┘
                ↓
           MODEL GATEWAY
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
    Ollama   OpenAI    Other
```

## Architectural rule

Keep domain logic independent from model providers.

Bad:

```text
agent.py → directly calls Ollama
```

Better:

```text
agent.py → ModelGateway → OpenAICompatibleProvider / NativeProvider / EmbeddingProvider
```

This is important because model comparison is one of the project's experiments.

Suggested provider structure:

```text
apps/api/app/models/
├── gateway.py
├── provider_interface.py
├── providers/
│   ├── openai_compatible.py
│   ├── ollama.py
│   ├── vllm.py
│   ├── native_runtime.py
│   └── embeddings.py
└── router.py
```

The OpenAI-compatible API should be treated as an adapter implementation, not the project-wide abstraction. The app talks to a stable interface, while the provider layer decides how to route to OpenAI, Ollama, vLLM, or another backend.

## Migration rule

Don't refactor the entire repository in one giant commit. Migrate toward the target as we implement M6.8 onward, keeping `db/` and `memory/`/`decision/` separated.
