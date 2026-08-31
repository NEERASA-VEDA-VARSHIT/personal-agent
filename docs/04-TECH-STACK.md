# 04 — Technology Stack

## Recommended initial stack

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn/ui

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- Alembic

### Database

- PostgreSQL
- pgvector

### Local model runtime

Start with:

- Ollama for developer ergonomics and OpenAI-compatible local inference

Keep the application behind a provider interface so we can later benchmark:

- llama.cpp
- vLLM
- LM Studio
- other local runtimes

The app should not depend on a hard-coded Ollama client. Instead, we should use an adapter layer that supports OpenAI-compatible endpoints via `base_url` when available. This keeps the model layer portable across local and cloud providers while still supporting the best developer experience for local testing.

### Embeddings

Use a local embedding model in strict-local mode.

### Background jobs

Start with FastAPI background tasks where adequate.

Move to:

- Redis + worker

only when workloads justify it.

### Observability

- OpenTelemetry
- structured technical logs
- latency/token/model metrics

Never log raw personal prompts by default.

### Testing

- pytest
- Playwright
- unit + integration + evaluation tests

### Deployment

Development:

```text
Docker Compose
├── frontend
├── backend
├── postgres
└── model runtime
```

## Why Python backend?

The AI ecosystem is strongest in Python for:

- model tooling;
- evaluation;
- embeddings;
- data processing;
- experimentation.

The frontend remains TypeScript.

## Why not LangChain everywhere?

We should avoid making the architecture framework-dependent.

Use frameworks where they reduce work, but keep these interfaces ours:

```text
MemoryStore
Retriever
ModelGateway
ToolRegistry
PolicyEngine
Evaluator
```

That makes the project easier to understand and defend in interviews.

## Model strategy

### V0

Use a strong cloud model during development for debugging and benchmark creation if needed.

### V1

Local model becomes the default.

### V2

Add a policy-controlled hybrid router.

### Fine-tuning

Do not fine-tune initially.

First optimize:

1. prompt;
2. structured outputs;
3. retrieval;
4. memory extraction;
5. reranking;
6. tool use;
7. evaluation.

Fine-tuning becomes justified only if we collect a high-quality dataset showing a repeatable model-behavior gap.

Potential future fine-tuning targets:

- memory extraction;
- classification;
- question-asking policy;
- response style.

Do not fine-tune the model to memorize private user data.
