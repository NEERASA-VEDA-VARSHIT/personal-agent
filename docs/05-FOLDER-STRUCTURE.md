# 05 — Repository Structure

```text
personal-agent/
│
├── apps/
│   ├── web/                         # Next.js frontend
│   │   ├── app/
│   │   ├── components/
│   │   ├── features/
│   │   └── lib/
│   │
│   └── api/                         # FastAPI backend
│       ├── app/
│       │   ├── api/
│       │   ├── agents/
│       │   ├── memory/
│       │   ├── models/
│       │   ├── tools/
│       │   ├── safety/
│       │   ├── policies/
│       │   ├── evaluation/
│       │   └── db/
│       └── tests/
│
├── packages/
│   ├── schemas/                     # Shared contracts
│   └── prompts/                     # Versioned prompt templates
│
├── memory/
│   ├── migrations/
│   ├── retrieval/
│   ├── extraction/
│   └── fixtures/
│
├── evaluation/
│   ├── datasets/
│   ├── benchmarks/
│   ├── metrics/
│   ├── scenarios/
│   └── reports/
│
├── docs/
│   ├── 01-PRD.md
│   ├── 02-ARCHITECTURE.md
│   ├── 03-MEMORY-DESIGN.md
│   ├── 04-TECH-STACK.md
│   ├── 05-FOLDER-STRUCTURE.md
│   ├── 06-UI-UX-FLOW.md
│   ├── 07-AGENT-POLICY.md
│   ├── 08-PRIVACY-SECURITY.md
│   ├── 09-EVALUATION.md
│   ├── 10-ROADMAP.md
│   ├── 11-RESEARCH-NOTES.md
│   └── 12-PORTFOLIO.md
│
├── infra/
│   ├── docker/
│   ├── compose/
│   └── scripts/
│
├── .github/
│   └── workflows/
│
├── docker-compose.yml
├── README.md
└── LICENSE
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
