# 02 — System Architecture

## 1. Architecture decision

Use a **layered local-first architecture** rather than choosing one memory technology for everything.

```text
┌─────────────────────────────────────────────────────────────┐
│                         Web UI                              │
│                 Next.js / React / TypeScript                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Application API                        │
│                         FastAPI                              │
├─────────────────────────────────────────────────────────────┤
│                     Agent Orchestrator                       │
│                                                             │
│  Intent → Safety → Memory → Tools → Reasoning → Response  │
└──────────────┬───────────────┬───────────────┬──────────────┘
               │               │               │
               ▼               ▼               ▼
        Memory Service     Tool Service     Model Gateway
               │                               │
     ┌─────────┼──────────┐             ┌──────┴──────┐
     ▼         ▼          ▼             ▼             ▼
 PostgreSQL  pgvector  Object Store   Local LLM     Cloud LLM
```

## 2. Why not a pure vector database?

Personal memory has several shapes:

- exact facts;
- events;
- changing facts;
- relationships;
- decisions;
- preferences;
- source conversations.

A vector similarity search is excellent for semantic recall but weak as the sole representation of changing truth.

Therefore:

**PostgreSQL = source of truth**  
**pgvector = semantic retrieval index**  
**Object storage/files = large raw artifacts**  
**Optional graph layer later = relationship/temporal reasoning**

## 3. Request flow

```text
User message
   ↓
Normalize + classify intent
   ↓
Safety/stakes assessment
   ↓
Retrieve candidate memories
   ↓
Filter by permissions/sensitivity/time
   ↓
Rerank evidence
   ↓
Decide: answer or ask?
   ↓
Reason using evidence
   ↓
Generate response
   ↓
Validate claims against evidence
   ↓
Optionally propose memory write
   ↓
Persist approved memory
```

## 4. High-stakes flow

For high-stakes personal decisions:

```text
Question
  ↓
Stakes assessment
  ↓
Known facts
  ↓
Relevant memories
  ↓
Unknowns
  ↓
Potentially decision-changing questions
  ↓
Options + criteria
  ↓
Tradeoffs
  ↓
Uncertainty
  ↓
User decision
```

The agent should not jump directly from question to recommendation.

## 5. Model gateway

The application should not call a model directly from business logic.

We should define a stable internal model interface instead of baking in a single provider contract.

```text
                    Agent
                      │
                      ▼
                Model Gateway
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
 OpenAICompatibleProvider  NativeProvider  EmbeddingProvider
       │                        │               │
   ┌───┼───────────────┐   ┌────┴────┐   ┌───────┴────────┐
   ▼   ▼               ▼   ▼         ▼   ▼               ▼
OpenAI  Ollama         vLLM   llama.cpp  local runtime   local embed   cloud embed
```

The important design decision is that the app depends on a common interface such as:

```text
class ModelProvider:
    generate(...)
    stream(...)
    embed(...)
```

OpenAI compatibility is an implementation detail, not the app's architectural centerpiece.

This means:

- the app should call `model.generate(...)`, not `ollama.generate(...)`;
- a provider may expose an OpenAI-compatible HTTP API at `base_url`;
- the provider adapter can wrap OpenAI, Ollama, vLLM, or a custom inference server;
- the app should remain agnostic to whether a model is local, cloud, or hybrid.

This is the right abstraction for experimentation. We can compare:

- local model vs cloud model;
- local model + memory vs cloud model + full history;
- small local model + retrieval vs large cloud model;
- routing policy for sensitive vs non-sensitive requests.

A provider may support OpenAI-compatible semantics via `base_url`, but we should not assume every provider is feature-equivalent. Compatibility exists at the HTTP layer, while abstraction exists at the application layer.

Example runtime config:

```text
MODEL_PROVIDER=ollama
MODEL_NAME=gpt-oss:20b
MODEL_BASE_URL=http://localhost:11434/v1
```

or:

```text
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4.1-mini
MODEL_BASE_URL=https://api.openai.com/v1
```

The gateway can choose the adapter without changing the agent contract.

## 6. Privacy boundary

Sensitive memory crosses only:

```text
UI → Local API → Local memory → Local model
```

The cloud provider is behind an explicit policy gate.

Strict mode:

```text
Cloud = DENY
```

Hybrid mode:

```text
Sensitive context = DENY
General/non-sensitive tasks = ALLOW
```

## 7. Future extension

A temporal graph can be introduced only when evaluation demonstrates that relational/time-based queries are a bottleneck.

Do not add Graphiti/Neo4j/etc. merely because "agents need graphs."
