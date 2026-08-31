# 12 — Portfolio Framing

## Project thesis

> **Can a privacy-first, local-first AI agent achieve useful long-term personalization without treating personal data as an opaque prompt history?**

## What makes it stronger than an LLM wrapper?

### 1. Memory architecture

Structured + semantic + temporal memory.

### 2. Privacy architecture

Sensitive context is protected by the system architecture, not just prompting.

### 3. Decision-support policy

The agent decides whether to answer or ask based on uncertainty, stakes and decision impact.

### 4. Evaluation

We measure retrieval, hallucination, advice quality, question efficiency and privacy leakage.

### 5. Experimental comparison

We compare multiple memory architectures and model strategies.

## Suggested portfolio story

### Problem

Long-term personalization requires sensitive personal data.

### Hypothesis

A local-first memory architecture can provide useful personalization while substantially reducing external data exposure.

### Approach

- local inference;
- structured memory;
- semantic retrieval;
- provenance;
- temporal validity;
- policy-controlled model routing;
- evaluation harness.

### Experiments

- vector vs hybrid memory;
- full history vs selective memory;
- local vs cloud model;
- different retrieval strategies;
- different question policies.

### Results

Report actual measured numbers rather than invented claims.

## Resume bullet template

> Built a privacy-first personal AI agent using [stack], implementing structured + vector long-term memory, provenance-aware retrieval, policy-controlled local/cloud inference, and an evaluation harness for retrieval quality, hallucination, question efficiency, and privacy leakage.

## Interview topics this project supports

- RAG
- embeddings
- vector search
- PostgreSQL
- agent architecture
- tool calling
- prompt engineering
- structured outputs
- local inference
- model routing
- privacy/security
- evaluation
- system design
- distributed systems
- Docker
- observability
