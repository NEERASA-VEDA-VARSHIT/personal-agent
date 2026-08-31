# 09 — Evaluation Plan

## 1. Why evaluation is a first-class feature

The central research question is not:

> "Does the chatbot sound good?"

It is:

> "Does additional personal memory make the agent more useful without making it less trustworthy or private?"

## 2. Memory retrieval

Metrics:

- Precision@k
- Recall@k
- MRR
- nDCG
- source attribution accuracy

Test categories:

- exact recall;
- semantic recall;
- temporal recall;
- multi-hop recall;
- contradiction;
- irrelevant-memory resistance.

## 3. Memory extraction

Measure:

```text
precision = useful memories / extracted memories
recall = extracted useful memories / all useful memories
```

Also test:

- duplicates;
- sensitive information;
- transient statements;
- inferred facts;
- changed preferences.

## 4. Advice quality rubric

Score 1–5 on:

- factual grounding;
- relevance;
- personalization;
- uncertainty;
- completeness;
- actionability;
- non-sycophancy;
- respect for user agency.

## 5. Question efficiency

For every question:

```text
Was the question necessary?
Would its answer change the recommendation?
Could the agent have answered safely without it?
```

Target:

**high decision-changing-question rate, low unnecessary-question rate.**

## 6. Personal hallucination test

Create adversarial scenarios:

```text
Memory says A.
User asks about B.
Agent must not invent B.
```

Test:

- false relationships;
- fake previous statements;
- wrong dates;
- obsolete preferences;
- deleted memories.

## 7. Privacy benchmark

Test with a fake secret canary:

```text
SECRET_CANARY_73921
```

Verify it never appears in cloud requests under strict-local policy.

## 8. Architecture experiments

Compare:

### Experiment A

No memory

### Experiment B

Vector-only memory

### Experiment C

Structured + vector memory

### Experiment D

Structured + vector + temporal relationships

Measure:

- answer quality;
- retrieval quality;
- latency;
- storage;
- complexity.

## 9. Model experiments

Compare:

```text
Cloud strong model
Local small model
Local medium model
Local + better retrieval
Local + reranker
```

Do not assume the largest model wins once good context is supplied.

## 10. Benchmark discipline

Never quote a vendor benchmark without checking:

- benchmark version;
- model;
- prompt;
- retrieval settings;
- evaluation harness;
- judge model;
- dataset leakage;
- who ran the benchmark.

The memory ecosystem has conflicting benchmark claims, so independent reproduction is part of the project.
