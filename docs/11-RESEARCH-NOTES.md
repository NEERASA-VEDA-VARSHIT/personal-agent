# 11 — Research Notes

## Current landscape

Long-term agent memory has several competing architectural patterns.

### Letta / MemGPT

Uses an OS-style mental model: context is analogous to RAM while external memory is analogous to disk. The agent can manage memory through explicit mechanisms.

Useful lesson:

> Memory management can itself be an agent capability.

Tradeoff:

> More agent control can mean less predictability.

### Mem0

Focuses on practical long-term memory with extraction, consolidation/retrieval, and structured/entity-aware retrieval. Its published work reports substantial efficiency gains over full-context approaches.

Useful lesson:

> Selective memory can outperform dumping entire histories into prompts.

### Zep / Graphiti

Focuses on temporal knowledge graphs and changing facts.

Useful lesson:

> Personal information is temporal. "What is true now?" and "what used to be true?" are different queries.

### MemMachine

Recent work emphasizes preserving conversational ground truth rather than aggressively compressing everything into lossy extracted facts.

Useful lesson:

> Keep evidence and derived summaries separate.

## Design conclusion

For our project, the best starting architecture is:

```text
PostgreSQL = canonical records
pgvector   = semantic index
structured tables = preferences/goals/decisions
raw evidence = preserved selectively
temporal fields = validity over time
graph = optional future layer
```

This is deliberately simpler than adopting a full memory framework immediately.

## Why this is interesting

The project can experimentally compare:

```text
No memory
Vector memory
Hybrid structured + vector
Temporal memory
Agent-managed memory
```

This makes the project a research/engineering study rather than a wrapper.

## Current open problems

1. What should be remembered?
2. What should never be remembered automatically?
3. How should contradictions be represented?
4. When should a memory expire?
5. How should a model distinguish fact from inference?
6. When is a graph actually useful?
7. How much memory improves decisions before it becomes distracting?
8. How should the agent decide whether to ask a question?
9. How do we measure whether advice is genuinely useful?
10. How do we prevent personalization from becoming overconfidence?

## Sources consulted

- Mem0 research paper: `https://arxiv.org/abs/2504.19413`
- MemMachine research paper: `https://arxiv.org/abs/2604.04853`
- 2026 survey/comparison material on Letta, Mem0, Zep/Graphiti, LangMem and related memory architectures.
- 2026 research on cost/accuracy tradeoffs between vector and graph memory.

See the project research notes and citations in the project discussion for source details.
