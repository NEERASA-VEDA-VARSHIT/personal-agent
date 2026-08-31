# 01 — Product Requirements Document

## 1. Product

**Working name:** Personal Agent  
**Category:** Privacy-first personal reflection and decision-support agent  
**Primary user:** One individual using the system as a private thinking partner.

## 2. Problem

General-purpose chatbots can provide useful advice, but a personal assistant becomes much more useful when it understands the user's history, goals, previous decisions, and changing circumstances.

That creates a major privacy problem: the more personal context the system needs, the more sensitive information it may need to process.

We want to explore:

> How much useful personalization can we achieve while keeping sensitive personal data local and under explicit user control?

## 3. Product principles

1. **Privacy before convenience**
2. **Memory is evidence, not truth**
3. **User controls memory**
4. **Never invent personal history**
5. **Ask only useful questions**
6. **Expose uncertainty and assumptions**
7. **Support decisions; do not make life decisions for the user**
8. **Safety escalates with stakes**
9. **Prefer reversible actions**
10. **Every important memory should have provenance**

## 4. Core use cases

### A. Reflection

User describes an event and asks for help understanding it.

### B. Decision support

User compares options and wants structured reasoning.

### C. Long-term continuity

User asks about something discussed previously.

### D. Pattern discovery

Agent identifies repeated themes only when supported by multiple pieces of evidence.

### E. Goal tracking

Agent helps the user maintain goals and review progress.

## 5. MVP

### Must have

- Local chat UI
- Local LLM inference
- Conversation history
- Explicit memory creation
- Semantic memory retrieval
- Structured memory records
- Memory inspection/edit/delete
- Source/provenance for recalled facts
- Decision-support response format
- Question-vs-answer policy
- Basic safety boundaries
- Evaluation harness

### Should have

- Hybrid model routing
- Calendar/task tools
- Decision journal
- Memory confidence
- Contradiction detection
- Export/import
- Encryption at rest

### Won't have initially

- Diagnosis
- Mental-health treatment
- Crisis intervention
- Fully autonomous life management
- Financial/legal/medical professional advice
- Automatic sending of messages
- Silent monitoring of private services
- Fine-tuning a foundation model from scratch

## 6. Example success scenario

User:

> "I have an internship offer. The salary is good, but I'm worried I won't learn much."

Agent:

1. identifies the decision;
2. retrieves relevant user priorities;
3. distinguishes known facts from assumptions;
4. identifies missing information;
5. asks a question only if its answer could materially change the recommendation;
6. presents a comparison;
7. lets the user decide;
8. optionally stores the decision and rationale.

## 7. Non-functional requirements

- Local-only mode must work without internet after model installation.
- Personal memories must never be sent to cloud providers in strict-local mode.
- User must be able to inspect and delete stored memories.
- Deleting a memory must make it unavailable to retrieval.
- Every recalled memory should have provenance.
- System should log technical events without logging sensitive content by default.
- Agent should be deterministic enough for evaluation while allowing model upgrades.

## 8. Product success metrics

- Memory retrieval precision
- Memory retrieval recall
- Personal-fact hallucination rate
- Contradiction rate
- Question efficiency
- Advice quality score
- Sycophancy rate
- User correction rate
- Privacy leakage rate
- p95 response latency
- Local inference success rate
