# Personal Agent — Project Documentation

## Working thesis

**Personal Agent** is a privacy-first personal reflection and decision-support AI that helps a user reason about experiences, choices, goals, and recurring patterns while keeping sensitive personal data under the user's control.

It is **not** a therapist, psychologist, medical system, or autonomous life manager.

## Documentation map

- `01-PRD.md` — product requirements and scope
- `02-ARCHITECTURE.md` — system architecture and data flow
- `03-MEMORY-DESIGN.md` — memory model, storage, write/read policies
- `04-TECH-STACK.md` — technology choices and tradeoffs
- `05-FOLDER-STRUCTURE.md` — proposed repository structure
- `06-UI-UX-FLOW.md` — screens and interaction flows
- `07-AGENT-POLICY.md` — answer/question/decision-support policy
- `08-PRIVACY-SECURITY.md` — privacy model and threat model
- `09-EVALUATION.md` — quality, memory, safety and privacy evaluation
- `10-ROADMAP.md` — staged implementation plan
- `11-RESEARCH-NOTES.md` — research findings and design implications
- `12-PORTFOLIO.md` — portfolio/resume/interview framing

## Core design principle

**Local-first, evidence-grounded, user-controlled memory.**

The first version should prefer a simple, inspectable architecture over a complex autonomous memory framework.
