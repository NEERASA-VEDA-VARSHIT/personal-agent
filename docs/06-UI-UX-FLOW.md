# 06 — UI / UX Flow

## 1. Main navigation

```text
Personal Agent
│
├── Chat
├── Memories
├── Decisions
├── Journal
├── Insights
└── Settings
```

## 2. Chat

Primary screen:

```text
┌─────────────────────────────────────────────┐
│ Personal Agent                 Local Mode ● │
├─────────────────────────────────────────────┤
│                                             │
│ Conversation                                │
│                                             │
│ User message                                │
│                                             │
│ Agent response                              │
│                                             │
│ [Sources: 3 memories]                       │
│                                             │
├─────────────────────────────────────────────┤
│ Ask anything...                    [Send]    │
└─────────────────────────────────────────────┘
```

## 3. Memory transparency

When memory is used:

> "I used 3 memories from your previous conversations."

User can expand:

```text
Why this memory?
Source: Aug 12 conversation
Confidence: High
Last confirmed: Aug 12
[View] [Edit] [Forget]
```

## 4. Question behavior

The agent should not ask questions just to sound conversational.

Example:

User:

> "Should I take this internship?"

If missing information cannot change the recommendation:

> Answer directly with caveats.

If a missing fact could flip the decision:

> Ask one focused question.

## 5. Decision mode

Optional structured view:

```text
Decision
────────────────────────
Question:
Should I take Internship A?

Criteria
Learning       ████████░░
Mentorship     ██████░░░░
Compensation   ██████████
Location       ███████░░░

Known
• ...

Unknown
• ...

Assumptions
• ...

Options
A ...
B ...

Current assessment
...

What could change it?
...
```

## 6. Memories screen

```text
Your Memory

Preferences
Goals
People
Experiences
Decisions
Patterns
```

Every item supports:

- view;
- edit;
- forget;
- mark incorrect;
- change sensitivity.

## 7. Privacy indicator

Persistent UI indicator:

```text
● Local only
```

or

```text
● Hybrid
```

Clicking it shows exactly what can leave the device.

## 8. Settings

- Strict local mode
- Hybrid mode
- Memory permissions
- Auto-save memories
- Data export
- Delete all data
- Model selection
- Technical telemetry
