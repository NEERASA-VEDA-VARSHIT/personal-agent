"""Deterministic retrieval dataset for M7 first experiment.

Each case:
- query
- relevant_ids (ground truth)
- stale_ids (superseded/expired, should not be retrieved for current intent)
- sensitive_ids (confidential, must not leak)
"""

from __future__ import annotations


# Minimal labelled set — enough to demonstrate Recall@K / MRR / stale & leakage.
# In a real M7 we expand to 100 cases; here we keep it deterministic and tiny
# so baselines can be tested without external LLM calls (hash embeddings).
RETRIEVAL_CASES = [
    {
        "query": "What are my career goals?",
        "relevant_contents": ["software engineering career", "improve backend skills"],
        "stale_contents": [],
        "sensitive_contents": [],
    },
    {
        "query": "Which language do I currently prefer?",
        "relevant_contents": ["I now prefer TypeScript"],
        "stale_contents": ["I prefer Python"],
        "sensitive_contents": [],
    },
    {
        "query": "How do I feel about public speaking?",
        "relevant_contents": ["I've started enjoying presentations"],
        "stale_contents": [],
        "sensitive_contents": [],
        "contradiction_paired": "I hate public speaking.",
    },
    {
        "query": "What are my career goals?",
        "relevant_contents": ["goal: improve backend skills"],
        "stale_contents": [],
        "sensitive_contents": ["My performance review was confidential"],
    },
    {
        "query": "I am interested in distributed systems",
        "relevant_contents": ["I am interested in distributed systems"],
        "stale_contents": [],
        "sensitive_contents": [],
        # source-quality variant handled in runner by creating two mems with same content but different source_type
        "source_quality_test": True,
    },
]
