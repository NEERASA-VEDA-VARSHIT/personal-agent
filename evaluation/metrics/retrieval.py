"""M7 — Retrieval metrics: Recall@K, Precision@K, MRR, stale-rate, leakage-rate."""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / k


def mrr(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    """Reciprocal rank of first relevant."""
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def stale_rate(retrieved_ids: list[int], stale_ids: set[int]) -> float:
    if not retrieved_ids:
        return 0.0
    stale = sum(1 for rid in retrieved_ids if rid in stale_ids)
    return stale / len(retrieved_ids)


def leakage_rate(retrieved_ids: list[int], sensitive_ids: set[int]) -> float:
    if not retrieved_ids:
        return 0.0
    leaked = sum(1 for rid in retrieved_ids if rid in sensitive_ids)
    return leaked / len(retrieved_ids)


def average_at_k(cases: list[dict], k: int, metric_fn) -> float:
    if not cases:
        return 0.0
    scores = [metric_fn(c["retrieved"], c["relevant"], k) for c in cases]
    return sum(scores) / len(scores)


def mean_mrr(cases: list[dict]) -> float:
    if not cases:
        return 0.0
    return sum(mrr(c["retrieved"], c["relevant"]) for c in cases) / len(cases)
