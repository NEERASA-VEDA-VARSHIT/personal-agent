"""Memory metrics placeholder — extraction precision/recall, etc."""

from __future__ import annotations


def extraction_precision(extracted: list, relevant: list) -> float:
    if not extracted:
        return 0.0
    return len(set(extracted) & set(relevant)) / len(extracted)


def extraction_recall(extracted: list, relevant: list) -> float:
    if not relevant:
        return 0.0
    return len(set(extracted) & set(relevant)) / len(relevant)
