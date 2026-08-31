"""Decision support metrics — grounding, calibration placeholders."""

from __future__ import annotations


def factual_grounding_score(found_facts: int, total_claims: int) -> float:
    if total_claims == 0:
        return 0.0
    return found_facts / total_claims
