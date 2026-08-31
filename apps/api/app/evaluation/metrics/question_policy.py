"""Question policy metrics."""

from __future__ import annotations


def unnecessary_question_rate(decisions: list[dict]) -> float:
    """Fraction of ASK where gold was ANSWER."""
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d["pred"] == "ASK" and d["gold"] == "ANSWER") / len(decisions)


def missed_question_rate(decisions: list[dict]) -> float:
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d["pred"] == "ANSWER" and d["gold"] == "ASK") / len(decisions)


def critical_capture_rate(decisions: list[dict]) -> float:
    critical = [d for d in decisions if d.get("is_critical")]
    if not critical:
        return 0.0
    return sum(1 for d in critical if d["pred"] == d["gold"]) / len(critical)
