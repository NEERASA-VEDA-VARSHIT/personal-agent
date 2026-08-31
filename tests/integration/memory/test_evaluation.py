"""Tests for M7 retrieval metrics."""

import sys
from pathlib import Path

# Make evaluation/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import unittest

from evaluation.metrics.retrieval import recall_at_k, precision_at_k, mrr, stale_rate, leakage_rate


class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_at_k(self) -> None:
        self.assertAlmostEqual(recall_at_k([1, 2, 3], [1, 2], k=2), 1.0)
        self.assertAlmostEqual(recall_at_k([1, 3], [1, 2], k=2), 0.5)
        self.assertAlmostEqual(recall_at_k([], [1], k=3), 0.0)

    def test_precision_at_k(self) -> None:
        self.assertAlmostEqual(precision_at_k([1, 2, 3], [1, 2], k=2), 1.0)
        self.assertAlmostEqual(precision_at_k([1, 3], [1, 2], k=2), 0.5)
        self.assertAlmostEqual(precision_at_k([3, 4], [1, 2], k=2), 0.0)

    def test_mrr(self) -> None:
        self.assertAlmostEqual(mrr([1, 2, 3], [2]), 0.5)
        self.assertAlmostEqual(mrr([2, 1, 3], [2]), 1.0)
        self.assertAlmostEqual(mrr([3, 4], [1, 2]), 0.0)

    def test_stale_rate(self) -> None:
        self.assertAlmostEqual(stale_rate([1, 2, 3], {2, 3}), 2 / 3)
        self.assertAlmostEqual(stale_rate([1], {2}), 0.0)
        self.assertAlmostEqual(stale_rate([], {1}), 0.0)

    def test_leakage_rate(self) -> None:
        self.assertAlmostEqual(leakage_rate([1, 2], {2}), 0.5)
        self.assertAlmostEqual(leakage_rate([1, 3], {2}), 0.0)


class TestRunnerIntegration(unittest.TestCase):
    def test_runner_produces_improvement(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "apps" / "api"))
        from evaluation.runner import run_retrieval_experiment

        results = run_retrieval_experiment(k=3, verbose=False)
        # hybrid_rerank should not be worse than vector_only on recall and mrr
        self.assertGreaterEqual(results["hybrid_rerank"]["recall@k"], results["vector_only"]["recall@k"])
        self.assertGreaterEqual(results["hybrid_rerank"]["mrr"], results["vector_only"]["mrr"])
        # and should have zero stale/leakage (policy filters)
        self.assertEqual(results["hybrid_rerank"]["stale_rate"], 0.0)
        self.assertEqual(results["hybrid_rerank"]["leakage_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
