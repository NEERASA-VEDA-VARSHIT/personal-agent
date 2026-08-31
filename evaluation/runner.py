"""M7 runner — compares Retrieval baselines on a deterministic dataset.

Usage:
    python -m evaluation.runner
    python evaluation/runner.py
"""

from __future__ import annotations

import hashlib
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import sys
from pathlib import Path

# Make `app` importable when running as `python -m evaluation.runner` or `python evaluation/runner.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Memory, MemorySource, User
from app.memory.lifecycle import MemoryLifecycle
from app.memory.retrieval import CandidateRetriever, RetrievalPipeline
from app.models.embeddings import EmbeddingService
from app.memory.policy import MemoryStatus, SourceType
from evaluation.baselines.vector_only import retrieve_vector_only
from evaluation.baselines.hybrid import retrieve_hybrid
from evaluation.baselines.hybrid_rerank import retrieve_hybrid_rerank
from evaluation.metrics.retrieval import recall_at_k, precision_at_k, mrr, stale_rate, leakage_rate


def _mock_gateway():
    gw = MagicMock()

    def mock_embed(texts):
        embs = []
        for t in texts:
            h = hashlib.md5(t.encode()).hexdigest()
            vec = [float(int(h[i : i + 2], 16)) / 256.0 for i in range(0, len(h), 2)]
            norm = sum(v * v for v in vec) ** 0.5
            if norm:
                vec = [v / norm for v in vec]
            embs.append(vec)
        return embs

    gw.embed = mock_embed
    return gw


def _make_mem(db, user_id: int, svc: EmbeddingService, content: str, **kw):
    mtype = kw.pop("mtype", "fact")
    status = kw.pop("status", MemoryStatus.ACTIVE.value)
    sensitivity = kw.pop("sensitivity", "private")
    source_type = kw.pop("source_type", SourceType.USER_STATED.value)
    confidence = kw.pop("confidence", 0.9)
    is_active = status == MemoryStatus.ACTIVE.value
    valid_from = kw.pop("valid_from", datetime.utcnow() - timedelta(days=1))
    valid_until = kw.pop("valid_until", None)
    mem = Memory(
        user_id=user_id,
        type=mtype,
        memory_type=mtype,
        content=content,
        sensitivity=sensitivity,
        confidence=confidence,
        status=status,
        is_active=is_active,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    db.add(mem)
    db.flush()
    src = MemorySource(memory_id=mem.id, source_type=source_type, source_ref="eval", confidence=confidence)
    db.add(src)
    svc.embed_memory(db, mem)
    return mem


def run_retrieval_experiment(k: int = 3, verbose: bool = True) -> dict:
    # Setup in-memory DB
    fd, path = tempfile.mkstemp(suffix=".db")
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    gw = _mock_gateway()
    svc = EmbeddingService(gateway=gw)
    retriever = CandidateRetriever(embedding_service=svc)
    pipeline = RetrievalPipeline(retriever=retriever)

    try:
        # Create user + memories covering the 5 cases
        user = User(username="eval_user")
        db.add(user)
        db.flush()
        uid = user.id

        # Case 1: career goals
        g1 = _make_mem(db, uid, svc, "software engineering career goal: become staff engineer", mtype="goal")
        g2 = _make_mem(db, uid, svc, "goal: improve backend skills and system design", mtype="goal")
        _make_mem(db, uid, svc, "I like hiking on weekends", mtype="fact")

        # Case 2: temporal — old Python (will be superseded) vs new TS
        old = _make_mem(db, uid, svc, "I prefer Python", mtype="preference", valid_from=datetime(2024, 1, 1))
        new = _make_mem(db, uid, svc, "I now prefer TypeScript", mtype="preference", valid_from=datetime(2026, 8, 31))
        db.commit()
        MemoryLifecycle.supersede(db, old, new, actor="eval")
        db.commit()

        # Case 3: contradiction — old hate, new enjoy
        hate = _make_mem(db, uid, svc, "I hate public speaking", mtype="preference")
        hate.created_at = datetime.utcnow() - timedelta(days=60)
        hate.updated_at = hate.created_at
        db.add(hate)
        enjoy = _make_mem(db, uid, svc, "I've started enjoying presentations", mtype="preference")
        db.commit()
        MemoryLifecycle.contradict(db, enjoy, hate)
        db.commit()

        # Case 4: sensitive
        sensitive = _make_mem(db, uid, svc, "My performance review was confidential: needs improvement", mtype="fact", sensitivity="confidential")
        normal_goal = _make_mem(db, uid, svc, "goal: improve backend skills", mtype="goal", sensitivity="private")
        db.commit()

        # Case 5: source quality — same content, different source_type, same timestamp
        now = datetime.utcnow()
        same = "I am interested in distributed systems"
        user_stated = _make_mem(db, uid, svc, same, mtype="fact", source_type=SourceType.USER_STATED.value, confidence=0.85)
        inferred = _make_mem(db, uid, svc, same, mtype="hypothesis", source_type=SourceType.MODEL_INFERRED.value, confidence=0.85)
        for m in (user_stated, inferred):
            m.created_at = now
            m.updated_at = now
            db.add(m)
        db.commit()

        # Build mapping from content -> id for relevant/stale lookups
        content_to_id = {m.content: m.id for m in db.query(Memory).filter(Memory.user_id == uid).all()}

        cases = [
            {
                "query": "What are my career goals?",
                "relevant": [g1.id, g2.id],
                "stale_set": set(),
                "sensitive_set": set(),
            },
            {
                "query": "Which language do I currently prefer?",
                "relevant": [new.id],
                "stale_set": {old.id},
                "sensitive_set": set(),
            },
            {
                "query": "How do I feel about public speaking?",
                "relevant": [enjoy.id],
                "stale_set": set(),
                "sensitive_set": set(),
            },
            {
                "query": "What are my career goals?",
                "relevant": [normal_goal.id],
                "stale_set": set(),
                "sensitive_set": {sensitive.id},
            },
            {
                "query": same,
                "relevant": [user_stated.id],
                "stale_set": set(),
                "sensitive_set": set(),
                # For source quality, we check ranking order separately
            },
        ]

        baselines = {
            "vector_only": lambda q: retrieve_vector_only(db, uid, q, svc, k=k),
            "hybrid": lambda q: retrieve_hybrid(db, uid, q, retriever, k=k),
            "hybrid_rerank": lambda q: retrieve_hybrid_rerank(db, uid, q, pipeline, k=k),
        }

        results = {}
        for name, fn in baselines.items():
            recs, precs, mrrs, stales, leaks = [], [], [], [], []
            for c in cases:
                retrieved = fn(c["query"])
                recs.append(recall_at_k(retrieved, c["relevant"], k))
                precs.append(precision_at_k(retrieved, c["relevant"], k))
                mrrs.append(mrr(retrieved, c["relevant"]))
                stales.append(stale_rate(retrieved, c["stale_set"]))
                leaks.append(leakage_rate(retrieved, c["sensitive_set"]))
            results[name] = {
                "recall@k": sum(recs) / len(recs),
                "precision@k": sum(precs) / len(precs),
                "mrr": sum(mrrs) / len(mrrs),
                "stale_rate": sum(stales) / len(stales),
                "leakage_rate": sum(leaks) / len(leaks),
                "cases": len(cases),
                "k": k,
            }

        if verbose:
            print(f"\n=== M7 Retrieval Experiment (k={k}, {len(cases)} cases) ===")
            for name, r in results.items():
                print(f"\n{name}:")
                for mk in ["recall@k", "precision@k", "mrr", "stale_rate", "leakage_rate"]:
                    print(f"  {mk}: {r[mk]:.3f}")

        return results
    finally:
        db.close()
        engine.dispose()
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    run_retrieval_experiment()
