"""pytest-benchmark microbenchmarks for indexing and query throughput.

Run with: pytest benchmarks/ --benchmark-only -v
(excluded from the default `pytest` run via testpaths=["tests"] in
pyproject.toml, so `pytest` alone never silently also runs benchmarks.)

Numbers are measured on whatever machine runs this file -- see the README
for the actual numbers measured on this project's development machine and
in CI, plus the standard caveat that CI runner throughput varies run to run
and is not directly comparable to a dedicated workstation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from engineering_rag.bm25 import BM25Index  # noqa: E402
from engineering_rag.corpus import load_corpus  # noqa: E402
from engineering_rag.pipeline import RAGPipeline  # noqa: E402

CORPUS_DIR = REPO_ROOT / "data" / "corpus"


def _documents() -> list[tuple[str, str]]:
    return load_corpus(CORPUS_DIR)


def test_bench_ingest_full_demo_corpus(benchmark):
    """End-to-end ingest throughput: chunk + tokenize + BM25-index all 25 demo documents."""
    documents = _documents()

    def _ingest():
        pipeline = RAGPipeline(chunk_size=120, chunk_overlap=30)
        return pipeline.ingest(documents)

    n_chunks = benchmark(_ingest)
    assert n_chunks > 0


def test_bench_query_single_question(benchmark):
    """Query throughput against the already-indexed full demo corpus."""
    pipeline = RAGPipeline(chunk_size=120, chunk_overlap=30)
    pipeline.ingest(_documents())

    def _query():
        return pipeline.retrieve("What is the maximum discharge pressure of the centrifugal pump?", top_k=5)

    results = benchmark(_query)
    assert len(results) == 5


def test_bench_bm25_score_single_document(benchmark):
    """Cost of scoring one already-tokenized document against one query (the innermost hot loop)."""
    idx = BM25Index()
    idx.index(_documents())

    def _score():
        return idx.score("centrifugal pump seal inspection cavitation", 0)

    benchmark(_score)


def test_bench_tokenize_a_typical_chunk(benchmark):
    """Tokenizer throughput on a realistically sized (~120-word) chunk of text."""
    from engineering_rag.tokenize import tokenize

    documents = _documents()
    sample_text = " ".join(documents[0][1].split()[:120])

    benchmark(lambda: tokenize(sample_text))
