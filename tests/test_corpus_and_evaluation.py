"""Tests against the real shipped demo corpus and labeled QA dataset.

These are integration tests: they exercise `load_corpus` on the actual
`data/corpus/` directory and `evaluate_retrieval` on the actual
`data/qa_dataset.json`, so a broken corpus file, a QA entry referencing a
document that doesn't exist, or a real retrieval regression would all be
caught here -- not just by the isolated unit tests in test_bm25.py /
test_pipeline.py.
"""

import json
from pathlib import Path

import pytest

from engineering_rag.corpus import load_corpus
from engineering_rag.evaluation import QAExample, evaluate_retrieval
from engineering_rag.pipeline import RAGPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
QA_FILE = REPO_ROOT / "data" / "qa_dataset.json"


def test_corpus_loads_and_is_non_trivial():
    docs = load_corpus(CORPUS_DIR)
    assert 15 <= len(docs) <= 30
    doc_ids = [doc_id for doc_id, _ in docs]
    assert len(doc_ids) == len(set(doc_ids))  # unique doc_ids
    for _doc_id, text in docs:
        assert len(text.split()) > 20  # not a stub/empty file
        assert "SYNTHETIC" in text.upper()  # every doc self-labels as synthetic


def test_missing_corpus_dir_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)  # empty directory, no .txt files


def test_qa_dataset_has_20_to_40_questions_referencing_real_documents():
    qa_raw = json.loads(QA_FILE.read_text(encoding="utf-8"))
    assert 20 <= len(qa_raw) <= 40

    doc_ids = {doc_id for doc_id, _ in load_corpus(CORPUS_DIR)}
    qids = [q["id"] for q in qa_raw]
    assert len(qids) == len(set(qids))  # unique question ids

    for q in qa_raw:
        assert q["question"].strip().endswith("?")
        assert len(q["relevant_doc_ids"]) >= 1
        for doc_id in q["relevant_doc_ids"]:
            assert doc_id in doc_ids, f"{q['id']} references unknown doc_id {doc_id!r}"


def test_every_corpus_document_is_referenced_by_at_least_one_question():
    # a QA set that never asks about some documents can't tell us anything
    # about retrieval quality for those documents.
    qa_raw = json.loads(QA_FILE.read_text(encoding="utf-8"))
    referenced = {doc_id for q in qa_raw for doc_id in q["relevant_doc_ids"]}
    all_docs = {doc_id for doc_id, _ in load_corpus(CORPUS_DIR)}
    assert referenced == all_docs


@pytest.fixture(scope="module")
def eval_result():
    documents = load_corpus(CORPUS_DIR)
    pipeline = RAGPipeline(chunk_size=120, chunk_overlap=30)
    pipeline.ingest(documents)
    qa_raw = json.loads(QA_FILE.read_text(encoding="utf-8"))
    examples = [QAExample(qid=q["id"], question=q["question"], relevant_doc_ids=q["relevant_doc_ids"]) for q in qa_raw]
    return evaluate_retrieval(pipeline, examples, k_values=(1, 3, 5))


def test_evaluation_runs_and_produces_bounded_metrics(eval_result):
    assert eval_result.n_questions == 40
    for p in eval_result.precision_at_k.values():
        assert 0.0 <= p <= 1.0
    for r in eval_result.recall_at_k.values():
        assert 0.0 <= r <= 1.0
    assert 0.0 <= eval_result.mrr <= 1.0


def test_recall_is_non_decreasing_in_k(eval_result):
    # recall@k can only go up (or stay flat) as k grows: strictly more
    # documents are being checked against the same relevant set.
    ks = sorted(eval_result.recall_at_k)
    for a, b in zip(ks, ks[1:], strict=False):
        assert eval_result.recall_at_k[b] >= eval_result.recall_at_k[a] - 1e-9


def test_retrieval_quality_is_well_above_random_baseline(eval_result):
    # Random baseline for MRR over 25 documents, one relevant on average,
    # is roughly 1/25 = 0.04 (harmonic-mean-ish, order of magnitude). A BM25
    # index that isn't fundamentally broken should clear that by a wide
    # margin on a QA set intentionally built with real term overlap.
    assert eval_result.mrr > 0.5
    assert eval_result.precision_at_k[1] > 0.5
