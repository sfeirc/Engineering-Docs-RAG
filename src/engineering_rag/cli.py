"""Command-line interface for engineering_rag."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import load_corpus
from .evaluation import QAExample, evaluate_retrieval
from .pipeline import RAGPipeline

# src/engineering_rag/cli.py -> src/engineering_rag -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS = _REPO_ROOT / "data" / "corpus"
DEFAULT_QA = _REPO_ROOT / "data" / "qa_dataset.json"


def _build_pipeline(corpus_dir: Path, chunk_size: int, chunk_overlap: int) -> tuple[RAGPipeline, int, int]:
    documents = load_corpus(corpus_dir)
    pipeline = RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    n_chunks = pipeline.ingest(documents)
    return pipeline, n_chunks, len(documents)


def cmd_query(args: argparse.Namespace) -> None:
    pipeline, n_chunks, n_docs = _build_pipeline(Path(args.corpus_dir), args.chunk_size, args.chunk_overlap)
    print(f"Indexed {n_chunks} chunks from {n_docs} documents in {args.corpus_dir}", file=sys.stderr)
    answer = pipeline.ask(args.question, top_k=args.top_k)
    print(answer.answer)


def cmd_evaluate(args: argparse.Namespace) -> None:
    pipeline, n_chunks, n_docs = _build_pipeline(Path(args.corpus_dir), args.chunk_size, args.chunk_overlap)
    qa_raw = json.loads(Path(args.qa_file).read_text(encoding="utf-8"))
    examples = [QAExample(qid=q["id"], question=q["question"], relevant_doc_ids=q["relevant_doc_ids"]) for q in qa_raw]
    result = evaluate_retrieval(pipeline, examples, k_values=(1, 3, 5))

    print(f"{n_docs} documents, {n_chunks} chunks indexed (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"{result.n_questions} labeled questions\n")
    print("k    precision@k   recall@k")
    for k in sorted(result.precision_at_k):
        print(f"{k:<4} {result.precision_at_k[k]:<13.3f} {result.recall_at_k[k]:.3f}")
    print(f"\nMRR: {result.mrr:.3f}")


def cmd_demo(args: argparse.Namespace) -> None:
    pipeline, n_chunks, n_docs = _build_pipeline(Path(args.corpus_dir), args.chunk_size, args.chunk_overlap)
    print(f"Indexed {n_chunks} chunks from {n_docs} documents.\n")
    demo_questions = [
        "What is the maximum discharge pressure of the centrifugal pump?",
        "What PPE is required before entering a confined space?",
        "What caused the HAZOP overfill deviation on the storage tank?",
    ]
    for q in demo_questions:
        print(f"Q: {q}")
        answer = pipeline.ask(q, top_k=2)
        print(answer.answer)
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="engineering-rag",
        description=(
            "BM25 retrieval over engineering documents (equipment datasheets, safety "
            "procedures, HAZOP excerpts), with a pluggable answer-generation backend "
            "that defaults to a deterministic, no-LLM extractive generator."
        ),
    )
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS), help="directory of .txt documents to index")
    parser.add_argument("--chunk-size", type=int, default=120, help="chunk size in words")
    parser.add_argument("--chunk-overlap", type=int, default=30, help="chunk overlap in words")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="Ask a question against the indexed corpus")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=3)
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("evaluate", help="Run precision@k/recall@k/MRR evaluation against the labeled QA set")
    p_eval.add_argument("--qa-file", default=str(DEFAULT_QA))
    p_eval.set_defaults(func=cmd_evaluate)

    p_demo = sub.add_parser("demo", help="Run a small set of demo questions end to end")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
