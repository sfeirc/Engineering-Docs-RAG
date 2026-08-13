"""Retrieval evaluation: precision@k, recall@k, and MRR against a labeled
question -> relevant-document-ids ground truth set.

These are standard information-retrieval metrics (see e.g. Manning, Raghavan
& Schuetze, *Introduction to Information Retrieval*, ch. 8):

- precision@k: of the top `k` documents returned, what fraction are
  actually relevant? Measures how much irrelevant material a user has to
  wade through.
- recall@k: of all documents actually relevant to the question, what
  fraction were found in the top `k`? Measures how much relevant material
  was missed.
- MRR (Mean Reciprocal Rank): the reciprocal of the rank of the *first*
  relevant document, averaged across all questions (0 if no relevant
  document appears anywhere in the ranked list). Rewards getting a relevant
  result near the very top of the ranking, specifically.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import RAGPipeline


@dataclass(frozen=True)
class QAExample:
    qid: str
    question: str
    relevant_doc_ids: list[str]


@dataclass(frozen=True)
class EvalResult:
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    n_questions: int
    per_question: list[dict]


def evaluate_retrieval(
    pipeline: RAGPipeline,
    examples: list[QAExample],
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> EvalResult:
    """Run `pipeline.retrieve_doc_ids` against every example and average the metrics.

    A single, sufficiently long ranked list is retrieved per question (long
    enough to cover every requested `k` and to find a relevant document
    anywhere in the corpus for MRR), so this is O(n_questions) BM25 queries,
    not O(n_questions * len(k_values)).
    """
    if not examples:
        raise ValueError("examples must be non-empty")
    max_k = max(k_values)
    rank_list_len = max(max_k, len(pipeline))

    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    reciprocal_ranks: list[float] = []
    per_question: list[dict] = []

    for ex in examples:
        relevant = set(ex.relevant_doc_ids)
        ranked = pipeline.retrieve_doc_ids(ex.question, top_k=rank_list_len)

        for k in k_values:
            hits = len(set(ranked[:k]) & relevant)
            precision_sums[k] += hits / k
            recall_sums[k] += (hits / len(relevant)) if relevant else 0.0

        rr = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        per_question.append(
            {
                "qid": ex.qid,
                "question": ex.question,
                "relevant_doc_ids": sorted(relevant),
                "ranked_top5": ranked[:5],
                "reciprocal_rank": rr,
            }
        )

    n = len(examples)
    return EvalResult(
        precision_at_k={k: precision_sums[k] / n for k in k_values},
        recall_at_k={k: recall_sums[k] / n for k in k_values},
        mrr=sum(reciprocal_ranks) / n,
        n_questions=n,
        per_question=per_question,
    )
