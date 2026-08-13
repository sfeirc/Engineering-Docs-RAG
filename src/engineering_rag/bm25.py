"""A from-scratch implementation of Okapi BM25 (Robertson & Zaragoza, 2009).

BM25 is the standard bag-of-words ranking function used by production search
engines (Lucene/Elasticsearch's default `BM25Similarity` implements this
exact formula). It is implemented here directly against its textbook
definition, rather than delegated to a library, because retrieval is the
part of this project meant to be the most rigorous and best-tested: every
term below is unit-tested against a worked, by-hand example in
`tests/test_bm25.py` (also reproduced in the README).

Formula, per query term `q` scored against document `D` in a corpus of `N`
documents::

    score(D, q) = idf(q) * f(q, D) * (k1 + 1)
                  ----------------------------------------------
                  f(q, D) + k1 * (1 - b + b * |D| / avgdl)

    idf(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))

where `f(q, D)` is the number of times `q` occurs in `D`, `|D|` is `D`'s
length in tokens, `avgdl` is the corpus's average document length, `n(q)` is
the number of documents containing `q` at least once, and `k1` (term-
frequency saturation, default 1.5) and `b` (length-normalization strength,
default 0.75) are the standard Robertson tuning constants.

The document's full score is the sum of this term over every query term
present in `D` (terms absent from `D` contribute zero, not a penalty).

The "+1" inside `idf`'s logarithm is a standard variant (used by Lucene and
Elasticsearch) that keeps idf non-negative even for a term occurring in more
than half the corpus -- the original Robertson-Sparck-Jones idf can go
negative in that case, which would make a very common term actively *hurt*
a document's score. That is an undesirable property for a general-purpose
index and is why this project uses the "+1" variant rather than the
textbook-original one.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .tokenize import tokenize


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float


class BM25Index:
    """An in-memory Okapi BM25 index over a set of (doc_id, text) documents."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()  # n(q): #docs containing term q
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        self._built = False

    def index(self, documents: list[tuple[str, str]]) -> None:
        """Build the index from a list of `(doc_id, text)` pairs.

        Calling this replaces any previously indexed content. Raises
        ValueError if `doc_id`s are not unique (BM25 scoring and downstream
        chunk lookup both assume a document is identified unambiguously).
        """
        doc_ids = [doc_id for doc_id, _ in documents]
        if len(doc_ids) != len(set(doc_ids)):
            dupes = sorted({d for d in doc_ids if doc_ids.count(d) > 1})
            raise ValueError(f"duplicate doc_id(s) passed to index(): {dupes}")

        self._doc_ids = []
        self._doc_len = []
        self._term_freqs = []
        self._doc_freq = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            self._doc_ids.append(doc_id)
            self._doc_len.append(len(tokens))
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            for term in tf:
                self._doc_freq[term] += 1

        self._n_docs = len(documents)
        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0
        self._built = True

    def _idf(self, term: str) -> float:
        n = self._doc_freq.get(term, 0)
        return math.log(1 + (self._n_docs - n + 0.5) / (n + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score of one already-indexed document (by position) against `query`."""
        if not self._built:
            raise RuntimeError("BM25Index.index() must be called before score()")
        query_terms = tokenize(query)
        tf = self._term_freqs[doc_index]
        dl = self._doc_len[doc_index]
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else f
            total += idf * (f * (self.k1 + 1)) / denom
        return total

    def query(self, query: str, top_k: int = 5) -> list[ScoredDoc]:
        """Return the `top_k` indexed documents ranked by BM25 score, descending.

        Ties are broken by original indexing order (Python's sort is stable,
        including under `reverse=True`, so equal-score documents keep their
        relative insertion order rather than being shuffled).
        """
        if not self._built:
            raise RuntimeError("BM25Index.index() must be called before query()")
        scored = [ScoredDoc(doc_id=self._doc_ids[i], score=self.score(query, i)) for i in range(self._n_docs)]
        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return self._n_docs
