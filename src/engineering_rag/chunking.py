"""Fixed-size, overlapping word-window chunking for retrieval indexing.

Chunking trades off two failure modes on a term-frequency index like BM25:

- chunks too large dilute term frequency and topic focus (a query about
  "seal replacement" would match a whole multi-topic document just as well
  as a passage that is *only* about seal replacement, because both contain
  the term at least once -- BM25's length normalization partially, but only
  partially, compensates for this);
- chunks too small, or non-overlapping, can split the terms relevant to a
  query across a chunk boundary, so neither chunk alone contains enough of
  the query's vocabulary to score well.

`chunk_text` slides a window over *whitespace-separated word tokens* (not
characters, and not sentences -- word counts are what BM25's length
normalization and this project's chunk-size parameter both operate on) with
a configurable overlap, so information straddling a boundary in one chunk
still appears whole in an adjacent chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One chunk of a source document.

    `start_word`/`end_word` are a half-open `[start_word, end_word)` index
    range into the document's whitespace-split word list, so chunk
    boundaries and overlap can be checked exactly (see tests/test_chunking.py).
    """

    text: str
    start_word: int
    end_word: int
    chunk_index: int


def chunk_text(text: str, *, chunk_size: int = 120, chunk_overlap: int = 30) -> list[Chunk]:
    """Split `text` into overlapping chunks of up to `chunk_size` words.

    Consecutive chunks overlap by exactly `chunk_overlap` words (except that
    the final chunk may be shorter than `chunk_size`, since it stops at the
    end of the document rather than padding). An empty/whitespace-only
    `text` returns an empty list.

    Raises ValueError if `chunk_size` <= 0, `chunk_overlap` < 0, or
    `chunk_overlap >= chunk_size` (the last would mean each window never
    advances, looping forever).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    n = len(words)
    chunks: list[Chunk] = []
    start = 0
    while True:
        end = min(start + chunk_size, n)
        chunks.append(
            Chunk(
                text=" ".join(words[start:end]),
                start_word=start,
                end_word=end,
                chunk_index=len(chunks),
            )
        )
        if end >= n:
            break
        start += step
    return chunks
