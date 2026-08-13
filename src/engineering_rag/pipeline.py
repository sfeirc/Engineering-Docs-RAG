"""End-to-end retrieval pipeline: ingest -> chunk -> index -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index
from .chunking import chunk_text
from .generator import ExtractiveGenerator, GeneratedAnswer, Generator, RetrievedPassage


@dataclass(frozen=True)
class IndexedChunk:
    doc_id: str
    chunk_index: int
    text: str


class RAGPipeline:
    """Ties chunking, BM25 indexing, and a pluggable `Generator` together.

    Chunks are indexed by a synthetic key `f"{doc_id}#{chunk_index}"` so that
    multiple chunks from the same source document can be scored and ranked
    independently, while `retrieve_doc_ids` can still map back to unique
    source documents for document-level evaluation (precision@k, recall@k,
    MRR against the labeled QA set -- see `engineering_rag.evaluation`).
    """

    def __init__(
        self,
        *,
        chunk_size: int = 120,
        chunk_overlap: int = 30,
        k1: float = 1.5,
        b: float = 0.75,
        generator: Generator | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._index = BM25Index(k1=k1, b=b)
        self._chunk_by_key: dict[str, IndexedChunk] = {}
        self.generator: Generator = generator or ExtractiveGenerator()

    def ingest(self, documents: list[tuple[str, str]]) -> int:
        """Chunk and index `documents` (a list of `(doc_id, text)` pairs).

        Replaces any previously ingested content. Returns the number of
        chunks indexed.
        """
        self._chunk_by_key = {}
        records: list[tuple[str, str]] = []
        for doc_id, text in documents:
            for chunk in chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap):
                key = f"{doc_id}#{chunk.chunk_index}"
                self._chunk_by_key[key] = IndexedChunk(doc_id=doc_id, chunk_index=chunk.chunk_index, text=chunk.text)
                records.append((key, chunk.text))
        self._index.index(records)
        return len(records)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedPassage]:
        """Return the `top_k` best-scoring chunks (not deduplicated by document)."""
        scored = self._index.query(question, top_k=top_k)
        passages = []
        for sd in scored:
            ic = self._chunk_by_key[sd.doc_id]
            passages.append(
                RetrievedPassage(doc_id=ic.doc_id, chunk_index=ic.chunk_index, text=ic.text, score=sd.score)
            )
        return passages

    def retrieve_doc_ids(self, question: str, top_k: int = 5) -> list[str]:
        """Return up to `top_k` unique source doc_ids, ranked by their best chunk.

        Since one document can contribute several chunks to the chunk-level
        ranking, this scans further than `top_k` chunks (5x, capped at the
        number of indexed chunks) so that `top_k` *distinct* documents can
        actually be found rather than being crowded out by several chunks of
        the single best-matching document.
        """
        scan_k = min(max(top_k * 5, top_k), len(self._index))
        scored = self._index.query(question, top_k=scan_k)
        seen: list[str] = []
        for sd in scored:
            doc_id = self._chunk_by_key[sd.doc_id].doc_id
            if doc_id not in seen:
                seen.append(doc_id)
            if len(seen) >= top_k:
                break
        return seen

    def ask(self, question: str, top_k: int = 5) -> GeneratedAnswer:
        """Retrieve then generate: the full RAG pipeline in one call."""
        passages = self.retrieve(question, top_k=top_k)
        return self.generator.generate(question, passages)

    def __len__(self) -> int:
        return len(self._index)
