"""engineering_rag: a from-scratch BM25 retrieval pipeline for engineering
documents, with a pluggable (extractive-by-default, no-LLM-required)
answer-generation interface.

See the project README for the full design rationale.
"""

from __future__ import annotations

from .bm25 import BM25Index, ScoredDoc
from .chunking import Chunk, chunk_text
from .generator import (
    ExtractiveGenerator,
    GeneratedAnswer,
    Generator,
    OpenAICompatibleGenerator,
    RetrievedPassage,
)
from .pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "BM25Index",
    "ScoredDoc",
    "Chunk",
    "chunk_text",
    "ExtractiveGenerator",
    "GeneratedAnswer",
    "Generator",
    "OpenAICompatibleGenerator",
    "RetrievedPassage",
    "RAGPipeline",
]
