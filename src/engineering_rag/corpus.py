"""Loading the synthetic demo corpus (or any other `*.txt` corpus directory)."""

from __future__ import annotations

from pathlib import Path


def load_corpus(corpus_dir: str | Path) -> list[tuple[str, str]]:
    """Load all `*.txt` documents from `corpus_dir`, sorted by filename.

    Each document's `doc_id` is its filename stem, e.g.
    `pump-cp100-datasheet.txt` -> `pump-cp100-datasheet`.

    Raises FileNotFoundError if the directory contains no `.txt` files
    (silently indexing zero documents would otherwise fail much later, and
    much less clearly, at query time).
    """
    corpus_dir = Path(corpus_dir)
    docs = [(path.stem, path.read_text(encoding="utf-8")) for path in sorted(corpus_dir.glob("*.txt"))]
    if not docs:
        raise FileNotFoundError(f"No .txt documents found in {corpus_dir}")
    return docs
