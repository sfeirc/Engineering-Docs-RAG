import pytest

from engineering_rag.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_document_is_a_single_chunk():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) == 1
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 50
    assert chunks[0].text == text


def test_chunk_size_is_respected():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    for c in chunks:
        assert c.end_word - c.start_word <= 100


def test_overlap_between_consecutive_chunks_is_exact():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):  # pairwise iteration; lengths differ by 1 by design
        # every non-final pair should advance by exactly (chunk_size - overlap) words...
        if nxt.end_word - nxt.start_word == 100:  # nxt is a "full" chunk, not the shrunk tail
            assert nxt.start_word - prev.start_word == 80
            # ... which means they share exactly `chunk_overlap` words at the boundary
            assert prev.end_word - nxt.start_word == 20


def test_full_word_coverage_no_gaps():
    n_words = 437
    text = " ".join(f"w{i}" for i in range(n_words))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=25)
    assert chunks[0].start_word == 0
    assert chunks[-1].end_word == n_words
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        # no gap: the next chunk must start at or before the previous chunk's end
        assert nxt.start_word <= prev.end_word


def test_last_chunk_may_be_shorter_than_chunk_size():
    # 250 words, chunk_size=100, overlap=20 -> step=80: starts at 0, 80, 160
    # chunk at 160 covers [160, 250) = 90 words, shorter than 100
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert chunks[-1].end_word - chunks[-1].start_word == 90
    assert chunks[-1].end_word == 250


def test_chunk_index_increments_sequentially():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_parameters_raise(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_no_overlap_chunks_tile_exactly():
    n_words = 300
    text = " ".join(f"w{i}" for i in range(n_words))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 3
    assert [(c.start_word, c.end_word) for c in chunks] == [(0, 100), (100, 200), (200, 300)]
