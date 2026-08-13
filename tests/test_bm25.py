"""Tests for the from-scratch BM25 implementation.

`test_hand_computed_scores_exact` reproduces, digit for digit, a small
worked example computed by hand against the formula documented in
`engineering_rag/bm25.py` and reproduced in the README -- see that file's
docstring for the exact derivation. It is the single most important test in
this project: if it fails, the retrieval half of this whole project's
measured evaluation numbers are not to be trusted.
"""

import math

import pytest

from engineering_rag.bm25 import BM25Index


def test_document_containing_query_terms_ranks_above_unrelated_document():
    idx = BM25Index()
    idx.index(
        [
            ("relevant", "the centrifugal pump seal requires periodic inspection and replacement"),
            ("unrelated", "the storage tank roof seal design uses a rim mounted wiper"),
            ("totally_unrelated", "gas turbine driven compressor exports product downstream"),
        ]
    )
    results = idx.query("centrifugal pump seal inspection", top_k=3)
    assert results[0].doc_id == "relevant"
    assert results[0].score > results[1].score > results[2].score
    assert results[2].doc_id == "totally_unrelated"


def test_hand_computed_scores_exact():
    """Worked example (also reproduced in the README).

    Corpus of 3 tiny 3-token documents, k1=1.5, b=0.75 (this project's
    defaults), avgdl=3:

        docA: "cat cat dog"    (2 occurrences of "cat")
        docB: "car train plane" (0 occurrences of "cat")
        docC: "cat bird fish"   (1 occurrence of "cat")

    Query: "cat". N=3, n(cat)=2 (docA and docC contain it).

        idf(cat) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.4700036...

        docA: f=2, dl=3
          denom = 2 + 1.5*(1 - 0.75 + 0.75*3/3) = 2 + 1.5*1 = 3.5
          score = idf * (2 * 2.5) / 3.5 = idf * 5/3.5 = 0.6714337...

        docC: f=1, dl=3
          denom = 1 + 1.5*(1 - 0.75 + 0.75*3/3) = 1 + 1.5 = 2.5
          score = idf * (1 * 2.5) / 2.5 = idf * 1 = 0.4700036...

        docB: f=0 -> score = 0.0 exactly (no "cat" occurrence contributes anything)
    """
    idx = BM25Index(k1=1.5, b=0.75)
    idx.index(
        [
            ("docA", "cat cat dog"),
            ("docB", "car train plane"),
            ("docC", "cat bird fish"),
        ]
    )

    n, df = 3, 2
    expected_idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    assert expected_idf == pytest.approx(0.4700036292457356, rel=1e-12)

    score_a = idx.score("cat", 0)
    score_b = idx.score("cat", 1)
    score_c = idx.score("cat", 2)

    assert score_a == pytest.approx(0.6714337560653366, rel=1e-9)
    assert score_c == pytest.approx(0.4700036292457356, rel=1e-9)
    assert score_b == pytest.approx(0.0, abs=1e-12)

    # ranking sanity: more occurrences of the query term -> higher score
    assert score_a > score_c > score_b

    ranked = idx.query("cat", top_k=3)
    assert [r.doc_id for r in ranked] == ["docA", "docC", "docB"]
    assert ranked[0].score == pytest.approx(score_a)
    assert ranked[1].score == pytest.approx(score_c)
    assert ranked[2].score == pytest.approx(score_b)


def test_idf_of_a_term_in_every_document_stays_non_negative():
    # A term appearing in ALL documents would make the textbook (non "+1")
    # Robertson-Sparck-Jones idf zero or negative; the "+1" variant used
    # here (see bm25.py docstring) keeps it strictly positive.
    idx = BM25Index()
    idx.index([("a", "common word here"), ("b", "common word there"), ("c", "common word everywhere")])
    idf_common = idx._idf("common")
    assert idf_common > 0.0


def test_term_frequency_saturation_k1():
    # Higher k1 should let repeated term occurrences matter more (BM25's
    # term-frequency saturation is *weaker* with higher k1).
    text_low_freq = "pump seal leak detected"
    text_high_freq = "pump pump pump pump seal leak detected"

    idx_low_k1 = BM25Index(k1=0.1, b=0.75)
    idx_low_k1.index([("low", text_low_freq), ("high", text_high_freq)])
    ratio_low_k1 = idx_low_k1.score("pump", 1) / idx_low_k1.score("pump", 0)

    idx_high_k1 = BM25Index(k1=3.0, b=0.75)
    idx_high_k1.index([("low", text_low_freq), ("high", text_high_freq)])
    ratio_high_k1 = idx_high_k1.score("pump", 1) / idx_high_k1.score("pump", 0)

    # with a small k1, term frequency saturates fast: the 4x-repeated-term
    # document scores only modestly higher than the single-occurrence one.
    # With a larger k1, the extra occurrences count for relatively more.
    assert ratio_high_k1 > ratio_low_k1


def test_length_normalization_b():
    # b=0 disables length normalization entirely: a long document repeating
    # the query term the same number of times as a short one should score
    # IDENTICALLY when b=0, but LOWER when b=1 (full length normalization
    # penalizes the longer document for its extra, irrelevant bulk).
    short_doc = "pump seal"
    long_doc = "pump seal " + " ".join(f"filler{i}" for i in range(50))

    idx_b0 = BM25Index(k1=1.5, b=0.0)
    idx_b0.index([("short", short_doc), ("long", long_doc)])
    assert idx_b0.score("pump", 0) == pytest.approx(idx_b0.score("pump", 1))

    idx_b1 = BM25Index(k1=1.5, b=1.0)
    idx_b1.index([("short", short_doc), ("long", long_doc)])
    assert idx_b1.score("pump", 0) > idx_b1.score("pump", 1)


def test_absent_query_term_contributes_zero_not_a_penalty():
    idx = BM25Index()
    idx.index([("a", "pump seal maintenance"), ("b", "compressor surge control")])
    # "compressor" doesn't appear in doc "a" at all
    score_with_only_matching_terms = idx.score("pump", 0)
    score_with_extra_absent_term = idx.score("pump compressor", 0)
    assert score_with_extra_absent_term == pytest.approx(score_with_only_matching_terms)


def test_query_before_index_raises():
    idx = BM25Index()
    with pytest.raises(RuntimeError):
        idx.query("anything")
    with pytest.raises(RuntimeError):
        idx.score("anything", 0)


def test_duplicate_doc_ids_rejected():
    idx = BM25Index()
    with pytest.raises(ValueError):
        idx.index([("dup", "text one"), ("dup", "text two")])


def test_top_k_truncates_and_len_reports_corpus_size():
    idx = BM25Index()
    idx.index([("a", "pump"), ("b", "pump"), ("c", "pump"), ("d", "pump")])
    assert len(idx) == 4
    assert len(idx.query("pump", top_k=2)) == 2


@pytest.mark.parametrize("bad_k1", [-1.0, -0.001])
def test_negative_k1_rejected(bad_k1):
    with pytest.raises(ValueError):
        BM25Index(k1=bad_k1)


@pytest.mark.parametrize("bad_b", [-0.1, 1.1])
def test_out_of_range_b_rejected(bad_b):
    with pytest.raises(ValueError):
        BM25Index(b=bad_b)
