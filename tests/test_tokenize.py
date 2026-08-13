from engineering_rag.tokenize import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("The Pump Model CP-100 delivers 250 m3/h.") == [
        "pump",
        "model",
        "cp-100",
        "delivers",
        "250",
        "m3/h",
    ]


def test_stopwords_removed_by_default():
    tokens = tokenize("the pump and the compressor")
    assert "the" not in tokens
    assert "and" not in tokens
    assert tokens == ["pump", "compressor"]


def test_stopwords_kept_when_disabled():
    tokens = tokenize("the pump and the compressor", remove_stopwords=False)
    assert tokens == ["the", "pump", "and", "the", "compressor"]


def test_hyphenated_and_slash_tokens_stay_whole():
    assert tokenize("PT-800 reads 0-60 bar g") == ["pt-800", "reads", "0-60", "bar", "g"]


def test_empty_string_yields_no_tokens():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_trailing_punctuation_is_stripped():
    tokens = tokenize("cavitation, vibration, and seal damage.")
    assert tokens == ["cavitation", "vibration", "seal", "damage"]
