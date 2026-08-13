import pytest

from engineering_rag.generator import (
    ExtractiveGenerator,
    Generator,
    OpenAICompatibleGenerator,
    RetrievedPassage,
)


def _passage(doc_id: str, score: float, text: str = "some passage text") -> RetrievedPassage:
    return RetrievedPassage(doc_id=doc_id, chunk_index=0, text=text, score=score)


def test_extractive_generator_is_a_generator():
    assert isinstance(ExtractiveGenerator(), Generator)


def test_extractive_generator_no_passages_returns_deterministic_no_result_message():
    gen = ExtractiveGenerator()
    result = gen.generate("anything", [])
    assert result.backend == "extractive"
    assert result.passages_used == []
    assert "No relevant passage" in result.answer


def test_extractive_generator_returns_passages_verbatim_in_order():
    gen = ExtractiveGenerator(max_passages=3)
    passages = [
        _passage("docA", 5.0, "first passage text"),
        _passage("docB", 3.0, "second passage text"),
    ]
    result = gen.generate("my question", passages)
    assert result.passages_used == passages
    assert "docA" in result.answer
    assert "docB" in result.answer
    assert "first passage text" in result.answer
    assert "second passage text" in result.answer
    # order preserved: docA (higher score, listed first) appears before docB
    assert result.answer.index("docA") < result.answer.index("docB")


def test_extractive_generator_respects_max_passages():
    gen = ExtractiveGenerator(max_passages=1)
    passages = [_passage("docA", 5.0), _passage("docB", 3.0), _passage("docC", 1.0)]
    result = gen.generate("q", passages)
    assert len(result.passages_used) == 1
    assert result.passages_used[0].doc_id == "docA"
    assert "docB" not in result.answer
    assert "docC" not in result.answer


def test_extractive_generator_is_deterministic():
    gen = ExtractiveGenerator()
    passages = [_passage("docA", 5.0, "text")]
    r1 = gen.generate("q", passages)
    r2 = gen.generate("q", passages)
    assert r1.answer == r2.answer


def test_extractive_generator_rejects_invalid_max_passages():
    with pytest.raises(ValueError):
        ExtractiveGenerator(max_passages=0)


def test_openai_compatible_generator_constructor_makes_no_network_call():
    # Constructing it must never raise/hang/attempt a connection -- only
    # calling .generate() would, and .generate() is intentionally NotImplemented.
    gen = OpenAICompatibleGenerator(
        base_url="https://example.invalid/v1",
        api_key="not-a-real-key",
        model="not-a-real-model",
    )
    assert gen.name == "openai-compatible"


def test_openai_compatible_generator_generate_raises_not_implemented_not_a_network_error():
    gen = OpenAICompatibleGenerator(base_url="https://example.invalid/v1", api_key="x", model="m")
    with pytest.raises(NotImplementedError):
        gen.generate("question", [_passage("docA", 1.0)])
