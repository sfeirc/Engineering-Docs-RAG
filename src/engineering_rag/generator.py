"""Pluggable answer-generation backends.

Retrieval (BM25 over chunked documents) answers "which passages are
relevant"; *generation* is the separate concern of turning those passages
into a natural-language answer. This project deliberately keeps the two
decoupled behind the `Generator` protocol so any generation backend --
extractive (this project's default, no LLM involved at all), an
OpenAI-compatible HTTP API, or a self-hosted inference server such as this
author's own `sfeirc/LLM-inference-server` -- can be swapped in without
touching the retrieval pipeline. See the README section "Why
extractive-by-default" for the full reasoning.

No code in this module makes a network call or requires an API key. That is
intentional: see the README for why no paid LLM call is made anywhere in
this project's tests, CI, or default demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedPassage:
    """One retrieved chunk, as handed from the retrieval pipeline to a generator."""

    doc_id: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    backend: str
    passages_used: list[RetrievedPassage]


@runtime_checkable
class Generator(Protocol):
    """Interface every generation backend must satisfy.

    Implement this to plug in an LLM: read `ExtractiveGenerator.generate`'s
    docstring for the exact contract (question in, ranked passages in,
    `GeneratedAnswer` out), then pass an instance of your class to
    `RAGPipeline(generator=...)` -- no other code changes are needed.
    """

    name: str

    def generate(self, question: str, passages: list[RetrievedPassage]) -> GeneratedAnswer: ...


class ExtractiveGenerator:
    """Default, LLM-free generator: surfaces the retrieved passages themselves.

    This is deliberately not "smart": it does no paraphrasing, summarizing,
    or synthesis. It takes the passages in the order retrieval already
    ranked them (by BM25 score) and returns them verbatim, each tagged with
    its source document id and score, so a human or a downstream system can
    verify the "answer" against its source directly rather than trusting an
    opaque generation step.

    Fully deterministic; requires no network access, API key, or GPU; safe
    to call in a unit test or a CI pipeline an unlimited number of times.
    """

    name = "extractive"

    def __init__(self, *, max_passages: int = 3) -> None:
        if max_passages < 1:
            raise ValueError("max_passages must be at least 1")
        self.max_passages = max_passages

    def generate(self, question: str, passages: list[RetrievedPassage]) -> GeneratedAnswer:
        used = passages[: self.max_passages]
        if not used:
            return GeneratedAnswer(
                answer="No relevant passage was found in the indexed corpus for this question.",
                backend=self.name,
                passages_used=[],
            )
        lines = [f'Top {len(used)} passage(s) retrieved for: "{question}"']
        for i, p in enumerate(used, start=1):
            lines.append(f"[{i}] ({p.doc_id}, score={p.score:.3f}): {p.text.strip()}")
        return GeneratedAnswer(answer="\n".join(lines), backend=self.name, passages_used=used)


class OpenAICompatibleGenerator:
    """Reference sketch for plugging in a real LLM.

    NOT called anywhere in this project's tests, CI, or default demo -- see
    the README for why. Any endpoint implementing the OpenAI
    chat-completions request/response shape works here, including a
    self-hosted server such as `sfeirc/LLM-inference-server`; this class
    just documents the integration point and constructor shape.

    Constructing this class makes no network call; only `.generate()` would,
    and `.generate()` is intentionally left unimplemented (raises
    `NotImplementedError`) rather than wired up to an HTTP client, so that
    importing or instantiating this class can never accidentally trigger a
    paid API call from a test, a CI job, or the CLI demo.
    """

    name = "openai-compatible"

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, question: str, passages: list[RetrievedPassage]) -> GeneratedAnswer:
        # Left as a documented integration point rather than a working call:
        # importing an HTTP client and making a network call from here is
        # exactly the "no paid LLM calls in this project" line this codebase
        # does not cross. See README, "Plugging in a real LLM", for the
        # exact prompt shape (system instruction + numbered passages +
        # question) this method is meant to send once implemented.
        raise NotImplementedError(
            "OpenAICompatibleGenerator is a documented integration point, not a wired-up "
            "backend. Implement generate() against your own OpenAI-compatible endpoint "
            "(e.g. sfeirc/LLM-inference-server) to use it -- see the README."
        )
