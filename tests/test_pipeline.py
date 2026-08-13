from engineering_rag.pipeline import RAGPipeline

DOCS = [
    ("pump-doc", "The centrifugal pump seal requires periodic inspection and replacement. " * 20),
    ("tank-doc", "The storage tank roof seal design uses a rim mounted wiper seal. " * 20),
    ("unrelated-doc", "Gas turbine driven compressor exports product downstream to the terminal. " * 20),
]


def test_ingest_returns_chunk_count():
    pipeline = RAGPipeline(chunk_size=30, chunk_overlap=5)
    n_chunks = pipeline.ingest(DOCS)
    assert n_chunks > len(DOCS)  # each repeated doc should split into multiple chunks
    assert len(pipeline) == n_chunks


def test_retrieve_returns_ranked_passages_with_correct_doc_id():
    pipeline = RAGPipeline(chunk_size=200, chunk_overlap=20)
    pipeline.ingest(DOCS)
    passages = pipeline.retrieve("centrifugal pump seal inspection", top_k=3)
    assert passages[0].doc_id == "pump-doc"
    assert passages[0].score >= passages[-1].score


def test_retrieve_doc_ids_deduplicates_across_chunks_of_same_document():
    # small chunk size -> pump-doc produces several chunks; retrieve_doc_ids
    # must still return each *document* only once.
    pipeline = RAGPipeline(chunk_size=15, chunk_overlap=3)
    pipeline.ingest(DOCS)
    doc_ids = pipeline.retrieve_doc_ids("centrifugal pump seal inspection", top_k=3)
    assert doc_ids[0] == "pump-doc"
    assert len(doc_ids) == len(set(doc_ids))  # no duplicates
    assert len(doc_ids) <= len(DOCS)


def test_ask_uses_the_configured_generator_and_returns_relevant_passages():
    pipeline = RAGPipeline(chunk_size=200, chunk_overlap=20)
    pipeline.ingest(DOCS)
    answer = pipeline.ask("centrifugal pump seal inspection", top_k=2)
    assert answer.backend == "extractive"
    assert answer.passages_used[0].doc_id == "pump-doc"


def test_re_ingesting_replaces_previous_index():
    pipeline = RAGPipeline(chunk_size=200, chunk_overlap=20)
    pipeline.ingest(DOCS)
    first_len = len(pipeline)
    pipeline.ingest(DOCS[:1])
    assert len(pipeline) < first_len


def test_empty_query_returns_something_without_crashing():
    pipeline = RAGPipeline(chunk_size=200, chunk_overlap=20)
    pipeline.ingest(DOCS)
    # a query with no recognizable tokens (all stopwords) should not crash;
    # every document scores 0 and ranking falls back to insertion order.
    result = pipeline.retrieve("the a of", top_k=3)
    assert len(result) == 3
