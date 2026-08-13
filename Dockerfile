FROM python:3.12-slim

LABEL org.opencontainers.image.title="Engineering-Docs-RAG" \
      org.opencontainers.image.description="BM25 retrieval over engineering documents (datasheets, safety procedures, HAZOP excerpts), with a pluggable no-LLM extractive generator."

WORKDIR /app

# engineering_rag has ZERO runtime dependencies (see pyproject.toml) -- the
# entire retrieval pipeline is built on the Python standard library. That
# means the package can simply be copied onto PYTHONPATH and run directly,
# with no `pip install` step at all, which keeps this build fully
# network-free. That matters in general, and specifically on this project's
# development machine, which transparently intercepts TLS inside Docker
# build containers (see README, "Docker -- what was actually tested,
# honestly") -- a `pip install` step here would be exactly the kind of
# network call that fails under that interception.
COPY src ./src
COPY data ./data

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "engineering_rag.cli"]
CMD ["demo"]
