import pytest
from pydantic import ValidationError

from backend.app.agent.rag_search import RAGSearchInput, rag_search_tool
from backend.app.rag.citations import Source
from backend.app.rag.pipeline import RagRetrievalContext, RetrievalInfo


class FakeRagPipeline:
    def __init__(self, retrieved: RagRetrievalContext | None = None) -> None:
        self.requests = []
        self.retrieved = retrieved or RagRetrievalContext(
            sources=[
                Source(
                    source_id="1",
                    title="RAG Evaluation Notes",
                    section="Citations",
                    source_uri="docs/rag-eval.md",
                    chunk_id="chunk-1",
                    score=0.91,
                )
            ],
            context="[1] RAG Evaluation Notes\nCheck retrieved chunks.",
            retrieval=RetrievalInfo(
                mode="hybrid_rrf_rerank",
                vector_top_k=3,
                sparse_top_k=3,
                fused_count=1,
                used_count=1,
                top_score=0.91,
            ),
        )

    async def retrieve_context(self, request):
        self.requests.append(request)
        return self.retrieved


@pytest.mark.asyncio
async def test_rag_search_tool_returns_sources_context_and_retrieval_metadata() -> None:
    pipeline = FakeRagPipeline()

    output = await rag_search_tool(
        RAGSearchInput(
            query=" How can I debug citations? ",
            workspace_id=" public ",
            top_k=3,
            metadata_filter={" topic ": " eval "},
        ),
        pipeline=pipeline,  # type: ignore[arg-type]
    )

    assert output.sources[0]["chunk_id"] == "chunk-1"
    assert output.context == "[1] RAG Evaluation Notes\nCheck retrieved chunks."
    assert output.top_score == 0.91
    assert output.refusal_recommended is False
    assert output.retrieval["used_count"] == 1

    assert len(pipeline.requests) == 1
    request = pipeline.requests[0]
    assert request.question == "How can I debug citations?"
    assert request.workspace_id == "public"
    assert request.vector_top_k == 3
    assert request.sparse_top_k == 3
    assert request.fused_top_k == 3
    assert request.rerank_top_n == 3
    assert request.metadata_filter == {"topic": " eval "}


def test_rag_search_input_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        RAGSearchInput(query=" ", top_k=0)
