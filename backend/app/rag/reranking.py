from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from backend.app.core.config import Settings, get_settings
from backend.app.rag.retrieval_models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    provider_name: str

    async def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        pass


class NoOpReranker:
    provider_name = "none"

    async def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        return [
            chunk.model_copy(update={"rank": rank})
            for rank, chunk in enumerate(chunks[:top_n], start=1)
        ]


def build_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()

    if settings.reranker_provider == "none":
        return NoOpReranker()

    raise ValueError(f"unsupported reranker provider: {settings.reranker_provider}")

