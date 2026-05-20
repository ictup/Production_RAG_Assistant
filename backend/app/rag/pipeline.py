import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings, get_settings
from backend.app.core.tracing import trace_span
from backend.app.rag.citations import Source, build_sources, validate_citations
from backend.app.rag.costs import estimate_provider_token_cost
from backend.app.rag.embeddings import (
    EmbeddingClient,
    EmbeddingUsage,
    build_embedding_client,
)
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.generation import Generator, build_generator
from backend.app.rag.metadata_filters import normalize_metadata_filter
from backend.app.rag.prompts import build_rag_prompt
from backend.app.rag.query_rewriting import (
    ConversationTurn,
    QueryRewriter,
    QueryRewriteResult,
    build_query_rewriter,
)
from backend.app.rag.refusal import (
    REFUSAL_ANSWER,
    RefusalInfo,
    should_refuse,
    should_refuse_question,
)
from backend.app.rag.reranking import Reranker, build_reranker
from backend.app.rag.sparse_retrieval import SparseRetriever
from backend.app.rag.vector_retrieval import VectorRetriever


class ChatPipelineRequest(BaseModel):
    question: str
    workspace_id: str = "public"
    chat_history: list[ConversationTurn] = Field(default_factory=list)
    metadata_filter: dict[str, object] = Field(default_factory=dict)
    vector_top_k: int | None = Field(default=None, gt=0)
    sparse_top_k: int | None = Field(default=None, gt=0)
    fused_top_k: int | None = Field(default=None, gt=0)
    rerank_top_n: int | None = Field(default=None, gt=0)
    rerank: bool = True

    @field_validator("question", "workspace_id")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("metadata_filter", mode="before")
    @classmethod
    def metadata_filter_must_be_object(cls, value: object) -> dict[str, object]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata_filter must be an object")
        return normalize_metadata_filter(value)


class QueryRewriteInfo(BaseModel):
    provider: str = "none"
    model: str = "none"
    rewritten: bool = False
    retrieval_query: str | None = None
    history_turn_count: int = 0


class RetrievalInfo(BaseModel):
    mode: str
    vector_top_k: int
    sparse_top_k: int
    fused_count: int
    used_count: int
    top_score: float | None
    metadata_filter: dict[str, object] = Field(default_factory=dict)
    query_rewrite: QueryRewriteInfo = Field(default_factory=QueryRewriteInfo)


class UsageInfo(BaseModel):
    model: str
    embedding_model: str
    latency_ms: int
    generator_provider: str = "unknown"
    embedding_provider: str = "unknown"
    embedding_latency_ms: int = 0
    generation_latency_ms: int = 0
    embedding_input_tokens: int = 0
    embedding_total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_cost_usd: float = Field(default=0.0, ge=0)
    embedding_cost_estimated: bool = False
    input_cost_usd: float = Field(default=0.0, ge=0)
    output_cost_usd: float = Field(default=0.0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0)
    cost_estimated: bool = False
    cost_currency: str = "USD"


class ChatPipelineResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval: RetrievalInfo
    usage: UsageInfo
    citation_valid: bool | None
    refusal: RefusalInfo | None = None


@dataclass(frozen=True)
class ChatPipelineStreamEvent:
    event_type: str
    delta: str = ""
    response: ChatPipelineResponse | None = None


class RagPipeline:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings | None = None,
        embedding_client: EmbeddingClient | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_client = embedding_client or build_embedding_client(
            self.settings
        )
        self.query_rewriter = query_rewriter or build_query_rewriter(self.settings)
        self.reranker = reranker or build_reranker(self.settings)
        self.generator = generator or build_generator(self.settings)

    async def answer_question(
        self,
        request: ChatPipelineRequest,
    ) -> ChatPipelineResponse:
        started_at = time.perf_counter()
        vector_top_k = request.vector_top_k or self.settings.vector_top_k
        sparse_top_k = request.sparse_top_k or self.settings.sparse_top_k
        fused_top_k = request.fused_top_k or self.settings.fused_top_k
        rerank_top_n = request.rerank_top_n or self.settings.rerank_top_n
        metadata_filter = request.metadata_filter

        with trace_span(
            "rag.question_guard",
            {"question_length": len(request.question)},
        ):
            question_refusal = should_refuse_question(request.question)
        if question_refusal is not None:
            return build_refusal_response(
                refusal=question_refusal,
                mode="question_guard",
                vector_top_k=vector_top_k,
                sparse_top_k=sparse_top_k,
                fused_count=0,
                top_score=None,
                metadata_filter=metadata_filter,
                model=self.generator.model_name,
                generator_provider=self.generator.provider_name,
                embedding_model=self.embedding_client.model_name,
                embedding_provider=self.embedding_client.provider_name,
                started_at=started_at,
                provider_price_table=self.settings.provider_price_table,
            )

        with trace_span(
            "rag.query_rewrite",
            {
                "provider": self.query_rewriter.provider_name,
                "model": self.query_rewriter.model_name,
                "history_turn_count": len(request.chat_history),
                "metadata_filter_keys": sorted(metadata_filter),
            },
        ):
            query_rewrite = await self.query_rewriter.rewrite(
                question=request.question,
                metadata_filter=metadata_filter,
                chat_history=request.chat_history,
            )
            retrieval_query = query_rewrite.rewritten_query

        with trace_span(
            "rag.embedding",
            {
                "provider": self.embedding_client.provider_name,
                "model": self.embedding_client.model_name,
                "query_rewritten": query_rewrite.rewritten,
            },
        ):
            embedding_started_at = time.perf_counter()
            query_embedding = await self.embedding_client.embed_query(retrieval_query)
            embedding_latency_ms = max(
                0,
                int((time.perf_counter() - embedding_started_at) * 1000),
            )
            embedding_usage = get_embedding_usage(self.embedding_client)
        with trace_span(
            "rag.vector_retrieval",
            {
                "workspace_id": request.workspace_id,
                "top_k": vector_top_k,
                "metadata_filter_keys": sorted(metadata_filter),
            },
        ):
            vector_results = await VectorRetriever(self.session).retrieve(
                query_embedding=query_embedding,
                top_k=vector_top_k,
                workspace_id=request.workspace_id,
                metadata_filter=metadata_filter,
            )
        with trace_span(
            "rag.sparse_retrieval",
            {
                "workspace_id": request.workspace_id,
                "top_k": sparse_top_k,
                "metadata_filter_keys": sorted(metadata_filter),
            },
        ):
            sparse_results = await SparseRetriever(self.session).retrieve(
                query=retrieval_query,
                top_k=sparse_top_k,
                workspace_id=request.workspace_id,
                metadata_filter=metadata_filter,
            )
        with trace_span(
            "rag.fusion",
            {
                "vector_count": len(vector_results),
                "sparse_count": len(sparse_results),
                "top_n": fused_top_k,
            },
        ):
            fused_results = reciprocal_rank_fusion(
                [vector_results, sparse_results],
                k=self.settings.rrf_k,
                top_n=fused_top_k,
            )

        with trace_span(
            "rag.retrieval_guard",
            {
                "fused_count": len(fused_results),
                "threshold": self.settings.refusal_score_threshold,
            },
        ):
            refusal = should_refuse(
                fused_results,
                threshold=self.settings.refusal_score_threshold,
            )
        if refusal is not None:
            return build_refusal_response(
                refusal=refusal,
                mode="hybrid_rrf",
                vector_top_k=vector_top_k,
                sparse_top_k=sparse_top_k,
                fused_count=len(fused_results),
                top_score=refusal.top_score,
                metadata_filter=metadata_filter,
                query_rewrite=query_rewrite,
                model=self.generator.model_name,
                generator_provider=self.generator.provider_name,
                embedding_model=self.embedding_client.model_name,
                embedding_provider=self.embedding_client.provider_name,
                started_at=started_at,
                embedding_latency_ms=embedding_latency_ms,
                embedding_input_tokens=embedding_usage.input_tokens,
                embedding_total_tokens=embedding_usage.total_tokens,
                provider_price_table=self.settings.provider_price_table,
            )

        with trace_span(
            "rag.rerank",
            {
                "enabled": request.rerank,
                "provider": self.reranker.provider_name,
                "input_count": len(fused_results),
                "top_n": rerank_top_n,
            },
        ):
            used_chunks = (
                await self.reranker.rerank(
                    query=retrieval_query,
                    chunks=fused_results,
                    top_n=rerank_top_n,
                )
                if request.rerank
                else fused_results[:rerank_top_n]
            )
        prompt = build_rag_prompt(request.question, used_chunks)
        with trace_span(
            "rag.generation",
            {
                "provider": self.generator.provider_name,
                "model": self.generator.model_name,
                "source_count": len(used_chunks),
            },
        ):
            generation_started_at = time.perf_counter()
            generated = await self.generator.generate(prompt)
            generation_latency_ms = max(
                0,
                int((time.perf_counter() - generation_started_at) * 1000),
            )
        sources = build_sources(used_chunks)
        with trace_span(
            "rag.citation_validation",
            {"source_count": len(sources)},
        ):
            citation_valid = validate_citations(generated.answer, len(sources))

        return ChatPipelineResponse(
            answer=generated.answer,
            sources=sources,
            retrieval=build_retrieval_info(
                mode="hybrid_rrf_rerank" if request.rerank else "hybrid_rrf",
                vector_top_k=vector_top_k,
                sparse_top_k=sparse_top_k,
                fused_count=len(fused_results),
                used_count=len(used_chunks),
                top_score=fused_results[0].score if fused_results else None,
                metadata_filter=metadata_filter,
                query_rewrite=query_rewrite,
            ),
            usage=build_usage_info(
                model=generated.model,
                generator_provider=self.generator.provider_name,
                embedding_model=self.embedding_client.model_name,
                embedding_provider=self.embedding_client.provider_name,
                started_at=started_at,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                embedding_latency_ms=embedding_latency_ms,
                embedding_input_tokens=embedding_usage.input_tokens,
                embedding_total_tokens=embedding_usage.total_tokens,
                generation_latency_ms=generation_latency_ms,
                provider_price_table=self.settings.provider_price_table,
            ),
            citation_valid=citation_valid,
            refusal=None,
        )

    async def stream_answer(
        self,
        request: ChatPipelineRequest,
    ) -> AsyncIterator[ChatPipelineStreamEvent]:
        started_at = time.perf_counter()
        vector_top_k = request.vector_top_k or self.settings.vector_top_k
        sparse_top_k = request.sparse_top_k or self.settings.sparse_top_k
        fused_top_k = request.fused_top_k or self.settings.fused_top_k
        rerank_top_n = request.rerank_top_n or self.settings.rerank_top_n
        metadata_filter = request.metadata_filter

        with trace_span(
            "rag.question_guard",
            {"question_length": len(request.question)},
        ):
            question_refusal = should_refuse_question(request.question)
        if question_refusal is not None:
            response = build_refusal_response(
                refusal=question_refusal,
                mode="question_guard",
                vector_top_k=vector_top_k,
                sparse_top_k=sparse_top_k,
                fused_count=0,
                top_score=None,
                metadata_filter=metadata_filter,
                model=self.generator.model_name,
                generator_provider=self.generator.provider_name,
                embedding_model=self.embedding_client.model_name,
                embedding_provider=self.embedding_client.provider_name,
                started_at=started_at,
                provider_price_table=self.settings.provider_price_table,
            )
            yield ChatPipelineStreamEvent(event_type="delta", delta=response.answer)
            yield ChatPipelineStreamEvent(event_type="completed", response=response)
            return

        with trace_span(
            "rag.query_rewrite",
            {
                "provider": self.query_rewriter.provider_name,
                "model": self.query_rewriter.model_name,
                "history_turn_count": len(request.chat_history),
                "metadata_filter_keys": sorted(metadata_filter),
            },
        ):
            query_rewrite = await self.query_rewriter.rewrite(
                question=request.question,
                metadata_filter=metadata_filter,
                chat_history=request.chat_history,
            )
            retrieval_query = query_rewrite.rewritten_query

        with trace_span(
            "rag.embedding",
            {
                "provider": self.embedding_client.provider_name,
                "model": self.embedding_client.model_name,
                "query_rewritten": query_rewrite.rewritten,
            },
        ):
            embedding_started_at = time.perf_counter()
            query_embedding = await self.embedding_client.embed_query(retrieval_query)
            embedding_latency_ms = max(
                0,
                int((time.perf_counter() - embedding_started_at) * 1000),
            )
            embedding_usage = get_embedding_usage(self.embedding_client)
        with trace_span(
            "rag.vector_retrieval",
            {
                "workspace_id": request.workspace_id,
                "top_k": vector_top_k,
                "metadata_filter_keys": sorted(metadata_filter),
            },
        ):
            vector_results = await VectorRetriever(self.session).retrieve(
                query_embedding=query_embedding,
                top_k=vector_top_k,
                workspace_id=request.workspace_id,
                metadata_filter=metadata_filter,
            )
        with trace_span(
            "rag.sparse_retrieval",
            {
                "workspace_id": request.workspace_id,
                "top_k": sparse_top_k,
                "metadata_filter_keys": sorted(metadata_filter),
            },
        ):
            sparse_results = await SparseRetriever(self.session).retrieve(
                query=retrieval_query,
                top_k=sparse_top_k,
                workspace_id=request.workspace_id,
                metadata_filter=metadata_filter,
            )
        with trace_span(
            "rag.fusion",
            {
                "vector_count": len(vector_results),
                "sparse_count": len(sparse_results),
                "top_n": fused_top_k,
            },
        ):
            fused_results = reciprocal_rank_fusion(
                [vector_results, sparse_results],
                k=self.settings.rrf_k,
                top_n=fused_top_k,
            )

        with trace_span(
            "rag.retrieval_guard",
            {
                "fused_count": len(fused_results),
                "threshold": self.settings.refusal_score_threshold,
            },
        ):
            refusal = should_refuse(
                fused_results,
                threshold=self.settings.refusal_score_threshold,
            )
        if refusal is not None:
            response = build_refusal_response(
                refusal=refusal,
                mode="hybrid_rrf",
                vector_top_k=vector_top_k,
                sparse_top_k=sparse_top_k,
                fused_count=len(fused_results),
                top_score=refusal.top_score,
                metadata_filter=metadata_filter,
                query_rewrite=query_rewrite,
                model=self.generator.model_name,
                generator_provider=self.generator.provider_name,
                embedding_model=self.embedding_client.model_name,
                embedding_provider=self.embedding_client.provider_name,
                started_at=started_at,
                embedding_latency_ms=embedding_latency_ms,
                embedding_input_tokens=embedding_usage.input_tokens,
                embedding_total_tokens=embedding_usage.total_tokens,
                provider_price_table=self.settings.provider_price_table,
            )
            yield ChatPipelineStreamEvent(event_type="delta", delta=response.answer)
            yield ChatPipelineStreamEvent(event_type="completed", response=response)
            return

        with trace_span(
            "rag.rerank",
            {
                "enabled": request.rerank,
                "provider": self.reranker.provider_name,
                "input_count": len(fused_results),
                "top_n": rerank_top_n,
            },
        ):
            used_chunks = (
                await self.reranker.rerank(
                    query=retrieval_query,
                    chunks=fused_results,
                    top_n=rerank_top_n,
                )
                if request.rerank
                else fused_results[:rerank_top_n]
            )
        prompt = build_rag_prompt(request.question, used_chunks)
        answer_parts: list[str] = []
        generated_model = self.generator.model_name
        input_tokens = 0
        output_tokens = 0

        with trace_span(
            "rag.generation_stream",
            {
                "provider": self.generator.provider_name,
                "model": self.generator.model_name,
                "source_count": len(used_chunks),
            },
        ):
            generation_started_at = time.perf_counter()
            async for generated_event in self.generator.stream(prompt):
                if generated_event.event_type == "delta" and generated_event.delta:
                    answer_parts.append(generated_event.delta)
                    yield ChatPipelineStreamEvent(
                        event_type="delta",
                        delta=generated_event.delta,
                    )
                    continue

                if generated_event.event_type == "completed":
                    if generated_event.answer is not None:
                        answer_parts = [generated_event.answer]
                    if generated_event.model is not None:
                        generated_model = generated_event.model
                    input_tokens = generated_event.input_tokens or 0
                    output_tokens = generated_event.output_tokens or 0

        generation_latency_ms = max(
            0,
            int((time.perf_counter() - generation_started_at) * 1000),
        )
        answer = "".join(answer_parts).strip()
        sources = build_sources(used_chunks)
        with trace_span(
            "rag.citation_validation",
            {"source_count": len(sources)},
        ):
            citation_valid = validate_citations(answer, len(sources))
        response = ChatPipelineResponse(
            answer=answer,
            sources=sources,
            retrieval=build_retrieval_info(
                mode="hybrid_rrf_rerank" if request.rerank else "hybrid_rrf",
                vector_top_k=vector_top_k,
                sparse_top_k=sparse_top_k,
                fused_count=len(fused_results),
                used_count=len(used_chunks),
                top_score=fused_results[0].score if fused_results else None,
                metadata_filter=metadata_filter,
                query_rewrite=query_rewrite,
            ),
            usage=build_usage_info(
                model=generated_model,
                generator_provider=self.generator.provider_name,
                embedding_model=self.embedding_client.model_name,
                embedding_provider=self.embedding_client.provider_name,
                started_at=started_at,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                embedding_latency_ms=embedding_latency_ms,
                embedding_input_tokens=embedding_usage.input_tokens,
                embedding_total_tokens=embedding_usage.total_tokens,
                generation_latency_ms=generation_latency_ms,
                provider_price_table=self.settings.provider_price_table,
            ),
            citation_valid=citation_valid,
            refusal=None,
        )
        yield ChatPipelineStreamEvent(event_type="completed", response=response)


def build_retrieval_info(
    *,
    mode: str,
    vector_top_k: int,
    sparse_top_k: int,
    fused_count: int,
    used_count: int,
    top_score: float | None,
    metadata_filter: dict[str, object] | None = None,
    query_rewrite: QueryRewriteResult | None = None,
) -> RetrievalInfo:
    return RetrievalInfo(
        mode=mode,
        vector_top_k=vector_top_k,
        sparse_top_k=sparse_top_k,
        fused_count=fused_count,
        used_count=used_count,
        top_score=top_score,
        metadata_filter=dict(metadata_filter or {}),
        query_rewrite=build_query_rewrite_info(query_rewrite),
    )


def build_query_rewrite_info(
    query_rewrite: QueryRewriteResult | None,
) -> QueryRewriteInfo:
    if query_rewrite is None:
        return QueryRewriteInfo()
    return QueryRewriteInfo(
        provider=query_rewrite.provider_name,
        model=query_rewrite.model_name,
        rewritten=query_rewrite.rewritten,
        retrieval_query=query_rewrite.rewritten_query,
        history_turn_count=query_rewrite.history_turn_count,
    )


def build_refusal_response(
    *,
    refusal: RefusalInfo,
    mode: str,
    vector_top_k: int,
    sparse_top_k: int,
    fused_count: int,
    top_score: float | None,
    metadata_filter: dict[str, object] | None = None,
    query_rewrite: QueryRewriteResult | None = None,
    model: str,
    embedding_model: str,
    generator_provider: str = "unknown",
    embedding_provider: str = "unknown",
    started_at: float,
    embedding_latency_ms: int = 0,
    embedding_input_tokens: int = 0,
    embedding_total_tokens: int = 0,
    provider_price_table: str = "",
) -> ChatPipelineResponse:
    return ChatPipelineResponse(
        answer=REFUSAL_ANSWER,
        sources=[],
        retrieval=build_retrieval_info(
            mode=mode,
            vector_top_k=vector_top_k,
            sparse_top_k=sparse_top_k,
            fused_count=fused_count,
            used_count=0,
            top_score=top_score,
            metadata_filter=metadata_filter,
            query_rewrite=query_rewrite,
        ),
        usage=build_usage_info(
            model=model,
            generator_provider=generator_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            started_at=started_at,
            embedding_latency_ms=embedding_latency_ms,
            embedding_input_tokens=embedding_input_tokens,
            embedding_total_tokens=embedding_total_tokens,
            provider_price_table=provider_price_table,
        ),
        citation_valid=None,
        refusal=refusal,
    )


def build_usage_info(
    *,
    model: str,
    embedding_model: str,
    started_at: float,
    generator_provider: str = "unknown",
    embedding_provider: str = "unknown",
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_input_tokens: int = 0,
    embedding_total_tokens: int = 0,
    embedding_latency_ms: int = 0,
    generation_latency_ms: int = 0,
    provider_price_table: str = "",
) -> UsageInfo:
    cost_estimate = estimate_provider_token_cost(
        provider=generator_provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_price_table=provider_price_table,
    )
    embedding_cost_estimate = estimate_provider_token_cost(
        provider=embedding_provider,
        model=embedding_model,
        input_tokens=embedding_input_tokens,
        output_tokens=0,
        raw_price_table=provider_price_table,
    )
    return UsageInfo(
        model=model,
        embedding_model=embedding_model,
        generator_provider=generator_provider,
        embedding_provider=embedding_provider,
        latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        embedding_latency_ms=embedding_latency_ms,
        generation_latency_ms=generation_latency_ms,
        embedding_input_tokens=embedding_input_tokens,
        embedding_total_tokens=embedding_total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_cost_usd=embedding_cost_estimate.total_cost_usd,
        embedding_cost_estimated=embedding_cost_estimate.estimated,
        input_cost_usd=cost_estimate.input_cost_usd,
        output_cost_usd=cost_estimate.output_cost_usd,
        total_cost_usd=(
            cost_estimate.total_cost_usd + embedding_cost_estimate.total_cost_usd
        ),
        cost_estimated=cost_estimate.estimated or embedding_cost_estimate.estimated,
    )


def get_embedding_usage(embedding_client: EmbeddingClient) -> EmbeddingUsage:
    return getattr(embedding_client, "last_usage", EmbeddingUsage())
