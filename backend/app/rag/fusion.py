from collections.abc import Sequence

from backend.app.rag.retrieval_models import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    top_n: int = 20,
) -> list[RetrievedChunk]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            chunks_by_id[chunk.chunk_id] = chunk
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (k + rank)
            )

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            fused_scores[chunk_id],
            -chunks_by_id[chunk_id].rank,
            chunks_by_id[chunk_id].chunk_id,
        ),
        reverse=True,
    )[:top_n]

    fused_chunks: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        source_chunk = chunks_by_id[chunk_id]
        fused_chunks.append(
            source_chunk.model_copy(
                update={
                    "score": fused_scores[chunk_id],
                    "rank": rank,
                    "retrieval_mode": "hybrid_rrf",
                }
            )
        )

    return fused_chunks

