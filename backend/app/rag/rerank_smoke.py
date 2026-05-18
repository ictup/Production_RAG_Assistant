import argparse
import asyncio

from backend.app.db.session import get_sessionmaker
from backend.app.rag.embeddings import build_embedding_client
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.reranking import build_reranker
from backend.app.rag.sparse_retrieval import SparseRetriever
from backend.app.rag.vector_retrieval import VectorRetriever
from backend.app.rag.vector_smoke import load_chunk_text_for_query


async def run_rerank_smoke(
    *,
    query: str,
    query_source_uri: str,
    expected_source_uri: str,
    workspace_id: str,
    top_n: int,
) -> int:
    query_text = await load_chunk_text_for_query(
        source_uri=query_source_uri,
        workspace_id=workspace_id,
    )
    embedding_client = build_embedding_client()
    query_embedding = await embedding_client.embed_query(query_text)

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        vector_results = await VectorRetriever(session).retrieve(
            query_embedding=query_embedding,
            top_k=10,
            workspace_id=workspace_id,
        )
        sparse_results = await SparseRetriever(session).retrieve(
            query=query,
            top_k=10,
            workspace_id=workspace_id,
        )

    fused_results = reciprocal_rank_fusion(
        [vector_results, sparse_results],
        top_n=20,
    )
    reranked_results = await build_reranker().rerank(
        query=query,
        chunks=fused_results,
        top_n=top_n,
    )

    for result in reranked_results:
        print(
            f"{result.rank}. score={result.score:.6f} "
            f"source={result.source_uri} section={result.section_title}"
        )

    if not reranked_results:
        raise SystemExit("rerank smoke failed: no results returned")

    if reranked_results[0].source_uri != expected_source_uri:
        raise SystemExit(
            "rerank smoke failed: "
            f"top source {reranked_results[0].source_uri!r} != {expected_source_uri!r}"
        )

    print("rerank smoke passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reranker smoke test.")
    parser.add_argument("--query", default="FlashAttention memory traffic")
    parser.add_argument(
        "--query-source-uri",
        default="llm_systems/flashattention.md",
    )
    parser.add_argument(
        "--expected-source-uri",
        default="llm_systems/flashattention.md",
    )
    parser.add_argument("--workspace-id", default="public")
    parser.add_argument("--top-n", type=int, default=2)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run_rerank_smoke(
        query=args.query,
        query_source_uri=args.query_source_uri,
        expected_source_uri=args.expected_source_uri,
        workspace_id=args.workspace_id,
        top_n=args.top_n,
    )


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

