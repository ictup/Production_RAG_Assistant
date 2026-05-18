import argparse
import asyncio

from backend.app.db.session import get_sessionmaker
from backend.app.rag.embeddings import build_embedding_client
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.sparse_retrieval import SparseRetriever
from backend.app.rag.vector_retrieval import VectorRetriever
from backend.app.rag.vector_smoke import load_chunk_text_for_query


async def run_hybrid_smoke(
    *,
    query: str,
    query_source_uri: str,
    expected_source_uri: str,
    workspace_id: str,
    vector_top_k: int,
    sparse_top_k: int,
    fused_top_n: int,
    rrf_k: int,
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
            top_k=vector_top_k,
            workspace_id=workspace_id,
        )
        sparse_results = await SparseRetriever(session).retrieve(
            query=query,
            top_k=sparse_top_k,
            workspace_id=workspace_id,
        )

    fused_results = reciprocal_rank_fusion(
        [vector_results, sparse_results],
        k=rrf_k,
        top_n=fused_top_n,
    )

    print("vector results:")
    for result in vector_results:
        print(f"- {result.rank}. {result.source_uri} {result.section_title}")
    print("sparse results:")
    for result in sparse_results:
        print(f"- {result.rank}. {result.source_uri} {result.section_title}")
    print("fused results:")
    for result in fused_results:
        print(
            f"- {result.rank}. score={result.score:.6f} "
            f"{result.source_uri} {result.section_title}"
        )

    if not fused_results:
        raise SystemExit("hybrid smoke failed: no fused results returned")

    if fused_results[0].source_uri != expected_source_uri:
        raise SystemExit(
            "hybrid smoke failed: "
            f"top source {fused_results[0].source_uri!r} != {expected_source_uri!r}"
        )

    print("hybrid smoke passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a hybrid RRF smoke test.")
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
    parser.add_argument("--vector-top-k", type=int, default=3)
    parser.add_argument("--sparse-top-k", type=int, default=3)
    parser.add_argument("--fused-top-n", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run_hybrid_smoke(
        query=args.query,
        query_source_uri=args.query_source_uri,
        expected_source_uri=args.expected_source_uri,
        workspace_id=args.workspace_id,
        vector_top_k=args.vector_top_k,
        sparse_top_k=args.sparse_top_k,
        fused_top_n=args.fused_top_n,
        rrf_k=args.rrf_k,
    )


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

