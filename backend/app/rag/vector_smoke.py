import argparse
import asyncio

from sqlalchemy import select

from backend.app.db.models import DocumentChunk
from backend.app.db.session import get_sessionmaker
from backend.app.rag.embeddings import build_embedding_client
from backend.app.rag.vector_retrieval import VectorRetriever


async def load_chunk_text_for_query(
    *,
    source_uri: str,
    workspace_id: str,
) -> str:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        text = await session.scalar(
            select(DocumentChunk.text)
            .where(DocumentChunk.workspace_id == workspace_id)
            .where(DocumentChunk.source_uri == source_uri)
            .order_by(DocumentChunk.chunk_index)
            .limit(1)
        )

    if text is None:
        raise SystemExit(f"no chunk found for source_uri={source_uri!r}")
    return text


async def run_vector_smoke(
    *,
    query_source_uri: str,
    expected_source_uri: str,
    workspace_id: str,
    top_k: int,
) -> int:
    query_text = await load_chunk_text_for_query(
        source_uri=query_source_uri,
        workspace_id=workspace_id,
    )
    embedding_client = build_embedding_client()
    query_embedding = await embedding_client.embed_query(query_text)

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        retriever = VectorRetriever(session)
        results = await retriever.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            workspace_id=workspace_id,
        )

    for result in results:
        print(
            f"{result.rank}. score={result.score:.6f} "
            f"source={result.source_uri} section={result.section_title}"
        )

    if not results:
        raise SystemExit("vector smoke failed: no results returned")

    if results[0].source_uri != expected_source_uri:
        raise SystemExit(
            "vector smoke failed: "
            f"top source {results[0].source_uri!r} != {expected_source_uri!r}"
        )

    print("vector smoke passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a vector retrieval smoke test.")
    parser.add_argument(
        "--query-source-uri",
        default="llm_systems/flashattention.md",
    )
    parser.add_argument(
        "--expected-source-uri",
        default="llm_systems/flashattention.md",
    )
    parser.add_argument("--workspace-id", default="public")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run_vector_smoke(
        query_source_uri=args.query_source_uri,
        expected_source_uri=args.expected_source_uri,
        workspace_id=args.workspace_id,
        top_k=args.top_k,
    )


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

