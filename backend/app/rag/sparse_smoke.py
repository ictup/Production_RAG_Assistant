import argparse
import asyncio

from backend.app.db.session import get_sessionmaker
from backend.app.rag.sparse_retrieval import SparseRetriever


async def run_sparse_smoke(
    *,
    query: str,
    expected_source_uri: str,
    workspace_id: str,
    top_k: int,
) -> int:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        retriever = SparseRetriever(session)
        results = await retriever.retrieve(
            query=query,
            top_k=top_k,
            workspace_id=workspace_id,
        )

    for result in results:
        print(
            f"{result.rank}. score={result.score:.6f} "
            f"source={result.source_uri} section={result.section_title}"
        )

    if not results:
        raise SystemExit("sparse smoke failed: no results returned")

    if results[0].source_uri != expected_source_uri:
        raise SystemExit(
            "sparse smoke failed: "
            f"top source {results[0].source_uri!r} != {expected_source_uri!r}"
        )

    print("sparse smoke passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sparse retrieval smoke test.")
    parser.add_argument("--query", default="KV cache")
    parser.add_argument(
        "--expected-source-uri",
        default="llm_systems/pagedattention.md",
    )
    parser.add_argument("--workspace-id", default="public")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run_sparse_smoke(
        query=args.query,
        expected_source_uri=args.expected_source_uri,
        workspace_id=args.workspace_id,
        top_k=args.top_k,
    )


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

