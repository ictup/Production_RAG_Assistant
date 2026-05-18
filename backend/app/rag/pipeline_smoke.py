import argparse
import asyncio

from backend.app.db.session import get_sessionmaker
from backend.app.rag.pipeline import ChatPipelineRequest, RagPipeline


async def run_pipeline_smoke(
    *,
    question: str,
    workspace_id: str,
) -> int:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        response = await RagPipeline(session=session).answer_question(
            ChatPipelineRequest(
                question=question,
                workspace_id=workspace_id,
                vector_top_k=5,
                sparse_top_k=5,
                fused_top_k=5,
                rerank_top_n=2,
            )
        )

    print(f"answer: {response.answer}")
    print(f"citation_valid: {response.citation_valid}")
    print(f"refusal: {response.refusal}")
    print(f"retrieval: {response.retrieval.model_dump()}")
    print("sources:")
    for source in response.sources:
        print(f"- [{source.source_id}] {source.source_uri} {source.section}")

    if response.refusal is not None:
        raise SystemExit("pipeline smoke failed: unexpected refusal")
    if not response.citation_valid:
        raise SystemExit("pipeline smoke failed: invalid citations")
    if not response.sources:
        raise SystemExit("pipeline smoke failed: no sources")

    print("pipeline smoke passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full RAG pipeline smoke test.")
    parser.add_argument(
        "--question",
        default="What problem does FlashAttention solve?",
    )
    parser.add_argument("--workspace-id", default="public")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run_pipeline_smoke(
        question=args.question,
        workspace_id=args.workspace_id,
    )


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

