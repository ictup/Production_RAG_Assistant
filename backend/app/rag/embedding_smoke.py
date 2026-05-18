import argparse
import asyncio
from collections.abc import Sequence

from backend.app.rag.embeddings import (
    EmbeddingClient,
    build_embedding_client,
    validate_embedding_batch,
)

DEFAULT_SMOKE_TEXTS = (
    "FlashAttention reduces memory traffic for attention computation.",
    "PagedAttention manages KV cache memory with paging.",
)


async def run_embedding_smoke(
    *,
    texts: Sequence[str],
    expected_dimension: int | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> int:
    client = embedding_client or build_embedding_client()
    embeddings = await client.embed_texts(texts)
    dimension = expected_dimension or client.dimension
    validate_embedding_batch(embeddings, expected_dimension=dimension)

    print(f"model: {client.model_name}")
    print(f"dimension: {dimension}")
    print(f"inputs: {len(texts)}")
    print(f"embeddings: {len(embeddings)}")
    print("embedding smoke passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an embedding provider smoke test."
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text to embed. Can be provided multiple times.",
    )
    parser.add_argument(
        "--expected-dimension",
        type=int,
        default=None,
        help="Expected embedding dimension. Defaults to configured client dimension.",
    )
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    texts = args.text or list(DEFAULT_SMOKE_TEXTS)
    return await run_embedding_smoke(
        texts=texts,
        expected_dimension=args.expected_dimension,
    )


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
