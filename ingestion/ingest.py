import argparse
import asyncio
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.repositories import DocumentRepository
from backend.app.db.session import get_sessionmaker
from backend.app.rag.embedding_pipeline import embed_chunks
from backend.app.rag.embeddings import EmbeddingClient, build_embedding_client
from ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_document,
)
from ingestion.hashing import compute_content_hash
from ingestion.parse_markdown import discover_markdown_files, load_markdown_document

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@dataclass
class IngestionStats:
    documents_discovered: int = 0
    documents_loaded: int = 0
    documents_inserted: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_inserted: int = 0
    dry_run: bool = False
    elapsed_seconds: float = 0.0


def resolve_source_root(input_path: Path) -> Path:
    return input_path if input_path.is_dir() else input_path.parent


async def prepare_document_chunks(
    path: Path,
    *,
    source_root: Path,
    workspace_id: str,
    embedding_client: EmbeddingClient,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[str, int, int]:
    raw_document = load_markdown_document(
        path,
        source_root=source_root,
        default_workspace_id=workspace_id,
    )
    chunks = chunk_document(
        raw_document,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    embeddings = await embed_chunks(chunks, embedding_client)
    return compute_content_hash(raw_document.text), len(chunks), len(embeddings)


async def ingest_markdown_files(
    input_path: Path | str,
    *,
    workspace_id: str = "public",
    source_root: Path | str | None = None,
    dry_run: bool = False,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    embedding_client: EmbeddingClient | None = None,
    session_factory: SessionFactory | None = None,
) -> IngestionStats:
    started_at = time.perf_counter()
    input_path = Path(input_path)
    resolved_source_root = Path(source_root) if source_root is not None else (
        resolve_source_root(input_path)
    )
    client = embedding_client or build_embedding_client()

    files = discover_markdown_files(input_path)
    stats = IngestionStats(documents_discovered=len(files), dry_run=dry_run)

    if dry_run:
        for file_path in files:
            _, chunks_count, embeddings_count = await prepare_document_chunks(
                file_path,
                source_root=resolved_source_root,
                workspace_id=workspace_id,
                embedding_client=client,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            stats.documents_loaded += 1
            stats.chunks_created += chunks_count
            stats.chunks_embedded += embeddings_count

        stats.elapsed_seconds = time.perf_counter() - started_at
        return stats

    factory = session_factory or get_sessionmaker()
    async with factory() as session:
        async with session.begin():
            repository = DocumentRepository(session)

            for file_path in files:
                raw_document = load_markdown_document(
                    file_path,
                    source_root=resolved_source_root,
                    default_workspace_id=workspace_id,
                )
                content_hash = compute_content_hash(raw_document.text)
                stats.documents_loaded += 1

                existing_document_id = await repository.get_document_id_by_hash(
                    content_hash
                )
                if existing_document_id is not None:
                    stats.documents_skipped += 1
                    continue

                chunks = chunk_document(
                    raw_document,
                    chunk_size_tokens=chunk_size_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                )
                embeddings = await embed_chunks(chunks, client)
                result = await repository.ingest_document(
                    raw_document,
                    chunks,
                    content_hash=content_hash,
                    chunk_embeddings=embeddings,
                )

                if result.inserted:
                    stats.documents_inserted += 1
                    stats.chunks_inserted += result.chunks_inserted
                else:
                    stats.documents_skipped += 1
                stats.chunks_created += len(chunks)
                stats.chunks_embedded += len(embeddings)

    stats.elapsed_seconds = time.perf_counter() - started_at
    return stats


def format_stats(stats: IngestionStats) -> str:
    mode = "dry-run" if stats.dry_run else "write"
    return "\n".join(
        [
            f"mode: {mode}",
            f"documents discovered: {stats.documents_discovered}",
            f"documents loaded: {stats.documents_loaded}",
            f"documents inserted: {stats.documents_inserted}",
            f"documents skipped: {stats.documents_skipped}",
            f"chunks created: {stats.chunks_created}",
            f"chunks embedded: {stats.chunks_embedded}",
            f"chunks inserted: {stats.chunks_inserted}",
            f"elapsed: {stats.elapsed_seconds:.2f}s",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Markdown documents.")
    parser.add_argument("--input", required=True, help="Markdown file or directory.")
    parser.add_argument("--workspace-id", default="public")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=DEFAULT_CHUNK_SIZE_TOKENS,
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
    )
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = await ingest_markdown_files(
        args.input,
        workspace_id=args.workspace_id,
        source_root=args.source_root,
        dry_run=args.dry_run,
        chunk_size_tokens=args.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
    )
    print(format_stats(stats))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
