import argparse
import asyncio
from dataclasses import dataclass

from sqlalchemy import func, select

from backend.app.db.models import Document, DocumentChunk
from backend.app.db.session import get_sessionmaker


@dataclass(frozen=True)
class IngestionSnapshot:
    documents_count: int
    chunks_count: int
    sample_sources: list[str]


async def load_ingestion_snapshot(limit: int = 5) -> IngestionSnapshot:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        documents_count = await session.scalar(select(func.count(Document.id)))
        chunks_count = await session.scalar(select(func.count(DocumentChunk.id)))
        sample_sources = (
            await session.scalars(
                select(Document.source_uri).order_by(Document.source_uri).limit(limit)
            )
        ).all()

    return IngestionSnapshot(
        documents_count=documents_count or 0,
        chunks_count=chunks_count or 0,
        sample_sources=list(sample_sources),
    )


def format_snapshot(snapshot: IngestionSnapshot) -> str:
    lines = [
        f"documents: {snapshot.documents_count}",
        f"chunks: {snapshot.chunks_count}",
        "sample sources:",
    ]
    if snapshot.sample_sources:
        lines.extend(f"- {source}" for source in snapshot.sample_sources)
    else:
        lines.append("- none")
    return "\n".join(lines)


def validate_snapshot(
    snapshot: IngestionSnapshot,
    *,
    min_documents: int,
    min_chunks: int,
) -> None:
    failures = []
    if snapshot.documents_count < min_documents:
        failures.append(
            f"documents {snapshot.documents_count} < required {min_documents}"
        )
    if snapshot.chunks_count < min_chunks:
        failures.append(f"chunks {snapshot.chunks_count} < required {min_chunks}")

    if failures:
        raise SystemExit("ingestion check failed: " + "; ".join(failures))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect ingested document counts.")
    parser.add_argument("--min-documents", type=int, default=0)
    parser.add_argument("--min-chunks", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = await load_ingestion_snapshot(limit=args.limit)
    print(format_snapshot(snapshot))
    validate_snapshot(
        snapshot,
        min_documents=args.min_documents,
        min_chunks=args.min_chunks,
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

