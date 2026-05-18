from pathlib import Path

import pytest

from backend.app.rag.embeddings import FakeEmbeddingClient
from ingestion.ingest import (
    IngestionStats,
    format_stats,
    ingest_markdown_files,
    resolve_source_root,
)


def write_markdown(path: Path, title: str, body: str) -> None:
    path.write_text(
        f"""---
title: "{title}"
topic: "systems"
---

# {title}

{body}
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_ingest_markdown_files_dry_run_loads_chunks_and_embeddings(
    tmp_path: Path,
) -> None:
    write_markdown(
        tmp_path / "flashattention.md",
        "FlashAttention",
        "FlashAttention reduces HBM traffic.",
    )
    write_markdown(
        tmp_path / "pagedattention.md",
        "PagedAttention",
        "PagedAttention stores KV cache in pages.",
    )

    stats = await ingest_markdown_files(
        tmp_path,
        dry_run=True,
        embedding_client=FakeEmbeddingClient(dimension=8),
        chunk_size_tokens=40,
        chunk_overlap_tokens=5,
    )

    assert stats.dry_run is True
    assert stats.documents_discovered == 2
    assert stats.documents_loaded == 2
    assert stats.documents_inserted == 0
    assert stats.documents_skipped == 0
    assert stats.chunks_created == 2
    assert stats.chunks_embedded == 2
    assert stats.chunks_inserted == 0


@pytest.mark.asyncio
async def test_ingest_markdown_files_dry_run_accepts_single_file(
    tmp_path: Path,
) -> None:
    document = tmp_path / "single.md"
    write_markdown(document, "Single", "One technical note.")

    stats = await ingest_markdown_files(
        document,
        dry_run=True,
        embedding_client=FakeEmbeddingClient(dimension=8),
        chunk_size_tokens=40,
        chunk_overlap_tokens=5,
    )

    assert stats.documents_discovered == 1
    assert stats.documents_loaded == 1
    assert stats.chunks_created == 1


def test_resolve_source_root_uses_directory_or_file_parent(tmp_path: Path) -> None:
    document = tmp_path / "doc.md"
    document.write_text("# Doc", encoding="utf-8")

    assert resolve_source_root(tmp_path) == tmp_path
    assert resolve_source_root(document) == tmp_path


def test_format_stats_includes_mode_and_counts() -> None:
    output = format_stats(
        IngestionStats(
            documents_discovered=2,
            documents_loaded=2,
            chunks_created=3,
            chunks_embedded=3,
            dry_run=True,
            elapsed_seconds=1.234,
        )
    )

    assert "mode: dry-run" in output
    assert "documents discovered: 2" in output
    assert "chunks embedded: 3" in output
    assert "elapsed: 1.23s" in output
