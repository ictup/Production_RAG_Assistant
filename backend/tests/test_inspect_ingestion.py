import pytest

from ingestion.inspect_ingestion import (
    IngestionSnapshot,
    format_snapshot,
    validate_snapshot,
)


def test_format_snapshot_includes_counts_and_sources() -> None:
    output = format_snapshot(
        IngestionSnapshot(
            documents_count=2,
            chunks_count=4,
            sample_sources=[
                "data/raw/llm_systems/flashattention.md",
                "data/raw/llm_systems/pagedattention.md",
            ],
        )
    )

    assert "documents: 2" in output
    assert "chunks: 4" in output
    assert "- data/raw/llm_systems/flashattention.md" in output


def test_validate_snapshot_passes_when_counts_meet_thresholds() -> None:
    validate_snapshot(
        IngestionSnapshot(documents_count=2, chunks_count=4, sample_sources=[]),
        min_documents=2,
        min_chunks=4,
    )


def test_validate_snapshot_fails_when_counts_are_too_low() -> None:
    with pytest.raises(SystemExit, match="ingestion check failed"):
        validate_snapshot(
            IngestionSnapshot(documents_count=0, chunks_count=0, sample_sources=[]),
            min_documents=1,
            min_chunks=1,
        )

