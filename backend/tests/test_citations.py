import uuid

from backend.app.rag.citations import (
    build_sources,
    extract_citations,
    validate_citations,
)
from backend.app.rag.retrieval_models import RetrievedChunk


def make_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        text=f"text {index}",
        title=f"Title {index}",
        section_title=f"Section {index}",
        source_uri=f"source-{index}.md",
        score=1.0 / index,
        rank=index,
        retrieval_mode="hybrid_rrf",
        metadata={},
    )


def test_extract_citations_returns_unique_numeric_ids() -> None:
    assert extract_citations("Answer [1] and more [2], repeated [1].") == {1, 2}


def test_validate_citations_requires_at_least_one_valid_citation() -> None:
    assert validate_citations("Answer [1].", num_sources=2) is True
    assert validate_citations("Answer with no source.", num_sources=2) is False


def test_validate_citations_rejects_out_of_range_citations() -> None:
    assert validate_citations("Answer [3].", num_sources=2) is False
    assert validate_citations("Answer [0].", num_sources=2) is False


def test_build_sources_assigns_backend_controlled_source_ids() -> None:
    chunks = [make_chunk(1), make_chunk(2)]

    sources = build_sources(chunks)

    assert [source.source_id for source in sources] == ["1", "2"]
    assert sources[0].title == "Title 1"
    assert sources[0].section == "Section 1"
    assert sources[0].source_uri == "source-1.md"
    assert sources[0].chunk_id == chunks[0].chunk_id
    assert sources[0].score == chunks[0].score

