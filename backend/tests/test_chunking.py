import re

import pytest

from ingestion.chunking import (
    chunk_document,
    count_tokens,
    split_markdown_sections,
)
from ingestion.models import RawDocument


def make_raw_document(text: str) -> RawDocument:
    return RawDocument(
        title="Attention Notes",
        source_uri="data/raw/attention.md",
        text=text,
        metadata={"topic": "attention"},
    )


def words(text: str) -> set[str]:
    return set(re.findall(r"token\d+", text))


def test_split_markdown_sections_tracks_headings() -> None:
    sections = split_markdown_sections(
        """# Attention

FlashAttention reduces memory traffic.

## Memory

PagedAttention manages KV cache pages.
"""
    )

    assert [section.title for section in sections] == ["Attention", "Memory"]
    assert sections[0].text.startswith("# Attention")
    assert sections[1].text.startswith("## Memory")


def test_split_markdown_sections_ignores_headings_inside_code_blocks() -> None:
    sections = split_markdown_sections(
        """```python
# This is not a Markdown heading
print("BF16")
```

# Real Section

Content.
"""
    )

    assert [section.title for section in sections] == [None, "Real Section"]


def test_chunk_document_keeps_chunks_under_size_and_adds_overlap() -> None:
    body = " ".join(f"token{index:02d}" for index in range(40))
    raw_document = make_raw_document(f"# Long Section\n\n{body}")

    chunks = chunk_document(
        raw_document,
        chunk_size_tokens=16,
        chunk_overlap_tokens=4,
    )

    assert len(chunks) > 1
    assert all(chunk.token_count <= 16 for chunk in chunks)
    assert words(chunks[0].text) & words(chunks[1].text)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.section_title for chunk in chunks} == {"Long Section"}


def test_chunk_document_preserves_headings_and_metadata() -> None:
    raw_document = make_raw_document(
        """# IO-aware attention

FlashAttention tiles attention to reduce HBM traffic.

# KV cache

PagedAttention stores KV cache in pages.
"""
    )

    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)

    assert [chunk.section_title for chunk in chunks] == [
        "IO-aware attention",
        "KV cache",
    ]
    assert chunks[0].text.startswith("# IO-aware attention")
    assert chunks[1].text.startswith("# KV cache")
    assert all(chunk.metadata == {"topic": "attention"} for chunk in chunks)
    assert all(chunk.source_uri == raw_document.source_uri for chunk in chunks)
    assert all(chunk.workspace_id == "public" for chunk in chunks)


def test_chunk_document_preserves_small_code_block() -> None:
    raw_document = make_raw_document(
        """# Code

```python
name = "BF16"

print(name)
```
"""
    )

    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)

    assert len(chunks) == 1
    assert 'name = "BF16"\n\nprint(name)' in chunks[0].text


def test_chunk_document_rejects_invalid_overlap() -> None:
    raw_document = make_raw_document("# Title\n\nContent")

    with pytest.raises(ValueError, match="smaller than chunk_size_tokens"):
        chunk_document(raw_document, chunk_size_tokens=10, chunk_overlap_tokens=10)


def test_count_tokens_returns_positive_count_for_technical_text() -> None:
    assert count_tokens("KV-cache BF16 FlashAttention") > 0

