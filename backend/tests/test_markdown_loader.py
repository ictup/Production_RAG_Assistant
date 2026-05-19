from pathlib import Path

import pytest

from ingestion.parse_markdown import (
    discover_markdown_files,
    load_markdown_document,
    load_markdown_text,
    split_front_matter,
)


def test_load_markdown_document_parses_front_matter_and_cleans_text(
    tmp_path: Path,
) -> None:
    document = tmp_path / "flashattention.md"
    document.write_text(
        """---
title: "FlashAttention Notes"
source_type: "markdown"
topic: "attention"
difficulty: "advanced"
tags: ["attention", "gpu", "memory"]
workspace_id: "public"
visibility: "public"
author: "Dao et al."
---

# IO-aware attention

[TOC]

FlashAttention preserves KV-cache and BF16 terms.
""",
        encoding="utf-8",
    )

    raw_document = load_markdown_document(document, source_root=tmp_path)

    assert raw_document.title == "FlashAttention Notes"
    assert raw_document.source_type == "markdown"
    assert raw_document.source_uri == "flashattention.md"
    assert raw_document.workspace_id == "public"
    assert raw_document.visibility == "public"
    assert raw_document.author == "Dao et al."
    assert raw_document.metadata == {
        "topic": "attention",
        "difficulty": "advanced",
        "tags": ["attention", "gpu", "memory"],
    }
    assert "[TOC]" not in raw_document.text
    assert "KV-cache" in raw_document.text
    assert "BF16" in raw_document.text


def test_load_markdown_document_infers_title_from_first_heading(tmp_path: Path) -> None:
    document = tmp_path / "pagedattention.md"
    document.write_text(
        "# PagedAttention\n\nPagedAttention improves KV cache memory management.",
        encoding="utf-8",
    )

    raw_document = load_markdown_document(document, source_root=tmp_path)

    assert raw_document.title == "PagedAttention"
    assert raw_document.metadata == {}
    assert raw_document.workspace_id == "public"


def test_load_markdown_text_uses_request_workspace_and_merges_metadata() -> None:
    markdown = """---
title: "Front Matter Title"
workspace_id: "ignored-workspace"
visibility: "private"
author: "Front Matter Author"
topic: "attention"
---

# Body Title

FlashAttention reduces HBM traffic.
"""

    raw_document = load_markdown_text(
        markdown,
        source_uri="uploads/flashattention.md",
        default_workspace_id="tenant-a",
        title="Request Title",
        metadata={"difficulty": "advanced"},
    )

    assert raw_document.title == "Request Title"
    assert raw_document.source_type == "markdown"
    assert raw_document.source_uri == "uploads/flashattention.md"
    assert raw_document.workspace_id == "tenant-a"
    assert raw_document.visibility == "private"
    assert raw_document.author == "Front Matter Author"
    assert raw_document.metadata == {
        "topic": "attention",
        "difficulty": "advanced",
    }
    assert "FlashAttention reduces HBM traffic." in raw_document.text


def test_discover_markdown_files_returns_sorted_markdown_paths(tmp_path: Path) -> None:
    first = tmp_path / "b.md"
    second = tmp_path / "nested" / "a.markdown"
    ignored = tmp_path / "notes.txt"
    second.parent.mkdir()
    first.write_text("# B", encoding="utf-8")
    second.write_text("# A", encoding="utf-8")
    ignored.write_text("ignore", encoding="utf-8")

    discovered = discover_markdown_files(tmp_path)

    assert discovered == sorted([first, second], key=lambda path: path.as_posix())


def test_split_front_matter_rejects_non_mapping_yaml() -> None:
    markdown = "---\n- not\n- a\n- mapping\n---\nBody"

    with pytest.raises(ValueError, match="YAML mapping"):
        split_front_matter(markdown)
