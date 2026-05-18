from ingestion.clean_text import clean_text


def test_clean_text_normalizes_newlines_and_collapses_excess_blank_lines() -> None:
    raw = "First line\r\n\r\n\r\n\r\nSecond line\rThird line"

    assert clean_text(raw) == "First line\n\nSecond line\nThird line"


def test_clean_text_preserves_code_blocks_and_technical_case() -> None:
    raw = """# KV-cache Notes

```python
name = "BF16"

print(name)
```

FlashAttention uses IO-aware attention.
"""

    cleaned = clean_text(raw)

    assert 'name = "BF16"\n\nprint(name)' in cleaned
    assert "KV-cache" in cleaned
    assert "FlashAttention" in cleaned


def test_clean_text_removes_obvious_navigation_noise() -> None:
    raw = """# Title

[TOC]

Useful content.

Back to top
"""

    assert clean_text(raw) == "# Title\n\nUseful content."

