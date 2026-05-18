from ingestion.hashing import compute_content_hash, normalize_for_hash


def test_normalize_for_hash_removes_line_trailing_space_and_outer_blank_lines() -> None:
    assert normalize_for_hash("\nA line   \r\nB line\t\n\n") == "A line\nB line"


def test_compute_content_hash_is_stable_for_line_endings_and_trailing_space() -> None:
    first = compute_content_hash("FlashAttention\r\nuses IO-aware attention.  \n")
    second = compute_content_hash("FlashAttention\nuses IO-aware attention.")

    assert first == second
    assert len(first) == 64


def test_compute_content_hash_changes_when_content_changes() -> None:
    first = compute_content_hash("KV cache")
    second = compute_content_hash("KV cache paging")

    assert first != second

