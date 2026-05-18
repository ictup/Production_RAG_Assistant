import hashlib

from ingestion.clean_text import normalize_newlines


def normalize_for_hash(text: str) -> str:
    normalized = normalize_newlines(text)
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip()


def compute_content_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

