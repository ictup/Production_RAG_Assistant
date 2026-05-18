from pathlib import Path
from typing import Any

import yaml

from ingestion.clean_text import clean_text, normalize_newlines
from ingestion.models import RawDocument

MARKDOWN_SUFFIXES = {".md", ".markdown"}
RESERVED_FRONT_MATTER_KEYS = {
    "author",
    "source_type",
    "source_uri",
    "title",
    "visibility",
    "workspace_id",
}


def discover_markdown_files(input_path: Path | str) -> list[Path]:
    path = Path(input_path)

    if path.is_file():
        return [path] if path.suffix.lower() in MARKDOWN_SUFFIXES else []

    if not path.exists():
        raise FileNotFoundError(path)

    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in MARKDOWN_SUFFIXES
    ]
    return sorted(files, key=lambda candidate: candidate.as_posix())


def split_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    normalized = normalize_newlines(markdown)

    if not normalized.startswith("---\n"):
        return {}, normalized

    lines = normalized.split("\n")
    closing_index = None

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in {"---", "..."}:
            closing_index = index
            break

    if closing_index is None:
        return {}, normalized

    front_matter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])
    parsed = yaml.safe_load(front_matter_text) or {}

    if not isinstance(parsed, dict):
        raise ValueError("Markdown front matter must be a YAML mapping")

    return parsed, body


def infer_title(markdown_body: str, fallback: str) -> str:
    for line in normalize_newlines(markdown_body).split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def build_source_uri(path: Path, source_root: Path | None = None) -> str:
    if source_root is not None:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            pass

    return path.as_posix()


def load_markdown_document(
    path: Path | str,
    *,
    source_root: Path | str | None = None,
    default_workspace_id: str = "public",
) -> RawDocument:
    file_path = Path(path)
    root = Path(source_root) if source_root is not None else None

    front_matter, body = split_front_matter(file_path.read_text(encoding="utf-8"))
    metadata = {
        key: value
        for key, value in front_matter.items()
        if key not in RESERVED_FRONT_MATTER_KEYS
    }

    source_uri = str(
        front_matter.get("source_uri") or build_source_uri(file_path, root)
    )

    return RawDocument(
        title=str(front_matter.get("title") or infer_title(body, file_path.stem)),
        source_type=str(front_matter.get("source_type") or "markdown"),
        source_uri=source_uri,
        text=clean_text(body),
        workspace_id=str(front_matter.get("workspace_id") or default_workspace_id),
        visibility=str(front_matter.get("visibility") or "public"),
        metadata=metadata,
        author=front_matter.get("author"),
    )
