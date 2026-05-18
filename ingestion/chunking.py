from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from ingestion.clean_text import is_code_fence, normalize_newlines
from ingestion.models import Chunk, RawDocument

DEFAULT_CHUNK_SIZE_TOKENS = 800
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownSection:
    title: str | None
    text: str


@lru_cache
def _get_tiktoken_encoding() -> Any | None:
    try:
        import tiktoken
    except ModuleNotFoundError:
        return None

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    encoding = _get_tiktoken_encoding()
    if encoding is not None:
        return len(encoding.encode(text))

    return len(TOKEN_PATTERN.findall(text))


def _trim_to_token_budget(
    text: str,
    max_tokens: int,
    *,
    keep: Literal["start", "end"],
) -> str:
    if max_tokens <= 0:
        return ""

    encoding = _get_tiktoken_encoding()
    if encoding is not None:
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text.strip()
        selected = tokens[:max_tokens] if keep == "start" else tokens[-max_tokens:]
        return encoding.decode(selected).strip()

    matches = list(TOKEN_PATTERN.finditer(text))
    if len(matches) <= max_tokens:
        return text.strip()

    if keep == "start":
        return text[: matches[max_tokens - 1].end()].strip()

    return text[matches[-max_tokens].start() :].strip()


def _split_text_by_token_windows(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if count_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []

    step = max(1, max_tokens - overlap_tokens)
    encoding = _get_tiktoken_encoding()

    if encoding is not None:
        tokens = encoding.encode(text)
        windows = []
        for start in range(0, len(tokens), step):
            window = encoding.decode(tokens[start : start + max_tokens]).strip()
            if window:
                windows.append(window)
            if start + max_tokens >= len(tokens):
                break
        return windows

    matches = list(TOKEN_PATTERN.finditer(text))
    windows = []
    for start in range(0, len(matches), step):
        end = min(start + max_tokens, len(matches))
        window = text[matches[start].start() : matches[end - 1].end()].strip()
        if window:
            windows.append(window)
        if end >= len(matches):
            break
    return windows


def _extract_heading_title(line: str) -> str | None:
    match = HEADING_PATTERN.match(line.strip())
    if match is None:
        return None
    return match.group(2).strip(" #")


def split_markdown_sections(text: str) -> list[MarkdownSection]:
    lines = normalize_newlines(text).split("\n")
    sections: list[MarkdownSection] = []
    current_title: str | None = None
    current_lines: list[str] = []
    in_code_block = False

    def flush() -> None:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(MarkdownSection(title=current_title, text=section_text))

    for line in lines:
        if is_code_fence(line):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        heading_title = None if in_code_block else _extract_heading_title(line)
        if heading_title is not None:
            flush()
            current_title = heading_title
            current_lines = [line]
            continue

        current_lines.append(line)

    flush()
    return sections


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current_lines: list[str] = []
    in_code_block = False

    def flush() -> None:
        block = "\n".join(current_lines).strip()
        if block:
            blocks.append(block)
        current_lines.clear()

    for line in normalize_newlines(text).split("\n"):
        if is_code_fence(line):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block and not line.strip():
            flush()
            continue

        current_lines.append(line)

    flush()
    return blocks


def _with_section_heading(
    text: str,
    section_title: str | None,
    *,
    preserve_headings: bool,
) -> str:
    text = text.strip()
    if not preserve_headings or section_title is None:
        return text

    if text.startswith("#"):
        return text

    return f"# {section_title}\n\n{text}"


def _join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip()).strip()


def _chunk_section(
    section: MarkdownSection,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    preserve_headings: bool,
) -> list[str]:
    blocks = _split_blocks(section.text)
    chunks: list[str] = []
    current_blocks: list[str] = []

    heading_budget = 0
    if preserve_headings and section.title is not None:
        heading_budget = count_tokens(f"# {section.title}\n\n")
    body_budget = max(1, chunk_size_tokens - heading_budget)

    def render(blocks_to_render: list[str]) -> str:
        return _with_section_heading(
            _join_blocks(blocks_to_render),
            section.title,
            preserve_headings=preserve_headings,
        )

    def is_heading_only(blocks_to_check: list[str]) -> bool:
        if section.title is None:
            return False
        return _extract_heading_title(_join_blocks(blocks_to_check)) == section.title

    def emit_current(*, keep_overlap: bool) -> None:
        nonlocal current_blocks

        rendered = render(current_blocks)
        if rendered:
            chunks.append(rendered)

        if keep_overlap and chunk_overlap_tokens > 0:
            raw_text = _join_blocks(current_blocks)
            overlap = _trim_to_token_budget(
                raw_text,
                chunk_overlap_tokens,
                keep="end",
            )
            current_blocks = [overlap] if overlap else []
        else:
            current_blocks = []

    for block in blocks:
        block_rendered = render([block])
        if count_tokens(block_rendered) > chunk_size_tokens:
            if current_blocks:
                if is_heading_only(current_blocks):
                    current_blocks = []
                else:
                    emit_current(keep_overlap=False)

            windows = _split_text_by_token_windows(
                block,
                max_tokens=body_budget,
                overlap_tokens=min(chunk_overlap_tokens, body_budget - 1),
            )
            chunks.extend(render([window]) for window in windows)
            continue

        candidate = [*current_blocks, block]
        if count_tokens(render(candidate)) <= chunk_size_tokens:
            current_blocks = candidate
            continue

        if is_heading_only(current_blocks):
            current_blocks = []
        else:
            emit_current(keep_overlap=True)
        candidate = [*current_blocks, block]
        if count_tokens(render(candidate)) <= chunk_size_tokens:
            current_blocks = candidate
        else:
            current_blocks = [block]

    if current_blocks:
        emit_current(keep_overlap=False)

    return chunks


def chunk_document(
    raw_document: RawDocument,
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    document_id: str | None = None,
    preserve_headings: bool = True,
) -> list[Chunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than zero")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must not be negative")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    sections = split_markdown_sections(raw_document.text)
    chunks: list[Chunk] = []

    for section in sections:
        for chunk_text in _chunk_section(
            section,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            preserve_headings=preserve_headings,
        ):
            chunks.append(
                Chunk(
                    document_id=document_id or raw_document.source_uri,
                    chunk_index=len(chunks),
                    text=chunk_text,
                    section_title=section.title,
                    token_count=count_tokens(chunk_text),
                    source_uri=raw_document.source_uri,
                    workspace_id=raw_document.workspace_id,
                    metadata=dict(raw_document.metadata),
                )
            )

    return chunks
