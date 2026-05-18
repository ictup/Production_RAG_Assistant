import re
from collections.abc import Sequence

from pydantic import BaseModel

from backend.app.rag.retrieval_models import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class Source(BaseModel):
    source_id: str
    title: str
    section: str | None
    source_uri: str
    chunk_id: str
    score: float


def extract_citations(answer: str) -> set[int]:
    return {int(match) for match in CITATION_PATTERN.findall(answer)}


def validate_citations(answer: str, num_sources: int) -> bool:
    citations = extract_citations(answer)
    if not citations:
        return False

    return all(1 <= citation <= num_sources for citation in citations)


def build_sources(chunks: Sequence[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            source_id=str(index),
            title=chunk.title,
            section=chunk.section_title,
            source_uri=chunk.source_uri,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

