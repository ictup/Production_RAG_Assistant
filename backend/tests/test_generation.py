import uuid

import pytest

from backend.app.core.config import Settings
from backend.app.rag.citations import validate_citations
from backend.app.rag.generation import (
    FakeGenerator,
    Generator,
    build_fake_answer,
    build_generator,
    extract_first_context_text,
    extract_question,
    first_sentence,
)
from backend.app.rag.prompts import build_rag_prompt
from backend.app.rag.retrieval_models import RetrievedChunk


def make_chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        text=text,
        title="FlashAttention Notes",
        section_title="FlashAttention",
        source_uri="llm_systems/flashattention.md",
        score=1.0,
        rank=1,
        retrieval_mode="hybrid_rrf",
        metadata={},
    )


@pytest.mark.asyncio
async def test_fake_generator_returns_cited_answer_and_usage() -> None:
    prompt = build_rag_prompt(
        "What problem does FlashAttention solve?",
        [
            make_chunk(
                "FlashAttention reduces memory traffic between HBM and SRAM. "
                "It tiles exact attention."
            )
        ],
    )
    generator = FakeGenerator(model_name="test-fake")

    generated = await generator.generate(prompt)

    assert generated.model == "test-fake"
    assert "FlashAttention reduces memory traffic" in generated.answer
    assert generated.answer.endswith("[1]")
    assert generated.input_tokens > 0
    assert generated.output_tokens > 0
    assert validate_citations(generated.answer, num_sources=1) is True


@pytest.mark.asyncio
async def test_fake_generator_rejects_blank_prompt() -> None:
    generator = FakeGenerator()

    with pytest.raises(ValueError, match="prompt"):
        await generator.generate("  ")


def test_extract_question_reads_question_section() -> None:
    assert extract_question("Context:\n[1]\n\nQuestion:\nWhat is RAG?") == (
        "What is RAG?"
    )


def test_extract_first_context_text_reads_first_numbered_block() -> None:
    prompt = (
        "[1]\nTitle: A\nText:\nFirst context.\n\n"
        "[2]\nTitle: B\nText:\nSecond context.\n\n"
        "Question:\nWhat matters?"
    )

    assert extract_first_context_text(prompt) == "First context."


def test_first_sentence_prefers_sentence_boundary() -> None:
    assert first_sentence("First sentence. Second sentence.") == "First sentence."


def test_build_fake_answer_always_adds_first_citation() -> None:
    answer = build_fake_answer(
        question="What is FlashAttention?",
        context_text="FlashAttention is IO-aware. Extra detail.",
    )

    assert answer == (
        "Based on the provided documents, the relevant answer is: "
        "FlashAttention is IO-aware. [1]"
    )


def test_build_generator_uses_settings() -> None:
    generator = build_generator(
        Settings(
            generator_provider="fake",
            llm_model="test-fake-llm",
        )
    )

    assert isinstance(generator, Generator)
    assert generator.model_name == "test-fake-llm"

