import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings
from ingestion.chunking import count_tokens

QUESTION_PATTERN = re.compile(r"\n\nQuestion:\n(?P<question>.+)\s*\Z", re.DOTALL)
FIRST_CONTEXT_TEXT_PATTERN = re.compile(
    r"\[1\]\n.*?Text:\n(?P<text>.*?)(?:\n\n\[\d+\]|\n\nQuestion:)",
    re.DOTALL,
)


class GeneratedAnswer(BaseModel):
    answer: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


@runtime_checkable
class Generator(Protocol):
    provider_name: str
    model_name: str

    async def generate(self, prompt: str) -> GeneratedAnswer:
        pass


class FakeGenerator:
    provider_name = "fake"

    def __init__(self, *, model_name: str = "fake-llm") -> None:
        self.model_name = model_name

    async def generate(self, prompt: str) -> GeneratedAnswer:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be blank")

        question = extract_question(prompt)
        context_text = extract_first_context_text(prompt)
        answer = build_fake_answer(question=question, context_text=context_text)

        return GeneratedAnswer(
            answer=answer,
            model=self.model_name,
            input_tokens=count_tokens(prompt),
            output_tokens=count_tokens(answer),
        )


def extract_question(prompt: str) -> str:
    match = QUESTION_PATTERN.search(prompt)
    if match is None:
        raise ValueError("prompt must contain a Question section")

    question = match.group("question").strip()
    if not question:
        raise ValueError("question must not be blank")
    return question


def extract_first_context_text(prompt: str) -> str:
    match = FIRST_CONTEXT_TEXT_PATTERN.search(prompt)
    if match is None:
        raise ValueError("prompt must contain a [1] context block with Text")

    text = match.group("text").strip()
    if not text:
        raise ValueError("first context block text must not be blank")
    return text


def first_sentence(text: str) -> str:
    normalized = " ".join(text.split())
    sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", normalized)
    if sentence_match is not None:
        return sentence_match.group(1)
    return normalized


def build_fake_answer(*, question: str, context_text: str) -> str:
    sentence = first_sentence(context_text)
    return (
        "Based on the provided documents, the relevant answer is: "
        f"{sentence} [1]"
    )


def build_generator(settings: Settings | None = None) -> Generator:
    settings = settings or get_settings()

    if settings.generator_provider == "fake":
        return FakeGenerator(model_name=settings.llm_model)

    raise ValueError(f"unsupported generator provider: {settings.generator_provider}")

