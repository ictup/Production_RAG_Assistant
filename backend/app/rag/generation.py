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
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "is",
    "it",
    "of",
    "the",
    "to",
    "what",
}
DEFAULT_ANSWER_SENTENCE_COUNT = 3


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
    sentences = split_sentences(text)
    if sentences:
        return sentences[0]
    return ""


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    sentences = re.findall(r".+?(?:[.!?])(?:\s|$)", normalized)
    consumed_length = sum(len(sentence) for sentence in sentences)
    remainder = normalized[consumed_length:].strip()
    if remainder:
        sentences.append(remainder)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_question_terms(question: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(question.casefold())
        if len(token) > 1 and token not in STOPWORDS
    }


def score_sentence(sentence: str, question_terms: set[str]) -> int:
    if not question_terms:
        return 0

    sentence_terms = set(TOKEN_PATTERN.findall(sentence.casefold()))
    return len(sentence_terms & question_terms)


def select_relevant_sentences(
    *,
    question: str,
    context_text: str,
    max_sentences: int = DEFAULT_ANSWER_SENTENCE_COUNT,
) -> list[str]:
    if max_sentences <= 0:
        raise ValueError("max_sentences must be greater than zero")

    sentences = split_sentences(context_text)
    if not sentences:
        return []

    question_terms = extract_question_terms(question)
    scored_indexes = [
        (score_sentence(sentence, question_terms), index)
        for index, sentence in enumerate(sentences)
    ]
    _, best_index = max(scored_indexes, key=lambda item: (item[0], -item[1]))

    start_index = max(0, best_index - 1)
    end_index = min(len(sentences), start_index + max_sentences)
    if end_index - start_index < max_sentences:
        start_index = max(0, end_index - max_sentences)

    return sentences[start_index:end_index]


def build_relevant_context_snippet(*, question: str, context_text: str) -> str:
    sentences = select_relevant_sentences(
        question=question,
        context_text=context_text,
    )
    if not sentences:
        raise ValueError("context_text must not be blank")
    return " ".join(sentences)


def build_fake_answer(*, question: str, context_text: str) -> str:
    snippet = build_relevant_context_snippet(
        question=question,
        context_text=context_text,
    )
    return (
        "Based on the provided documents, the relevant answer is: "
        f"{snippet} [1]"
    )


def build_generator(settings: Settings | None = None) -> Generator:
    settings = settings or get_settings()

    if settings.generator_provider == "fake":
        return FakeGenerator(model_name=settings.llm_model)

    raise ValueError(f"unsupported generator provider: {settings.generator_provider}")
