import pytest

from backend.app.core.config import Settings
from backend.app.rag.pipeline_smoke import build_pipeline_smoke_settings


def make_settings() -> Settings:
    return Settings(
        embedding_provider="fake",
        embedding_model="test-fake-embedding",
        generator_provider="fake",
        query_rewriter_provider="none",
        reranker_provider="none",
        query_rewrite_model="gpt-default",
        llm_model="fake-llm",
        reranker_model="gpt-default",
        query_rewrite_max_output_tokens=64,
        openai_api_key="test-key",
        openai_max_output_tokens=512,
    )


def test_build_pipeline_smoke_settings_applies_runtime_overrides() -> None:
    settings = build_pipeline_smoke_settings(
        make_settings(),
        embedding_provider="openai",
        generator_provider="openai",
        query_rewriter_provider="openai",
        reranker_provider="openai",
        query_rewrite_model="gpt-rewrite",
        llm_model="gpt-test",
        reranker_model="gpt-rerank",
        query_rewrite_max_output_tokens=32,
        openai_max_output_tokens=123,
    )

    assert settings.embedding_provider == "openai"
    assert settings.generator_provider == "openai"
    assert settings.query_rewriter_provider == "openai"
    assert settings.reranker_provider == "openai"
    assert settings.query_rewrite_model == "gpt-rewrite"
    assert settings.llm_model == "gpt-test"
    assert settings.reranker_model == "gpt-rerank"
    assert settings.query_rewrite_max_output_tokens == 32
    assert settings.openai_max_output_tokens == 123


def test_build_pipeline_smoke_settings_keeps_base_settings_without_overrides() -> None:
    base_settings = make_settings()

    settings = build_pipeline_smoke_settings(base_settings)

    assert settings is base_settings


def test_build_pipeline_smoke_settings_rejects_invalid_output_limit() -> None:
    with pytest.raises(ValueError, match="openai_max_output_tokens"):
        build_pipeline_smoke_settings(
            make_settings(),
            openai_max_output_tokens=0,
        )


def test_build_pipeline_smoke_settings_rejects_invalid_rewrite_output_limit() -> None:
    with pytest.raises(ValueError, match="query_rewrite_max_output_tokens"):
        build_pipeline_smoke_settings(
            make_settings(),
            query_rewrite_max_output_tokens=0,
        )
