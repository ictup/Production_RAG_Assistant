import pytest

from backend.app.rag.costs import (
    estimate_provider_token_cost,
    parse_provider_price_table,
)


def test_parse_provider_price_table_reads_generation_prices() -> None:
    prices = parse_provider_price_table(
        "openai:gpt-test:input=0.50,output=1.00;"
        "fake:fake-llm:input=0,output=0"
    )

    assert prices[("openai", "gpt-test")].input_per_1m_tokens_usd == 0.5
    assert prices[("openai", "gpt-test")].output_per_1m_tokens_usd == 1.0
    assert prices[("fake", "fake-llm")].input_per_1m_tokens_usd == 0.0


@pytest.mark.parametrize(
    "raw_price_table",
    [
        "openai",
        "openai:gpt-test",
        "openai:gpt-test:input=0.5",
        "openai:gpt-test:input=-1,output=1",
        "openai:gpt-test:prompt=1,output=1",
    ],
)
def test_parse_provider_price_table_rejects_invalid_entries(
    raw_price_table: str,
) -> None:
    with pytest.raises(ValueError):
        parse_provider_price_table(raw_price_table)


def test_estimate_provider_token_cost_uses_prices_per_million_tokens() -> None:
    estimate = estimate_provider_token_cost(
        provider="openai",
        model="gpt-test",
        input_tokens=10,
        output_tokens=5,
        raw_price_table="openai:gpt-test:input=0.50,output=1.00",
    )

    assert estimate.estimated is True
    assert estimate.input_cost_usd == 0.000005
    assert estimate.output_cost_usd == 0.000005
    assert estimate.total_cost_usd == 0.00001


def test_estimate_provider_token_cost_returns_unestimated_without_price() -> None:
    estimate = estimate_provider_token_cost(
        provider="openai",
        model="gpt-unpriced",
        input_tokens=10,
        output_tokens=5,
        raw_price_table="openai:gpt-test:input=0.50,output=1.00",
    )

    assert estimate.estimated is False
    assert estimate.total_cost_usd == 0.0


def test_estimate_provider_token_cost_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="token counts"):
        estimate_provider_token_cost(
            provider="openai",
            model="gpt-test",
            input_tokens=-1,
            output_tokens=0,
            raw_price_table="openai:gpt-test:input=0.50,output=1.00",
        )
