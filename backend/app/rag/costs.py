from dataclasses import dataclass

TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ProviderTokenPrices:
    input_per_1m_tokens_usd: float
    output_per_1m_tokens_usd: float


@dataclass(frozen=True)
class ProviderCostEstimate:
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    estimated: bool


def parse_provider_price_table(
    raw_price_table: str,
) -> dict[tuple[str, str], ProviderTokenPrices]:
    prices: dict[tuple[str, str], ProviderTokenPrices] = {}
    for raw_entry in raw_price_table.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue

        provider, model, raw_prices = parse_price_entry_header(entry)
        prices[(provider, model)] = parse_token_prices(raw_prices)

    return prices


def parse_price_entry_header(entry: str) -> tuple[str, str, str]:
    parts = entry.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "provider price entries must use provider:model:key=value syntax"
        )

    provider = parts[0].strip()
    model = parts[1].strip()
    if not provider or not model:
        raise ValueError("provider price entries require provider and model")
    return provider, model, parts[2]


def parse_token_prices(raw_prices: str) -> ProviderTokenPrices:
    values: dict[str, float] = {}
    for raw_pair in raw_prices.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue

        key, separator, raw_value = pair.partition("=")
        key = key.strip()
        if separator == "" or key not in {"input", "output"}:
            raise ValueError(
                "provider price values must use input=<usd>,output=<usd>"
            )

        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError("provider price values must be numeric") from exc
        if value < 0:
            raise ValueError("provider price values must not be negative")
        values[key] = value

    if "input" not in values or "output" not in values:
        raise ValueError("provider price entries require input and output prices")

    return ProviderTokenPrices(
        input_per_1m_tokens_usd=values["input"],
        output_per_1m_tokens_usd=values["output"],
    )


def estimate_provider_token_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    raw_price_table: str,
) -> ProviderCostEstimate:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must not be negative")

    prices = parse_provider_price_table(raw_price_table)
    token_prices = prices.get((provider, model))
    if token_prices is None:
        return ProviderCostEstimate(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
            estimated=False,
        )

    input_cost_usd = (
        input_tokens / TOKENS_PER_MILLION * token_prices.input_per_1m_tokens_usd
    )
    output_cost_usd = (
        output_tokens / TOKENS_PER_MILLION * token_prices.output_per_1m_tokens_usd
    )
    return ProviderCostEstimate(
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=input_cost_usd + output_cost_usd,
        estimated=True,
    )
