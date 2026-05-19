from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.observability.metrics import (
    MetricsRegistry,
    escape_label_value,
    metrics_registry,
)


def assert_metric_line(output: str, line: str) -> None:
    assert line in output


def test_escape_label_value_escapes_prometheus_special_characters() -> None:
    assert escape_label_value('a"b\\c\nd') == 'a\\"b\\\\c\\nd'


def test_metrics_registry_records_request_count_and_latency_histogram() -> None:
    registry = MetricsRegistry(latency_buckets=(0.1, 1.0))

    registry.observe_http_request(
        method="GET",
        path="/health",
        status_code=200,
        latency_seconds=0.05,
    )

    output = registry.render_prometheus()
    assert_metric_line(
        output,
        'rag_requests_total{method="GET",path="/health",status_code="200"} 1',
    )
    assert_metric_line(
        output,
        'rag_request_latency_seconds_bucket'
        '{method="GET",path="/health",le="0.1"} 1',
    )
    assert_metric_line(
        output,
        'rag_request_latency_seconds_bucket'
        '{method="GET",path="/health",le="1"} 1',
    )
    assert_metric_line(
        output,
        'rag_request_latency_seconds_bucket'
        '{method="GET",path="/health",le="+Inf"} 1',
    )
    assert_metric_line(
        output,
        'rag_request_latency_seconds_count{method="GET",path="/health"} 1',
    )
    assert_metric_line(
        output,
        'rag_request_latency_seconds_sum{method="GET",path="/health"} 0.05',
    )


def test_metrics_registry_records_rag_business_metrics() -> None:
    registry = MetricsRegistry()

    registry.observe_rag_response(
        refusal_reason="no_retrieved_chunks",
        citation_valid=None,
    )
    registry.observe_rag_response(
        refusal_reason=None,
        citation_valid=False,
    )

    output = registry.render_prometheus()
    assert_metric_line(
        output,
        'rag_refusals_total{reason="no_retrieved_chunks"} 1',
    )
    assert_metric_line(output, "rag_citation_invalid_total 1")


def test_metrics_registry_records_provider_errors() -> None:
    registry = MetricsRegistry()

    registry.observe_provider_error(
        provider="openai",
        operation="OpenAI response request",
        category="rate_limit",
    )

    output = registry.render_prometheus()
    assert_metric_line(
        output,
        'rag_provider_errors_total{provider="openai",'
        'operation="OpenAI response request",category="rate_limit"} 1',
    )


def test_metrics_registry_records_provider_latency_and_tokens() -> None:
    registry = MetricsRegistry(latency_buckets=(0.01, 0.1))

    registry.observe_provider_latency(
        provider="openai",
        operation="generation",
        model="gpt-5.4-nano",
        latency_seconds=0.05,
    )
    registry.observe_provider_tokens(
        provider="openai",
        model="gpt-5.4-nano",
        token_type="input",
        tokens=12,
    )
    registry.observe_provider_tokens(
        provider="openai",
        model="gpt-5.4-nano",
        token_type="output",
        tokens=4,
    )
    registry.observe_provider_cost(
        provider="openai",
        model="gpt-5.4-nano",
        cost_usd=0.00001,
    )

    output = registry.render_prometheus()
    assert_metric_line(
        output,
        'rag_provider_latency_seconds_bucket'
        '{provider="openai",operation="generation",'
        'model="gpt-5.4-nano",le="0.01"} 0',
    )
    assert_metric_line(
        output,
        'rag_provider_latency_seconds_bucket'
        '{provider="openai",operation="generation",'
        'model="gpt-5.4-nano",le="0.1"} 1',
    )
    assert_metric_line(
        output,
        'rag_provider_latency_seconds_count'
        '{provider="openai",operation="generation",model="gpt-5.4-nano"} 1',
    )
    assert_metric_line(
        output,
        'rag_provider_latency_seconds_sum'
        '{provider="openai",operation="generation",model="gpt-5.4-nano"} 0.05',
    )
    assert_metric_line(
        output,
        'rag_provider_tokens_total'
        '{provider="openai",model="gpt-5.4-nano",token_type="input"} 12',
    )
    assert_metric_line(
        output,
        'rag_provider_tokens_total'
        '{provider="openai",model="gpt-5.4-nano",token_type="output"} 4',
    )
    assert_metric_line(
        output,
        'rag_provider_cost_usd_total'
        '{provider="openai",model="gpt-5.4-nano"} 1e-05',
    )


def test_metrics_route_returns_prometheus_text() -> None:
    metrics_registry.reset()
    metrics_registry.observe_http_request(
        method="GET",
        path="/health",
        status_code=200,
        latency_seconds=0.01,
    )
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE rag_requests_total counter" in response.text
    assert 'path="/health"' in response.text


def test_metrics_middleware_records_http_request() -> None:
    metrics_registry.reset()
    client = TestClient(create_app())

    health_response = client.get("/health")
    metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert metrics_response.status_code == 200
    assert (
        'rag_requests_total{method="GET",path="/health",status_code="200"} 1'
        in metrics_response.text
    )
