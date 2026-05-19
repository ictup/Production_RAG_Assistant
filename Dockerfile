# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY alembic.ini ./
COPY backend ./backend
COPY ingestion ./ingestion
COPY evals ./evals
COPY data ./data

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    groupadd --system app && \
    useradd --system --gid app --home-dir /app app && \
    chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
