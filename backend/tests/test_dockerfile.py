from pathlib import Path

DOCKERFILE_PATH = Path("Dockerfile")
DOCKERIGNORE_PATH = Path(".dockerignore")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_backend_dockerfile_exists_and_runs_api() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert DOCKERFILE_PATH.exists()
    assert "ghcr.io/astral-sh/uv:python3.11-bookworm-slim" in content
    assert "uv sync --frozen --no-dev" in content
    assert "EXPOSE 8000" in content
    assert "backend.app.main:app" in content
    assert "0.0.0.0" in content


def test_backend_dockerfile_uses_non_root_runtime_user_and_healthcheck() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "useradd --system" in content
    assert "USER app" in content
    assert "HEALTHCHECK" in content
    assert "http://127.0.0.1:8000/health" in content


def test_dockerignore_excludes_local_state_and_keeps_env_example() -> None:
    lines = set(read_lines(DOCKERIGNORE_PATH))

    for required_pattern in {
        ".git/",
        ".venv/",
        ".uv-cache/",
        ".env",
        ".env.*",
        "!.env.example",
        "evals/reports/*.json",
    }:
        assert required_pattern in lines
