from pathlib import Path

ENV_EXAMPLE_PATH = Path(".env.example")
CONFIGURATION_DOC_PATH = Path("docs/CONFIGURATION.md")
README_PATH = Path("README.md")
HANDOFF_PATH = Path("docs/PROJECT_HANDOFF.md")
SECRET_MANAGER_DOC_PATH = Path("docs/SECRET_MANAGER_MAPPING.md")


def load_env_example_keys() -> set[str]:
    keys: set[str] = set()

    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            key, separator, _value = stripped.partition("=")
            if separator:
                keys.add(key)

    return keys


def test_configuration_doc_covers_every_env_example_key() -> None:
    documented_config = CONFIGURATION_DOC_PATH.read_text(encoding="utf-8")

    missing_keys = [
        key
        for key in sorted(load_env_example_keys())
        if f"`{key}`" not in documented_config
    ]

    assert missing_keys == []


def test_configuration_doc_is_linked_from_readme_and_handoff() -> None:
    assert "docs/CONFIGURATION.md" in README_PATH.read_text(encoding="utf-8")
    assert "docs/CONFIGURATION.md" in HANDOFF_PATH.read_text(encoding="utf-8")


def test_secret_manager_mapping_is_linked_from_entry_docs() -> None:
    expected_link = "docs/SECRET_MANAGER_MAPPING.md"

    assert expected_link in CONFIGURATION_DOC_PATH.read_text(encoding="utf-8")
    assert expected_link in README_PATH.read_text(encoding="utf-8")
    assert expected_link in HANDOFF_PATH.read_text(encoding="utf-8")


def test_configuration_doc_warns_against_printing_secrets() -> None:
    documented_config = CONFIGURATION_DOC_PATH.read_text(encoding="utf-8")

    assert "config --quiet" in documented_config
    assert "python -m backend.app.core.config_check --production" in documented_config
    assert "never commit real secrets" in documented_config.lower()


def test_env_example_has_commented_openai_provider_preset_without_real_key() -> None:
    env_example = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "# Real OpenAI provider preset for local .env only:" in env_example
    assert "# EMBEDDING_PROVIDER=openai" in env_example
    assert "# GENERATOR_PROVIDER=openai" in env_example
    assert "# QUERY_REWRITER_PROVIDER=openai" in env_example
    assert "# RERANKER_PROVIDER=openai" in env_example
    assert "# OPENAI_API_KEY=<set locally only>" in env_example
    assert "# OPENAI_EMBEDDING_MODEL=text-embedding-3-small" in env_example
    assert "# LLM_MODEL=gpt-5.4-nano" in env_example
    assert "# QUERY_REWRITE_MODEL=gpt-5.4-nano" in env_example
    assert "# RERANKER_MODEL=gpt-5.4-nano" in env_example
    assert ("s" + "k-") not in env_example


def test_secret_manager_mapping_classifies_sensitive_runtime_values() -> None:
    secret_mapping = SECRET_MANAGER_DOC_PATH.read_text(encoding="utf-8")

    required_secret_values = [
        "`API_KEYS`",
        "`API_KEY_WORKSPACE_ACCESS`",
        "`POSTGRES_PASSWORD`",
        "`DATABASE_URL`",
        "`SYNC_DATABASE_URL`",
        "`OPENAI_API_KEY`",
    ]
    missing_secret_values = [
        value for value in required_secret_values if value not in secret_mapping
    ]

    assert missing_secret_values == []
    assert "## Managed Secrets" in secret_mapping
    assert "## Plain Runtime Configuration" in secret_mapping
    assert "## Rotation Workflow" in secret_mapping
    assert (
        "uv run python -m backend.app.core.config_check --production"
        in secret_mapping
    )
    assert (
        "Never use a command that prints resolved environment values"
        in secret_mapping
    )
