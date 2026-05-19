from pathlib import Path

ENV_EXAMPLE_PATH = Path(".env.example")
CONFIGURATION_DOC_PATH = Path("docs/CONFIGURATION.md")
README_PATH = Path("README.md")
HANDOFF_PATH = Path("docs/PROJECT_HANDOFF.md")


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


def test_configuration_doc_warns_against_printing_secrets() -> None:
    documented_config = CONFIGURATION_DOC_PATH.read_text(encoding="utf-8")

    assert "config --quiet" in documented_config
    assert "never commit real secrets" in documented_config.lower()
