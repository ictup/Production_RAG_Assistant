import argparse
from dataclasses import dataclass, field

from backend.app.core.config import Settings, get_settings

OPENAI_PROVIDERS = {
    "embedding_provider": "openai",
    "generator_provider": "openai",
    "query_rewriter_provider": "openai",
    "reranker_provider": "openai",
}
INSECURE_API_KEYS = {"dev-key", "changeme", "change-me", "test", "password"}


@dataclass(frozen=True)
class ConfigCheckIssue:
    level: str
    variable: str
    message: str


@dataclass
class ConfigCheckReport:
    mode: str
    issues: list[ConfigCheckIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ConfigCheckIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ConfigCheckIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors


def check_settings(
    settings: Settings,
    *,
    strict_production: bool | None = None,
) -> ConfigCheckReport:
    production_mode = (
        settings.env.strip().lower() == "production"
        if strict_production is None
        else strict_production
    )
    report = ConfigCheckReport(mode="production" if production_mode else "local")

    _check_api_keys(settings, report=report, production_mode=production_mode)
    _check_openai_requirements(settings, report=report)
    _check_browser_boundary(settings, report=report)
    if production_mode:
        _check_production_database_urls(settings, report=report)
        _check_production_workspace_access(settings, report=report)
        _check_production_observability(settings, report=report)

    return report


def _check_api_keys(
    settings: Settings,
    *,
    report: ConfigCheckReport,
    production_mode: bool,
) -> None:
    api_keys = split_csv(settings.api_keys)
    if not api_keys:
        report.issues.append(
            ConfigCheckIssue(
                level="error",
                variable="API_KEYS",
                message="at least one API key must be configured",
            )
        )
        return

    if production_mode:
        insecure_keys = [
            key
            for key in api_keys
            if key.strip().lower() in INSECURE_API_KEYS or len(key.strip()) < 16
        ]
        if insecure_keys:
            report.issues.append(
                ConfigCheckIssue(
                    level="error",
                    variable="API_KEYS",
                    message=(
                        "production API keys must not use dev placeholders "
                        "or short tokens"
                    ),
                )
            )


def _check_openai_requirements(
    settings: Settings,
    *,
    report: ConfigCheckReport,
) -> None:
    enabled_openai_providers = [
        variable
        for variable, expected_value in OPENAI_PROVIDERS.items()
        if getattr(settings, variable) == expected_value
    ]
    if enabled_openai_providers and not is_nonblank(settings.openai_api_key):
        report.issues.append(
            ConfigCheckIssue(
                level="error",
                variable="OPENAI_API_KEY",
                message=(
                    "OPENAI_API_KEY is required when any OpenAI provider is enabled"
                ),
            )
        )


def _check_browser_boundary(
    settings: Settings,
    *,
    report: ConfigCheckReport,
) -> None:
    origins = split_csv(settings.cors_allowed_origins)
    if settings.cors_allow_credentials and "*" in origins:
        report.issues.append(
            ConfigCheckIssue(
                level="error",
                variable="CORS_ALLOW_CREDENTIALS",
                message="credentials must not be enabled with wildcard CORS origins",
            )
        )


def _check_production_database_urls(
    settings: Settings,
    *,
    report: ConfigCheckReport,
) -> None:
    if "localhost" in settings.database_url or "127.0.0.1" in settings.database_url:
        report.issues.append(
            ConfigCheckIssue(
                level="warning",
                variable="DATABASE_URL",
                message="production DATABASE_URL should not point at localhost",
            )
        )
    if (
        "localhost" in settings.sync_database_url
        or "127.0.0.1" in settings.sync_database_url
    ):
        report.issues.append(
            ConfigCheckIssue(
                level="warning",
                variable="SYNC_DATABASE_URL",
                message="production SYNC_DATABASE_URL should not point at localhost",
            )
        )


def _check_production_workspace_access(
    settings: Settings,
    *,
    report: ConfigCheckReport,
) -> None:
    if not is_nonblank(settings.api_key_workspace_access):
        report.issues.append(
            ConfigCheckIssue(
                level="warning",
                variable="API_KEY_WORKSPACE_ACCESS",
                message=(
                    "production deployments should explicitly scope API keys "
                    "to allowed workspaces"
                ),
            )
        )


def _check_production_observability(
    settings: Settings,
    *,
    report: ConfigCheckReport,
) -> None:
    if not settings.rate_limit_enabled:
        report.issues.append(
            ConfigCheckIssue(
                level="warning",
                variable="RATE_LIMIT_ENABLED",
                message="production deployments should enable rate limiting",
            )
        )


def split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def is_nonblank(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def format_report(report: ConfigCheckReport) -> str:
    lines = [
        f"configuration check mode: {report.mode}",
        f"errors: {len(report.errors)}",
        f"warnings: {len(report.warnings)}",
    ]
    for issue in report.issues:
        lines.append(f"- {issue.level}: {issue.variable}: {issue.message}")
    if report.passed:
        lines.append("configuration check passed")
    else:
        lines.append("configuration check failed")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate runtime configuration without printing secret values."
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Apply strict production checks regardless of ENV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = check_settings(
        get_settings(),
        strict_production=True if args.production else None,
    )
    print(format_report(report))
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
