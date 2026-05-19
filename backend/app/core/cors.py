from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from backend.app.core.config import Settings


def parse_cors_allowed_origins(raw_origins: str) -> list[str]:
    return [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]


def add_cors_middleware(app: FastAPI, settings: Settings) -> None:
    allowed_origins = parse_cors_allowed_origins(settings.cors_allowed_origins)
    allowed_origin_regex = settings.cors_allowed_origin_regex or None

    if not allowed_origins and allowed_origin_regex is None:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=allowed_origin_regex,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-Workspace-ID",
        ],
        expose_headers=["X-Request-ID"],
    )
