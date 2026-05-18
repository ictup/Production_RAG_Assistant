from fastapi import FastAPI

from backend.app.api import routes_chat, routes_health
from backend.app.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Production RAG Assistant",
        version=settings.app_version,
    )
    app.state.settings = settings

    app.include_router(routes_health.router)
    app.include_router(routes_chat.router)
    return app


app = create_app()
