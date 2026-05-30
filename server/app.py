from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routers import health, v1


def create_app() -> FastAPI:
    app = FastAPI(title="Genni API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(v1.router, prefix="/api/v1")

    return app


app = create_app()
