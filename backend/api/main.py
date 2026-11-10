"""The EnergyIQ REST API. Run from backend/ with: uvicorn api.main:app --reload
    http://127.0.0.1:8000/docs"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import connect
from api.deps import DB
from api.predictions import ModelStore
from api.routers import anomalies, buildings, models, weather
from api.schemas import Health
from energyiq import config

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"


def create_app(db_path: Path | None = None, models_dir: Path | None = None) -> FastAPI:
    db_path = db_path or Path(os.environ.get("ENERGYIQ_DB_PATH", config.DB_PATH))
    models_dir = models_dir or Path(os.environ.get("ENERGYIQ_MODELS_DIR", config.REPLAY_MODELS_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # DuckDB allows one read-write process, so the file stays locked while the API runs
        app.state.db = connect(db_path)
        app.state.models = ModelStore(models_dir)
        try:
            yield
        finally:
            app.state.db.close()

    app = FastAPI(
        title="EnergyIQ API",
        description="Building energy use, predicted usage, anomalies, weather and model performance.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("ENERGYIQ_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(","),
        allow_methods=["GET", "PATCH"],
        allow_headers=["Content-Type"],
    )
    for module in (buildings, anomalies, weather, models):
        app.include_router(module.router)

    @app.get("/health", response_model=Health, tags=["health"])
    def health(db: DB):
        readings, flagged = db.execute(
            "SELECT count(*), count(*) FILTER (WHERE is_anomalous) FROM meter_readings"
        ).fetchone()
        return {
            "status": "ok",
            "readings": readings,
            "flagged_readings": flagged,
            "models_loaded": len(app.state.models.bundles),
        }

    return app


app = create_app()
