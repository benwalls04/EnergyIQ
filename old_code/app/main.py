import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .db import init_db
from .api.routes import router as api_router

app = FastAPI(title="Capstone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(api_router, prefix="/api")


# ---------------------------------------------------------------------------
# CLI: python -m app.main ingest [/data]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        from .ingest import main as run_ingest

        data_root = sys.argv[2] if len(sys.argv) > 2 else "/data"
        run_ingest(data_root)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
