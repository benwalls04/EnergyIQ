# EnergyIQ API

A FastAPI service that serves the database to the dashboard. It reads the data that `build_db --replay` creates and computes predictions on request. It doesn't train models or flag anomalies.

## Run it

From `backend/`:

```bash
python -m scripts.build_db --replay   # once, or after the data/model changes
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs to see every endpoint and try them out.

- **One process at a time:** DuckDB lets only one process use the database file. Stop the API before rebuilding, and close the VS Code DuckDB extension before starting it.
- **Use `--replay`:** a plain build won't work. The API needs the anomaly and threshold tables that only `--replay` creates.
- **Times are UTC:** times without an offset are read as UTC, and every response is in UTC.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | Reading and flagged counts, models loaded |
| `GET /api/buildings` | Buildings and their meters |
| `GET /api/buildings/{id}` | One building and its meters |
| `GET /api/buildings/{id}/consumption` | Metered usage per interval (needs `start`, `end`) |
| `GET /api/buildings/{id}/predictions` | Actual vs. predicted usage (from 2025-09-01 on) |
| `GET /api/anomalies` | Flagged readings, filterable and paged |
| `GET /api/anomalies/summary` | Anomaly counts by building, utility and status |
| `GET /api/anomalies/{meter_id}/{reading_time}` | One flagged reading with its bounds |
| `PATCH /api/anomalies/{meter_id}/{reading_time}` | Update its review `status` and/or `notes` |
| `GET /api/weather` | Campus weather per hour, day or month |
| `GET /api/models` | Replay models and the period each predicts |
| `GET /api/models/performance` | Accuracy of each model (slow on the first call, then cached) |

`interval=15min` is limited to 93 days and `interval=hour` to 2 years.

## Files

| File | Purpose |
|---|---|
| `main.py` | Creates the app: opens the DB, loads models, adds routers and CORS |
| `db.py` | DuckDB connection, startup checks, the `anomaly_reviews` table, `fetch_dicts` |
| `deps.py` | Shared helpers: DB cursor, time range checks, building lookup, readings filter |
| `schemas.py` | Request/response shapes and allowed values (utility, interval, status) |
| `predictions.py` | `ModelStore`: loads replay models, predicts usage, scores accuracy |
| `routers/` | One file per group of endpoints above |

## Settings

These environment variables are optional:

- `ENERGYIQ_DB_PATH`: the database file (default `data/db/energyiq.duckdb`)
- `ENERGYIQ_MODELS_DIR`: the folder of replay models (default `energyiq/ml/models/replay`)
- `ENERGYIQ_CORS_ORIGINS`: comma-separated list of frontend URLs allowed to call the API (default `localhost:3000` and `localhost:5173`)