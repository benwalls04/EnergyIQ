"""DuckDB access for the API: one long-lived connection, with a cursor per request."""
import math
import threading
from collections.abc import Iterator
from pathlib import Path

import duckdb
from fastapi import Request

REBUILD_HINT = "Stop the API and run `python -m scripts.build_db --replay` from backend/."

CREATE_REVIEWS = """
    CREATE TABLE IF NOT EXISTS anomaly_reviews (
        meterid BIGINT NOT NULL,
        readingtime TIMESTAMP NOT NULL,
        status VARCHAR NOT NULL,
        notes VARCHAR,
        updated_at TIMESTAMP NOT NULL,
        PRIMARY KEY (meterid, readingtime)
    )
"""

# DuckDB fails a transaction that conflicts with another, so writes take turns
write_lock = threading.Lock()


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Opens the database and prepares it, failing with a usable message if it can't serve the API."""
    # duckdb.connect would silently create an empty file
    if not db_path.exists():
        raise RuntimeError(f"No database at {db_path}. {REBUILD_HINT}")
    try:
        conn = duckdb.connect(str(db_path))
    except duckdb.IOException as e:
        raise RuntimeError(f"Can't open {db_path}; close any other tool using it. ({e})") from e

    tables = {name for (name,) in conn.execute("SHOW TABLES").fetchall()}
    columns = {row[0] for row in conn.execute("DESCRIBE meter_readings").fetchall()} \
        if "meter_readings" in tables else set()
    if "group_thresholds" not in tables or not {"is_anomalous", "severity"} <= columns:
        conn.close()
        raise RuntimeError(f"{db_path} was built with an older schema. {REBUILD_HINT}")

    conn.execute(CREATE_REVIEWS)
    return conn


def get_db(request: Request) -> Iterator[duckdb.DuckDBPyConnection]:
    cursor = request.app.state.db.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


def fetch_dicts(cursor: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    """
    Query results as dicts. NaN and +-inf become None since JSON can't carry them; severity is
    inf when a group's bound equals its median.
    """
    result = cursor.execute(sql, params or [])
    names = [d[0] for d in result.description]
    return [
        {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in zip(names, row)}
        for row in result.fetchall()
    ]
