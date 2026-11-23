from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.db import fetch_dicts, write_lock
from api.deps import DB, to_naive_utc
from api.schemas import (
    Anomaly,
    AnomalyPage,
    AnomalySort,
    AnomalyStatus,
    AnomalySummary,
    AnomalyUpdate,
    Direction,
    Utility,
)

router = APIRouter(prefix="/api/anomalies", tags=["anomalies"])

# readings are flagged by the newest thresholds trained before them; a reading with no review yet is open
FROM_FLAGGED = """
    FROM meter_readings m
    JOIN buildings b ON b.id = m.building_id
    LEFT JOIN anomaly_reviews r ON r.meterid = m.meterid AND r.readingtime = m.readingtime
"""
SELECT_ANOMALY = f"""
    WITH cutoffs AS (SELECT DISTINCT train_end, CAST(train_end AS TIMESTAMP) AS cutoff FROM group_thresholds)
    SELECT m.meterid, m.readingtime, m.building_id, b.sitename, m.utility, m.readingunits AS units,
           m.readingvalue AS value, m.severity,
           CASE WHEN m.severity > 0 THEN 'spike' ELSE 'drop' END AS direction,
           t.median, t.lower_bound, t.upper_bound, c.train_end,
           coalesce(r.status, 'open') AS status, r.notes, r.updated_at
    {FROM_FLAGGED}
    ASOF LEFT JOIN cutoffs c ON m.readingtime >= c.cutoff
    LEFT JOIN group_thresholds t
        ON t.train_end = c.train_end AND t.meterid = m.meterid
        AND t.month = month(m.readingtime)
        AND t.is_weekend = CAST(isodow(m.readingtime) >= 6 AS INTEGER)
        AND t.hour_bucket = hour(m.readingtime) // 4
"""
# an infinite severity can't be ranked against the rest and is often a small departure, so it goes last
SORTS = {
    "severity": "isinf(m.severity), abs(m.severity) DESC, m.readingtime DESC, m.meterid",
    "time": "m.readingtime DESC, m.meterid",
}


def anomaly_filters(
    building_id: int | None = None,
    utility: Utility | None = None,
    meter_id: int | None = None,
    status: AnomalyStatus | None = None,
    direction: Direction | None = None,
    min_severity: Annotated[float | None, Query(ge=0, description="Minimum absolute severity; leaves out readings whose severity is null")] = None,
    start: Annotated[datetime | None, Query(description="Inclusive, UTC unless an offset is given")] = None,
    end: Annotated[datetime | None, Query(description="Exclusive, UTC unless an offset is given")] = None,
) -> tuple[str, list]:
    conditions, params = ["m.is_anomalous"], []
    for condition, value in (
        ("m.building_id = ?", building_id),
        ("m.utility = ?", utility),
        ("m.meterid = ?", meter_id),
        ("coalesce(r.status, 'open') = ?", status),
        ("NOT isinf(m.severity) AND abs(m.severity) >= ?", min_severity),
        ("m.readingtime >= ?", to_naive_utc(start) if start else None),
        ("m.readingtime < ?", to_naive_utc(end) if end else None),
    ):
        if value is not None:
            conditions.append(condition)
            params.append(value)
    if direction is not None:
        conditions.append("m.severity > 0" if direction == "spike" else "m.severity < 0")
    return " AND ".join(conditions), params


Filters = Annotated[tuple[str, list], Depends(anomaly_filters)]


def _get_anomaly(db, meter_id: int, reading_time: datetime) -> dict:
    rows = fetch_dicts(
        db,
        f"{SELECT_ANOMALY} WHERE m.is_anomalous AND m.meterid = ? AND m.readingtime = ?",
        [meter_id, to_naive_utc(reading_time)],
    )
    if not rows:
        raise HTTPException(404, f"Meter {meter_id} has no anomaly at {reading_time}")
    return rows[0]


@router.get("", response_model=AnomalyPage)
def list_anomalies(
    db: DB,
    filters: Filters,
    sort: AnomalySort = "severity",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Flagged readings, most severe first by default. Filters combine with AND."""
    where, params = filters
    total = db.execute(f"SELECT count(*) {FROM_FLAGGED} WHERE {where}", params).fetchone()[0]
    items = fetch_dicts(
        db,
        f"{SELECT_ANOMALY} WHERE {where} ORDER BY {SORTS[sort]} LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@router.get("/summary", response_model=AnomalySummary)
def summarize_anomalies(db: DB, filters: Filters):
    """Counts per building, utility and status, for the same filters as the list."""
    where, params = filters
    groups = fetch_dicts(db, f"""
        SELECT m.building_id, b.sitename, m.utility, coalesce(r.status, 'open') AS status, count(*) AS count
        {FROM_FLAGGED}
        WHERE {where}
        GROUP BY ALL ORDER BY m.building_id, m.utility, status
    """, params)
    by_status = {status: 0 for status in AnomalyStatus.__args__}
    for group in groups:
        by_status[group["status"]] += group["count"]
    return {"total": sum(by_status.values()), "by_status": by_status, "groups": groups}


@router.get("/{meter_id}/{reading_time}", response_model=Anomaly)
def read_anomaly(meter_id: int, reading_time: datetime, db: DB):
    return _get_anomaly(db, meter_id, reading_time)


@router.patch("/{meter_id}/{reading_time}", response_model=Anomaly)
def update_anomaly(meter_id: int, reading_time: datetime, update: AnomalyUpdate, db: DB):
    """Changes the review's status and/or notes. Fields left out of the body keep their value."""
    fields = update.model_fields_set
    # read inside the lock too, or two partial updates at once could undo each other's field
    with write_lock:
        current = _get_anomaly(db, meter_id, reading_time)
        db.execute("""
            INSERT INTO anomaly_reviews (meterid, readingtime, status, notes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (meterid, readingtime) DO UPDATE
            SET status = excluded.status, notes = excluded.notes, updated_at = excluded.updated_at
        """, [
            meter_id,
            current["readingtime"],
            update.status if "status" in fields else current["status"],
            update.notes if "notes" in fields else current["notes"],
            datetime.now(UTC).replace(tzinfo=None),
        ])
    return _get_anomaly(db, meter_id, reading_time)
