from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import fetch_dicts
from api.deps import DB, TimeRange, check_range_size, get_building, reading_filters, required_range
from api.predictions import READING_COLUMNS
from api.schemas import Building, Consumption, Interval, Predictions, Utility

router = APIRouter(prefix="/api/buildings", tags=["buildings"])

BUCKET_SQL = {
    "15min": "time_bucket(INTERVAL 15 MINUTE, m.readingtime)",
    "hour": "date_trunc('hour', m.readingtime)",
    "day": "date_trunc('day', m.readingtime)",
    "month": "date_trunc('month', m.readingtime)",
}
PANDAS_FLOOR = {"15min": "15min", "hour": "h", "day": "D"}

ExcludeFlagged = Annotated[bool, Query(
    description="Leave out flagged readings. Some meters report glitches around 1e9 kWh, which swamp totals")]


def _meters(db, building_id: int) -> list[dict]:
    return fetch_dicts(db, """
        SELECT meterid, utility, any_value(readingunits) AS units, min(readingtime) AS first_reading,
               max(readingtime) AS last_reading, count(*) AS n_readings
        FROM meter_readings WHERE building_id = ?
        GROUP BY meterid, utility ORDER BY utility, meterid
    """, [building_id])


@router.get("", response_model=list[Building])
def list_buildings(db: DB):
    """Every building with its meters and the dates each has data for."""
    buildings = fetch_dicts(db, "SELECT id, sitename, simscode FROM buildings ORDER BY id")
    return [{**building, "meters": _meters(db, building["id"])} for building in buildings]


@router.get("/{building_id}", response_model=Building)
def read_building(building_id: int, db: DB):
    return {**get_building(db, building_id), "meters": _meters(db, building_id)}


@router.get("/{building_id}/consumption", response_model=Consumption)
def read_consumption(
    building_id: int,
    db: DB,
    time_range: Annotated[TimeRange, Depends(required_range)],
    utility: Utility | None = None,
    meter_id: int | None = None,
    interval: Interval = "hour",
    exclude_flagged: ExcludeFlagged = False,
):
    """Metered usage summed across the building's meters, one series per utility."""
    building = get_building(db, building_id)
    check_range_size(time_range, interval)
    where, params = reading_filters(building_id, time_range, utility, meter_id, exclude_flagged)
    points = fetch_dicts(db, f"""
        SELECT {BUCKET_SQL[interval]} AS time, m.utility, m.readingunits AS units, sum(m.readingvalue) AS value,
               count(*) AS n_readings, count(*) FILTER (WHERE m.is_anomalous) AS n_flagged
        FROM meter_readings m
        WHERE {where}
        GROUP BY ALL ORDER BY time, m.utility
    """, params)
    return {"building_id": building_id, "sitename": building["sitename"], "interval": interval, "points": points}


@router.get("/{building_id}/predictions", response_model=Predictions)
def read_predictions(
    building_id: int,
    request: Request,
    db: DB,
    time_range: Annotated[TimeRange, Depends(required_range)],
    utility: Utility | None = None,
    meter_id: int | None = None,
    interval: Interval = "hour",
    exclude_flagged: ExcludeFlagged = False,
):
    """
    Actual against predicted usage, summed like /consumption. Predictions start at the first
    replay model's train_end; there's no out-of-sample model for anything earlier.
    """
    building = get_building(db, building_id)
    check_range_size(time_range, interval)
    store = request.app.state.models
    if not store.bundles:
        raise HTTPException(503, "No replay models found. Run `python -m scripts.build_db --replay`.")

    time_range = TimeRange(max(time_range.start, store.first_prediction_time.to_pydatetime()), time_range.end)
    points = []
    if time_range.start < time_range.end:
        where, params = reading_filters(building_id, time_range, utility, meter_id, exclude_flagged)
        readings = db.execute(f"""
            SELECT {READING_COLUMNS}
            FROM meter_readings m JOIN buildings b ON b.id = m.building_id
            WHERE {where}
        """, params).fetchdf()
        if len(readings):
            points = _aggregate(store.predict(readings), interval)
    return {"building_id": building_id, "sitename": building["sitename"], "interval": interval, "points": points}


def _aggregate(scored: pd.DataFrame, interval: Interval) -> list[dict]:
    scored = scored.dropna(subset=["predicted"])
    if interval == "month":
        time = scored["readingtime"].dt.to_period("M").dt.start_time
    else:
        time = scored["readingtime"].dt.floor(PANDAS_FLOOR[interval])
    grouped = scored.assign(time=time).groupby(["time", "utility", "readingunits"], as_index=False, dropna=False).agg(
        actual=("readingvalue", "sum"),
        predicted=("predicted", "sum"),
        n_readings=("readingvalue", "size"),
        n_flagged=("is_anomalous", "sum"),
    ).rename(columns={"readingunits": "units"})
    grouped["residual"] = grouped["actual"] - grouped["predicted"]
    return grouped.astype(object).where(grouped.notna(), None).to_dict("records")
