"""Query parameters and lookups shared by the routers."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Query

from api.db import get_db
from api.schemas import Interval, Utility

DB = Annotated[object, Depends(get_db)]

# keeps a single response to a size a dashboard can draw; day and month are always small enough
MAX_RANGE = {"15min": timedelta(days=93), "hour": timedelta(days=732)}


def to_naive_utc(value: datetime) -> datetime:
    """The database's timestamps are naive UTC, so a time given with an offset is converted first."""
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


@dataclass
class TimeRange:
    start: datetime
    end: datetime


def required_range(
    start: Annotated[datetime, Query(description="Inclusive, UTC unless an offset is given")],
    end: Annotated[datetime, Query(description="Exclusive, UTC unless an offset is given")],
) -> TimeRange:
    time_range = TimeRange(to_naive_utc(start), to_naive_utc(end))
    if time_range.start >= time_range.end:
        raise HTTPException(422, "start must be before end")
    return time_range


def check_range_size(time_range: TimeRange, interval: Interval | str) -> None:
    limit = MAX_RANGE.get(interval)
    if limit is not None and time_range.end - time_range.start > limit:
        raise HTTPException(422, f"interval={interval} is limited to {limit.days} days; use a coarser interval")


def get_building(db, building_id: int) -> dict:
    row = db.execute("SELECT id, sitename, simscode FROM buildings WHERE id = ?", [building_id]).fetchone()
    if row is None:
        raise HTTPException(404, f"No building with id {building_id}")
    return {"id": row[0], "sitename": row[1], "simscode": row[2]}


def reading_filters(
    building_id: int, time_range: TimeRange, utility: Utility | None, meter_id: int | None, exclude_flagged: bool,
) -> tuple[str, list]:
    """WHERE clause and parameters for one building's readings, on the alias m."""
    conditions = ["m.building_id = ?", "m.readingtime >= ?", "m.readingtime < ?"]
    params = [building_id, time_range.start, time_range.end]
    if utility is not None:
        conditions.append("m.utility = ?")
        params.append(utility)
    if meter_id is not None:
        conditions.append("m.meterid = ?")
        params.append(meter_id)
    if exclude_flagged:
        conditions.append("NOT m.is_anomalous")
    return " AND ".join(conditions), params
