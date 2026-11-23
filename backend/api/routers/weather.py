from typing import Annotated

from fastapi import APIRouter, Depends

from api.db import fetch_dicts
from api.deps import DB, TimeRange, check_range_size, required_range
from api.schemas import Weather, WeatherInterval

router = APIRouter(prefix="/api/weather", tags=["weather"])

AVERAGED = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
            "wind_speed_10m", "wind_speed_100m", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "direct_normal_irradiance"]
WIND_DIRECTIONS = ["wind_direction_10m", "wind_direction_100m"]


@router.get("", response_model=Weather)
def read_weather(
    db: DB,
    time_range: Annotated[TimeRange, Depends(required_range)],
    interval: WeatherInterval = "hour",
):
    """
    Campus weather. Day and month points average each measure, except precipitation, which is
    summed, and wind direction, which is null because averaging compass bearings is meaningless.
    """
    check_range_size(time_range, interval)
    params = [time_range.start, time_range.end]
    if interval == "hour":
        columns = ", ".join(["precipitation", *AVERAGED, *WIND_DIRECTIONS])
        points = fetch_dicts(db, f"""
            SELECT date AS time, {columns} FROM weather
            WHERE date >= ? AND date < ? ORDER BY date
        """, params)
    else:
        averages = ", ".join(f"avg({column}) AS {column}" for column in AVERAGED)
        directions = ", ".join(f"NULL AS {column}" for column in WIND_DIRECTIONS)
        points = fetch_dicts(db, f"""
            SELECT date_trunc('{interval}', date) AS time, sum(precipitation) AS precipitation, {averages}, {directions}
            FROM weather
            WHERE date >= ? AND date < ?
            GROUP BY 1 ORDER BY 1
        """, params)
    return {"interval": interval, "points": points}
