from pathlib import Path
import json

from fastapi import APIRouter, Query
from ..db import get_conn

router = APIRouter()

# ---- Buildings to exclude from all API responses -------------------------
EXCLUDED_BUILDING_IDS = ("069", "079")  # OSU Electric Substation, McCracken Power Plant

# ---- Utility name mapping (frontend → DB) --------------------------------
UTILITY_MAP = {
    "electricity": "ELECTRICITY",
    "steam": "STEAM",
    "gas": "GAS",
    "heat": "HEAT",
    "chilled_water": "COOLING",
}


def _resolve_utility(utility: str) -> str:
    return UTILITY_MAP.get(utility.lower(), utility.upper())


# ---- Unit label computation -----------------------------------------------

def _display_units(metric: str, raw_units: str | None) -> str | None:
    """Return a human-readable unit label based on the selected metric."""
    if metric == "occupancy_intensity":
        return "WiFi clients"
    if raw_units is None:
        return None
    if metric == "energy_per_sqft":
        return f"{raw_units}/sqft"
    if metric == "energy_per_occupant":
        return f"{raw_units}/occupant"
    return raw_units


# ---- Metric computation ---------------------------------------------------

def _compute_metric(metric: str, energy, sqft, occupancy):
    """Compute value for a given Story 2.1 metric."""
    if metric == "energy_intensity":
        return float(energy) if energy is not None else None
    if metric == "occupancy_intensity":
        return int(occupancy) if occupancy is not None else None
    if metric == "energy_per_sqft":
        if energy is None or not sqft:
            return None
        return float(energy) / float(sqft)
    if metric == "energy_per_occupant":
        if energy is None or not occupancy:
            return None
        return float(energy) / float(occupancy)
    # fallback – raw energy
    return float(energy) if energy is not None else None


# ---- Endpoints ------------------------------------------------------------

@router.get("/health")
def health():
    return {"ok": True}


@router.get("/buildings")
def buildings():
    con = get_conn()
    placeholders = ", ".join(["?"] * len(EXCLUDED_BUILDING_IDS))
    rows = con.execute(f"""
      SELECT building_id, name, lat, lon, sqft, floors
      FROM buildings
      WHERE building_id NOT IN ({placeholders})
      ORDER BY name;
    """, list(EXCLUDED_BUILDING_IDS)).fetchall()
    con.close()

    return [
        {
            "building_id": r[0],
            "name": r[1],
            "lat": r[2],
            "lon": r[3],
            "sqft": r[4],
            "floors": r[5],
        }
        for r in rows
    ]


@router.get("/map")
def map_snapshot(
    ts: str = Query(..., description="ISO timestamp"),
    utility: str = Query("electricity"),
    metric: str = Query("energy_intensity"),
):
    """
    Returns one value per building for a given timestamp + utility + metric.
    Metric choices (Story 2.1):
      - energy_intensity
      - occupancy_intensity
      - energy_per_sqft
      - energy_per_occupant
    """
    db_utility = _resolve_utility(utility)

    placeholders = ", ".join(["?"] * len(EXCLUDED_BUILDING_IDS))
    con = get_conn()
    rows = con.execute(
        f"""
        SELECT
          b.building_id,
          b.name,
          b.lat,
          b.lon,
          b.sqft,
          m.energy_value,
          m.energy_units,
          m.occupancy,
          m.max_occupancy
        FROM buildings b
        LEFT JOIN measurements_hourly m
          ON m.building_id = b.building_id
         AND m.ts = CAST(? AS TIMESTAMP)
         AND m.utility = ?
        WHERE b.building_id NOT IN ({placeholders})
        ORDER BY b.name;
        """,
        [ts, db_utility, *EXCLUDED_BUILDING_IDS],
    ).fetchall()
    con.close()

    return [
        {
            "building_id": r[0],
            "name": r[1],
            "lat": r[2],
            "lon": r[3],
            "value": _compute_metric(metric, r[5], r[4], r[7]),
            "units": _display_units(metric, r[6]),
            "max_occupancy": r[8],
        }
        for r in rows
    ]


@router.get("/building/{building_id}/series")
def building_series(
    building_id: str,
    start: str = Query(..., description="ISO timestamp start"),
    end: str = Query(..., description="ISO timestamp end"),
    utility: str = Query("electricity"),
    metric: str = Query("energy_intensity"),
):
    """Time-series data for a single building."""
    db_utility = _resolve_utility(utility)

    con = get_conn()
    rows = con.execute(
        """
        SELECT
          m.ts,
          b.sqft,
          m.energy_value,
          m.occupancy,
          m.energy_units,
          m.max_occupancy
        FROM measurements_hourly m
        JOIN buildings b
          ON b.building_id = m.building_id
        WHERE m.building_id = ?
          AND m.utility = ?
          AND m.ts >= CAST(? AS TIMESTAMP)
          AND m.ts <  CAST(? AS TIMESTAMP)
        ORDER BY m.ts;
        """,
        [building_id, db_utility, start, end],
    ).fetchall()
    con.close()

    return [
        {
            "ts": r[0].isoformat(),
            "value": _compute_metric(metric, r[2], r[1], r[3]),
            "units": _display_units(metric, r[4]),
            "max_occupancy": r[5],
        }
        for r in rows
    ]


@router.get("/timestamps")
def get_available_timestamps(utility: str = Query("electricity")):
    """Return all distinct timestamps available for a utility."""
    db_utility = _resolve_utility(utility)

    con = get_conn()
    rows = con.execute(
        """
        SELECT DISTINCT ts
        FROM measurements_hourly
        WHERE utility = ?
        ORDER BY ts;
        """,
        [db_utility],
    ).fetchall()
    con.close()

    return [r[0].isoformat() for r in rows]


@router.get("/campus/summary")
def campus_summary(
    ts: str = Query(..., description="ISO timestamp"),
    utility: str = Query("electricity"),
    metric: str = Query("energy_intensity"),
):
    """Campus-wide aggregate statistics for a given timestamp + utility + metric."""
    db_utility = _resolve_utility(utility)

    placeholders = ", ".join(["?"] * len(EXCLUDED_BUILDING_IDS))
    con = get_conn()

    # Count buildings that have any data for this utility (not just this timestamp)
    buildings_metered: int = con.execute(
        f"""
        SELECT COUNT(DISTINCT building_id)
        FROM measurements_hourly
        WHERE utility = ?
          AND building_id NOT IN ({placeholders});
        """,
        [db_utility, *EXCLUDED_BUILDING_IDS],
    ).fetchone()[0]

    rows = con.execute(
        f"""
        SELECT
          b.building_id,
          b.name,
          b.sqft,
          m.energy_value,
          m.energy_units,
          m.occupancy
        FROM buildings b
        LEFT JOIN measurements_hourly m
          ON m.building_id = b.building_id
         AND m.ts = CAST(? AS TIMESTAMP)
         AND m.utility = ?
        WHERE b.building_id NOT IN ({placeholders})
        ORDER BY b.name;
        """,
        [ts, db_utility, *EXCLUDED_BUILDING_IDS],
    ).fetchall()
    con.close()

    total_buildings = len(rows)
    named_values: list[tuple[str, float]] = []
    units: str | None = None

    for r in rows:
        building_id, name, sqft, energy_value, energy_units, occupancy = r
        v = _compute_metric(metric, energy_value, sqft, occupancy)
        if v is not None:
            named_values.append((name, v))
            if units is None:
                units = _display_units(metric, energy_units)

    if not named_values:
        return {
            "total_buildings": total_buildings,
            "buildings_metered": buildings_metered,
            "buildings_reporting": 0,
            "avg_value": None,
            "total_value": None,
            "max_value": None,
            "max_building": None,
            "min_value": None,
            "min_building": None,
            "units": units,
        }

    vals = [v for _, v in named_values]
    max_name, max_val = max(named_values, key=lambda x: x[1])
    min_name, min_val = min(named_values, key=lambda x: x[1])

    return {
        "total_buildings": total_buildings,
        "buildings_metered": buildings_metered,
        "buildings_reporting": len(named_values),
        "avg_value": sum(vals) / len(vals),
        "total_value": sum(vals),
        "max_value": max_val,
        "max_building": max_name,
        "min_value": min_val,
        "min_building": min_name,
        "units": units,
    }


@router.get("/buildings/geojson")
def buildings_geojson():
    """Serve the Columbus_Buildings GeoJSON file, excluding filtered buildings."""
    p = Path("/data/Columbus_Buildings.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["features"] = [
        f for f in data.get("features", [])
        if f.get("properties", {}).get("Building Number") not in EXCLUDED_BUILDING_IDS
    ]
    return data


@router.get("/weather")
def get_weather(
    start: str = Query(..., description="ISO timestamp start"),
    end: str = Query(..., description="ISO timestamp end"),
):
    """Return hourly weather observations between start (inclusive) and end (exclusive)."""
    con = get_conn()
    rows = con.execute(
        """
        SELECT
          ts,
          temperature_2m,
          temperature_unit,
          apparent_temperature,
          relative_humidity_2m,
          precipitation,
          direct_radiation,
          cloud_cover,
          wind_speed_10m,
          wind_direction_10m
        FROM weather_hourly
        WHERE ts >= CAST(? AS TIMESTAMP)
          AND ts <  CAST(? AS TIMESTAMP)
        ORDER BY ts;
        """,
        [start, end],
    ).fetchall()
    con.close()

    return [
        {
            "ts": r[0].isoformat(),
            "temperature_2m": r[1],
            "temperature_unit": r[2],
            "apparent_temperature": r[3],
            "relative_humidity_2m": r[4],
            "precipitation": r[5],
            "direct_radiation": r[6],
            "cloud_cover": r[7],
            "wind_speed_10m": r[8],
            "wind_direction_10m": r[9],
        }
        for r in rows
    ]
