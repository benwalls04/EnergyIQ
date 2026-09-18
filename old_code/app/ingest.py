"""
Data ingestion pipeline – DuckDB-native CSV loading (fast).

Loads CSV files exported from AWS Athena into DuckDB for serving via API.

Usage (inside container):
    python -m app.ingest            # default /data root
    python -m app.ingest /path/to   # custom data root
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

from .db import get_conn, init_db
from .settings import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_zeros(s: str) -> str:
    return s.lstrip("0") or "0"


# Buildings to exclude from ingestion and all API responses (power plants, substations)
EXCLUDED_BUILDING_IDS = {"069", "079"}


# ---------------------------------------------------------------------------
# Phase 1 – Select curated buildings (pure-Python, only touches GeoJSON + 1 CSV header scan)
# ---------------------------------------------------------------------------

def select_curated_buildings(
    con: duckdb.DuckDBPyConnection,
    geojson_path: str,
    electricity_csv: str,
) -> dict[str, str]:
    """Return mapping *stripped_simscode → geojson_building_number* for ALL
    buildings with both GeoJSON geometry and electricity data, ranked by row count.
    Excludes power plants and substations defined in EXCLUDED_BUILDING_IDS.
    """

    # --- GeoJSON building numbers ----------------------------------------
    with open(geojson_path, encoding="utf-8") as fh:
        geojson = json.load(fh)

    geojson_map: dict[str, str] = {}  # stripped → original
    for feat in geojson.get("features", []):
        bnum = feat.get("properties", {}).get("Building Number")
        if bnum and bnum not in ("Never", "None", "Pending") and bnum not in EXCLUDED_BUILDING_IDS:
            geojson_map[_strip_zeros(bnum)] = bnum

    # --- Use DuckDB to count simscodes quickly ---------------------------
    rows = con.execute(f"""
        SELECT simscode, COUNT(*) AS cnt
        FROM read_csv_auto('{electricity_csv}', header=true, all_varchar=true)
        GROUP BY simscode
        ORDER BY cnt DESC;
    """).fetchall()

    matches: list[tuple[str, str, int]] = []
    for simscode, cnt in rows:
        stripped = _strip_zeros(simscode)
        if stripped in geojson_map:
            matches.append((stripped, geojson_map[stripped], cnt))

    # All matched buildings, sorted by data completeness (descending)
    matches.sort(key=lambda x: x[2], reverse=True)
    curated: dict[str, str] = {}
    for stripped, building_id, _cnt in matches:
        curated[stripped] = building_id

    return curated


# ---------------------------------------------------------------------------
# Phase 2 – Ingest building metadata via DuckDB
# ---------------------------------------------------------------------------

def ingest_buildings(
    con: duckdb.DuckDBPyConnection,
    sims_csv: str,
    curated: dict[str, str],
) -> int:
    """Load Building_sims.csv into a temp table, then INSERT curated rows."""

    # Build a mapping table in DuckDB
    mapping_rows = [(stripped, building_id) for stripped, building_id in curated.items()]
    con.execute("CREATE OR REPLACE TEMP TABLE curated_map (stripped VARCHAR, building_id VARCHAR);")
    con.executemany("INSERT INTO curated_map VALUES (?, ?);", mapping_rows)

    # Load raw CSV into temp table
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE raw_sims AS
        SELECT * FROM read_csv_auto('{sims_csv}', header=true, all_varchar=true);
    """)

    # Insert matched buildings
    con.execute("""
        INSERT INTO buildings
            (building_id, building_number_stripped, name, lat, lon,
             sqft, floors, construction_date, campus, status)
        SELECT
            cm.building_id,
            cm.stripped,
            rs.buildingname,
            TRY_CAST(rs.latitude AS DOUBLE),
            TRY_CAST(rs.longitude AS DOUBLE),
            TRY_CAST(rs.grossarea AS BIGINT),
            TRY_CAST(rs.floorsaboveground AS INTEGER),
            TRY_CAST(rs.constructiondate AS DATE),
            rs.campusname,
            rs.status
        FROM raw_sims rs
        JOIN curated_map cm
          ON cm.stripped = LTRIM(rs.buildingnumber, '0')
        ON CONFLICT DO NOTHING;
    """)

    # Handle edge case: buildingnumber = '000...'
    # Already covered by _strip_zeros logic above

    count = con.execute("SELECT COUNT(*) FROM buildings;").fetchone()[0]

    con.execute("DROP TABLE IF EXISTS raw_sims;")
    con.execute("DROP TABLE IF EXISTS curated_map;")

    return count


# ---------------------------------------------------------------------------
# Phase 3 – Ingest utility CSVs using DuckDB native CSV reader
# ---------------------------------------------------------------------------

def ingest_utility_csv(
    con: duckdb.DuckDBPyConnection,
    csv_path: str,
    utility_name: str,
    curated: dict[str, str],
) -> int:
    """Load one utility CSV using DuckDB's read_csv_auto with a JOIN filter."""

    # Ensure curated_map exists
    _ensure_curated_map(con, curated)

    inserted = con.execute(f"""
        INSERT INTO measurements_hourly
            (building_id, ts, utility, energy_value, energy_units)
        SELECT
            cm.building_id,
            CAST(REPLACE(raw.hour_ts, ' UTC', '') AS TIMESTAMP),
            '{utility_name}',
            TRY_CAST(raw.hourly_value AS DOUBLE),
            raw.readingunits
        FROM read_csv_auto('{csv_path}', header=true, all_varchar=true) raw
        JOIN curated_map cm
          ON cm.stripped = LTRIM(raw.simscode, '0')
        ON CONFLICT DO NOTHING;
    """).fetchone()

    return con.execute(f"""
        SELECT COUNT(*) FROM measurements_hourly WHERE utility = '{utility_name}';
    """).fetchone()[0]


def _ensure_curated_map(con: duckdb.DuckDBPyConnection, curated: dict[str, str]) -> None:
    """Create the curated_map temp table if it doesn't already exist."""
    try:
        con.execute("SELECT 1 FROM curated_map LIMIT 1;")
    except duckdb.CatalogException:
        mapping_rows = [(stripped, building_id) for stripped, building_id in curated.items()]
        con.execute("CREATE TEMP TABLE curated_map (stripped VARCHAR, building_id VARCHAR);")
        con.executemany("INSERT INTO curated_map VALUES (?, ?);", mapping_rows)


# ---------------------------------------------------------------------------
# Phase 4 – Ingest occupancy (WIFI.csv)
# ---------------------------------------------------------------------------

def ingest_occupancy(
    con: duckdb.DuckDBPyConnection,
    wifi_csv: str,
    curated: dict[str, str],
) -> int:
    """Load WIFI daily occupancy and replicate to each hour.

    Strategy: For each WIFI daily row, UPDATE the occupancy column on all
    measurements_hourly rows for that building+day. Rows without a matching
    measurement are ignored (occupancy-only buildings don't have energy data).
    """

    _ensure_curated_map(con, curated)

    # Load WIFI into temp
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE raw_wifi AS
        SELECT
            cm.building_id,
            CAST(REPLACE(w.day_ts, ' UTC', '') AS TIMESTAMP) AS day_ts,
            TRY_CAST(w.avg_clients AS INTEGER) AS avg_clients,
            TRY_CAST(w.max_clients AS INTEGER) AS max_clients
        FROM read_csv_auto('{wifi_csv}', header=true, all_varchar=true) w
        JOIN curated_map cm
          ON cm.stripped = LTRIM(w.building_number, '0');
    """)

    # Generate hourly timestamps from daily and update matching measurements
    con.execute("""
        UPDATE measurements_hourly m
        SET occupancy = w.avg_clients,
            max_occupancy = w.max_clients
        FROM raw_wifi w
        WHERE m.building_id = w.building_id
          AND m.ts >= w.day_ts
          AND m.ts <  w.day_ts + INTERVAL 1 DAY;
    """)

    occ_count = con.execute(
        "SELECT COUNT(*) FROM measurements_hourly WHERE occupancy IS NOT NULL;"
    ).fetchone()[0]

    con.execute("DROP TABLE IF EXISTS raw_wifi;")
    return occ_count


# ---------------------------------------------------------------------------
# Phase 4b – Outlier filtering (IQR-based, per building per utility)
# ---------------------------------------------------------------------------

def filter_outliers(con: duckdb.DuckDBPyConnection) -> int:
    """NULL out extreme outlier energy readings using IQR per (building_id, utility).

    A value is considered an outlier if it exceeds Q3 + 3 * IQR for its
    building+utility combination.  Uses a single DuckDB window-function UPDATE
    so no row-by-row Python iteration is required.

    Returns the number of rows that were nulled out.
    """

    # Count rows before so we can report how many were nulled
    before = con.execute(
        "SELECT COUNT(*) FROM measurements_hourly WHERE energy_value IS NOT NULL;"
    ).fetchone()[0]

    con.execute("""
        UPDATE measurements_hourly
        SET energy_value = NULL
        WHERE energy_value IS NOT NULL
          AND energy_value > (
            SELECT q3 + 3.0 * iqr
            FROM (
              SELECT
                building_id,
                utility,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY energy_value) AS q3,
                (  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY energy_value)
                 - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY energy_value)
                ) AS iqr
              FROM measurements_hourly
              WHERE energy_value IS NOT NULL
              GROUP BY building_id, utility
            ) stats
            WHERE stats.building_id = measurements_hourly.building_id
              AND stats.utility     = measurements_hourly.utility
          );
    """)

    after = con.execute(
        "SELECT COUNT(*) FROM measurements_hourly WHERE energy_value IS NOT NULL;"
    ).fetchone()[0]

    return before - after


# ---------------------------------------------------------------------------
# Phase 5 – Ingest weather (Weather.csv)
# ---------------------------------------------------------------------------

def ingest_weather(
    con: duckdb.DuckDBPyConnection,
    weather_csv: str,
) -> int:
    """Load Weather.csv into weather_hourly table."""
    con.execute(f"""
        INSERT INTO weather_hourly
            (ts, temperature_2m, temperature_unit, apparent_temperature,
             dew_point_2m, relative_humidity_2m, precipitation, precipitation_unit,
             direct_radiation, wind_speed_10m, wind_speed_100m, wind_speed_unit,
             wind_direction_10m, cloud_cover)
        SELECT
            CAST(REPLACE(w.ts, ' UTC', '') AS TIMESTAMP),
            TRY_CAST(w.temperature_2m AS DOUBLE),
            w.temperature_unit,
            TRY_CAST(w.apparent_temperature AS DOUBLE),
            TRY_CAST(w.dew_point_2m AS DOUBLE),
            TRY_CAST(w.relative_humidity_2m AS DOUBLE),
            TRY_CAST(w.precipitation AS DOUBLE),
            w.precipitation_unit,
            TRY_CAST(w.direct_radiation AS DOUBLE),
            TRY_CAST(w.wind_speed_10m AS DOUBLE),
            TRY_CAST(w.wind_speed_100m AS DOUBLE),
            w.wind_speed_unit,
            TRY_CAST(w.wind_direction_10m AS DOUBLE),
            TRY_CAST(w.cloud_cover AS DOUBLE)
        FROM read_csv_auto('{weather_csv}', header=true, all_varchar=true) w
        ON CONFLICT DO NOTHING;
    """)
    return con.execute("SELECT COUNT(*) FROM weather_hourly;").fetchone()[0]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_ingestion_summary(con: duckdb.DuckDBPyConnection) -> None:
    bcount = con.execute("SELECT COUNT(*) FROM buildings;").fetchone()[0]
    mcount = con.execute("SELECT COUNT(*) FROM measurements_hourly;").fetchone()[0]
    utilities = con.execute(
        "SELECT utility, COUNT(*) FROM measurements_hourly GROUP BY utility ORDER BY utility;"
    ).fetchall()
    ts_range = con.execute(
        "SELECT MIN(ts), MAX(ts) FROM measurements_hourly;"
    ).fetchone()
    occ_count = con.execute(
        "SELECT COUNT(*) FROM measurements_hourly WHERE occupancy IS NOT NULL;"
    ).fetchone()[0]
    wcount = con.execute("SELECT COUNT(*) FROM weather_hourly;").fetchone()[0]

    print("\n=== Ingestion Summary ===")
    print(f"  Buildings:            {bcount}")
    print(f"  Measurement rows:     {mcount:,}")
    print(f"  Rows with occupancy:  {occ_count:,}")
    print(f"  Weather rows:         {wcount:,}")
    if ts_range and ts_range[0]:
        print(f"  Timestamp range:      {ts_range[0]} → {ts_range[1]}")
    print("  Rows per utility:")
    for util, cnt in utilities:
        print(f"    {util:20s} {cnt:>10,}")
    print("=========================\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(data_root: str = "/data") -> None:
    root = Path(data_root)
    geojson_path = str(root / "Columbus_Buildings.json")
    sims_csv = str(root / "Building_sims.csv")
    electricity_csv = str(root / "Electricity.csv")
    wifi_csv = str(root / "WIFI.csv")
    weather_csv = str(root / "Weather.csv")

    utility_files = [
        (str(root / "Electricity.csv"), "ELECTRICITY"),
        (str(root / "Steam.csv"), "STEAM"),
        (str(root / "Gas.csv"), "GAS"),
        (str(root / "Heat.csv"), "HEAT"),
        (str(root / "Cooling.csv"), "COOLING"),
    ]

    # ---- Ensure tables exist --------------------------------------------
    print("Initializing database schema …")
    init_db()

    con = get_conn()

    # Drop old data so ingestion is idempotent
    con.execute("DELETE FROM measurements_hourly;")
    con.execute("DELETE FROM buildings;")
    con.execute("DELETE FROM weather_hourly;")

    # ---- Step 1: Select curated buildings --------------------------------
    print("Selecting curated buildings …")
    curated = select_curated_buildings(con, geojson_path, electricity_csv)
    print(f"  Selected {len(curated)} buildings")

    if not curated:
        print("ERROR: No buildings matched. Check GeoJSON Building Numbers vs. CSV simscodes.")
        con.close()
        return

    # ---- Step 2: Ingest building metadata --------------------------------
    print("Ingesting building metadata …")
    bcount = ingest_buildings(con, sims_csv, curated)
    print(f"  Inserted {bcount} building records")

    # ---- Step 3: Ingest utility CSVs ------------------------------------
    for csv_path, utility_name in utility_files:
        p = Path(csv_path)
        if not p.exists():
            print(f"  SKIP {utility_name} – file not found: {csv_path}")
            continue
        print(f"Ingesting {utility_name} …")
        row_count = ingest_utility_csv(con, csv_path, utility_name, curated)
        print(f"  {utility_name}: {row_count:,} rows in DB")

    # ---- Step 3b: Filter outliers ----------------------------------------
    print("Filtering outliers (IQR per building per utility) …")
    nulled = filter_outliers(con)
    print(f"  Nulled {nulled:,} outlier readings")

    # ---- Step 4: Ingest occupancy ----------------------------------------
    if Path(wifi_csv).exists():
        print("Ingesting occupancy (WIFI) …")
        occ_count = ingest_occupancy(con, wifi_csv, curated)
        print(f"  Rows with occupancy: {occ_count:,}")
    else:
        print(f"  SKIP occupancy – file not found: {wifi_csv}")

    # ---- Step 5: Ingest weather ------------------------------------------
    if Path(weather_csv).exists():
        print("Ingesting weather …")
        wcount = ingest_weather(con, weather_csv)
        print(f"  Weather rows: {wcount:,}")
    else:
        print(f"  SKIP weather – file not found: {weather_csv}")

    # ---- Step 6: Summary -------------------------------------------------
    print_ingestion_summary(con)
    con.close()
    print("Done!")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/data"
    main(data_root)
