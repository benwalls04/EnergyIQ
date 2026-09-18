import duckdb
from .settings import settings


def get_conn() -> duckdb.DuckDBPyConnection:
    # Single-process dev server: simple connection per request is fine.
    return duckdb.connect(settings.duckdb_path)


def init_db() -> None:
    con = get_conn()

    con.execute("""
      CREATE TABLE IF NOT EXISTS buildings (
        building_id VARCHAR PRIMARY KEY,          -- GeoJSON Building Number (e.g. "0004")
        building_number_stripped VARCHAR,          -- For matching (e.g. "4")
        name VARCHAR,
        lat DOUBLE,
        lon DOUBLE,
        sqft BIGINT,
        floors INTEGER,
        construction_date DATE,
        campus VARCHAR,
        status VARCHAR
      );
    """)

    con.execute("""
      CREATE TABLE IF NOT EXISTS measurements_hourly (
        building_id VARCHAR,
        ts TIMESTAMP,
        utility VARCHAR,          -- ELECTRICITY | STEAM | GAS | HEAT | COOLING
        energy_value DOUBLE,
        energy_units VARCHAR,     -- kWh, MMBtu, etc.
        occupancy INTEGER,
        max_occupancy INTEGER,
        PRIMARY KEY (building_id, ts, utility)
      );
    """)

    # Indexes for fast queries
    # Migrate: add max_occupancy if it was added after initial schema creation
    con.execute("""
      ALTER TABLE measurements_hourly
        ADD COLUMN IF NOT EXISTS max_occupancy INTEGER;
    """)

    con.execute("""
      CREATE INDEX IF NOT EXISTS idx_measurements_ts
        ON measurements_hourly(ts, utility);
    """)
    con.execute("""
      CREATE INDEX IF NOT EXISTS idx_measurements_building
        ON measurements_hourly(building_id, utility);
    """)

    con.execute("""
      CREATE TABLE IF NOT EXISTS weather_hourly (
        ts TIMESTAMP PRIMARY KEY,
        temperature_2m DOUBLE,
        temperature_unit VARCHAR,
        apparent_temperature DOUBLE,
        dew_point_2m DOUBLE,
        relative_humidity_2m DOUBLE,
        precipitation DOUBLE,
        precipitation_unit VARCHAR,
        direct_radiation DOUBLE,
        wind_speed_10m DOUBLE,
        wind_speed_100m DOUBLE,
        wind_speed_unit VARCHAR,
        wind_direction_10m DOUBLE,
        cloud_cover DOUBLE
      );
    """)
    con.execute("""
      CREATE INDEX IF NOT EXISTS idx_weather_ts
        ON weather_hourly(ts);
    """)

    con.close()
