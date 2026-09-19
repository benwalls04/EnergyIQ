"""Loads raw CSVs from data/ into a DuckDB database with three tables:
buildings, meter_readings, and weather.
"""

import pathlib

import duckdb
import pandas as pd

DB_FILENAME = "energyiq.duckdb"

TIME_SEGMENTS = [
    "sept2024-feb2025",
    "march2025-aug2025",
    "sept2025-feb2026",
    "march2026-sept2026",
]


def find_project_root(marker: str = ".git") -> pathlib.Path:
    """Walk upward from the current working directory until a folder containing `marker` is found."""
    current = pathlib.Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find project root (no '{marker}' found in {current} or any parent directory)"
    )


def load_energy_df(project_root: pathlib.Path) -> pd.DataFrame:
    data_path = project_root / "data"

    all_dfs = []
    for subfolder_name in TIME_SEGMENTS:
        subfolder_path = data_path / subfolder_name
        for csv_file in subfolder_path.glob("*.csv"):
            all_dfs.append(pd.read_csv(csv_file))

    energy_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    energy_df["readingtime"] = pd.to_datetime(energy_df["readingtime"])

    # drop rows where reading value is not given
    energy_df = energy_df.dropna(subset=["readingvalue"])
    energy_df = energy_df.sort_values("readingtime")

    before = len(energy_df)
    energy_df = energy_df.drop_duplicates(subset=["meterid", "readingtime"], keep="first")
    if len(energy_df) != before:
        print(f"energy: dropped {before - len(energy_df)} duplicate (meterid, readingtime) rows")

    return energy_df


def load_weather_df(project_root: pathlib.Path) -> pd.DataFrame:
    subfolder_path = project_root / "data" / "weather_data"

    all_dfs = [pd.read_csv(csv_file) for csv_file in subfolder_path.glob("*.csv")]
    weather_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df = weather_df.drop(columns=["partition_0"])
    weather_df = weather_df.sort_values("date")

    before = len(weather_df)
    weather_df = weather_df.drop_duplicates(subset=["date"], keep="first")
    if len(weather_df) != before:
        print(f"weather: dropped {before - len(weather_df)} duplicate 'date' rows")

    return weather_df


def build_database(db_path: pathlib.Path, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> None:
    con = duckdb.connect(str(db_path))

    con.execute("CREATE OR REPLACE SEQUENCE seq_building_id START 1")
    con.execute("""
        CREATE OR REPLACE TABLE buildings (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_building_id'),
            sitename VARCHAR NOT NULL,
            simscode VARCHAR NOT NULL UNIQUE
        )
    """)
    con.execute("""
        INSERT INTO buildings (sitename, simscode)
        SELECT DISTINCT sitename, simscode FROM energy_df
    """)

    con.execute("""
        CREATE OR REPLACE TABLE meter_readings (
            meterid BIGINT NOT NULL,
            building_id INTEGER NOT NULL REFERENCES buildings(id),
            utility VARCHAR NOT NULL,
            readingtime TIMESTAMP NOT NULL,
            readingvalue DOUBLE,
            readingunits VARCHAR,
            PRIMARY KEY (meterid, readingtime)
        )
    """)
    con.execute("""
        INSERT INTO meter_readings (meterid, building_id, utility, readingtime, readingvalue, readingunits)
        SELECT e.meterid, b.id, e.utility, e.readingtime, e.readingvalue, e.readingunits
        FROM energy_df e
        JOIN buildings b ON b.simscode = e.simscode
    """)
    con.execute("""
        CREATE INDEX idx_meter_readings_building_utility_time
        ON meter_readings (building_id, utility, readingtime)
    """)

    con.execute("""
        CREATE OR REPLACE TABLE weather (
            "date" TIMESTAMP PRIMARY KEY,
            latitude DOUBLE,
            longitude DOUBLE,
            temperature_2m DOUBLE,
            shortwave_radiation DOUBLE,
            direct_radiation DOUBLE,
            diffuse_radiation DOUBLE,
            direct_normal_irradiance DOUBLE,
            relative_humidity_2m DOUBLE,
            dew_point_2m DOUBLE,
            precipitation DOUBLE,
            wind_speed_10m DOUBLE,
            wind_speed_100m DOUBLE,
            wind_direction_100m DOUBLE,
            wind_direction_10m DOUBLE,
            cloud_cover DOUBLE,
            apparent_temperature DOUBLE
        )
    """)
    con.execute("INSERT INTO weather SELECT * FROM weather_df")

    for table in ("buildings", "meter_readings", "weather"):
        (count,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"{table}: {count} rows")

    con.close()


def main() -> None:
    project_root = find_project_root()
    print(f"Project root: {project_root}")

    energy_df = load_energy_df(project_root)
    weather_df = load_weather_df(project_root)

    db_path = project_root / "db" / DB_FILENAME

    build_database(db_path, energy_df, weather_df)
    print(f"Wrote {db_path}")


if __name__ == "__main__":
    main()