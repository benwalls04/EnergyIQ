"""Load raw energy and weather CSVs, clean them, and build the DuckDB database."""

import logging
import pathlib

import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from energyiq.config import TIME_SEGMENTS, WEATHER_DIR
from energyiq.models import Base, Building, MeterReading, Weather

logger = logging.getLogger(__name__)


def _read_csvs(paths: list[pathlib.Path], label: str, dtype: dict[str, type] | None = None) -> pd.DataFrame:
    """Read and concatenate CSV files, failing clearly if none were found."""
    if not paths:
        raise FileNotFoundError(f"No {label} CSV files found")
    return pd.concat((pd.read_csv(p, dtype=dtype) for p in paths), ignore_index=True)


def _drop_duplicates(df: pd.DataFrame, subset: list[str], label: str) -> pd.DataFrame:
    """Drop duplicate rows on `subset`, keeping the first, and log how many were removed."""
    deduped = df.drop_duplicates(subset=subset, keep="first")
    dropped = len(df) - len(deduped)
    if dropped:
        logger.info("%s: dropped %d duplicate %s rows", label, dropped, tuple(subset))
    return deduped


def _in_range(df: pd.DataFrame, col: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Rows with `col` in [start, end); a None bound is open."""
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df[col] >= start
    if end is not None:
        keep &= df[col] < end
    return df[keep]


def load_meter_readings_df(data_dir: pathlib.Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    # ISO dates compare correctly as strings, so only overlapping folders are read
    paths = [p for folder, seg_start, seg_end in TIME_SEGMENTS
             if (end is None or seg_start < end) and (start is None or seg_end > start)
             for p in sorted((data_dir / folder).glob("*.csv"))]
    # keep simscode as a string so leading zeros survive ("082", not 82)
    df = _read_csvs(paths, "energy", dtype={"simscode": str})

    df["readingtime"] = pd.to_datetime(df["readingtime"])
    df = _in_range(df, "readingtime", start, end)
    df = df.dropna(subset=["readingvalue"]).sort_values("readingtime")
    return _drop_duplicates(df, ["meterid", "readingtime"], "energy")


def load_weather_df(data_dir: pathlib.Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    # weather files are yearly, not per segment, so read them all and filter on date
    paths = sorted((data_dir / WEATHER_DIR).glob("*.csv"))
    df = _read_csvs(paths, "weather")

    df["date"] = pd.to_datetime(df["date"])
    df = _in_range(df, "date", start, end)
    df = df.drop(columns=["partition_0"], errors="ignore").sort_values("date")
    return _drop_duplicates(df, ["date"], "weather")


def build_database(db_path: pathlib.Path, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict[str, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"duckdb:///{db_path}")
    try:
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()
    return insert_readings(db_path, energy_df, weather_df)


def insert_readings(db_path: pathlib.Path, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict[str, int]:
    engine = create_engine(f"duckdb:///{db_path}")

    try:
        with Session(engine) as session:
            # buildings: small, so go through the ORM. Earlier batches may have added them already
            known = set(session.scalars(select(Building.simscode)))
            buildings = energy_df[["sitename", "simscode"]].drop_duplicates()
            session.add_all(
                Building(sitename=row.sitename, simscode=row.simscode)
                for row in buildings.itertuples(index=False)
                if row.simscode not in known
            )
            session.flush()

            # meter_readings + weather: large, so bulk-load straight from pandas via DuckDB
            duck = session.connection().connection.driver_connection
            duck.register("energy_df", energy_df)
            duck.register("weather_df", weather_df)
            try:
                duck.execute("""
                    INSERT INTO meter_readings
                        (meterid, readingtime, building_id, utility, readingvalue, readingunits)
                    SELECT e.meterid, e.readingtime, b.id, e.utility, e.readingvalue, e.readingunits
                    FROM energy_df e
                    JOIN buildings b ON b.simscode = e.simscode
                """)
                weather_cols = ", ".join(f'"{c.name}"' for c in Weather.__table__.columns)
                duck.execute(f"INSERT INTO weather ({weather_cols}) SELECT {weather_cols} FROM weather_df")
            finally:
                duck.unregister("energy_df")
                duck.unregister("weather_df")

            session.commit()

            return {
                model.__tablename__: session.scalar(select(func.count()).select_from(model))
                for model in (Building, MeterReading, Weather)
            }
    finally:
        engine.dispose()


def insert_anomalies(db_path: pathlib.Path, anomaly_df: pd.DataFrame) -> None:
    engine = create_engine(f"duckdb:///{db_path}")

    try:
        with Session(engine) as session:
            duck = session.connection().connection.driver_connection
            duck.register("anomaly_df", anomaly_df)
            try:
                duck.execute("""
                    INSERT INTO anomalies
                        (meterid, readingtime, building_id, utility, actual, predicted,
                         residual, threshold, severity, segment)
                    SELECT a.meterid, a.readingtime, m.building_id, a.utility, a.actual, a.predicted,
                           a.residual, a.threshold, a.severity, a.segment
                    FROM anomaly_df a
                    JOIN (SELECT DISTINCT meterid, building_id FROM meter_readings) m
                    ON m.meterid = a.meterid
                """)
            finally:
                duck.unregister("anomaly_df")

            session.commit()
    finally:
        engine.dispose()
