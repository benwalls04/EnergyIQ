from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import db as db_module
from app.api import routes as routes_module


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("duckdb") / "test.duckdb"


@pytest.fixture(scope="session")
def seeded_db(db_path: Path) -> Path:
    con = duckdb.connect(str(db_path))

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS buildings (
          building_id VARCHAR PRIMARY KEY,
          building_number_stripped VARCHAR,
          name VARCHAR,
          lat DOUBLE,
          lon DOUBLE,
          sqft BIGINT,
          floors INTEGER,
          construction_date DATE,
          campus VARCHAR,
          status VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements_hourly (
          building_id VARCHAR,
          ts TIMESTAMP,
          utility VARCHAR,
          energy_value DOUBLE,
          energy_units VARCHAR,
          occupancy INTEGER,
          PRIMARY KEY (building_id, ts, utility)
        );
        """
    )

    con.execute(
        """
        INSERT INTO buildings
            (building_id, building_number_stripped, name, lat, lon, sqft, floors, construction_date, campus, status)
        VALUES
            ('0001', '1', 'Alpha Hall', 40.0000, -83.0000, 1000, 4, DATE '2000-01-01', 'Main', 'Active'),
            ('0002', '2', 'Beta Hall', 40.0010, -83.0010, 2000, 6, DATE '2005-01-01', 'Main', 'Active');
        """
    )

    con.execute(
        """
        INSERT INTO measurements_hourly
            (building_id, ts, utility, energy_value, energy_units, occupancy)
        VALUES
            ('0001', TIMESTAMP '2025-01-01 05:00:00', 'ELECTRICITY', 100.0, 'kWh', 50),
            ('0001', TIMESTAMP '2025-01-01 06:00:00', 'ELECTRICITY', 120.0, 'kWh', 60),
            ('0002', TIMESTAMP '2025-01-01 05:00:00', 'ELECTRICITY', 300.0, 'kWh', 150),
            ('0002', TIMESTAMP '2025-01-01 05:00:00', 'GAS', 20.0, 'ccf', 150);
        """
    )

    con.close()
    return db_path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, seeded_db: Path) -> TestClient:
    def _get_conn() -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(seeded_db))

    monkeypatch.setattr(db_module, "get_conn", _get_conn)
    monkeypatch.setattr(routes_module, "get_conn", _get_conn)

    with TestClient(app) as test_client:
        yield test_client
