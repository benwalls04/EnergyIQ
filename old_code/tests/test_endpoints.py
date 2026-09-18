from pathlib import Path

import pytest


def test_health(client) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_buildings_shape(client) -> None:
    res = client.get("/api/buildings")
    assert res.status_code == 200

    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 2

    first = data[0]
    expected_keys = {"building_id", "name", "lat", "lon", "sqft", "floors"}
    assert expected_keys.issubset(first.keys())


def test_buildings_geojson(client) -> None:
    if not Path("/data/Columbus_Buildings.json").exists():
        pytest.xfail("GeoJSON fixture not available at /data/Columbus_Buildings.json")

    res = client.get("/api/buildings/geojson")
    assert res.status_code == 200
    payload = res.json()
    assert isinstance(payload, dict)
    assert "features" in payload


def test_timestamps_per_utility(client) -> None:
    electricity = client.get("/api/timestamps", params={"utility": "electricity"})
    assert electricity.status_code == 200
    e_data = electricity.json()
    assert e_data == sorted(e_data)
    assert e_data == ["2025-01-01T05:00:00", "2025-01-01T06:00:00"]

    gas = client.get("/api/timestamps", params={"utility": "gas"})
    assert gas.status_code == 200
    assert gas.json() == ["2025-01-01T05:00:00"]


def test_map_energy_intensity(client) -> None:
    res = client.get(
        "/api/map",
        params={
            "ts": "2025-01-01T05:00:00",
            "utility": "electricity",
            "metric": "energy_intensity",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 2
    assert all({"building_id", "name", "lat", "lon", "value", "units"}.issubset(d.keys()) for d in data)

    by_id = {d["building_id"]: d for d in data}
    assert isinstance(by_id["0001"]["value"], float)
    assert by_id["0001"]["value"] == 100.0
    assert by_id["0002"]["value"] == 300.0


def test_map_energy_per_sqft(client) -> None:
    res = client.get(
        "/api/map",
        params={
            "ts": "2025-01-01T05:00:00",
            "utility": "electricity",
            "metric": "energy_per_sqft",
        },
    )
    assert res.status_code == 200
    by_id = {d["building_id"]: d for d in res.json()}

    assert by_id["0001"]["value"] == 0.1
    assert by_id["0002"]["value"] == 0.15


def test_map_missing_ts_returns_422(client) -> None:
    res = client.get("/api/map", params={"utility": "electricity", "metric": "energy_intensity"})
    assert res.status_code == 422


def test_building_series_happy_path(client) -> None:
    res = client.get(
        "/api/building/0001/series",
        params={
            "start": "2025-01-01T05:00:00",
            "end": "2025-01-01T07:00:00",
            "utility": "electricity",
            "metric": "energy_intensity",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert [row["ts"] for row in data] == ["2025-01-01T05:00:00", "2025-01-01T06:00:00"]
    assert [row["value"] for row in data] == [100.0, 120.0]


def test_building_series_unknown_building_returns_empty(client) -> None:
    res = client.get(
        "/api/building/9999/series",
        params={
            "start": "2025-01-01T05:00:00",
            "end": "2025-01-01T07:00:00",
            "utility": "electricity",
            "metric": "energy_intensity",
        },
    )
    assert res.status_code == 200
    assert res.json() == []


def test_building_series_start_after_end_returns_empty(client) -> None:
    res = client.get(
        "/api/building/0001/series",
        params={
            "start": "2025-01-01T07:00:00",
            "end": "2025-01-01T05:00:00",
            "utility": "electricity",
            "metric": "energy_intensity",
        },
    )
    assert res.status_code == 200
    assert res.json() == []
