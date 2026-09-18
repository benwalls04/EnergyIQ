from app.api.routes import _compute_metric, _resolve_utility
from app.ingest import _strip_zeros


def test_compute_metric_energy_intensity() -> None:
    assert _compute_metric("energy_intensity", 123.4, 1000, 40) == 123.4
    assert _compute_metric("energy_intensity", None, 1000, 40) is None


def test_compute_metric_occupancy_intensity() -> None:
    assert _compute_metric("occupancy_intensity", 123.4, 1000, 42) == 42
    assert _compute_metric("occupancy_intensity", 123.4, 1000, None) is None


def test_compute_metric_energy_per_sqft() -> None:
    assert _compute_metric("energy_per_sqft", 100, 200, 20) == 0.5
    assert _compute_metric("energy_per_sqft", 100, 0, 20) is None
    assert _compute_metric("energy_per_sqft", 100, None, 20) is None
    assert _compute_metric("energy_per_sqft", None, 200, 20) is None


def test_compute_metric_energy_per_occupant() -> None:
    assert _compute_metric("energy_per_occupant", 100, 200, 20) == 5.0
    assert _compute_metric("energy_per_occupant", 100, 200, 0) is None
    assert _compute_metric("energy_per_occupant", 100, 200, None) is None
    assert _compute_metric("energy_per_occupant", None, 200, 20) is None


def test_resolve_utility_maps_frontend_values() -> None:
    assert _resolve_utility("electricity") == "ELECTRICITY"
    assert _resolve_utility("steam") == "STEAM"
    assert _resolve_utility("gas") == "GAS"
    assert _resolve_utility("heat") == "HEAT"
    assert _resolve_utility("chilled_water") == "COOLING"


def test_resolve_utility_unknown_uppercases() -> None:
    assert _resolve_utility("foo") == "FOO"


def test_strip_zeros_handles_edge_cases() -> None:
    assert _strip_zeros("0004") == "4"
    assert _strip_zeros("0") == "0"
    assert _strip_zeros("000") == "0"
    assert _strip_zeros("7059") == "7059"
