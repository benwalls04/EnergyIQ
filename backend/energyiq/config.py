from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "db" / "energyiq.duckdb"

MODELS_DIR = Path(__file__).resolve().parent / "ml" / "models"
EXPERIMENT_MODELS_DIR = MODELS_DIR / "experiments" 
REPLAY_MODELS_DIR = MODELS_DIR / "replay" 

UTILITIES = ["heat", "electricity"]

DEFAULT_FEATURES = {
    "electricity": {"inc_lags", "inc_breaks", "inc_weekdays", "inc_events"},
    "heat": {"inc_lags", "inc_temp_history", "inc_usage_history", "inc_seasons"},
}

HGB_DEFAULTS = {
    "random_state": 42,
    "loss": "absolute_error",
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "early_stopping": False, 
}

ANOMALY_K = 8

EXPERIMENT_TRAIN_END = "2025-09-01"
REPLAY_INIT_TRAIN_END = "2025-09-01"

TIME_SEGMENTS = [
    ("sep2023-feb2024", "2023-09-01", "2024-03-01"),
    ("mar2024-aug2024", "2024-03-01", "2024-09-01"),
    ("sept2024-feb2025", "2024-09-01", "2025-03-01"),
    ("march2025-aug2025", "2025-03-01", "2025-09-01"),
    ("sept2025-feb2026", "2025-09-01", "2026-03-01"),
    ("march2026-sept2026", "2026-03-01", "2026-09-01"),
]
WEATHER_DIR = "weather_data"
