# scripts/build_db.py
from energyiq import config
from energyiq.ingest import load_meter_readings_df, load_weather_df, build_database
from energyiq.ml.train import train, preprocess
from energyiq import anomaly_detector
import argparse

# train: Sep 2023 - Aug 2025, test/replay: Sep 2025 - Aug 2026 (one year)
TRAIN_END = "2025-09-01"

TIME_SEGMENTS = [
    "sep2023-feb2024",
    "mar2024-aug2024",
    "sept2024-feb2025",
    "march2025-aug2025",
    "sept2025-feb2026",
    "march2026-sept2026",
]
WEATHER_DIR = "weather_data"

def build_full() -> None:
    energy_df = load_meter_readings_df(config.DATA_DIR)
    weather_df = load_weather_df(config.DATA_DIR)
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_database(config.DB_PATH, energy_df, weather_df)
    print(f"Wrote {config.DB_PATH}")

def build_with_replays() -> None: 
    # train first iteration, default params from config
    model, _, _ = train()
    detecor = AnomalyDetector()

    # for t in time segment, get replays, then retrain
    for t in TIME_SEGMENTS[2:]:
        X, Y, _ = preprocess(t)
        Y_pred = model.predict(X)

        #

        load_meter_readings_df([t])
        load_weather_df([t])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for a building")
    parser.add_argument(
        "--replay", 
        action="store_true",
        help="Build the database with replays instead of one by one"
    )
    if parser.replay:
        build_with_replays()
    else: 
        build_full()