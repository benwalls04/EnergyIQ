# scripts/build_db.py
from energyiq.config import DB_PATH, DATA_DIR, TIME_SEGMENTS, UTILITIES, REPLAY_INIT_TRAIN_END, REPLAY_MODELS_DIR
from energyiq.ingest import load_meter_readings_df, load_weather_df, build_database, insert_readings, insert_anomalies
from energyiq.ml.train import train_replay, save_replay_model
from energyiq.AnomalyDetector import AnomalyDetector
import argparse

def build_full() -> None:
    energy_df = load_meter_readings_df(DATA_DIR)
    weather_df = load_weather_df(DATA_DIR)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_database(DB_PATH, energy_df, weather_df)
    print(f"Wrote {DB_PATH}")

def train_replay_models(train_end):
    """Trains and saves one all-building model per utility on readings before train_end; returns their folder."""
    for u in UTILITIES:
        print(f"\n-- training {u} (data before {train_end}) --")
        save_replay_model(train_replay(u, train_end), u, train_end)
    return REPLAY_MODELS_DIR / train_end

def build_with_replays() -> None:
    # initial state: the first year of data, and models trained on it
    print(f"\n{'=' * 60}\nINITIAL STATE: data before {REPLAY_INIT_TRAIN_END}\n{'=' * 60}")
    build_database(
        DB_PATH,
        load_meter_readings_df(DATA_DIR, end=REPLAY_INIT_TRAIN_END),
        load_weather_df(DATA_DIR, end=REPLAY_INIT_TRAIN_END),
    )
    model_dir = train_replay_models(REPLAY_INIT_TRAIN_END)

    # each later segment is a batch: detect on its CSVs with the models trained before it, dump
    # its readings, weather and anomalies to the DB, then retrain on everything so far for the
    # next batch. The last models are kept as the current ones
    for folder, start, end in TIME_SEGMENTS:
        if start < REPLAY_INIT_TRAIN_END:
            continue
        print(f"\n{'=' * 60}\nBATCH {folder}: {start} to {end}\n{'=' * 60}")

        print(f"\n-- detecting anomalies with {model_dir.name} models --")
        anomalies = AnomalyDetector(model_dir).get_anomalies(start, end)

        print("\n-- writing readings, weather and anomalies to the DB --")
        insert_readings(
            DB_PATH,
            load_meter_readings_df(DATA_DIR, start, end),
            load_weather_df(DATA_DIR, start, end),
        )
        # after the readings, since each anomaly's building is looked up from its meter's readings
        insert_anomalies(DB_PATH, anomalies.assign(segment=folder))
        print(f"{folder}: {len(anomalies)} anomalies")

        model_dir = train_replay_models(end)
    print(f"\n{'=' * 60}\nDONE: wrote {DB_PATH}\n{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the DuckDB database")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Build the database one segment at a time, detecting anomalies and retraining between segments"
    )
    args = parser.parse_args()
    if args.replay:
        build_with_replays()
    else:
        build_full()
