# scripts/build_db.py
from energyiq.config import DB_PATH, DATA_DIR, TIME_SEGMENTS, TRAIN_END, ANOMALY_K
from energyiq.ingest import load_meter_readings_df, load_weather_df, build_database, insert_readings, \
    insert_thresholds, mark_anomalies
from energyiq.ml.preprocess import preprocess_window
from energyiq.ml.train import train_replay, save_replay_model, predict
from energyiq.ml.thresholds import flag_anomalies
import argparse

def build_full() -> None:
    energy_df = load_meter_readings_df(DATA_DIR)
    weather_df = load_weather_df(DATA_DIR)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_database(DB_PATH, energy_df, weather_df)
    print(f"Wrote {DB_PATH}")

def train_and_store(train_end):
    """Trains on everything before train_end, saves the model and writes its thresholds to the DB."""
    print(f"\n-- training on data before {train_end} --")
    bundle, thresholds = train_replay(train_end)
    save_replay_model(bundle, train_end)
    n = insert_thresholds(DB_PATH, thresholds, train_end, ANOMALY_K)
    print(f"wrote {n} group thresholds (train_end {train_end})")
    return bundle, thresholds

def build_with_replays() -> None:
    # initial state: the data before TRAIN_END, and a model trained on it
    print(f"\n{'=' * 60}\nINITIAL STATE: data before {TRAIN_END}\n{'=' * 60}")
    build_database(
        DB_PATH,
        load_meter_readings_df(DATA_DIR, end=TRAIN_END),
        load_weather_df(DATA_DIR, end=TRAIN_END),
    )
    bundle, thresholds = train_and_store(TRAIN_END)

    # each later segment is a batch: add its readings to the DB, flag them with the model and
    # thresholds trained before it, then retrain on everything so far for the next batch
    for folder, start, end in TIME_SEGMENTS:
        if start < TRAIN_END:
            continue
        print(f"\n{'=' * 60}\nBATCH {folder}: {start} to {end}\n{'=' * 60}")

        print("\n-- writing readings and weather to the DB --")
        insert_readings(
            DB_PATH,
            load_meter_readings_df(DATA_DIR, start, end),
            load_weather_df(DATA_DIR, start, end),
        )

        print(f"\n-- flagging with the {bundle['train_end']} model and thresholds --")
        X, y, meta = preprocess_window(start, end)
        y_pred = predict(bundle["model"], X)
        n = mark_anomalies(DB_PATH, flag_anomalies(meta, y, y_pred, thresholds))
        print(f"{folder}: {n} of {len(y)} readings flagged ({n / len(y):.2%})")

        bundle, thresholds = train_and_store(end)
    print(f"\n{'=' * 60}\nDONE: wrote {DB_PATH}\n{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the DuckDB database")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Build the database one segment at a time, flagging anomalies and retraining between segments"
    )
    args = parser.parse_args()
    if args.replay:
        build_with_replays()
    else:
        build_full()