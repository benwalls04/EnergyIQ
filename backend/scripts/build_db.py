# scripts/build_db.py
from energyiq import config
from energyiq.ingest import load_meter_readings_df, load_weather_df, build_database

def main() -> None:
    energy_df = load_meter_readings_df(config.DATA_DIR)
    weather_df = load_weather_df(config.DATA_DIR)
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_database(config.DB_PATH, energy_df, weather_df)
    print(f"Wrote {config.DB_PATH}")

if __name__ == "__main__":
    main()