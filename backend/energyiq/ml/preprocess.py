import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from energyiq import config

SEASONS = {12: "winter", 1: "winter", 2: "winter",
           3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer",
           9: "fall", 10: "fall", 11: "fall"}

HOUR_BUCKET_HOURS = 4

FEATURE_COLS = ["season", "is_weekend", "hour_bucket", "utility", "meterid", "building_id",
                "hour_sin", "hour_cos", "month_sin", "month_cos"]
CATEGORICAL_COLS = ["season", "hour_bucket", "utility", "meterid", "building_id"]
GROUP_COLS = ["meterid", "month", "is_weekend", "hour_bucket"]
META_COLS = ["utility", "sitename", *GROUP_COLS]

ZERO_RUN_MAX_SPAN = pd.Timedelta(hours=1)

OUTLIER_K = 15


def load_readings(start=None, end=None):
    conditions, params = ["TRUE"], {}
    if start is not None:
        conditions.append("m.readingtime >= :start")
        params["start"] = pd.Timestamp(start)
    if end is not None:
        conditions.append("m.readingtime < :end")
        params["end"] = pd.Timestamp(end)

    engine = create_engine(f"duckdb:///{config.DB_PATH}")
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(f"""
                SELECT m.*, b.sitename
                FROM meter_readings m
                JOIN buildings b ON m.building_id = b.id
                WHERE {' AND '.join(conditions)}
            """), conn, params=params)
    finally:
        engine.dispose()

def clean_zero_runs(df, train_end):
    df = df.sort_values(["meterid", "readingtime"]).reset_index(drop=True)
    is_train = df["readingtime"] < train_end

    is_zero = df["readingvalue"].eq(0) & is_train
    prev_zero = is_zero.groupby(df["meterid"]).shift(fill_value=False)
    run_id = (is_zero != prev_zero).cumsum()
    run_times = df["readingtime"].groupby([df["meterid"], run_id])
    run_span = run_times.transform("max") - run_times.transform("min")

    short = is_zero & (run_span < ZERO_RUN_MAX_SPAN)
    long = is_zero & (run_span >= ZERO_RUN_MAX_SPAN)

    df.loc[short, "readingvalue"] = np.nan
    filled = df.groupby([df["meterid"], is_train])["readingvalue"].transform(
        lambda s: s.interpolate(limit_area="inside")
    )
    df.loc[short, "readingvalue"] = filled[short]
    df = df[~long]

    print(f"zero cleanup (training rows only): filled {short.sum()} readings in short runs, "
          f"dropped {long.sum()} readings in long runs")
    return df

def drop_outliers(df, train_end):
    """Drops training readings more than OUTLIER_K MADs from their group's (GROUP_COLS) median."""
    df = df[df["readingvalue"] >= 0]  # negative usage is impossible (also drops NaN)
    train = df[df["readingtime"] < train_end]

    groups = [train[c] for c in GROUP_COLS]
    abs_dev = (train["readingvalue"] - train["readingvalue"].groupby(groups).transform("median")).abs()
    mad = abs_dev.groupby(groups).transform("median")
    # flat groups fall back to their building's median
    building_med = train.groupby("sitename")["readingvalue"].transform(lambda s: s.abs().median())
    mad = mad.where(mad != 0, building_med)

    outlier = abs_dev > OUTLIER_K * mad
    print(f"dropping {outlier.sum()} of {len(train)} training rows as outliers")
    return df.drop(index=train.index[outlier])

def average_training_hours(df, train_end):
    is_train = df["readingtime"] < train_end
    train = df[is_train].copy()
    hour = train["readingtime"].dt.floor("h")
    train["readingvalue"] = train.groupby([train["meterid"], hour])["readingvalue"].transform("mean")
    train["readingtime"] = hour
    train = train.drop_duplicates(subset=["meterid", "readingtime"])
    print(f"averaged {is_train.sum()} training readings to {len(train)} meter-hours")
    return pd.concat([train, df[~is_train]], ignore_index=True)

def add_features(df):
    hour = df["readingtime"].dt.hour
    month = df["readingtime"].dt.month
    df["month"] = month  # for grouping only, not a feature
    df["season"] = month.map(SEASONS)
    df["is_weekend"] = (df["readingtime"].dt.dayofweek >= 5).astype(int)
    df["hour_bucket"] = hour // HOUR_BUCKET_HOURS
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df

def to_model_inputs(df):
    """(X, y, meta) indexed by time. meta holds the group columns, kept out of X."""
    df = df.sort_values("readingtime").set_index("readingtime")
    X = pd.get_dummies(df[FEATURE_COLS], columns=CATEGORICAL_COLS, dtype=int)
    return X, df["readingvalue"], df[META_COLS]

def preprocess(train_end=config.TRAIN_END, load_end=None):
    """
    Readings before load_end (all of them if None), for training and testing.
    Rows before train_end are training rows and get cleaned; later rows stay raw for testing.
    """
    df = add_features(load_readings(end=load_end))
    df = clean_zero_runs(df, train_end)
    df = drop_outliers(df, train_end)
    df = average_training_hours(df, train_end)
    return to_model_inputs(df)

def preprocess_window(start, end):
    """Raw readings in [start, end), uncleaned, for flagging: each is judged as it was recorded."""
    return to_model_inputs(add_features(load_readings(start=start, end=end)))