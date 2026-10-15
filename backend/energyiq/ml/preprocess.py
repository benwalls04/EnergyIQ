import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from energyiq import config

BREAKS = [
    ("10/12/23", "10/13/23"), ("11/22/23", "11/24/23"), ("12/15/23", "1/7/24"),
    ("3/11/24", "3/15/24"),   ("5/1/24", "8/19/24"),
    ("10/10/24", "10/11/24"), ("11/27/24", "11/29/24"), ("12/13/24", "1/5/25"),
    ("3/10/25", "3/14/25"),   ("4/30/25", "8/18/25"),
    ("10/16/25", "10/17/25"), ("11/26/25", "11/28/25"), ("12/19/25", "1/11/26"),
    ("3/16/26", "3/20/26"),   ("5/5/26", "8/24/26"),
    ("10/15/26", "10/16/26"), ("11/25/26", "11/27/26"),
    ("12/18/26", "1/10/27"),  # was 11/18/26, check against the calendar
    ("3/15/27", "3/19/27"),   ("5/5/27", "8/23/27"),
]

SEASONS = {12: "winter", 1: "winter", 2: "winter",
           3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer",
           9: "fall", 10: "fall", 11: "fall"}

WEATHER_DROP = ["direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
              "wind_speed_100m", "wind_direction_100m", "wind_direction_10m",
              "relative_humidity_2m", "apparent_temperature"]

# columns from the DB join that are never features
NON_FEATURE_COLS = ["id", "readingunits", "sitename", "latitude", "longitude",
                    "building_id", "date", "utility"]

# categorical columns that get one-hot encoded when present
CATEGORICAL_COLS = ["meterid", "hour", "season"]


# ---------------------------------------------------------------- shared steps

def load_readings(building_name, utility_type):
    """One building's readings for one utility, joined to the weather for that hour."""
    engine = create_engine(f"duckdb:///{config.DB_PATH}")
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id FROM buildings WHERE sitename = :name"),
                {"name": building_name},
            ).first()
            if row is None:
                raise ValueError(f"No building named {building_name!r}")

            df = pd.read_sql(text("""
                SELECT m.*, b.sitename, w.*
                FROM meter_readings m
                JOIN buildings b ON m.building_id = b.id
                JOIN weather w
                  ON date_trunc('hour', m.readingtime) = date_trunc('hour', w.date)
                WHERE m.building_id = :bId
            """), conn, params={"bId": row[0]})
    finally:
        engine.dispose()

    df = df[df["utility"] == utility_type.upper()]
    if df.empty:
        raise ValueError(f"No {utility_type} readings for {building_name}")
    return df.drop(columns=WEATHER_DROP)

def clean_zero_runs(df, col="readingvalue", group="meterid", max_fill=3):
    """
    Zeros that come from meter dropouts, not real usage:
    - runs of <= max_fill consecutive zeros are interpolated from the neighbouring readings
    - longer runs (outages) are dropped
    Assumes one reading per meter per 15-minute interval.
    """
    df = df.sort_values([group, "readingtime"]).reset_index(drop=True)

    is_zero = df[col].eq(0)
    prev_zero = is_zero.groupby(df[group]).shift(fill_value=False)
    run_id = (is_zero != prev_zero).cumsum()
    run_len = is_zero.groupby([df[group], run_id]).transform("size")

    short = is_zero & (run_len <= max_fill)
    long = is_zero & (run_len > max_fill)

    df.loc[short, col] = np.nan
    df[col] = df.groupby(group)[col].transform(
        lambda s: s.interpolate(limit=max_fill, limit_area="inside")
    )
    df = df[~long]

    print(f"zero cleanup: filled {short.sum()} readings in short runs, "
          f"dropped {long.sum()} readings in long runs")
    return df

def drop_outliers(df, col="readingvalue", group="meterid", k=10):
    df = df[df[col] >= 0]  # negative usage is impossible (also drops NaN)
    med = df.groupby(group)[col].transform("median")
    mad = df.groupby(group)[col].transform(lambda s: (s - s.median()).abs().median())
    mad = mad.replace(0, df[col].abs().median())  # flat meters
    keep = (df[col] - med).abs() <= k * mad
    print(f"dropping {(~keep).sum()} of {len(df)} rows as outliers")
    return df[keep]

def add_time_of_day(df, baseline=False):
    """
    baseline (linear models): the hour as a category, one-hot encoded later, so each
    hour gets its own coefficient.
    otherwise (trees): sin/cos of the hour, which trees can split into any hour range.
    """
    hour = df["readingtime"].dt.hour
    if baseline:
        df["hour"] = hour
    else:
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def to_model_inputs(df):
    """One-hot encode, drop non-feature columns, and split into (X, y) indexed by time."""
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=categorical, dtype=int)
    df = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    df = df.sort_values("readingtime").set_index("readingtime")

    X = df.drop(columns=["readingvalue"])
    X = X.loc[:, X.nunique() > 1]  # drop constant columns
    return X, df["readingvalue"]


# ---------------------------------------------------------------- heat features

def add_temp_history(df, temp_col="temperature_2m", temp_windows=(6, 24, 72)):
    """
    - temp_mean_{h}h: mean temperature over the last h hours, for each h in temp_windows
    - temp_min_24h:   minimum temperature over the last 24 hours
    """
    # one temperature per hour, then time-based rolling windows
    df["_hour"] = df["readingtime"].dt.floor("h")
    temp = df.groupby("_hour")[temp_col].first().sort_index()

    temp_feats = pd.DataFrame(index=temp.index)
    for h in temp_windows:
        temp_feats[f"temp_mean_{h}h"] = temp.rolling(f"{h}h").mean()
    temp_feats["temp_min_24h"] = temp.rolling("24h").min()

    return df.join(temp_feats, on="_hour").drop(columns="_hour")

def add_usage_history(df, usage_window=24, usage_offset=24):
    """
    usage_mean_{w}h: the meter's mean usage over a w-hour window ending `usage_offset`
    hours before each reading (no lookahead). Rows without enough history are dropped.
    """
    df = df.sort_values(["meterid", "readingtime"]).reset_index(drop=True)

    col = f"usage_mean_{usage_window}h"
    roll = (df.set_index("readingtime")
              .groupby("meterid")["readingvalue"]
              .rolling(f"{usage_window}h", min_periods=usage_window * 2)  # >= half the 15-min readings
              .mean()
              .rename(col)
              .reset_index())
    roll["readingtime"] += pd.Timedelta(hours=usage_offset)
    df = df.merge(roll, on=["meterid", "readingtime"], how="left")

    return df.dropna(subset=[col])

def preprocess_heat(df, inc_seasons=False, inc_temp_history=False, inc_usage_history=False,
                    inc_lags=False, lag_hours=(24, 168)):
    if inc_lags:
        df = add_lag_features(df, lag_hours)
    if inc_temp_history:
        df = add_temp_history(df)
    if inc_usage_history:
        df = add_usage_history(df)
    if inc_seasons:
        df["season"] = df["readingtime"].dt.month.map(SEASONS)
    return df


# --------------------------------------------------------- electricity features

def add_lag_features(df, lag_hours=(24, 168)):
    """Value of the same meter exactly `h` hours earlier; robust to gaps."""
    base = df[["meterid", "readingtime", "readingvalue"]]
    for h in lag_hours:
        lagged = (base.assign(readingtime=base["readingtime"] + pd.Timedelta(hours=h))
                      .rename(columns={"readingvalue": f"lag_{h}h"}))
        df = df.merge(lagged, on=["meterid", "readingtime"], how="left")
    return df.dropna(subset=[f"lag_{h}h" for h in lag_hours])

def in_classes(times):
    dates = times.dt.normalize()
    on_break = pd.Series(False, index=times.index)
    for start, end in BREAKS:
        s = pd.to_datetime(start, format="%m/%d/%y")
        e = pd.to_datetime(end, format="%m/%d/%y")
        on_break |= dates.between(s, e)  # inclusive on both ends
    return (~on_break).astype(int)

def preprocess_electricity(df, inc_lags=False, inc_weekends=False, inc_breaks=False,
                           lag_hours=(24, 168)):
    if inc_lags:
        df = add_lag_features(df, lag_hours)
    if inc_weekends:
        df["is_weekend"] = (df["readingtime"].dt.dayofweek >= 5).astype(int)
    if inc_breaks:
        df["in_classes"] = in_classes(df["readingtime"])
    return df


# ------------------------------------------------------------------ entry point

def preprocess(building_name, utility_type, baseline=False, **features):
    """
    (X, y) for one building and utility, sorted and indexed by readingtime.

    baseline=True one-hot encodes the hour (linear models); otherwise sin/cos.
    Feature flags, all off by default:
      heat:        inc_seasons, inc_temp_history, inc_usage_history
      electricity: inc_lags, inc_weekends, inc_breaks
    """
    utility_type = utility_type.lower()
    if utility_type not in ("heat", "electricity"):
        raise ValueError(f"utility_type must be 'heat' or 'electricity', got {utility_type!r}")

    df = load_readings(building_name, utility_type)
    df = clean_zero_runs(df)
    df = drop_outliers(df)

    if utility_type == "heat":
        df = preprocess_heat(df, **features)
    else:
        df = preprocess_electricity(df, **features)

    df = add_time_of_day(df, baseline)
    return to_model_inputs(df)


if __name__ == "__main__":
    X, y = preprocess("Ohio Stadium", "electricity", inc_lags=True, inc_breaks=True)
    print(X.shape, list(X.columns))