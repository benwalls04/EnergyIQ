import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from energyiq import config
from energyiq.ml.evaluation import inject_anomalies

# train: Sep 2023 - Aug 2025, test/replay: Sep 2025 - Aug 2026 (one year)
TRAIN_END = "2025-09-01"

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
NON_FEATURE_COLS = ["id", "readingunits", "latitude", "longitude",
                    "building_id", "date", "utility"]

# categorical columns that get one-hot encoded when present.
# sitename is constant for a single building, so its dummy is dropped as a constant
# column and only survives as a feature when training on All
CATEGORICAL_COLS = ["meterid", "sitename", "hour", "season", "weekday", "month"]

# known-anomaly labels for scoring the detector: the injected kind ("" if none),
# and whether the reading broke the drop_outliers rule. Never features.
LABEL_COLS = ["injected", "outlier"]


# ---------------------------------------------------------------- shared steps

def load_readings(building_name, utility_type):
    """One building's readings for one utility, joined to the weather for that hour."""
    engine = create_engine(f"duckdb:///{config.DB_PATH}")
    try:
        with engine.connect() as conn:
            if building_name != "All":
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
            else: 
                df = pd.read_sql(text("""
                    SELECT m.*, b.sitename, w.*
                    FROM meter_readings m
                    JOIN buildings b ON m.building_id = b.id
                    JOIN weather w
                    ON date_trunc('hour', m.readingtime) = date_trunc('hour', w.date)
                """), conn)
    finally:
        engine.dispose()

    df = df[df["utility"] == utility_type.upper()]
    if df.empty:
        raise ValueError(f"No {utility_type} readings for {building_name} building")
    return df.drop(columns=WEATHER_DROP)

def load_readings_from_csv(building_name, utility_type):
    pass 

def clean_zero_runs(df, col="readingvalue", group="meterid", max_span=pd.Timedelta(hours=1)):
    """
    Zeros that come from meter dropouts, not real usage:
    - runs of zeros spanning less than max_span (first to last zero) are interpolated
      from the neighbouring readings
    - longer runs (outages) are dropped
    Measured in time rather than readings, since the raw data mixes hourly and 15-minute
    readings: a lone hourly zero or up to four 15-minute zeros count as short.
    Training period only: in the test/replay period a run of zeros may be a real anomaly,
    so those readings are left as they are.
    """
    df = df.sort_values([group, "readingtime"]).reset_index(drop=True)
    is_train = df["readingtime"] < TRAIN_END

    is_zero = df[col].eq(0) & is_train
    prev_zero = is_zero.groupby(df[group]).shift(fill_value=False)
    run_id = (is_zero != prev_zero).cumsum()
    run_times = df["readingtime"].groupby([df[group], run_id])
    run_span = run_times.transform("max") - run_times.transform("min")

    short = is_zero & (run_span < max_span)
    long = is_zero & (run_span >= max_span)

    # interpolate within the training period only, and only write back the short-run zeros,
    # so test readings are never filled or used to fill
    df.loc[short, col] = np.nan
    filled = df.groupby([df[group], is_train])[col].transform(
        lambda s: s.interpolate(limit_area="inside")
    )
    df.loc[short, col] = filled[short]
    df = df[~long]

    print(f"zero cleanup (training rows only): filled {short.sum()} readings in short runs, "
          f"dropped {long.sum()} readings in long runs")
    return df

def group_keys(meters, times, utility_type):
    """
    Keys for comparable readings, shared by outlier removal and anomaly thresholds:
    meter + season for heat, meter + weekday + break for electricity.
    Returned as plain arrays so they group positionally, whatever the index.
    """
    times = pd.Series(np.asarray(times))
    meters = np.asarray(meters)
    if utility_type == "heat":
        return [meters, times.dt.month.map(SEASONS).to_numpy()]
    return [meters, times.dt.dayofweek.to_numpy(), in_classes(times).to_numpy()]

def drop_outliers(df, utility_type, col="readingvalue", k=10):
    """
    Readings more than k MADs from their group's median, with medians and MADs from training rows only.
    Training outliers are dropped. Test/replay outliers are kept, since those are the anomalies to
    detect, and marked True in an `outlier` column as known anomalies for scoring the detector.
    """
    df = df[df[col] >= 0].copy()  # negative usage is impossible (also drops NaN)
    is_train = df["readingtime"] < TRAIN_END
    train = df[is_train]

    groups = train[col].groupby(group_keys(train["meterid"], train["readingtime"], utility_type))
    group_med = groups.median()
    group_mad = groups.apply(lambda s: (s - s.median()).abs().median())
    # flat groups fall back to their building's median, so pooled runs match per-building ones
    building_med = train.groupby("sitename")[col].apply(lambda s: s.abs().median())

    keys = pd.MultiIndex.from_arrays(group_keys(df["meterid"], df["readingtime"], utility_type))
    med = group_med.reindex(keys).to_numpy()
    mad = group_mad.reindex(keys).to_numpy()
    mad = np.where(mad == 0, df["sitename"].map(building_med).to_numpy(), mad)

    # groups never seen in training have a NaN cutoff, so they're never outliers
    outlier = np.abs(df[col].to_numpy() - med) > k * mad
    drop = outlier & is_train.to_numpy()
    df["outlier"] = outlier & ~is_train.to_numpy()
    print(f"dropping {drop.sum()} of {is_train.sum()} training rows as outliers, "
          f"labelling {df['outlier'].sum()} of {(~is_train).sum()} test rows as outliers")
    return df[~drop]

def hourly_means(df, col="readingvalue"):
    """Each meter's mean reading per clock hour, indexed by (meterid, readingtime floored to the hour)."""
    return df.groupby(["meterid", df["readingtime"].dt.floor("h")])[col].mean()

def average_training_hours(df, col="readingvalue"):
    """
    Training readings averaged to one row per meter per clock hour, so the hourly and 15-minute
    stretches of the raw data count alike. Test rows stay raw, since every reading is scored
    on its own; both are on the same scale.
    Run after the features are built: every feature is the same for all readings in an hour
    (weather is hourly, lags/usage history come from hourly means), so only the target is averaged.
    """
    is_train = df["readingtime"] < TRAIN_END
    train = df[is_train].copy()
    hour = train["readingtime"].dt.floor("h")
    train[col] = train.groupby([train["meterid"], hour])[col].transform("mean")
    train["readingtime"] = hour
    train = train.drop_duplicates(subset=["meterid", "readingtime"])
    print(f"averaged {is_train.sum()} training readings to {len(train)} meter-hours")
    return pd.concat([train, df[~is_train]], ignore_index=True)

def add_time_of_day(df, baseline=False):
    hour = df["readingtime"].dt.hour
    if baseline:
        df["hour"] = hour
    else:
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def to_model_inputs(df):
    """
    One-hot encode, drop non-feature columns, and split into (X, y, labels) indexed by time.
    labels holds the anomaly labels (LABEL_COLS), kept out of X so they can't leak into the model.
    """
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=categorical, dtype=int)
    df = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    df = df.sort_values("readingtime").set_index("readingtime")

    X = df.drop(columns=["readingvalue", *LABEL_COLS])
    X = X.loc[:, X.nunique() > 1]  # drop constant columns
    return X, df["readingvalue"], df[LABEL_COLS]


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
    hours before each reading's hour (no lookahead). Built from hourly means, so hourly and
    15-minute data get the same feature. Rows with under half the window's hours of history
    are dropped.
    """
    col = f"usage_mean_{usage_window}h"
    roll = (hourly_means(df).reset_index()
              .set_index("readingtime")
              .groupby("meterid")["readingvalue"]
              .rolling(f"{usage_window}h", min_periods=usage_window // 2)
              .mean()
              .rename(col)
              .reset_index())
    roll["readingtime"] += pd.Timedelta(hours=usage_offset)

    df["_hour"] = df["readingtime"].dt.floor("h")
    df = df.merge(roll.rename(columns={"readingtime": "_hour"}), on=["meterid", "_hour"], how="left")
    return df.drop(columns="_hour").dropna(subset=[col])

def preprocess_heat(df, inc_seasons=False, inc_months=False, inc_temp_history=False, inc_usage_history=False,
                    inc_lags=False, lag_hours=(24, 168)):
    if inc_lags:
        df = add_lag_features(df, lag_hours)
    if inc_temp_history:
        df = add_temp_history(df)
    if inc_usage_history:
        df = add_usage_history(df)
    if inc_seasons:
        df["season"] = df["readingtime"].dt.month.map(SEASONS)
    if inc_months:
        df["month"] = df["readingtime"].dt.month
    return df


# --------------------------------------------------------- electricity features

def add_lag_features(df, lag_hours=(24, 168)):
    """
    The same meter's mean reading over the clock hour `h` hours earlier; robust to gaps.
    Hourly means, so 15-minute and hourly readings get equally smooth lags.
    """
    base = hourly_means(df).reset_index().rename(columns={"readingtime": "_hour"})
    df["_hour"] = df["readingtime"].dt.floor("h")
    for h in lag_hours:
        lagged = (base.assign(_hour=base["_hour"] + pd.Timedelta(hours=h))
                      .rename(columns={"readingvalue": f"lag_{h}h"}))
        df = df.merge(lagged, on=["meterid", "_hour"], how="left")
    return df.drop(columns="_hour").dropna(subset=[f"lag_{h}h" for h in lag_hours])

def in_classes(times):
    dates = times.dt.normalize()
    on_break = pd.Series(False, index=times.index)
    for start, end in BREAKS:
        s = pd.to_datetime(start, format="%m/%d/%y")
        e = pd.to_datetime(end, format="%m/%d/%y")
        on_break |= dates.between(s, e)  # inclusive on both ends
    return (~on_break).astype(int)

def preprocess_electricity(df, inc_lags=False, inc_weekends=False, inc_weekdays=False, inc_breaks=False,
                           lag_hours=(24, 168)):
    if inc_lags:
        df = add_lag_features(df, lag_hours)
    if inc_weekends:
        df["is_weekend"] = (df["readingtime"].dt.dayofweek >= 5).astype(int)
    if inc_weekdays:
        df["weekday"] = (df["readingtime"].dt.dayofweek).astype(int)
    if inc_breaks:
        df["in_classes"] = in_classes(df["readingtime"])
    return df


# ------------------------------------------------------------------ entry point

def preprocess(building_name, utility_type, baseline=False, inject=False, inject_seed=0, from_csv=False, **features):
    """
    (X, y, labels) for one building and utility, sorted and indexed by readingtime.
    labels has the LABEL_COLS anomaly labels for each row.

    baseline=True one-hot encodes the hour (linear models); otherwise sin/cos.
    inject=True adds synthetic anomalies to the test period (see evaluation.py), before
    the features are built so lag/history features see them too.
    Feature flags, all off by default:
      heat:        inc_seasons, inc_temp_history, inc_usage_history
      electricity: inc_lags, inc_weekends, inc_breaks
    """
    utility_type = utility_type.lower()
    if utility_type not in ("heat", "electricity"):
        raise ValueError(f"utility_type must be 'heat' or 'electricity', got {utility_type!r}")

    if from_csv:
        load_readings_from_csv(building_name, utility_type)
    else:
        df = load_readings(building_name, utility_type)
    df = clean_zero_runs(df)
    df = drop_outliers(df, utility_type)
    if inject:
        df = inject_anomalies(df, TRAIN_END, seed=inject_seed)
    else:
        df["injected"] = ""

    if utility_type == "heat":
        df = preprocess_heat(df, **features)
    else:
        df = preprocess_electricity(df, **features)

    df = average_training_hours(df)
    df = add_time_of_day(df, baseline)
    return to_model_inputs(df)