import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from energyiq import config
from energyiq.ml.evaluation import inject_anomalies

# academic breaks, (first day, last day) inclusive
BREAKS = [
    ("2023-10-12", "2023-10-13"), ("2023-11-22", "2023-11-24"), ("2023-12-15", "2024-01-07"),
    ("2024-03-11", "2024-03-15"), ("2024-05-01", "2024-08-19"),
    ("2024-10-10", "2024-10-11"), ("2024-11-27", "2024-11-29"), ("2024-12-13", "2025-01-05"),
    ("2025-03-10", "2025-03-14"), ("2025-04-30", "2025-08-18"),
    ("2025-10-16", "2025-10-17"), ("2025-11-26", "2025-11-28"), ("2025-12-19", "2026-01-11"),
    ("2026-03-16", "2026-03-20"), ("2026-05-05", "2026-08-24"),
    ("2026-10-15", "2026-10-16"), ("2026-11-25", "2026-11-27"),
    ("2026-12-18", "2027-01-10"),  # was 11/18/26, check against the calendar
    ("2027-03-15", "2027-03-19"), ("2027-05-05", "2027-08-23"),
]

# Ohio Stadium events, (start, approx. end) in Eastern local time. readingtime is UTC, so these
# are converted before matching. Published schedules, so calendar data, not occupancy
STADIUM_EVENTS = [
    # 2023 football
    ("2023-09-09 12:00", "2023-09-09 15:30"),  # vs. Youngstown State
    ("2023-09-16 16:00", "2023-09-16 19:30"),  # vs. Western Kentucky
    ("2023-10-07 12:00", "2023-10-07 15:30"),  # vs. Maryland
    ("2023-10-21 12:00", "2023-10-21 15:30"),  # vs. Penn State
    ("2023-11-11 19:30", "2023-11-11 23:00"),  # vs. Michigan State
    ("2023-11-18 16:00", "2023-11-18 19:30"),  # vs. Minnesota
    # 2024 spring and summer
    ("2024-04-13 12:00", "2024-04-13 14:30"),  # spring game
    ("2024-05-05 12:00", "2024-05-05 15:00"),  # spring commencement
    ("2024-06-22 10:00", "2024-06-22 23:30"),  # Buckeye Country Superfest day 1 (tailgate from 10)
    ("2024-06-23 15:00", "2024-06-23 23:00"),  # Buckeye Country Superfest day 2
    # 2024 football
    ("2024-08-31 15:30", "2024-08-31 19:00"),  # vs. Akron
    ("2024-09-07 19:30", "2024-09-07 23:00"),  # vs. Western Michigan
    ("2024-09-21 12:00", "2024-09-21 15:30"),  # vs. Marshall
    ("2024-10-05 15:30", "2024-10-05 19:00"),  # vs. Iowa
    ("2024-10-12 08:00", "2024-10-12 21:00"),  # Buckeye Invitational (high school bands)
    ("2024-10-26 12:00", "2024-10-26 15:30"),  # vs. Nebraska
    ("2024-11-09 12:00", "2024-11-09 15:30"),  # vs. Purdue
    ("2024-11-23 12:00", "2024-11-23 15:30"),  # vs. Indiana
    ("2024-11-30 12:00", "2024-11-30 15:45"),  # vs. Michigan
    ("2024-12-21 20:00", "2024-12-21 23:30"),  # CFP first round vs. Tennessee
    # 2025 spring and summer
    ("2025-03-01 16:00", "2025-03-01 21:00"),  # NHL Stadium Series (gates 4, puck drop 6)
    ("2025-04-12 12:00", "2025-04-12 14:30"),  # spring game
    ("2025-05-04 12:00", "2025-05-04 15:00"),  # spring commencement
    ("2025-06-21 11:00", "2025-06-21 23:00"),  # Buckeye Country Superfest (fan fest from 11)
    # 2025 football
    ("2025-08-30 12:00", "2025-08-30 15:30"),  # vs. Texas
    ("2025-09-06 15:30", "2025-09-06 19:00"),  # vs. Grambling State
    ("2025-09-13 19:00", "2025-09-13 22:30"),  # vs. Ohio
    ("2025-10-04 15:30", "2025-10-04 19:00"),  # vs. Minnesota
    ("2025-10-11 08:00", "2025-10-11 21:00"),  # Buckeye Invitational (high school bands)
    ("2025-11-01 12:00", "2025-11-01 15:30"),  # vs. Penn State
    ("2025-11-15 15:30", "2025-11-15 19:00"),  # vs. UCLA
    ("2025-11-22 12:00", "2025-11-22 15:30"),  # vs. Rutgers
    # 2026 spring and summer
    ("2026-04-18 12:00", "2026-04-18 14:30"),  # spring game
    ("2026-04-25 17:45", "2026-04-25 21:45"),  # Luke Combs
    ("2026-05-03 12:00", "2026-05-03 15:00"),  # spring commencement
    ("2026-05-20 19:00", "2026-05-20 23:00"),  # Bruno Mars
    ("2026-06-13 14:30", "2026-06-13 23:00"),  # Buckeye Country Superfest
    ("2026-07-15 19:30", "2026-07-15 23:30"),  # AC/DC
    # 2026 football
    ("2026-09-05 12:30", "2026-09-05 16:00"),  # vs. Ball State
    ("2026-09-19 12:00", "2026-09-19 15:30"),  # vs. Kent State
    ("2026-09-26 12:00", "2026-09-26 15:30"),  # vs. Illinois
    ("2026-10-03 08:00", "2026-10-03 21:00"),  # Buckeye Invitational (high school bands)
    ("2026-10-10 16:15", "2026-10-10 19:45"),  # vs. Maryland
    ("2026-10-17 16:00", "2026-10-17 23:00"),  # Twenty One Pilots (gates 4, show 6)
]
STADIUM_SITENAME = "Ohio Stadium"
# load ramps up hours before an event (gates, lights, concessions) and stays up after (teardown)
EVENT_PRE_HOURS = 6
EVENT_POST_HOURS = 3

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
CATEGORICAL_COLS = ["meterid", "sitename", "hour", "season", "weekday", "month"]

# known-anomaly labels for scoring the detector: the injected kind ("" if none),
# and whether the reading broke the drop_outliers rule. Never features.
LABEL_COLS = ["injected", "outlier"]

# per-row info returned alongside X (labels, plus what thresholds and reports group by)
META_COLS = ["meterid", "sitename", "event_day", *LABEL_COLS]

LAG_HOURS = (24, 168)
TEMP_WINDOWS = (6, 24, 72)
USAGE_WINDOW = 24  
USAGE_OFFSET = 24 
HISTORY_LOOKBACK = pd.Timedelta(hours=max(LAG_HOURS) + 24)
ZERO_RUN_MAX_SPAN = pd.Timedelta(hours=1)

OUTLIER_K = 15


# ---------------------------------------------------------------- shared steps

def load_readings(building_name, utility_type, start=None, end=None, exclude_anomalies=True):
    conditions, params = ["m.utility = :utility"], {"utility": utility_type.upper()}
    if exclude_anomalies:
        conditions.append("a.id IS NULL")
    if building_name != "All":
        conditions.append("b.sitename = :name")
        params["name"] = building_name
    if start is not None:
        conditions.append("m.readingtime >= :start")
        params["start"] = pd.Timestamp(start)
    if end is not None:
        conditions.append("m.readingtime < :end")
        params["end"] = pd.Timestamp(end)

    engine = create_engine(f"duckdb:///{config.DB_PATH}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT m.*, b.sitename, w.*
                FROM meter_readings m
                JOIN buildings b ON m.building_id = b.id
                JOIN weather w
                ON date_trunc('hour', m.readingtime) = date_trunc('hour', w.date)
                LEFT JOIN anomalies a
                ON a.meterid = m.meterid AND a.readingtime = m.readingtime
                WHERE {' AND '.join(conditions)}
            """), conn, params=params)
    finally:
        engine.dispose()

    if df.empty:
        raise ValueError(f"No {utility_type} readings for building {building_name!r}")
    return df.drop(columns=WEATHER_DROP)

def clean_zero_runs(df, train_end):
    """
    Zeros that come from meter dropouts, not real usage:
    - runs of zeros spanning less than ZERO_RUN_MAX_SPAN (first to last zero) are interpolated
      from the neighbouring readings
    - longer runs (outages) are dropped
    Measured in time rather than readings, since the raw data mixes hourly and 15-minute
    readings: a lone hourly zero or up to four 15-minute zeros count as short.
    Training period only: in the test/replay period a run of zeros may be a real anomaly,
    so those readings are left as they are.
    """
    df = df.sort_values(["meterid", "readingtime"]).reset_index(drop=True)
    is_train = df["readingtime"] < train_end

    is_zero = df["readingvalue"].eq(0) & is_train
    prev_zero = is_zero.groupby(df["meterid"]).shift(fill_value=False)
    run_id = (is_zero != prev_zero).cumsum()
    run_times = df["readingtime"].groupby([df["meterid"], run_id])
    run_span = run_times.transform("max") - run_times.transform("min")

    short = is_zero & (run_span < ZERO_RUN_MAX_SPAN)
    long = is_zero & (run_span >= ZERO_RUN_MAX_SPAN)

    # interpolate within the training period only, and only write back the short-run zeros,
    # so test readings are never filled or used to fill
    df.loc[short, "readingvalue"] = np.nan
    filled = df.groupby([df["meterid"], is_train])["readingvalue"].transform(
        lambda s: s.interpolate(limit_area="inside")
    )
    df.loc[short, "readingvalue"] = filled[short]
    df = df[~long]

    print(f"zero cleanup (training rows only): filled {short.sum()} readings in short runs, "
          f"dropped {long.sum()} readings in long runs")
    return df

def group_keys(meters, times, utility_type, event_day=None):
    """
    Keys for comparable readings, shared by outlier removal and anomaly thresholds:
    meter + season for heat, meter + weekday + break for electricity.
    Electricity readings on a stadium event day (event_day, 0/1 per row) get one group per meter
    instead, so event days (setup and teardown included) are judged against other event days.
    Returned as plain arrays so they group positionally, whatever the index.
    """
    times = pd.Series(np.asarray(times))
    meters = np.asarray(meters)
    if utility_type == "heat":
        return [meters, times.dt.month.map(SEASONS).to_numpy()]
    weekday = times.dt.dayofweek.to_numpy()
    classes = in_classes(times).to_numpy()
    if event_day is not None:
        event = np.asarray(event_day) == 1
        weekday = np.where(event, -1, weekday)
        classes = np.where(event, -1, classes)
    return [meters, weekday, classes]

def drop_outliers(df, utility_type, train_end, by_events=False):
    """
    Readings more than OUTLIER_K MADs from their group's median, with medians and MADs from
    training rows only. Training outliers are dropped. Test/replay outliers are kept, since those
    are the anomalies to detect, and marked True in an `outlier` column as known anomalies for
    scoring the detector.
    """
    df = df[df["readingvalue"] >= 0].copy()  # negative usage is impossible (also drops NaN)
    is_train = df["readingtime"] < train_end
    # by_events: stadium event days get their own groups, as the thresholds do (see group_keys).
    # Features aren't built yet, so they're found here
    event_day = (stadium_events(df["readingtime"], df["sitename"])[0].to_numpy()
                 if by_events else np.zeros(len(df), dtype=int))
    train = df[is_train]

    groups = train["readingvalue"].groupby(group_keys(train["meterid"], train["readingtime"], utility_type,
                                                      event_day[is_train.to_numpy()]))
    group_med = groups.median()
    group_mad = groups.apply(lambda s: (s - s.median()).abs().median())
    # flat groups fall back to their building's median, so pooled runs match per-building ones
    building_med = train.groupby("sitename")["readingvalue"].apply(lambda s: s.abs().median())

    keys = pd.MultiIndex.from_arrays(group_keys(df["meterid"], df["readingtime"], utility_type, event_day))
    med = group_med.reindex(keys).to_numpy()
    mad = group_mad.reindex(keys).to_numpy()
    mad = np.where(mad == 0, df["sitename"].map(building_med).to_numpy(), mad)

    # groups never seen in training have a NaN cutoff, so they're never outliers
    outlier = np.abs(df["readingvalue"].to_numpy() - med) > OUTLIER_K * mad
    drop = outlier & is_train.to_numpy()
    df["outlier"] = outlier & ~is_train.to_numpy()
    print(f"dropping {drop.sum()} of {is_train.sum()} training rows as outliers, "
          f"labelling {df['outlier'].sum()} of {(~is_train).sum()} test rows as outliers")
    return df[~drop]

def hourly_means(df):
    return df.groupby(["meterid", df["readingtime"].dt.floor("h")])["readingvalue"].mean()

def average_training_hours(df, train_end):
    """
    Training readings averaged to one row per meter per clock hour, so the hourly and 15-minute
    stretches of the raw data count alike. Test rows stay raw, since every reading is scored
    on its own; both are on the same scale.
    Run after the features are built: every feature is the same for all readings in an hour
    (weather is hourly, lags/usage history come from hourly means), so only the target is averaged.
    """
    is_train = df["readingtime"] < train_end
    train = df[is_train].copy()
    hour = train["readingtime"].dt.floor("h")
    train["readingvalue"] = train.groupby([train["meterid"], hour])["readingvalue"].transform("mean")
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
    One-hot encode, drop non-feature columns, and split into (X, y, meta) indexed by time.
    meta holds the META_COLS that are present: the anomaly labels (LABEL_COLS), kept out of X so
    they can't leak into the model, and each row's meter, building and event day as plain values.
    """
    df = df.sort_values("readingtime").set_index("readingtime")
    meta = df[[c for c in META_COLS if c in df.columns]].copy()

    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=categorical, dtype=int)
    df = df.drop(columns=NON_FEATURE_COLS, errors="ignore")

    X = df.drop(columns=["readingvalue", *LABEL_COLS])
    X = X.loc[:, X.nunique() > 1]  # drop constant columns
    return X, df["readingvalue"], meta


# ---------------------------------------------------------------- heat features

def add_temp_history(df, temp_col="temperature_2m"):
    """
    - temp_mean_{h}h: mean temperature over the last h hours, for each h in TEMP_WINDOWS
    - temp_min_24h:   minimum temperature over the last 24 hours
    """
    # one temperature per hour, then time-based rolling windows
    df["_hour"] = df["readingtime"].dt.floor("h")
    temp = df.groupby("_hour")[temp_col].first().sort_index()

    temp_feats = pd.DataFrame(index=temp.index)
    for h in TEMP_WINDOWS:
        temp_feats[f"temp_mean_{h}h"] = temp.rolling(f"{h}h").mean()
    temp_feats["temp_min_24h"] = temp.rolling("24h").min()

    return df.join(temp_feats, on="_hour").drop(columns="_hour")

def add_usage_history(df, drop_missing=True):
    """
    usage_mean_{w}h: the meter's mean usage over a USAGE_WINDOW-hour window ending USAGE_OFFSET
    hours before each reading's hour (no lookahead). Built from hourly means, so hourly and
    15-minute data get the same feature. Rows with under half the window's hours of history
    are dropped, or with drop_missing=False left NaN (HGB handles missing values).
    """
    col = f"usage_mean_{USAGE_WINDOW}h"
    roll = (hourly_means(df).reset_index()
              .set_index("readingtime")
              .groupby("meterid")["readingvalue"]
              .rolling(f"{USAGE_WINDOW}h", min_periods=USAGE_WINDOW // 2)
              .mean()
              .rename(col)
              .reset_index())
    roll["readingtime"] += pd.Timedelta(hours=USAGE_OFFSET)

    df["_hour"] = df["readingtime"].dt.floor("h")
    df = df.merge(roll.rename(columns={"readingtime": "_hour"}), on=["meterid", "_hour"], how="left")
    df = df.drop(columns="_hour")
    return df.dropna(subset=[col]) if drop_missing else df

def preprocess_heat(df, inc_seasons=False, inc_months=False, inc_temp_history=False, inc_usage_history=False,
                    inc_lags=False, drop_missing=True):
    if inc_lags:
        df = add_lag_features(df, drop_missing)
    if inc_temp_history:
        df = add_temp_history(df)
    if inc_usage_history:
        df = add_usage_history(df, drop_missing)
    if inc_seasons:
        df["season"] = df["readingtime"].dt.month.map(SEASONS)
    if inc_months:
        df["month"] = df["readingtime"].dt.month
    return df


# --------------------------------------------------------- electricity features

def add_lag_features(df, drop_missing=True):
    """
    The same meter's mean reading over the clock hour `h` hours earlier, for each h in LAG_HOURS;
    robust to gaps. Hourly means, so 15-minute and hourly readings get equally smooth lags.
    Rows missing a lag are dropped, or with drop_missing=False left NaN (HGB handles missing values).
    """
    base = hourly_means(df).reset_index().rename(columns={"readingtime": "_hour"})
    df["_hour"] = df["readingtime"].dt.floor("h")
    for h in LAG_HOURS:
        lagged = (base.assign(_hour=base["_hour"] + pd.Timedelta(hours=h))
                      .rename(columns={"readingvalue": f"lag_{h}h"}))
        df = df.merge(lagged, on=["meterid", "_hour"], how="left")
    df = df.drop(columns="_hour")
    lag_cols = [f"lag_{h}h" for h in LAG_HOURS]
    return df.dropna(subset=lag_cols) if drop_missing else df

def in_classes(times):
    """1 for readings on a class day, 0 during an academic break."""
    dates = times.dt.normalize()
    on_break = pd.Series(False, index=times.index)
    for start, end in BREAKS:
        on_break |= dates.between(pd.Timestamp(start), pd.Timestamp(end))  # inclusive on both ends
    return (~on_break).astype(int)

def _eastern_to_utc(times):
    return times.dt.tz_localize("America/New_York").dt.tz_convert("UTC").dt.tz_localize(None)

def stadium_events(times, sitenames):
    """
    - event_day: 1 if the reading's Eastern local date has a stadium event
    - at_event:  1 from EVENT_PRE_HOURS before an event's start to EVENT_POST_HOURS after its end
    Both are 0 for every building but Ohio Stadium, so the pooled model can't spread the effect.
    """
    events = pd.DataFrame(STADIUM_EVENTS, columns=["start", "end"]).apply(pd.to_datetime)
    start = _eastern_to_utc(events["start"]) - pd.Timedelta(hours=EVENT_PRE_HOURS)
    end = _eastern_to_utc(events["end"]) + pd.Timedelta(hours=EVENT_POST_HOURS)

    local_dates = times.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    event_day = local_dates.isin(events["start"].dt.normalize())
    at_event = pd.Series(False, index=times.index)
    for s, e in zip(start, end):
        at_event |= times.between(s, e)

    is_stadium = sitenames.eq(STADIUM_SITENAME)
    return (event_day & is_stadium).astype(int), (at_event & is_stadium).astype(int)

def preprocess_electricity(df, inc_lags=False, inc_weekends=False, inc_weekdays=False, inc_breaks=False,
                           inc_events=False, drop_missing=True):
    if inc_lags:
        df = add_lag_features(df, drop_missing)
    if inc_weekends:
        df["is_weekend"] = (df["readingtime"].dt.dayofweek >= 5).astype(int)
    if inc_weekdays:
        df["weekday"] = (df["readingtime"].dt.dayofweek).astype(int)
    if inc_breaks:
        df["in_classes"] = in_classes(df["readingtime"])
    if inc_events:
        df["event_day"], df["at_event"] = stadium_events(df["readingtime"], df["sitename"])
    return df


# ------------------------------------------------------------------ entry point

def preprocess(building_name, utility_type, baseline=False, inject=False, inference=False,
               train_end=config.EXPERIMENT_TRAIN_END, data_end=None, **features):
    utility_type = utility_type.lower()
    if utility_type not in ("heat", "electricity"):
        raise ValueError(f"utility_type must be 'heat' or 'electricity', got {utility_type!r}")

    if inference:
        # earlier flagged anomalies included, so they don't leave gaps in the lags
        df = load_readings(building_name, utility_type, start=pd.Timestamp(train_end) - HISTORY_LOOKBACK,
                           end=data_end, exclude_anomalies=False)
        df["outlier"] = False
    else:
        df = load_readings(building_name, utility_type, end=data_end)
        df = clean_zero_runs(df, train_end)
        df = drop_outliers(df, utility_type, train_end, by_events=features.get("inc_events", False))
    if inject:
        df = inject_anomalies(df, train_end)
    else:
        df["injected"] = ""

    if utility_type == "heat":
        df = preprocess_heat(df, drop_missing=not inference, **features)
    else:
        df = preprocess_electricity(df, drop_missing=not inference, **features)

    if inference:
        df = df[df["readingtime"] >= train_end]  # the lookback was only there for the features
    else:
        df = average_training_hours(df, train_end)
    df = add_time_of_day(df, baseline)
    return to_model_inputs(df)
