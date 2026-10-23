"""
Fixed per-group anomaly bounds from training readings, grouped by GROUP_COLS.
Each group gets its median reading and a spread on each side (median deviation above / below
the median), so spikes and drops are judged separately:
    lower = max(0, median - k * lower_spread)
    upper = median + k * upper_spread
"""
import numpy as np
import pandas as pd

from energyiq.config import ANOMALY_K
from energyiq.ml.preprocess import GROUP_COLS


def fit_thresholds(meta_train, y_train, k=ANOMALY_K):
    """One row per group: GROUP_COLS, n_train, median, lower_bound and upper_bound."""
    frame = meta_train[GROUP_COLS].reset_index(drop=True).assign(y=np.asarray(y_train, dtype=float))
    keys = [frame[c] for c in GROUP_COLS]
    groups = frame["y"].groupby(keys)
    dev = frame["y"] - groups.transform("median")

    # a side with no deviations at all has a NaN median, so a spread of 0
    lower_spread = (-dev).where(dev < 0).groupby(keys).median().fillna(0)
    upper_spread = dev.where(dev > 0).groupby(keys).median().fillna(0)
    median = groups.median()

    thresholds = pd.DataFrame({
        "n_train": groups.size(),
        "median": median,
        "lower_bound": np.maximum(0, median - k * lower_spread),
        "upper_bound": median + k * upper_spread,
    })

    print(f"thresholds: {len(thresholds)} groups, smallest n_train {thresholds['n_train'].min()}, "
          f"{(lower_spread == 0).sum()} with lower_spread 0, {(upper_spread == 0).sum()} with "
          f"upper_spread 0 (k={k})")
    return thresholds.reset_index()


def thresholds_for(meta, thresholds):
    """Each reading's group row of thresholds, positionally. Groups never seen in training are all NaN."""
    keys = pd.MultiIndex.from_frame(meta[GROUP_COLS])
    return thresholds.set_index(GROUP_COLS).reindex(keys)


def flag_anomalies(meta, y, thresholds):
    """
    The readings outside their group's bounds, as (meterid, readingtime, severity).
    severity = (reading - median) / (distance from the median to the bound on that side):
    above 1 a spike, below -1 a drop. A bound equal to the median gives +-inf.
    """
    y = np.asarray(y, dtype=float)
    rows = thresholds_for(meta, thresholds)
    median = rows["median"].to_numpy()
    lower, upper = rows["lower_bound"].to_numpy(), rows["upper_bound"].to_numpy()
    flagged = (y > upper) | (y < lower)  # NaN bounds compare False

    dev = y - median
    with np.errstate(divide="ignore", invalid="ignore"):
        severity = dev / np.where(dev > 0, upper - median, median - lower)

    return pd.DataFrame({
        "meterid": meta["meterid"].to_numpy()[flagged],
        "readingtime": meta.index[flagged],
        "severity": severity[flagged],
    })