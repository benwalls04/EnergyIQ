"""
Per-group anomaly bounds on the model's residuals (actual - predicted), grouped by GROUP_COLS.
Each group gets its median residual and a spread on each side (median deviation above / below
that median), so spikes and drops are judged separately. lower_bound / upper_bound hold the
k-scaled spreads, and a reading is flagged when its residual falls outside
    [median - lower_bound, median + upper_bound]
"""
import numpy as np
import pandas as pd

from energyiq.config import ANOMALY_K
from energyiq.ml.preprocess import GROUP_COLS


def fit_thresholds(meta_train, residuals, k=ANOMALY_K):
    """
    One row per group: GROUP_COLS, n_train, median, lower_bound and upper_bound, where median is the
    median residual and lower_bound / upper_bound are k * the lower / upper spreads.
    In-sample (training) residuals give somewhat tight spreads; k makes up for that.
    """
    frame = meta_train[GROUP_COLS].reset_index(drop=True).assign(r=np.asarray(residuals, dtype=float))
    keys = [frame[c] for c in GROUP_COLS]
    groups = frame["r"].groupby(keys)
    dev = frame["r"] - groups.transform("median")

    # a side with no deviations at all has a NaN median, so a spread of 0
    lower_spread = (-dev).where(dev < 0).groupby(keys).median().fillna(0)
    upper_spread = dev.where(dev > 0).groupby(keys).median().fillna(0)

    thresholds = pd.DataFrame({
        "n_train": groups.size(),
        "median": groups.median(),
        "lower_bound": k * lower_spread,
        "upper_bound": k * upper_spread,
    })

    print(f"thresholds: {len(thresholds)} groups, smallest n_train {thresholds['n_train'].min()}, "
          f"{(lower_spread == 0).sum()} with lower_spread 0, {(upper_spread == 0).sum()} with "
          f"upper_spread 0 (k={k})")
    return thresholds.reset_index()


def thresholds_for(meta, thresholds):
    """Each reading's group row of thresholds, positionally. Groups never seen in training are all NaN."""
    keys = pd.MultiIndex.from_frame(meta[GROUP_COLS])
    return thresholds.set_index(GROUP_COLS).reindex(keys)


def flag_anomalies(meta, y, y_pred, thresholds):
    """
    The readings whose residual (y - y_pred) is outside their group's bounds, as
    (meterid, readingtime, severity).
    severity = (residual - median) / (that side's bound): above 1 a spike, below -1 a drop.
    A bound of 0 gives +-inf.
    """
    residual = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
    rows = thresholds_for(meta, thresholds)
    median = rows["median"].to_numpy()
    lower_bound, upper_bound = rows["lower_bound"].to_numpy(), rows["upper_bound"].to_numpy()

    dev = residual - median
    flagged = (dev > upper_bound) | (dev < -lower_bound)  # NaN bounds compare False

    with np.errstate(divide="ignore", invalid="ignore"):
        severity = dev / np.where(dev > 0, upper_bound, lower_bound)

    return pd.DataFrame({
        "meterid": meta["meterid"].to_numpy()[flagged],
        "readingtime": meta.index[flagged],
        "severity": severity[flagged],
    })