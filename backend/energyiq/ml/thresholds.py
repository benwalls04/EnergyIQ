"""
Context-aware anomaly thresholds. Readings are grouped with comparable ones (preprocess.group_keys), and a
reading is anomalous when its |residual| is more than k x its group's training MAE.
Everything here takes the `meta` frame from preprocess (meterid, event_day, ... indexed by time).
"""
import numpy as np
import pandas as pd

from energyiq.ml.preprocess import group_keys


def _meta_keys(meta, utility_type):
    # event_day is only in meta for electricity built with inc_events
    return group_keys(meta["meterid"], meta.index, utility_type, meta.get("event_day"))


def compute_group_mae(meta_train, train_resid, utility_type):
    """
    Training MAE per group (see group_keys), and the overall training MAE as a fallback for
    groups never seen in training.
    """
    abs_resid = np.abs(train_resid)
    return abs_resid.groupby(_meta_keys(meta_train, utility_type)).mean(), float(abs_resid.mean())


def thresholds_for(meta, group_mae, fallback_mae, utility_type, k):
    """Per-row anomaly threshold: k x the training MAE of the row's group."""
    thresholds = group_mae.reindex(pd.MultiIndex.from_arrays(_meta_keys(meta, utility_type))).to_numpy()
    return np.where(np.isnan(thresholds), fallback_mae, thresholds) * k


def flag_anomalies(meta_train, train_resid, meta_test, test_resid, utility_type, k):
    """
    True for test rows whose |residual| is more than k x the training MAE of their group.
    Thresholds come from training residuals only, so no test information is used to decide
    what's anomalous.
    """
    group_mae, fallback_mae = compute_group_mae(meta_train, train_resid, utility_type)
    return np.abs(np.asarray(test_resid)) > thresholds_for(meta_test, group_mae, fallback_mae, utility_type, k)
