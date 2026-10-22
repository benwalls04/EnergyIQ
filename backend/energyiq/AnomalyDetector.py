import joblib
import numpy as np
import pandas as pd

from energyiq.config import UTILITIES, DEFAULT_FEATURES, ANOMALY_K
from energyiq.ml.preprocess import preprocess
from energyiq.ml.evaluation import decode_one_hot
from energyiq.ml.train import thresholds_for

class AnomalyDetector:
  def __init__(self, model_dir, k=ANOMALY_K):
    """Loads model_dir / <utility>.joblib for every utility, as saved by save_replay_model."""
    self.k = k
    self.bundles = {u: joblib.load(model_dir / f"{u}.joblib") for u in UTILITIES}

  def detect_anomalies(self, bundle, utility_type, X, y) -> pd.DataFrame:
    """One row per reading whose |residual| is more than k x its group's training MAE."""
    # columns can differ from training (constant one-hots are dropped per run), so line them up
    y_pred = bundle["model"].predict(X.reindex(columns=bundle["feature_columns"], fill_value=0))
    resid = y.to_numpy() - y_pred
    # thresholds from X as preprocessed, so each row's meter decodes from its own one-hot columns
    thresholds = thresholds_for(X, bundle["group_mae"], bundle["fallback_mae"], utility_type, self.k)
    flagged = np.abs(resid) > thresholds

    return pd.DataFrame({
      "meterid": decode_one_hot(X, "meterid_", "only meter")[flagged].astype("int64"),
      "readingtime": X.index[flagged],
      "utility": utility_type.upper(),
      "actual": y.to_numpy()[flagged],
      "predicted": y_pred[flagged],
      "residual": resid[flagged],
      "threshold": thresholds[flagged],
      "severity": np.abs(resid[flagged]) / thresholds[flagged],
    })

  def get_anomalies(self, start, end) -> pd.DataFrame:
    """
    Anomalies among the readings in [start, end), read straight from that segment's CSVs.
    Every feature looks backwards, so each reading's decision uses nothing after its own timestamp.
    """
    all_anomalies = []
    for u, bundle in self.bundles.items():
      features = {name: True for name in DEFAULT_FEATURES[u]}
      X, y, _ = preprocess("All", u, inference=True, train_end=start, data_end=end, **features)
      all_anomalies.append(self.detect_anomalies(bundle, u, X, y))
    return pd.concat(all_anomalies, ignore_index=True)
