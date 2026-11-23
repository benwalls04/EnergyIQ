"""
Expected usage from the saved replay models. Predictions aren't stored in the database, so they're
computed on request. Each reading uses the newest model trained before it, the same way the replay
judges a segment, so a prediction never relies on data from its own future.
"""
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from energyiq.ml.evaluation import TRIM_PCT, accuracy_metrics, trim_mask
from energyiq.ml.preprocess import CATEGORICAL_COLS, FEATURE_COLS, add_features

READING_COLUMNS = "m.meterid, m.readingtime, m.building_id, m.utility, m.readingvalue, m.readingunits, " \
                  "m.is_anomalous, b.sitename"


class ModelStore:
    def __init__(self, models_dir: Path):
        # the old replay/<date>/<utility>.joblib folders aren't matched by this glob
        bundles = [joblib.load(path) for path in sorted(models_dir.glob("*.joblib"))]
        self.bundles = sorted(bundles, key=lambda bundle: bundle["train_end"])
        self._performance = None
        self._lock = threading.Lock()

    def window(self, index: int) -> tuple[pd.Timestamp, pd.Timestamp | None]:
        """The period a model predicts: from its train_end until the next model takes over."""
        start = pd.Timestamp(self.bundles[index]["train_end"])
        if index + 1 == len(self.bundles):
            return start, None
        return start, pd.Timestamp(self.bundles[index + 1]["train_end"])

    @property
    def first_prediction_time(self) -> pd.Timestamp | None:
        return self.window(0)[0] if self.bundles else None

    def predict(self, readings: pd.DataFrame) -> pd.DataFrame:
        """readings with `predicted` and `train_end` added. Rows from before the first model are dropped."""
        parts = []
        for i, bundle in enumerate(self.bundles):
            start, end = self.window(i)
            in_window = readings["readingtime"] >= start
            if end is not None:
                in_window &= readings["readingtime"] < end
            if in_window.any():
                parts.append(_predict_with(bundle, readings[in_window]))
        if not parts:
            return readings.iloc[0:0].assign(predicted=np.nan, train_end=None)
        return pd.concat(parts, ignore_index=True)

    def performance(self, cursor) -> list[dict]:
        """
        Out-of-sample accuracy of each model on the period it predicted. Scoring means predicting
        every reading in the period, so it's computed once per process; the data can't change
        underneath the API since a rebuild needs it stopped.
        """
        with self._lock:
            if self._performance is None:
                self._performance = [self._score(cursor, i) for i in range(len(self.bundles))]
            return self._performance

    def _score(self, cursor, index: int) -> dict:
        start, end = self.window(index)
        conditions, params = "m.readingtime >= ?", [start]
        if end is not None:
            conditions += " AND m.readingtime < ?"
            params.append(end)
        readings = cursor.execute(f"""
            SELECT {READING_COLUMNS}
            FROM meter_readings m JOIN buildings b ON b.id = m.building_id
            WHERE {conditions}
        """, params).fetchdf()

        groups = []
        if len(readings):
            scored = self.predict(readings).dropna(subset=["predicted"])
            groups.append({"utility": None, "building_id": None, "sitename": None, **_metrics(scored)})
            for utility, part in scored.groupby("utility"):
                groups.append({"utility": utility, "building_id": None, "sitename": None, **_metrics(part)})
            for (utility, building_id, sitename), part in scored.groupby(["utility", "building_id", "sitename"]):
                groups.append({"utility": utility, "building_id": int(building_id), "sitename": sitename,
                               **_metrics(part)})
        return {
            "train_end": self.bundles[index]["train_end"],
            "evaluated_from": start,
            "evaluated_until": end,
            "groups": groups,
        }


def _predict_with(bundle: dict, readings: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    features = list(model.feature_names_in_)
    # not to_model_inputs: it re-sorts by time, and predictions have to stay lined up with these rows
    df = add_features(readings.copy())
    X = pd.get_dummies(df[FEATURE_COLS], columns=CATEGORICAL_COLS, dtype=int)
    df["predicted"] = model.predict(X.reindex(columns=features, fill_value=0))
    # all-zero meter columns would still give a number, just not one that means anything
    known_meter = df["meterid"].map(lambda meterid: f"meterid_{meterid}" in features)
    df.loc[~known_meter, "predicted"] = np.nan
    df["train_end"] = bundle["train_end"]
    return df


def _metrics(scored: pd.DataFrame) -> dict:
    y_true = scored["readingvalue"].to_numpy()
    y_pred = scored["predicted"].to_numpy()
    kept = trim_mask(y_true, y_pred, TRIM_PCT)
    raw_err = y_true - y_pred
    accuracy = accuracy_metrics(y_true[kept], y_pred[kept])
    return {
        # R2 is undefined on a single reading
        **{k.lower(): v if np.isfinite(v) else None for k, v in accuracy.items()},
        "mae_untrimmed": float(np.abs(raw_err).mean()),
        "n_readings": len(scored),
        "n_flagged": int(scored["is_anomalous"].sum()),
    }
