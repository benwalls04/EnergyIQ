"""
Evaluation for a trained model and its anomaly detector:
- synthetic anomalies injected into the test period, so the detector is scored against known answers
- accuracy on the real (non-injected) test readings, plus a trimmed MAE
- detection metrics against the known anomalies (injected + drop_outliers-rule rows)
- console breakdowns of the test MAE
evaluate() returns every metric as one dict.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# percent of the largest test residuals dropped from trimmed_MAE
TRIM_PCT = 2

# fixed so every run is scored against the same injected anomalies
INJECT_SEED = 0

# events per meter per year of test data. Only short, sharp anomalies are in scope;
# sustained ones (multi-day offsets, missed night setbacks) aren't detected by this project.
EVENTS_PER_YEAR = {"spike": 12, "dropout": 6}
MAX_TRIES = 20  # attempts to place an event without overlapping an earlier one

# ------------------------------------------------------------- injection

def plan_event(kind, t, rng):
    """
    A random event on one meter's test times `t`: (mask over t, function old -> new values).
    - spike:   x1.5-3 for 1-3 hours
    - dropout: zero for 2-6 hours
    """
    start = t[rng.integers(len(t))]
    hour = np.timedelta64(1, "h")
    match kind:
        case "spike":
            mask = (t >= start) & (t < start + rng.integers(1, 4) * hour)
            factor = rng.uniform(1.5, 3.0)
            return mask, lambda old: old * factor
        case "dropout":
            mask = (t >= start) & (t < start + rng.integers(2, 7) * hour)
            return mask, lambda old: np.zeros_like(old)
    raise ValueError(f"Unknown anomaly kind {kind!r}")


def inject_anomalies(df, start, seed=INJECT_SEED, col="readingvalue"):
    """
    Adds synthetic anomalies to each meter's readings at or after `start`, and labels them
    in an `injected` column ("" for untouched rows). Events on a meter don't overlap.
    Only rows an event changes by more than 10% are labelled, so e.g. a dropout on a meter
    that already reads zero doesn't count as an anomaly to find.
    Run before lag/history features are built, so those see the injected values too,
    as they would with a real anomaly.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    values = df[col].to_numpy(dtype=float).copy()
    labels = np.full(len(df), "", dtype=object)
    times = df["readingtime"].to_numpy()
    is_test = times >= np.datetime64(start)

    for meter, idx in df.groupby("meterid").indices.items():
        idx = idx[is_test[idx]]
        if len(idx) == 0:
            continue
        t = times[idx]
        years = (t.max() - t.min()) / np.timedelta64(365, "D")
        free = np.ones(len(idx), dtype=bool)

        for kind, per_year in EVENTS_PER_YEAR.items():
            for _ in range(round(per_year * years)):
                for _ in range(MAX_TRIES):
                    mask, change = plan_event(kind, t, rng)
                    if mask.any() and free[mask].all():
                        break
                else:
                    continue  # no free spot found, skip this event
                rows = idx[mask]
                old = values[rows]
                new = change(old)
                values[rows] = new
                labels[rows[~np.isclose(new, old, rtol=0.1, atol=0)]] = kind
                free[mask] = False

    df[col] = values
    df["injected"] = labels
    counts = {kind: int((labels == kind).sum()) for kind in EVENTS_PER_YEAR}
    print(f"injected anomalies (test rows): {counts}")
    return df


# ------------------------------------------------------------- metrics

def decode_one_hot(X, prefix, fallback):
    """Category per row from its one-hot columns, or `fallback` if they were dropped as constant."""
    cols = [c for c in X.columns if c.startswith(prefix)]
    if cols:
        return X[cols].idxmax(axis=1).str.removeprefix(prefix).to_numpy()
    return pd.Series(fallback, index=X.index).to_numpy()


def trim_mask(y_true, y_pred, trim_pct):
    """
    False for the rows with the largest trim_pct% of absolute residuals, True for the rest.
    The share dropped is fixed, so unlike scoring only unflagged rows, a worse model can't
    remove more of its own error.
    """
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n_drop = int(len(abs_err) * trim_pct / 100)
    keep = np.ones(len(abs_err), dtype=bool)
    if n_drop:  # [-0:] would select every row
        keep[np.argsort(abs_err)[-n_drop:]] = False
    return keep


def accuracy_metrics(y_true, y_pred, trim_pct):
    kept = trim_mask(y_true, y_pred, trim_pct)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "trimmed_MAE": mean_absolute_error(y_true[kept], y_pred[kept]),
        "trim_pct": trim_pct,
    }


def detection_metrics(flagged, injected, outlier):
    """
    Scores the flags against the known anomalies: injected rows plus test rows that broke
    the drop_outliers rule. Real anomalies the outlier rule misses still count as false
    positives, so precision and false_positive_rate are lower and upper bounds respectively.
    """
    truth = (injected != "") | outlier
    tp = (flagged & truth).sum()
    precision = tp / flagged.sum() if flagged.any() else np.nan
    recall = tp / truth.sum() if truth.any() else np.nan
    metrics = {
        "n_anomalies": int(flagged.sum()),
        "anomaly_rate": flagged.mean(),
        "n_labelled": int(truth.sum()),
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan,
        "false_positive_rate": (flagged & ~truth).sum() / (~truth).sum(),
        "recall_outlier": flagged[outlier].mean() if outlier.any() else np.nan,
    }
    # per injected kind, since spikes and dropouts can be easier or harder to catch
    for kind in EVENTS_PER_YEAR:
        is_kind = injected == kind
        metrics[f"recall_{kind}"] = flagged[is_kind].mean() if is_kind.any() else np.nan
    return metrics


def print_mae_breakdowns(X_test, y_test, y_pred, building, label="all rows"):
    """Test MAE per building and month, and per building and meter, console output only."""
    abs_err = (y_test - y_pred).abs()

    # a single building or single-meter utility has no sitename/meterid columns left
    buildings = decode_one_hot(X_test, "sitename_", building)
    meters = decode_one_hot(X_test, "meterid_", "only meter")

    print(f"\nTest MAE by month ({label}):")
    by_month = abs_err.groupby([abs_err.index.to_period("M"), buildings]).mean()
    print(by_month.unstack().round(3).to_string())  # months as rows, buildings as columns

    print(f"\nTest MAE by meter ({label}):")
    print(abs_err.groupby([buildings, meters]).mean().round(3).to_string())


def print_detection_report(X_test, flagged, injected, outlier, building):
    """Confusion counts, recall per kind of known anomaly, and detection per building, console output only."""
    truth = (injected != "") | outlier
    tp, fp = (flagged & truth).sum(), (flagged & ~truth).sum()
    fn, tn = (~flagged & truth).sum(), (~flagged & ~truth).sum()

    print("\nDetection report")
    print(f"{'':>16}{'labelled':>12}{'unlabelled':>12}")
    print(f"{'flagged':>16}{tp:>12}{fp:>12}")
    print(f"{'not flagged':>16}{fn:>12}{tn:>12}")
    print("(unlabelled = neither injected nor an outlier-rule reading, so 'flagged, unlabelled' "
          "may include real anomalies)")

    # a row can be both injected and an outlier, so it counts under both kinds here
    kinds = {"outlier": outlier, **{kind: injected == kind for kind in EVENTS_PER_YEAR}}
    by_kind = pd.DataFrame(
        [(kind, is_kind.sum(), (flagged & is_kind).sum()) for kind, is_kind in kinds.items()],
        columns=["kind", "labelled", "caught"],
    ).set_index("kind")
    by_kind["recall"] = by_kind["caught"] / by_kind["labelled"].replace(0, np.nan)
    print("\nRecall by kind of anomaly:")
    print(by_kind.round(3).to_string())

    rows = pd.DataFrame({
        "building": decode_one_hot(X_test, "sitename_", building),
        "flagged": flagged,
        "labelled": truth,
        "caught": flagged & truth,
        "false_flag": flagged & ~truth,
        "unlabelled": ~truth,
    })
    by_building = rows.groupby("building")[["flagged", "labelled", "caught", "false_flag", "unlabelled"]].sum()
    by_building["recall"] = by_building["caught"] / by_building["labelled"].replace(0, np.nan)
    by_building["precision"] = by_building["caught"] / by_building["flagged"].replace(0, np.nan)
    by_building["false_positive_rate"] = by_building["false_flag"] / by_building["unlabelled"]
    print("\nDetection by building:")
    print(by_building.drop(columns="unlabelled").round(3).to_string())


# ------------------------------------------------------------- entry point

def evaluate(X_test, y_test, y_pred, flagged, test_labels, y_train, train_pred, building,
             trim_pct=TRIM_PCT, verbose=False):
    """
    All evaluation metrics for one run, as a dict.
    Accuracy is scored on the real test readings only, so the injected anomalies don't count
    as model error. Detection is scored on every test row. test_labels holds the LABEL_COLS
    from preprocess for the test rows; verbose prints the MAE breakdowns and detection report.
    """
    injected = test_labels["injected"].to_numpy()
    outlier = test_labels["outlier"].to_numpy()
    real = injected == ""
    X_real, y_real, pred_real = X_test[real], y_test[real], y_pred[real]

    metrics = {
        **accuracy_metrics(y_real, pred_real, trim_pct),
        "train_MAE": mean_absolute_error(y_train, train_pred),
        **detection_metrics(flagged, injected, outlier),
    }

    if verbose:
        kept = trim_mask(y_real, pred_real, trim_pct)
        print_mae_breakdowns(X_real, y_real, pred_real, building, label="real readings")
        print_mae_breakdowns(X_real[kept], y_real[kept], pred_real[kept], building,
                             label=f"real readings, largest {trim_pct}% of residuals removed")
        print_detection_report(X_test, flagged, injected, outlier, building)
    return metrics
