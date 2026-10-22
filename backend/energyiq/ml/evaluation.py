"""
Evaluation for a trained model and its anomaly detector:
- synthetic anomalies injected into the test period, so the detector is scored against known answers
- accuracy (MAE, RMSE, R2) on the real (non-injected) test readings, largest residuals trimmed
- detection metrics against the known anomalies (injected + drop_outliers-rule rows)
- console breakdowns of the test MAE
evaluate() returns every metric as one dict.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# percent of the largest test residuals dropped before scoring accuracy
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
    """MAE, RMSE and R2 on the rows kept after trimming the largest trim_pct% of residuals."""
    kept = trim_mask(y_true, y_pred, trim_pct)
    y_true, y_pred = y_true[kept], y_pred[kept]
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def confusion(flagged, injected, outlier):
    """
    (truth, tp, fp, fn, tn) for the flags against the known anomalies: injected rows plus
    test rows that broke the drop_outliers rule.
    """
    truth = (injected != "") | outlier
    return (truth, (flagged & truth).sum(), (flagged & ~truth).sum(),
            (~flagged & truth).sum(), (~flagged & ~truth).sum())


def detection_metrics(flagged, injected, outlier):
    """
    Scores the flags against the known anomalies (see confusion). Real anomalies the outlier
    rule misses still count as false positives, so precision and false_positive_rate are lower
    and upper bounds respectively.
    """
    truth, tp, fp, fn, tn = confusion(flagged, injected, outlier)
    metrics = {
        "n_anomalies": int(flagged.sum()),
        "n_labelled": int(truth.sum()),
        "precision": float(tp / flagged.sum()) if flagged.any() else np.nan,
        "false_positive_rate": float(fp / (fp + tn)),
        "recall_outlier": float(flagged[outlier].mean()) if outlier.any() else np.nan,
    }
    # recall per kind rather than overall, since the kinds have very different row counts
    for kind in EVENTS_PER_YEAR:
        is_kind = injected == kind
        metrics[f"recall_{kind}"] = float(flagged[is_kind].mean()) if is_kind.any() else np.nan
    return metrics


def flags_per_meter_week(meta_test, flagged):
    """Flagged readings per meter per week of test data: the alert volume a facilities contact would see."""
    n_meters = meta_test["meterid"].nunique()
    weeks = (meta_test.index.max() - meta_test.index.min()) / pd.Timedelta(weeks=1)
    return float(flagged.sum() / n_meters / weeks)


def print_mae_breakdowns(meta, y_test, y_pred, label="all rows"):
    """Overall test MAE/RMSE/R2, then MAE per building and month, and per building and meter, console output only."""
    abs_err = (y_test - y_pred).abs()
    buildings = meta["sitename"].to_numpy()
    meters = meta["meterid"].to_numpy()

    print(f"\nTest accuracy ({label}): MAE {mean_absolute_error(y_test, y_pred):.3f}, "
          f"RMSE {mean_squared_error(y_test, y_pred) ** 0.5:.3f}, R2 {r2_score(y_test, y_pred):.3f}")

    print(f"\nTest MAE by month ({label}):")
    by_month = abs_err.groupby([abs_err.index.to_period("M"), buildings]).mean()
    print(by_month.unstack().round(3).to_string())  # months as rows, buildings as columns

    print(f"\nTest MAE by meter ({label}):")
    print(abs_err.groupby([buildings, meters]).mean().round(3).to_string())

    # normalized by each meter's mean usage, so meters of different sizes compare;
    # residual = actual - predicted, so positive bias means the model under-predicts
    resid = y_test - y_pred
    frame = pd.DataFrame({"actual": y_test, "resid": resid, "abs": resid.abs(), "sq": resid ** 2})
    by_meter = frame.groupby([buildings, meters])
    mean_usage = by_meter["actual"].mean()
    normalized = pd.DataFrame({
        "mean_usage": mean_usage,
        "MAE_%": by_meter["abs"].mean() / mean_usage * 100,
        "CV(RMSE)_%": by_meter["sq"].mean() ** 0.5 / mean_usage * 100,
        "NMBE_%": by_meter["resid"].mean() / mean_usage * 100,
    })
    print(f"\nNormalized error by meter ({label}; ASHRAE Guideline 14 hourly targets: "
          f"CV(RMSE) <= 30%, NMBE within +/-10%):")
    print(normalized.round(2).to_string())

    by_month_building = frame.groupby([frame.index.to_period("M"), buildings])
    bias = by_month_building["resid"].sum() / by_month_building["actual"].sum() * 100
    print(f"\nBias (NMBE %) by month ({label}; positive = under-predicting):")
    print(bias.unstack().round(2).to_string())


def print_detection_report(meta_test, flagged, injected, outlier):
    """Confusion counts, recall per kind of known anomaly, and detection per building, console output only."""
    truth, tp, fp, fn, tn = confusion(flagged, injected, outlier)

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
        "building": meta_test["sitename"].to_numpy(),
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

def evaluate(y_test, y_pred, flagged, meta_test, y_train, train_pred, verbose=False):
    """
    All evaluation metrics for one run, as a dict.
    Accuracy is scored on the real test readings only, so the injected anomalies don't count
    as model error. Detection is scored on every test row. meta_test holds preprocess's meta
    for the test rows; verbose prints the MAE breakdowns and detection report.
    """
    injected = meta_test["injected"].to_numpy()
    outlier = meta_test["outlier"].to_numpy()
    real = injected == ""
    meta_real, y_real, pred_real = meta_test[real], y_test[real], y_pred[real]

    metrics = {
        **accuracy_metrics(y_real, pred_real, TRIM_PCT),
        "train_MAE": float(mean_absolute_error(y_train, train_pred)),
        **detection_metrics(flagged, injected, outlier),
        "flags_per_meter_week": flags_per_meter_week(meta_test, flagged),
    }

    if verbose:
        kept = trim_mask(y_real, pred_real, TRIM_PCT)
        print_mae_breakdowns(meta_real[kept], y_real[kept], pred_real[kept],
                             label=f"real readings, largest {TRIM_PCT}% of residuals removed")
        print_detection_report(meta_test, flagged, injected, outlier)
    return metrics
