import argparse
import joblib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error
)
from energyiq.ml.preprocess import preprocess, group_keys, TRAIN_END  # one split date shared with outlier removal

# percent of the largest test residuals dropped from trimmed_MAE
TRIM_PCT = 2

# best from the stage-A search; model_params passed to train() override these
HGB_DEFAULTS = {
    "random_state": 42,
    "loss": "absolute_error",
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "early_stopping": False,  # its holdout is a shuffled 10%, not time-ordered
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for a building")

    parser.add_argument(
        "--building",
        choices=["All", "Ohio Stadium", "Thompson Memorial Library", "Scott House"],
        default="All",
        help="Building to train on: All, Ohio Stadium, Thompson Memorial Library, or Scott House",
    )
    parser.add_argument(
        "--model-type",
        choices=["regression", "ridge", "hgb"],
        default="regression",
        help="Model to train (default: regression)",
    )
    parser.add_argument(
        "--utility-type",
        choices=["electricity", "heat"],
        default="electricity",
        help="Utility type (default: electricity)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the trained model",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extra output",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=None,
        help="Time-decay half-life in days (default: no weighting)",
    )
    parser.add_argument(
        "--anomaly-k",
        type=float,
        default=3,
        help=f"Flag test residuals above k x their group's training MAE (default: {3})",
    )
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=TRIM_PCT,
        help=f"Percent of the largest test residuals dropped from trimmed_MAE (default: {TRIM_PCT})",
    )

    # feature flags passed through to preprocess
    shared = parser.add_argument_group("heat and electricity features")
    shared.add_argument(
        "--lags",
        action="store_true",
        help="Add usage 24h and 168h earlier",
    )

    heat = parser.add_argument_group("heat features")
    heat.add_argument(
        "--seasons",
        action="store_true",
        help="Add one-hot season columns",
    )
    heat.add_argument(
        "--months",
        action="store_true",
        help="Add a months flag",
    )
    heat.add_argument(
        "--temp-history",
        action="store_true",
        help="Add rolling temperature means/min over recent hours",
    )
    heat.add_argument(
        "--usage-history",
        action="store_true",
        help="Add the meter's mean usage over a past 24h window",
    )

    electricity = parser.add_argument_group("electricity features")
    electricity.add_argument(
        "--weekends",
        action="store_true",
        help="Add an is_weekend flag",
    )
    electricity.add_argument(
        "--breaks",
        action="store_true",
        help="Add an in_classes flag (0 during academic breaks)",
    )
    electricity.add_argument(
        "--weekdays",
        action="store_true",
        help="Add a weekday flag",
    )

    return parser.parse_args()

def save_run(model, metrics, metadata, out_dir, metric="MAE"):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = metadata["trained_at"]

    bundle = {"model": model, "metadata": {**metadata, "metrics": metrics}}
    run_path = out_dir / f"{metadata['model_type']}_{stamp}.joblib"
    joblib.dump(bundle, run_path)
    print("Saved to", run_path)

    best_path = out_dir / "best.joblib"
    best_info_path = out_dir / "best.json"

    best_score = None
    if best_info_path.exists():
        best_score = json.loads(best_info_path.read_text())["metrics"][metric]

    if best_score is None or metrics[metric] < best_score:
        shutil.copyfile(run_path, best_path)
        best_info_path.write_text(json.dumps(bundle["metadata"], indent=2, default=str))
        print(f"New best {metric}: {metrics[metric]:.3f} (previous: {best_score})")
    else:
        print(f"Kept existing best {metric}: {best_score:.3f}")

def time_decay_weights(index, half_life_days):
    age_days = (index.max() - index) / pd.Timedelta(days=1)
    return 0.5 ** (age_days / half_life_days)

def decode_one_hot(X, prefix, fallback):
  """Category per row from its one-hot columns, or `fallback` if they were dropped as constant."""
  cols = [c for c in X.columns if c.startswith(prefix)]
  if cols:
    return X[cols].idxmax(axis=1).str.removeprefix(prefix).to_numpy()
  return pd.Series(fallback, index=X.index).to_numpy()

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

def flag_anomalies(X_train, train_resid, X_test, test_resid, utility_type, k):
  """
  True for test rows whose |residual| is more than k x the training MAE of their group
  (meter + season for heat, meter + weekday + break for electricity). Thresholds come from
  training residuals only, so no test information is used to decide what's anomalous.
  """
  train_keys = group_keys(decode_one_hot(X_train, "meterid_", "only meter"), X_train.index, utility_type)
  test_keys = group_keys(decode_one_hot(X_test, "meterid_", "only meter"), X_test.index, utility_type)

  group_mae = np.abs(train_resid).groupby(train_keys).mean()
  thresholds = group_mae.reindex(pd.MultiIndex.from_arrays(test_keys)).to_numpy()
  # a group never seen in training falls back to the overall training MAE
  thresholds = np.where(np.isnan(thresholds), np.abs(train_resid).mean(), thresholds) * k

  return np.abs(np.asarray(test_resid)) > thresholds

def train(model_type, utility_type, building, log_metrics, half_life=None,
          model_params=None, anomaly_k=3, trim_pct=TRIM_PCT, **features):
  
  # linear models get one-hot hours, trees get sin/cos
  X, y = preprocess(building, utility_type, baseline=model_type != "hgb", **features)

  if log_metrics:
    print(f"{len(X)} rows, {X.shape[1]} features: {list(X.columns)}")

  # split on dates: train on the first year, score on the test/replay year
  train_rows = X.index < TRAIN_END
  X_train, y_train = X[train_rows], y[train_rows]
  X_test, y_test = X[~train_rows], y[~train_rows]

  match model_type:
    case "regression":
      model = LinearRegression()
    case "ridge":
      model = Ridge(alpha=1.0)
    case "hgb":
      model = HistGradientBoostingRegressor(**{**HGB_DEFAULTS, **(model_params or {})})
    case _:
      raise ValueError(f"Unknown model type: {model_type}")

  # optional time decay, more recent entries weighted more
  weights = None
  if half_life is not None:
      weights = time_decay_weights(X_train.index, half_life)

  start = time.perf_counter()
  model.fit(X_train, y_train, sample_weight=weights)
  fit_seconds = time.perf_counter() - start
  y_pred = model.predict(X_test)
  train_pred = model.predict(X_train)

  anomalous = flag_anomalies(X_train, y_train - train_pred, X_test, y_test - y_pred,
                             utility_type, anomaly_k)
  print(f"flagged {anomalous.sum()} of {len(y_test)} test rows as anomalies "
        f"({anomalous.mean():.2%}, k={anomaly_k})")

  kept = trim_mask(y_test, y_pred, trim_pct)

  mse =mean_squared_error(y_test, y_pred)
  metrics = {
    "MSE": mse,
    "RMSE": mse ** 0.5,
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "trimmed_MAE": mean_absolute_error(y_test[kept], y_pred[kept]),
    "trim_pct": trim_pct,
    "train_MAE": mean_absolute_error(y_train, train_pred),
    "anomaly_k": anomaly_k,
    "n_anomalies": int(anomalous.sum()),
    "anomaly_rate": anomalous.mean(),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_features": X.shape[1],
    "fit_seconds": fit_seconds,
  }

  if log_metrics:
    print(metrics)
    print_mae_breakdowns(X_test, y_test, y_pred, building)
    print_mae_breakdowns(X_test[kept], y_test[kept], y_pred[kept], building,
                         label=f"largest {trim_pct}% of residuals removed")

  return model, metrics, list(X.columns)

if __name__=="__main__":
  args = parse_args()
  # only pass the flags that were set, so a heat flag on an electricity run errors out
  all_flags = {
    "inc_seasons": args.seasons,
    "inc_temp_history": args.temp_history,
    "inc_usage_history": args.usage_history,
    "inc_lags": args.lags,
    "inc_weekends": args.weekends,
    "inc_breaks": args.breaks,
    "inc_months": args.months,
    "inc_weekdays": args.weekdays,
  }
  feature_kwargs = {name: True for name, on in all_flags.items() if on}

  model, metrics, feature_columns = train(
    model_type=args.model_type,
    utility_type=args.utility_type,
    building=args.building,
    log_metrics=args.verbose,
    half_life=args.half_life,
    anomaly_k=args.anomaly_k,
    trim_pct=args.trim_pct,
    **feature_kwargs,
  )

  if args.save:
    metadata = {
        "building": args.building,
        "utility_type": args.utility_type,
        "model_type": args.model_type,
        "half_life": args.half_life,
        # everything needed to rebuild the same features at prediction time
        "features": feature_kwargs,
        "feature_columns": feature_columns,
        "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    out_dir = Path(__file__).resolve().parent / "models" / args.building / args.utility_type
    save_run(model, metrics, metadata, out_dir)