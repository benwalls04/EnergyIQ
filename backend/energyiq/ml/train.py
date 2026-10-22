import argparse
import joblib
import json
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from energyiq.config import (ANOMALY_K, DEFAULT_FEATURES, HGB_DEFAULTS, EXPERIMENT_TRAIN_END,
                             EXPERIMENT_MODELS_DIR, REPLAY_MODELS_DIR)
from energyiq.ml.preprocess import preprocess, group_keys
from energyiq.ml.evaluation import evaluate, decode_one_hot, TRIM_PCT, INJECT_SEED

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
        "--anomaly-k",
        type=float,
        default=ANOMALY_K,
        help=f"Flag test residuals above k x their group's training MAE (default: {ANOMALY_K})",
    )
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=TRIM_PCT,
        help=f"Percent of the largest test residuals dropped before scoring MAE/RMSE/R2 (default: {TRIM_PCT})",
    )

    # feature flags passed through to preprocess. Each has a --no- form; left unset, it
    # falls back to DEFAULT_FEATURES for the utility
    shared = parser.add_argument_group("heat and electricity features")
    shared.add_argument(
        "--lags",
        action=argparse.BooleanOptionalAction,
        help="Add usage 24h and 168h earlier (on by default for both)",
    )

    heat = parser.add_argument_group("heat features")
    heat.add_argument(
        "--seasons",
        action=argparse.BooleanOptionalAction,
        help="Add one-hot season columns (on by default)",
    )
    heat.add_argument(
        "--months",
        action=argparse.BooleanOptionalAction,
        help="Add a months flag",
    )
    heat.add_argument(
        "--temp-history",
        action=argparse.BooleanOptionalAction,
        help="Add rolling temperature means/min over recent hours (on by default)",
    )
    heat.add_argument(
        "--usage-history",
        action=argparse.BooleanOptionalAction,
        help="Add the meter's mean usage over a past 24h window (on by default)",
    )

    electricity = parser.add_argument_group("electricity features")
    electricity.add_argument(
        "--weekends",
        action=argparse.BooleanOptionalAction,
        help="Add an is_weekend flag",
    )
    electricity.add_argument(
        "--breaks",
        action=argparse.BooleanOptionalAction,
        help="Add an in_classes flag (0 during academic breaks) (on by default)",
    )
    electricity.add_argument(
        "--weekdays",
        action=argparse.BooleanOptionalAction,
        help="Add a weekday flag (on by default)",
    )
    electricity.add_argument(
        "--events",
        action=argparse.BooleanOptionalAction,
        help="Add Ohio Stadium event_day and at_event flags (on by default)",
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

def save_replay_model(bundle, utility_type, train_end):
  """Saves to REPLAY_MODELS_DIR / train_end / <utility>.joblib, overwriting an earlier replay's model."""
  out_dir = REPLAY_MODELS_DIR / train_end
  out_dir.mkdir(parents=True, exist_ok=True)
  joblib.dump(bundle, out_dir / f"{utility_type}.joblib")
  print("Saved to", out_dir / f"{utility_type}.joblib")

def compute_group_mae(X_train, train_resid, utility_type):
  """
  Training MAE per group (meter + season for heat, meter + weekday + break for electricity),
  and the overall training MAE as a fallback for groups never seen in training.
  """
  keys = group_keys(decode_one_hot(X_train, "meterid_", "only meter"), X_train.index, utility_type,
                    X_train.get("event_day"))
  abs_resid = np.abs(train_resid)
  return abs_resid.groupby(keys).mean(), float(abs_resid.mean())

def thresholds_for(X, group_mae, fallback_mae, utility_type, k):
  """Per-row anomaly threshold: k x the training MAE of the row's group."""
  # event_day is missing when it was dropped as constant (no events in X): no event groups then
  keys = group_keys(decode_one_hot(X, "meterid_", "only meter"), X.index, utility_type, X.get("event_day"))
  thresholds = group_mae.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy()
  return np.where(np.isnan(thresholds), fallback_mae, thresholds) * k

def flag_anomalies(X_train, train_resid, X_test, test_resid, utility_type, k):
  """
  True for test rows whose |residual| is more than k x the training MAE of their group.
  Thresholds come from training residuals only, so no test information is used to decide
  what's anomalous.
  """
  group_mae, fallback_mae = compute_group_mae(X_train, train_resid, utility_type)
  return np.abs(np.asarray(test_resid)) > thresholds_for(X_test, group_mae, fallback_mae, utility_type, k)

def train(model_type, utility_type, building, log_metrics,
          model_params=None, anomaly_k=ANOMALY_K, trim_pct=TRIM_PCT, **features):
  
  # linear models get one-hot hours, trees get sin/cos
  X, y, labels = preprocess(building, utility_type, baseline=model_type != "hgb",
                            inject=True, inject_seed=INJECT_SEED, **features)

  if log_metrics:
    print(f"{len(X)} rows, {X.shape[1]} features: {list(X.columns)}")

  # split on dates: train on the first year, score on the test/replay year
  train_rows = X.index < EXPERIMENT_TRAIN_END
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

  start = time.perf_counter()
  model.fit(X_train, y_train)
  fit_seconds = time.perf_counter() - start
  y_pred = model.predict(X_test)
  train_pred = model.predict(X_train)

  anomalous = flag_anomalies(X_train, y_train - train_pred, X_test, y_test - y_pred,
                             utility_type, anomaly_k)
  print(f"flagged {anomalous.sum()} of {len(y_test)} test rows as anomalies "
        f"({anomalous.mean():.2%}, k={anomaly_k})")

  metrics = {
    **evaluate(X_test, y_test, y_pred, anomalous, labels[~train_rows], y_train, train_pred,
               building, trim_pct=trim_pct, verbose=log_metrics),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_features": X.shape[1],
    "fit_seconds": fit_seconds,
  }

  if log_metrics:
    print(metrics)

  return model, metrics, list(X.columns)

def train_replay(utility_type, train_end):
  """
  The replay model for one utility: HGB with HGB_DEFAULTS and DEFAULT_FEATURES, fit on every
  building's readings before train_end. No test period and no injected anomalies.
  Returned as a bundle with what AnomalyDetector needs: the model, its feature columns, and
  the per-group training MAEs its thresholds come from.
  """
  features = {name: True for name in DEFAULT_FEATURES[utility_type]}
  X, y, _ = preprocess("All", utility_type, train_end=train_end, data_end=train_end, **features)

  model = HistGradientBoostingRegressor(**HGB_DEFAULTS)
  model.fit(X, y)
  train_resid = y - model.predict(X)
  group_mae, fallback_mae = compute_group_mae(X, train_resid, utility_type)

  return {
    "model": model,
    "feature_columns": list(X.columns),
    "group_mae": group_mae,
    "fallback_mae": fallback_mae,
    "metadata": {
      "utility_type": utility_type,
      "train_end": train_end,
      "features": features,
      "model_params": HGB_DEFAULTS,
      "n_train": len(X),
      "train_MAE": fallback_mae,
      "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    },
  }

if __name__=="__main__":
  args = parse_args()
  all_flags = {
    "inc_seasons": args.seasons,
    "inc_temp_history": args.temp_history,
    "inc_usage_history": args.usage_history,
    "inc_lags": args.lags,
    "inc_weekends": args.weekends,
    "inc_breaks": args.breaks,
    "inc_months": args.months,
    "inc_weekdays": args.weekdays,
    "inc_events": args.events,
  }
  # a flag left unset (None) is on if it's a default for this utility; only the ones that
  # end up on are passed, so a heat-only flag on an electricity run still errors out
  defaults = DEFAULT_FEATURES[args.utility_type]
  feature_kwargs = {name: True for name, on in all_flags.items()
                    if (name in defaults if on is None else on)}

  model, metrics, feature_columns = train(
    model_type=args.model_type,
    utility_type=args.utility_type,
    building=args.building,
    log_metrics=args.verbose,
    anomaly_k=args.anomaly_k,
    trim_pct=args.trim_pct,
    **feature_kwargs,
  )

  if args.save:
    metadata = {
        "building": args.building,
        "utility_type": args.utility_type,
        "model_type": args.model_type,
        "features": feature_kwargs,
        "feature_columns": feature_columns,
        # settings that shape the metrics, so runs with different ones aren't compared blindly
        "anomaly_k": args.anomaly_k,
        "trim_pct": args.trim_pct,
        "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    out_dir = EXPERIMENT_MODELS_DIR / args.utility_type
    save_run(model, metrics, metadata, out_dir)