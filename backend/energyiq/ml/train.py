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
from energyiq.ml.preprocess import preprocess, group_keys, TRAIN_END  # one split date shared with outlier removal
from energyiq.ml.evaluation import evaluate, decode_one_hot, TRIM_PCT, INJECT_SEED

# a test residual is an anomaly when it's more than this many times its group's training MAE
ANOMALY_K = 3

# features on unless turned off with --no-<flag>
DEFAULT_FEATURES = {
    "electricity": {"inc_lags", "inc_breaks", "inc_weekdays"},
    "heat": {"inc_lags", "inc_temp_history", "inc_usage_history", "inc_seasons"},
}

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

def train(model_type, utility_type, building, log_metrics,
          model_params=None, anomaly_k=ANOMALY_K, trim_pct=TRIM_PCT, **features):
  
  # linear models get one-hot hours, trees get sin/cos
  X, y, labels = preprocess(building, utility_type, baseline=model_type != "hgb",
                            inject=True, inject_seed=INJECT_SEED, **features)

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

    out_dir = Path(__file__).resolve().parent / "models" / args.utility_type
    save_run(model, metrics, metadata, out_dir)