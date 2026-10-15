import argparse
import joblib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error
)
from energyiq.ml.preprocess import preprocess

# train: Sep 2024 - Aug 2025, test/replay: Sep 2025 - Aug 2026 (a full year each)
TRAIN_END = "2025-09-01"
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
        "building",
        help="Building to train on",
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

def print_mae_breakdowns(X_test, y_test, y_pred):
  """Test MAE per month and per meter, console output only."""
  abs_err = (y_test - y_pred).abs()

  print("\nTest MAE by month:")
  print(abs_err.groupby(abs_err.index.to_period("M")).mean().round(3).to_string())

  # meterid is one-hot encoded; a single-meter utility has no meterid columns left
  meter_cols = [c for c in X_test.columns if c.startswith("meterid_")]
  if meter_cols:
    meters = X_test[meter_cols].idxmax(axis=1).str.removeprefix("meterid_")
  else:
    meters = pd.Series("only meter", index=X_test.index)

  print("\nTest MAE by meter:")
  print(abs_err.groupby(meters.to_numpy()).mean().round(3).to_string())

def train(model_type, utility_type, building, log_metrics, half_life=None,
          model_params=None, **features):
  
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

  mse = mean_squared_error(y_test, y_pred)
  metrics = {
    "MSE": mse,
    "RMSE": mse ** 0.5,
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "train_MAE": mean_absolute_error(y_train, model.predict(X_train)),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_features": X.shape[1],
    "fit_seconds": fit_seconds,
  }

  if log_metrics:
    print(metrics)
    print_mae_breakdowns(X_test, y_test, y_pred)

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
  }
  feature_kwargs = {name: True for name, on in all_flags.items() if on}

  model, metrics, feature_columns = train(
    model_type=args.model_type,
    utility_type=args.utility_type,
    building=args.building,
    log_metrics=args.verbose,
    half_life=args.half_life,
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