import argparse
import joblib
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error
)
from energyiq.ml.preprocess import preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for a building")

    parser.add_argument(
        "building",
        help="Building to train on",
    )
    parser.add_argument(
        "--model-type",
        choices=["regression", "hgb"],        
        default="regression",
        help="Model to train (default: regression)",
    )
    parser.add_argument(
        "--utility-type",
        choices=["electricity", "heat", "both"],        
        default="both",
        help="Utility type",
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

def train(model_type, utility_type, building, log_metrics, half_life=None):
  X, y = preprocess(building, utility_type)

  # no shuffle, preprocess needs to sort the data in chronological order
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
  )

  match model_type:
    case "linear_regression":
      model = LinearRegression()
    case "hgb":
      model = HistGradientBoostingRegressor(random_state=42)
    case _:
      model = LinearRegression() 

  # optinal time decay, more recent entries weighted more
  weights = None
  if half_life is not None:
      weights = time_decay_weights(X_train.index, half_life)

  model.fit(X_train, y_train, sample_weight=weights)
  y_pred = model.predict(X_test)

  metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
  }

  if log_metrics: 
    print(metrics)

  return model, metrics

if __name__=="__main__":
  args = parse_args()
  model, metrics = train(
    model_type=args.model_type,
    utility_type=args.utility_type if args.utility_type != "both" else None,
    building=args.building,
    log_metrics=args.verbose,
    half_life=args.half_life
  )

  if args.save:
    metadata = {
        "building": args.building,
        "utility_type": args.utility_type,
        "model_type": args.model_type,
        "half_life": args.half_life,
        "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    out_dir = Path(__file__).resolve().parent / "models" / args.building / args.utility_type
    save_run(model, metrics, metadata, out_dir)