import argparse
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    mean_squared_error
)
from ml.preprocess import preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for a building")

    parser.add_argument(
        "building",
        help="Building to train on",
    )
    parser.add_argument(
        "--model-type",
        choices=["regression"],        
        default="regression",
        help="Model to train (default: regression)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the trained model to this path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extra output",
    )

    return parser.parse_args()

def train(model_type, building, log_metrics):
  X, y = preprocess(building)

  # no shuffle, preprocess needs to sort the data in chronological order
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
  )

  match model_type:
    case "regression":
      model = LinearRegression()
    case _:
      model = LinearRegression() 

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MAPE": mean_absolute_percentage_error(y_test, y_pred)
  }

  if log_metrics: 
    print(metrics)

  return model, metrics

if __name__=="__main__":
  args = parse_args()
  model, metrics = train(
    model_type=args.model_type,
    building=args.building,
    log_metrics=args.verbose
  )

  if args.save:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "models" / args.building
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.model_type}_{timestamp}.joblib"


    bundle = {
      "model": model,
      "metadata": {
        "building": args.building,
        "model_type": args.model_type,
        "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": metrics
      }
    }

    joblib.dump(bundle, path)
    print("Saved to ", path)