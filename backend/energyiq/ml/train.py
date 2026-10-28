import argparse
import joblib
import json
import shutil
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor
from energyiq.config import ANOMALY_K, HGB_DEFAULTS, TRAIN_END, EXPERIMENT_MODELS_DIR, REPLAY_MODELS_DIR
from energyiq.ml.preprocess import preprocess
from energyiq.ml.evaluation import evaluate
from energyiq.ml.thresholds import fit_thresholds

def parse_args():
    parser = argparse.ArgumentParser(description="Train the expected-usage model for every building and utility")

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

def predict(model, X):
    """Predictions for X, its dummy columns lined up with the ones the model was fit on."""
    return model.predict(X.reindex(columns=model.feature_names_in_, fill_value=0))

def train(log_metrics, model_params=None):
    X, y, meta = preprocess()

    # split on dates: train on the first year, score on the test/replay year
    train_rows = X.index < TRAIN_END
    X_train, y_train, meta_train = X[train_rows], y[train_rows], meta[train_rows]
    X_test, y_test, meta_test = X[~train_rows], y[~train_rows], meta[~train_rows]

    model = HistGradientBoostingRegressor(**{**HGB_DEFAULTS, **(model_params or {})})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    thresholds = fit_thresholds(meta_train, y_train - train_pred, ANOMALY_K)

    metrics = {
        **evaluate(y_test, y_pred, y_train, train_pred, meta_test, verbose=log_metrics),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
    }

    if log_metrics:
        print(f"features: {list(X.columns)}")
        print(metrics)

    return model, metrics, thresholds

def train_replay(train_end):
    X, y, meta = preprocess(train_end=train_end, load_end=train_end)

    model = HistGradientBoostingRegressor(**HGB_DEFAULTS)
    model.fit(X, y)
    thresholds = fit_thresholds(meta, y - model.predict(X), ANOMALY_K)

    bundle = {"model": model, "train_end": train_end}
    return bundle, thresholds

def save_replay_model(bundle, train_end):
    REPLAY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPLAY_MODELS_DIR / f"{train_end}.joblib"
    joblib.dump(bundle, path)
    print("Saved to", path)

if __name__ == "__main__":
    args = parse_args()

    model, metrics, thresholds = train(log_metrics=args.verbose)

    if args.save:
        metadata = {
            "model_type": "hgb",
            "model_params": model.get_params(),
            "train_end": TRAIN_END,
            "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        save_run(model, metrics, metadata, EXPERIMENT_MODELS_DIR)