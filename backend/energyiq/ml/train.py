import argparse
import joblib
import json
import shutil
import time
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from energyiq.config import (ANOMALY_K, DEFAULT_FEATURES, HGB_DEFAULTS, EXPERIMENT_TRAIN_END,
                             EXPERIMENT_MODELS_DIR, REPLAY_MODELS_DIR, default_features)
from energyiq.ml.preprocess import preprocess
from energyiq.ml.thresholds import compute_group_mae, flag_anomalies
from energyiq.ml.evaluation import evaluate, TRIM_PCT

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
        choices=["regression", "hgb"],
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

    shared = parser.add_argument_group("heat and electricity features")
    shared.add_argument(
        "--lags", dest="inc_lags",
        action=argparse.BooleanOptionalAction,
        help="Add usage 24h and 168h earlier (on by default for both)",
    )

    heat = parser.add_argument_group("heat features")
    heat.add_argument(
        "--seasons", dest="inc_seasons",
        action=argparse.BooleanOptionalAction,
        help="Add one-hot season columns (on by default)",
    )
    heat.add_argument(
        "--months", dest="inc_months",
        action=argparse.BooleanOptionalAction,
        help="Add a months flag",
    )
    heat.add_argument(
        "--temp-history", dest="inc_temp_history",
        action=argparse.BooleanOptionalAction,
        help="Add rolling temperature means/min over recent hours (on by default)",
    )
    heat.add_argument(
        "--usage-history", dest="inc_usage_history",
        action=argparse.BooleanOptionalAction,
        help="Add the meter's mean usage over a past 24h window (on by default)",
    )

    electricity = parser.add_argument_group("electricity features")
    electricity.add_argument(
        "--weekends", dest="inc_weekends",
        action=argparse.BooleanOptionalAction,
        help="Add an is_weekend flag",
    )
    electricity.add_argument(
        "--breaks", dest="inc_breaks",
        action=argparse.BooleanOptionalAction,
        help="Add an in_classes flag (0 during academic breaks) (on by default)",
    )
    electricity.add_argument(
        "--weekdays", dest="inc_weekdays",
        action=argparse.BooleanOptionalAction,
        help="Add a weekday flag (on by default)",
    )
    electricity.add_argument(
        "--events", dest="inc_events",
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
    out_dir = REPLAY_MODELS_DIR / train_end
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / f"{utility_type}.joblib")
    print("Saved to", out_dir / f"{utility_type}.joblib")

def train(model_type, utility_type, building, log_metrics,
          model_params=None, anomaly_k=ANOMALY_K, **features):

    # linear models get one-hot hours, trees get sin/cos
    X, y, meta = preprocess(building, utility_type, baseline=model_type != "hgb", inject=True, **features)

    if log_metrics:
        print(f"{len(X)} rows, {X.shape[1]} features: {list(X.columns)}")

    # split on dates: train on the first year, score on the test/replay year
    train_rows = X.index < EXPERIMENT_TRAIN_END
    X_train, y_train, meta_train = X[train_rows], y[train_rows], meta[train_rows]
    X_test, y_test, meta_test = X[~train_rows], y[~train_rows], meta[~train_rows]

    match model_type:
        case "regression":
            model = LinearRegression()
        case "hgb":
            model = HistGradientBoostingRegressor(**{**HGB_DEFAULTS, **(model_params or {})})
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    anomalous = flag_anomalies(meta_train, y_train - train_pred, meta_test, y_test - y_pred,
                               utility_type, anomaly_k)
    print(f"flagged {anomalous.sum()} of {len(y_test)} test rows as anomalies "
          f"({anomalous.mean():.2%}, k={anomaly_k})")

    metrics = {
        **evaluate(y_test, y_pred, anomalous, meta_test, y_train, train_pred, verbose=log_metrics),
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
    Returned as a bundle with what AnomalyDetector needs: the model, its features and feature
    columns, and the per-group training MAEs its thresholds come from.
    """
    features = default_features(utility_type)
    X, y, meta = preprocess("All", utility_type, train_end=train_end, data_end=train_end, **features)

    model = HistGradientBoostingRegressor(**HGB_DEFAULTS)
    model.fit(X, y)
    train_resid = y - model.predict(X)
    group_mae, fallback_mae = compute_group_mae(meta, train_resid, utility_type)

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

if __name__ == "__main__":
    args = parse_args()
    # a flag left unset (None) is on if it's a default for this utility; only the ones that
    # end up on are passed, so a heat-only flag on an electricity run still errors out
    defaults = DEFAULT_FEATURES[args.utility_type]
    feature_kwargs = {name: True for name, on in vars(args).items()
                      if name.startswith("inc_") and (name in defaults if on is None else on)}

    model, metrics, feature_columns = train(
        model_type=args.model_type,
        utility_type=args.utility_type,
        building=args.building,
        log_metrics=args.verbose,
        anomaly_k=args.anomaly_k,
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
            "trim_pct": TRIM_PCT,
            "trained_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        out_dir = EXPERIMENT_MODELS_DIR / args.utility_type
        save_run(model, metrics, metadata, out_dir)
