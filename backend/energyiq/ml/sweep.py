"""Grid searches, every config logged as its own W&B run.
Stage 0: feature sets per (utility, model) at default hyperparameters.
Stage A: HGB hyperparameters on the default features (DEFAULT_FEATURES)."""
import argparse
import wandb
from itertools import product

from energyiq.config import EXPERIMENT_TRAIN_END, ANOMALY_K, UTILITIES, default_features
from energyiq.ml.train import train
from energyiq.ml.evaluation import TRIM_PCT

MODELS = ["regression", "hgb"]

# preprocess feature flags searched in stage 0
FEATURE_FLAGS = {
    "heat": ["inc_seasons", "inc_temp_history", "inc_usage_history"],
    "electricity": ["inc_lags", "inc_weekends", "inc_breaks", "inc_events"],
}

HGB_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [200, 500],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 100],
}


def features_name(features):
    """e.g. {"inc_lags": True, "inc_breaks": False} -> "lags"."""
    on = [flag.removeprefix("inc_") for flag, value in features.items() if value]
    return "+".join(on) or "none"


def plan_stage0(building):
    runs = []
    for utility, flags in FEATURE_FLAGS.items():
        for values in product([False, True], repeat=len(flags)):
            features = dict(zip(flags, values))
            for model_type in MODELS:
                runs.append({
                    "name": f"{model_type}-{utility}-{features_name(features)}",
                    "model_type": model_type,
                    "utility_type": utility,
                    "features": features,
                })
    return runs


def plan_stageA(building):
    runs = []
    for utility in UTILITIES:
        for values in product(*HGB_GRID.values()):
            params = dict(zip(HGB_GRID, values))
            # HGB's built-in early stopping holds out a shuffled 10%, so turn it off
            params["early_stopping"] = False
            runs.append({
                "name": f"hgb-{utility}-lr{params['learning_rate']}-it{params['max_iter']}"
                        f"-leaves{params['max_leaf_nodes']}-msl{params['min_samples_leaf']}",
                "model_type": "hgb",
                "utility_type": utility,
                "features": default_features(utility),
                "model_params": params,
            })
    return runs


def run_config(building, run):
    return train(
        model_type=run["model_type"],
        utility_type=run["utility_type"],
        building=building,
        log_metrics=False,
        model_params=run.get("model_params"),
        **run["features"],
    )


def log_to_wandb(args, run, model, metrics):
    config = {
        "building": args.building,
        "utility": run["utility_type"],
        "model": run["model_type"],
        **run["features"],
        "anomaly_k": ANOMALY_K,
        "trim_pct": TRIM_PCT,
        "model_params": model.get_params(),
        "train_end": EXPERIMENT_TRAIN_END,
        "stage": args.stage,
    }
    wandb.init(project=args.project, group=f"stage{args.stage}/{args.building}",
               name=run["name"], config=config, tags=[args.building, f"stage{args.stage}"], reinit=True)
    try:
        wandb.log(metrics)
    finally:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search, logged to W&B")
    parser.add_argument("building", help="Building to sweep")
    parser.add_argument("--stage", choices=["0", "A"], default="0",
                        help="0: feature sets, A: HGB hyperparameters (default: 0)")
    parser.add_argument("--project", default="energyiq", help="W&B project")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N configs (smoke testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned configs and exit")
    return parser.parse_args()


PLANS = {"0": plan_stage0, "A": plan_stageA}
def main():
    args = parse_args()
    runs = PLANS[args.stage](args.building)
    print(f"Planned: {len(runs)} runs")
    runs = runs[:args.limit]

    if args.dry_run:
        for run in runs:
            print(" ", run["name"])
        return

    for run in runs:
        try:
            model, metrics, _ = run_config(args.building, run)
        except Exception as e:
            print(f"FAILED {run['name']}: {e!r}")
            continue
        log_to_wandb(args, run, model, metrics)


if __name__ == "__main__":
    main()