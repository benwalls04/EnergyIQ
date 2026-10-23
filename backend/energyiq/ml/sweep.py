import argparse
import wandb
from itertools import product

from energyiq.config import TRAIN_END, ANOMALY_K
from energyiq.ml.train import train
from energyiq.ml.evaluation import TRIM_PCT

HGB_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [200, 500],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 100],
}

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
        "model_params": model.get_params(),
        "train_end": TRAIN_END,
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
    parser.add_argument("--project", default="energyiq", help="W&B project")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N configs (smoke testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned configs and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    ## work in progress
    return 


if __name__ == "__main__":
    main()