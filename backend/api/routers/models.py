from fastapi import APIRouter, HTTPException, Request

from api.deps import DB
from api.schemas import ModelInfo, PerformanceReport
from energyiq.ml.evaluation import TRIM_PCT

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models(request: Request):
    """The replay models and the period each one predicts. The last has no end: it's the current model."""
    store = request.app.state.models
    models = []
    for i, bundle in enumerate(store.bundles):
        start, end = store.window(i)
        models.append({
            "train_end": bundle["train_end"],
            "predicts_from": start,
            "predicts_until": end,
            "n_features": len(bundle["model"].feature_names_in_),
        })
    return models


@router.get("/performance", response_model=PerformanceReport)
def read_performance(request: Request, db: DB):
    """
    Accuracy of each model on the period it predicted, overall, per utility and per building and
    utility. The first call scores every reading, so it takes a few seconds; later calls are cached.
    A model with no later readings yet has empty groups.
    """
    store = request.app.state.models
    if not store.bundles:
        raise HTTPException(503, "No replay models found. Run `python -m scripts.build_db --replay`.")
    return {"trim_pct": TRIM_PCT, "models": store.performance(db)}
