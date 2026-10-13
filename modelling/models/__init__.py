"""
Model registry used by `modelling/train.py --model {name}`.

Teammates: add your model class here once it subclasses BaseModel
(Rule 3 in agent.md -- data access goes through modelling/prepare_data.py).

    "logistic"      -> Javier  (Phase 3+)
    "random_forest" -> Russell (Phase 3+)
    "xgboost"       -> Sasi   (Phase 4)
"""
from .base import BaseModel, MedianImputer
from .baseline import ConstantPriorModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "baseline": ConstantPriorModel,
}

__all__ = ["BaseModel", "MedianImputer", "ConstantPriorModel",
           "MODEL_REGISTRY", "get_model"]


def get_model(name: str, **kwargs) -> BaseModel:
    """Instantiate a registered model by its CLI name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}. "
            "Register it in modelling/models/__init__.py."
        )
    return MODEL_REGISTRY[name](**kwargs)
