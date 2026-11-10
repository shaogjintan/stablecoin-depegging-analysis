"""Estimator registry for modelling.training.train_models and benchmarks.

Model implementations expose a common fitting and prediction interface.
Sample preparation lives in data_processing/; training settings and logistic
C selection live in modelling/training/. Existing class module paths remain
stable so retained joblib model files can be loaded.
"""
from .base import BaseModel, MedianImputer
from .baseline import ConstantPriorModel, PersistenceModel
from .random_forest_model import RFModel
from .xgboost_model import XGBModel
from .logreg import LogRegModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "baseline": ConstantPriorModel,
    "persistence": PersistenceModel,
    "xgboost": XGBModel,
    "random_forest": RFModel,
    "logreg": LogRegModel,
}

__all__ = ["BaseModel", "MedianImputer", "ConstantPriorModel",
           "PersistenceModel", "XGBModel", "RFModel", "LogRegModel", "MODEL_REGISTRY",
           "get_model"]


def get_model(name: str, **kwargs) -> BaseModel:
    """Instantiate a registered model by its CLI name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}. "
            "Register it in modelling/models/__init__.py."
        )
    return MODEL_REGISTRY[name](**kwargs)
