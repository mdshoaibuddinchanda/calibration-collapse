from .base import BaseClassifier
from .logistic import LogisticClassifier
from .random_forest import RandomForestClassifier_
from .mlp import MLPClassifier_
from .gradient_boosting import GradientBoostingClassifier_

MODEL_REGISTRY: dict[str, type[BaseClassifier]] = {
    "logistic_regression": LogisticClassifier,
    "random_forest": RandomForestClassifier_,
    "mlp": MLPClassifier_,
    "gradient_boosting": GradientBoostingClassifier_,
}


def get_model(name: str, seed: int = 42, **kwargs) -> BaseClassifier:
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name_lower](seed=seed, **kwargs)


__all__ = [
    "BaseClassifier",
    "LogisticClassifier",
    "RandomForestClassifier_",
    "MLPClassifier_",
    "GradientBoostingClassifier_",
    "MODEL_REGISTRY",
    "get_model",
]
