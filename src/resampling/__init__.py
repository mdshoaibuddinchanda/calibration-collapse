from .base import BaseResampler
from .smote_resampler import SMOTEResampler
from .adasyn_resampler import ADASYNResampler
from .borderline_resampler import BorderlineSMOTEResampler
from .class_weight import ClassWeightResampler

RESAMPLER_REGISTRY: dict[str, type[BaseResampler]] = {
    "none": ClassWeightResampler,   # passthrough with no weighting
    "smote": SMOTEResampler,
    "adasyn": ADASYNResampler,
    "borderline_smote": BorderlineSMOTEResampler,
    "class_weight": ClassWeightResampler,
}


def get_resampler(name: str, **kwargs) -> BaseResampler:
    name_lower = name.lower()
    if name_lower not in RESAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown resampler '{name}'. Available: {list(RESAMPLER_REGISTRY.keys())}"
        )
    if name_lower == "none":
        return ClassWeightResampler(use_class_weight=False, **kwargs)
    return RESAMPLER_REGISTRY[name_lower](**kwargs)


__all__ = [
    "BaseResampler",
    "SMOTEResampler",
    "ADASYNResampler",
    "BorderlineSMOTEResampler",
    "ClassWeightResampler",
    "RESAMPLER_REGISTRY",
    "get_resampler",
]
