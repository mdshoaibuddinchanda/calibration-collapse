from .base import BaseCalibrator
from .temperature_scaling import TemperatureScaling, _LOGIT_MODELS
from .isotonic import IsotonicCalibrator
from .platt import PlattScaling
from .per_class_adaptive import PerClassAdaptiveCalibrator

CALIBRATOR_REGISTRY: dict[str, type[BaseCalibrator]] = {
    "temperature_scaling": TemperatureScaling,
    "isotonic": IsotonicCalibrator,
    "platt": PlattScaling,
    "per_class_adaptive": PerClassAdaptiveCalibrator,
}


def get_calibrator(name: str, model_name: str = "unknown", **kwargs) -> BaseCalibrator:
    """
    Instantiate a calibrator by name.

    model_name is used to select the correct temperature scaling mode:
      - LR and MLP use logit-mode (use_logit=True)
      - RF and GBM use power-mode (use_logit=False)
    """
    name_lower = name.lower()
    if name_lower not in CALIBRATOR_REGISTRY:
        raise ValueError(
            f"Unknown calibrator '{name}'. Available: {list(CALIBRATOR_REGISTRY.keys())}"
        )
    if name_lower == "temperature_scaling":
        use_logit = model_name.lower() in _LOGIT_MODELS
        return TemperatureScaling(use_logit=use_logit)
    if name_lower == "per_class_adaptive":
        use_logit = model_name.lower() in _LOGIT_MODELS
        return PerClassAdaptiveCalibrator(use_logit=use_logit)
    return CALIBRATOR_REGISTRY[name_lower](**kwargs)


__all__ = [
    "BaseCalibrator",
    "TemperatureScaling",
    "IsotonicCalibrator",
    "PlattScaling",
    "PerClassAdaptiveCalibrator",
    "CALIBRATOR_REGISTRY",
    "get_calibrator",
]
