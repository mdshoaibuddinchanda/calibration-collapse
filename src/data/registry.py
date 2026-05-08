"""
Central registry mapping dataset names → validated DatasetConfig objects.
Acts as the single source of truth for all datasets the system knows about.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "name", "path", "target_column", "positive_class",
    "missing_value_strategy",
}

VALID_MISSING_STRATEGIES = {"median", "mean", "mode", "drop"}


@dataclass
class DatasetConfig:
    name: str
    path: Path
    target_column: str
    positive_class: int | str
    feature_columns: Optional[list[str]]
    missing_value_strategy: str
    imbalance_ratio: Optional[float]
    encoding: str = "utf-8"
    separator: str = ","
    expected_imbalance_range: tuple[float, float] = (1.0, 1000.0)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.missing_value_strategy not in VALID_MISSING_STRATEGIES:
            raise ValueError(
                f"Dataset '{self.name}': missing_value_strategy must be one of "
                f"{VALID_MISSING_STRATEGIES}, got '{self.missing_value_strategy}'"
            )
        if isinstance(self.expected_imbalance_range, list):
            self.expected_imbalance_range = tuple(self.expected_imbalance_range)


class DatasetRegistry:
    """
    Loads all dataset YAML configs from a directory, validates them,
    and provides a lookup interface for the rest of the pipeline.
    """

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = Path(config_dir)
        self._registry: dict[str, DatasetConfig] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Scan config_dir for *.yaml files (excluding _template) and register each."""
        yaml_files = [
            f for f in self._config_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]
        if not yaml_files:
            logger.warning("No dataset configs found in %s", self._config_dir)
        for yaml_file in yaml_files:
            try:
                cfg = self._parse_yaml(yaml_file)
                self.register(cfg)
            except Exception as exc:
                logger.error("Failed to load dataset config %s: %s", yaml_file, exc)

    def _parse_yaml(self, yaml_file: Path) -> DatasetConfig:
        with open(yaml_file, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        missing = REQUIRED_FIELDS - set(raw.keys())
        if missing:
            raise ValueError(
                f"Dataset config '{yaml_file}' is missing required fields: {missing}"
            )

        return DatasetConfig(
            name=raw["name"],
            path=Path(raw["path"]),
            target_column=raw["target_column"],
            positive_class=raw["positive_class"],
            feature_columns=raw.get("feature_columns"),
            missing_value_strategy=raw["missing_value_strategy"],
            imbalance_ratio=raw.get("imbalance_ratio"),
            encoding=raw.get("encoding", "utf-8"),
            separator=raw.get("separator", ","),
            expected_imbalance_range=tuple(
                raw.get("expected_imbalance_range", [1.0, 1000.0])
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, config: DatasetConfig) -> None:
        if config.name in self._registry:
            logger.warning("Dataset '%s' already registered — overwriting.", config.name)
        self._registry[config.name] = config
        logger.debug("Registered dataset: %s", config.name)

    def get(self, name: str) -> DatasetConfig:
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Dataset '{name}' not found in registry. "
                f"Available datasets: {available}"
            )
        return self._registry[name]

    def list_registered(self) -> list[str]:
        return sorted(self._registry.keys())

    def validate_all(self) -> list[str]:
        """
        Validate that every registered dataset's CSV file exists on disk.
        Returns a list of validation error strings (empty = all OK).
        """
        errors: list[str] = []
        for name, cfg in self._registry.items():
            if not cfg.path.exists():
                errors.append(
                    f"Dataset '{name}': file not found at '{cfg.path}'"
                )
        return errors

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"DatasetRegistry({self.list_registered()})"
