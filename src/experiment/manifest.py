"""
Experiment manifest — complete record of everything about an experiment run.
Written to reports/summaries/{experiment_id}_manifest.json.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_package_versions() -> dict[str, str]:
    packages = [
        "numpy", "pandas", "scikit-learn", "imbalanced-learn",
        "matplotlib", "scipy", "pyyaml", "omegaconf", "joblib", "torch",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


@dataclass
class ExperimentManifest:
    experiment_id: str
    run_id: str
    timestamp: str
    git_hash: str
    python_version: str
    os_info: str
    dependencies: dict[str, str]
    config: dict
    dataset_report: Optional[dict] = None
    split_indices: Optional[dict] = None
    resampling_metadata: Optional[dict] = None
    model_params: Optional[dict] = None
    calibrator_params: Optional[dict] = None
    audit_results: Optional[dict] = None
    results: Optional[dict] = None
    output_files: list[str] = field(default_factory=list)
    status: str = "running"   # running | completed | failed
    error_message: Optional[str] = None

    @classmethod
    def create(cls, experiment_id: str, config: dict) -> "ExperimentManifest":
        ts = datetime.now().isoformat()
        # Include microseconds to prevent collision when runs start in the same second
        run_id = f"{experiment_id}_run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        return cls(
            experiment_id=experiment_id,
            run_id=run_id,
            timestamp=ts,
            git_hash=_get_git_hash(),
            python_version=sys.version,
            os_info=platform.platform(),
            dependencies=_get_package_versions(),
            config=config,
        )

    def mark_completed(self) -> None:
        self.status = "completed"

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error_message = error

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.run_id}_manifest.json"
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)
        logger.info("Manifest saved to %s", out_path)
        return out_path

    def to_dict(self) -> dict:
        return asdict(self)
