"""
Calibration Stress Test Suite — Controlled Synthetic Benchmark Generator.

These are NOT toy datasets. Each generator targets a specific calibration
failure mechanism, with parameterized severity levels. This makes them
scientific instruments for mechanism isolation, not data augmentation.

Design principles:
  - Every generator accepts severity: 'mild' | 'moderate' | 'severe'
  - Every generator accepts explicit params for full reproducibility
  - Severity levels map to published calibration instability thresholds
  - All outputs are deterministic: same params + seed → identical file + SHA256

Failure mechanisms covered:
  Mode                  Mechanism targeted
  ─────────────────────────────────────────────────────────────────────
  extreme_imbalance     IR sensitivity (IR = 10 / 50 / 100)
  noisy_minority        Label corruption robustness (p = 0.05 / 0.15 / 0.30)
  boundary_overlap      Decision boundary uncertainty (σ = 2.0 / 1.5 / 0.8)
  confidence_collapse   Confidence zone collapse (zone = 0.45 / 0.30 / 0.15)
  feature_corruption    Feature instability (rate = 0.10 / 0.25 / 0.50)
  distribution_shift    Covariate shift (Δμ = 0.5 / 1.5 / 3.0)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Severity = Literal["mild", "moderate", "severe"]


# ---------------------------------------------------------------------------
# Severity parameter tables — the scientific backbone of the benchmark
# ---------------------------------------------------------------------------

# Each entry: (imbalance_ratio, mechanism_param)
_SEVERITY: dict[str, dict[Severity, tuple]] = {
    "extreme_imbalance": {
        "mild":     (10.0,  None),
        "moderate": (50.0,  None),
        "severe":   (100.0, None),
    },
    "noisy_minority": {
        "mild":     (10.0, 0.05),   # 5% label flip
        "moderate": (10.0, 0.15),   # 15% label flip
        "severe":   (10.0, 0.30),   # 30% label flip
    },
    "boundary_overlap": {
        "mild":     (10.0, 2.0),    # σ=2.0 → classes well separated
        "moderate": (10.0, 1.5),    # σ=1.5 → moderate overlap
        "severe":   (10.0, 0.8),    # σ=0.8 → heavy overlap
    },
    "confidence_collapse": {
        "mild":     (20.0, 0.45),   # predictions near 0.45 — close to boundary
        "moderate": (20.0, 0.30),   # predictions near 0.30 — moderate collapse
        "severe":   (20.0, 0.15),   # predictions near 0.15 — severe collapse
    },
    "feature_corruption": {
        "mild":     (10.0, 0.10),   # 10% cells zeroed
        "moderate": (10.0, 0.25),   # 25% cells zeroed
        "severe":   (10.0, 0.50),   # 50% cells zeroed
    },
    "distribution_shift": {
        "mild":     (10.0, 0.5),    # Δμ = 0.5
        "moderate": (10.0, 1.5),    # Δμ = 1.5
        "severe":   (10.0, 3.0),    # Δμ = 3.0
    },
}

# Confidence zone targets for targeted collapse experiments
CONFIDENCE_ZONES = [0.1, 0.3, 0.5, 0.7, 0.9]


class GenerationMode(str, Enum):
    EXTREME_IMBALANCE    = "extreme_imbalance"
    NOISY_MINORITY       = "noisy_minority"
    BOUNDARY_OVERLAP     = "boundary_overlap"
    CONFIDENCE_COLLAPSE  = "confidence_collapse"
    FEATURE_CORRUPTION   = "feature_corruption"
    DISTRIBUTION_SHIFT   = "distribution_shift"
    CONFIDENCE_ZONE      = "confidence_zone"   # new: targeted zone collapse


@dataclass
class SyntheticConfig:
    mode: GenerationMode
    n_samples: int = 5000
    n_features: int = 10
    seed: int = 42
    severity: Optional[Severity] = None   # if set, overrides explicit params

    # Explicit params (used when severity=None, or overridden by severity)
    imbalance_ratio: float = 100.0
    noise_rate: float = 0.20
    overlap_sigma: float = 1.5
    collapse_region: float = 0.30
    collapse_variance: float = 0.05       # NEW: controls width of collapse zone
    corruption_rate: float = 0.30
    shift_magnitude: float = 2.0
    confidence_zone: float = 0.5          # NEW: for CONFIDENCE_ZONE mode

    def __post_init__(self) -> None:
        """Apply severity presets if severity is set."""
        if self.severity is not None:
            mode_key = self.mode.value
            if mode_key in _SEVERITY:
                ir, param = _SEVERITY[mode_key][self.severity]
                self.imbalance_ratio = ir
                if param is not None:
                    if mode_key == "noisy_minority":
                        self.noise_rate = param
                    elif mode_key == "boundary_overlap":
                        self.overlap_sigma = param
                    elif mode_key == "confidence_collapse":
                        self.collapse_region = param
                    elif mode_key == "feature_corruption":
                        self.corruption_rate = param
                    elif mode_key == "distribution_shift":
                        self.shift_magnitude = param


class SyntheticDataGenerator:
    """
    Calibration Stress Test Suite generator.

    Usage — severity-based (recommended for experiments):
        cfg = SyntheticConfig(
            mode=GenerationMode.CONFIDENCE_COLLAPSE,
            severity='severe',
            seed=42,
        )
        path = generator.generate(cfg)

    Usage — explicit params (for fine-grained control):
        cfg = SyntheticConfig(
            mode=GenerationMode.CONFIDENCE_COLLAPSE,
            collapse_region=0.15,
            collapse_variance=0.02,
            imbalance_ratio=20.0,
            seed=42,
        )

    Usage — confidence zone sweep:
        for zone in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cfg = SyntheticConfig(
                mode=GenerationMode.CONFIDENCE_ZONE,
                confidence_zone=zone,
                seed=42,
            )
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, config: SyntheticConfig) -> Path:
        """Generate dataset, write CSV + .meta, return CSV path."""
        rng = np.random.default_rng(config.seed)

        dispatch = {
            GenerationMode.EXTREME_IMBALANCE:   self._extreme_imbalance,
            GenerationMode.NOISY_MINORITY:       self._noisy_minority,
            GenerationMode.BOUNDARY_OVERLAP:     self._boundary_overlap,
            GenerationMode.CONFIDENCE_COLLAPSE:  self._confidence_collapse,
            GenerationMode.FEATURE_CORRUPTION:   self._feature_corruption,
            GenerationMode.DISTRIBUTION_SHIFT:   self._distribution_shift,
            GenerationMode.CONFIDENCE_ZONE:      self._confidence_zone,
        }

        df = dispatch[config.mode](config, rng)
        out_path = self._output_path(config)
        df.to_csv(out_path, index=False)

        sha256 = self._sha256_file(out_path)
        self._write_meta(config, out_path, sha256, len(df))

        logger.info(
            "Generated '%s' (severity=%s): %d samples → %s",
            config.mode.value, config.severity or "explicit", len(df), out_path.name,
        )
        return out_path

    def generate_severity_sweep(
        self,
        mode: GenerationMode,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> list[Path]:
        """
        Generate all three severity levels for a given mode.
        Returns list of 3 CSV paths: [mild, moderate, severe].
        """
        paths = []
        for sev in ("mild", "moderate", "severe"):
            cfg = SyntheticConfig(
                mode=mode, severity=sev,  # type: ignore[arg-type]
                n_samples=n_samples, seed=seed,
            )
            paths.append(self.generate(cfg))
        return paths

    def generate_confidence_zone_sweep(
        self,
        imbalance_ratio: float = 20.0,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> list[Path]:
        """
        Generate confidence-zone collapse datasets for all 5 zones.
        Zones: [0.1, 0.3, 0.5, 0.7, 0.9]

        This produces a calibration degradation curve across confidence regions —
        a key contribution for mechanism-level analysis.
        """
        paths = []
        for zone in CONFIDENCE_ZONES:
            cfg = SyntheticConfig(
                mode=GenerationMode.CONFIDENCE_ZONE,
                confidence_zone=zone,
                imbalance_ratio=imbalance_ratio,
                n_samples=n_samples,
                seed=seed,
            )
            paths.append(self.generate(cfg))
        return paths

    # ------------------------------------------------------------------
    # Generation modes
    # ------------------------------------------------------------------

    def _extreme_imbalance(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        IR sensitivity benchmark.
        Well-separated classes — isolates the effect of IR alone on calibration.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority

        X_maj = rng.normal(loc=0.0, scale=1.0, size=(n_majority, cfg.n_features))
        X_min = rng.normal(loc=3.0, scale=1.0, size=(n_minority, cfg.n_features))
        X = np.vstack([X_maj, X_min])
        y = np.array([0] * n_majority + [1] * n_minority)
        return self._to_df(X, y, cfg.n_features)

    def _noisy_minority(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Label corruption robustness benchmark.
        Flips minority labels at rate noise_rate — tests calibration under
        label noise, which is common in real imbalanced datasets.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority

        X_maj = rng.normal(loc=0.0, scale=1.0, size=(n_majority, cfg.n_features))
        X_min = rng.normal(loc=2.5, scale=1.0, size=(n_minority, cfg.n_features))
        X = np.vstack([X_maj, X_min])
        y = np.array([0] * n_majority + [1] * n_minority)

        flip_mask = rng.random(n_minority) < cfg.noise_rate
        minority_indices = np.arange(n_majority, n_majority + n_minority)
        y[minority_indices[flip_mask]] = 0

        df = self._to_df(X, y, cfg.n_features)
        df["noise_rate"] = cfg.noise_rate   # metadata column for analysis
        return df

    def _boundary_overlap(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Decision boundary uncertainty benchmark.
        Controls class overlap via overlap_sigma — lower sigma = more overlap.
        Tests calibration near the decision boundary.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority

        X_maj = rng.normal(loc=0.0, scale=cfg.overlap_sigma, size=(n_majority, cfg.n_features))
        X_min = rng.normal(loc=cfg.overlap_sigma, scale=cfg.overlap_sigma, size=(n_minority, cfg.n_features))
        X = np.vstack([X_maj, X_min])
        y = np.array([0] * n_majority + [1] * n_minority)
        return self._to_df(X, y, cfg.n_features)

    def _confidence_collapse(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Confidence zone collapse benchmark.
        Samples concentrated near collapse_region with variance collapse_variance.
        Directly targets the core hypothesis: models collapse predictions to a
        narrow confidence band, causing severe miscalibration.

        collapse_region controls WHERE the collapse occurs.
        collapse_variance controls HOW TIGHT the collapse is.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority

        # Both classes have features concentrated near collapse_region
        # The tighter collapse_variance, the harder calibration becomes
        X = rng.normal(
            loc=cfg.collapse_region,
            scale=cfg.collapse_variance,
            size=(cfg.n_samples, cfg.n_features),
        )
        y = np.array([0] * n_majority + [1] * n_minority)
        rng.shuffle(y)

        df = self._to_df(X, y, cfg.n_features)
        df["collapse_region"] = cfg.collapse_region
        df["collapse_variance"] = cfg.collapse_variance
        return df

    def _feature_corruption(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Feature instability benchmark.
        Randomly zeros out feature values at corruption_rate.
        Tests calibration robustness to missing/corrupted features.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority

        X_maj = rng.normal(loc=0.0, scale=1.0, size=(n_majority, cfg.n_features))
        X_min = rng.normal(loc=2.0, scale=1.0, size=(n_minority, cfg.n_features))
        X = np.vstack([X_maj, X_min])

        corrupt_mask = rng.random(X.shape) < cfg.corruption_rate
        X[corrupt_mask] = 0.0

        y = np.array([0] * n_majority + [1] * n_minority)
        return self._to_df(X, y, cfg.n_features)

    def _distribution_shift(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Covariate shift benchmark.
        Train-like and test-like samples from shifted distributions.
        split_hint column: 0=train-like, 1=test-like.
        Tests calibration under distribution shift — common in deployment.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority

        n_maj_tr = n_majority // 2
        n_min_tr = n_minority // 2
        n_maj_te = n_majority - n_maj_tr
        n_min_te = n_minority - n_min_tr

        X_maj_tr = rng.normal(loc=0.0, scale=1.0, size=(n_maj_tr, cfg.n_features))
        X_min_tr = rng.normal(loc=2.0, scale=1.0, size=(n_min_tr, cfg.n_features))
        X_maj_te = rng.normal(loc=cfg.shift_magnitude, scale=1.0, size=(n_maj_te, cfg.n_features))
        X_min_te = rng.normal(loc=2.0 + cfg.shift_magnitude, scale=1.0, size=(n_min_te, cfg.n_features))

        X = np.vstack([X_maj_tr, X_min_tr, X_maj_te, X_min_te])
        y = np.array([0]*n_maj_tr + [1]*n_min_tr + [0]*n_maj_te + [1]*n_min_te)
        split_hint = np.array([0]*(n_maj_tr + n_min_tr) + [1]*(n_maj_te + n_min_te))

        df = self._to_df(X, y, cfg.n_features)
        df["split_hint"] = split_hint
        df["shift_magnitude"] = cfg.shift_magnitude
        return df

    def _confidence_zone(
        self, cfg: SyntheticConfig, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Targeted confidence zone benchmark (NEW).

        Generates samples whose true posterior probability is concentrated
        near a specific confidence zone (0.1, 0.3, 0.5, 0.7, 0.9).

        This allows measuring:
          - calibration error as a function of confidence zone
          - bin instability at specific confidence levels
          - adaptive calibration recovery per zone

        The zone is implemented by controlling the Bayes-optimal decision
        boundary: samples near zone p* have true P(y=1|x) ≈ p*.
        """
        n_minority = max(10, int(cfg.n_samples / (1 + cfg.imbalance_ratio)))
        n_majority = cfg.n_samples - n_minority
        zone = cfg.confidence_zone

        # Generate a 1D latent score that determines true class probability
        # Latent score concentrated near logit(zone) → P(y=1|x) ≈ zone
        logit_zone = np.log(zone / (1 - zone + 1e-8))
        latent = rng.normal(loc=logit_zone, scale=0.3, size=cfg.n_samples)

        # True probability from latent score
        p_true = 1.0 / (1.0 + np.exp(-latent))

        # Generate labels from true probability (Bernoulli)
        y_full = (rng.random(cfg.n_samples) < p_true).astype(int)

        # Enforce imbalance by subsampling majority
        minority_idx = np.where(y_full == 1)[0]
        majority_idx = np.where(y_full == 0)[0]

        if len(minority_idx) == 0 or len(majority_idx) == 0:
            # Fallback: assign labels directly
            y_full = np.array([0] * n_majority + [1] * n_minority)
            rng.shuffle(y_full)
        else:
            n_min_keep = min(len(minority_idx), n_minority)
            n_maj_keep = min(len(majority_idx), n_majority)
            keep_idx = np.concatenate([
                rng.choice(minority_idx, n_min_keep, replace=False),
                rng.choice(majority_idx, n_maj_keep, replace=False),
            ])
            keep_idx = np.sort(keep_idx)
            latent = latent[keep_idx]
            p_true = p_true[keep_idx]
            y_full = y_full[keep_idx]

        # Features: latent score + noise (so model can learn the zone)
        n = len(y_full)
        X = np.column_stack([
            latent.reshape(-1, 1),
            rng.normal(0, 1, (n, cfg.n_features - 1)),
        ])

        df = self._to_df(X, y_full, cfg.n_features)
        df["true_prob"] = p_true[:n]
        df["confidence_zone"] = zone
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_df(X: np.ndarray, y: np.ndarray, n_features: int) -> pd.DataFrame:
        cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = y
        return df

    def _output_path(self, cfg: SyntheticConfig) -> Path:
        mode_str = cfg.mode.value
        seed_str = f"seed{cfg.seed}"
        n_str = f"n{cfg.n_samples}"

        if cfg.severity is not None:
            fname = f"{mode_str}_{cfg.severity}_{n_str}_{seed_str}.csv"
        elif cfg.mode == GenerationMode.CONFIDENCE_ZONE:
            zone_str = f"zone{cfg.confidence_zone:.2f}".replace(".", "p")
            fname = f"{mode_str}_{zone_str}_{n_str}_{seed_str}.csv"
        else:
            fname = f"{mode_str}_{n_str}_{seed_str}.csv"
        return self._output_dir / fname

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_meta(
        self,
        cfg: SyntheticConfig,
        csv_path: Path,
        sha256: str,
        n_rows: int,
    ) -> None:
        meta = {
            "generated_at": datetime.now().isoformat(),
            "mode": cfg.mode.value,
            "severity": cfg.severity,
            "n_samples": cfg.n_samples,
            "n_features": cfg.n_features,
            "seed": cfg.seed,
            "params": {
                "imbalance_ratio": cfg.imbalance_ratio,
                "noise_rate": cfg.noise_rate,
                "overlap_sigma": cfg.overlap_sigma,
                "collapse_region": cfg.collapse_region,
                "collapse_variance": cfg.collapse_variance,
                "corruption_rate": cfg.corruption_rate,
                "shift_magnitude": cfg.shift_magnitude,
                "confidence_zone": cfg.confidence_zone,
            },
            "output_rows": n_rows,
            "sha256": sha256,
        }
        meta_path = csv_path.with_suffix(".meta")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
