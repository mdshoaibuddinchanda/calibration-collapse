"""
Unit tests for synthetic data generator — Calibration Stress Test Suite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import (
    SyntheticDataGenerator, SyntheticConfig, GenerationMode, CONFIDENCE_ZONES
)


class TestDeterminism:
    def test_same_seed_same_output(self, tmp_path):
        """Same params + seed must produce identical files."""
        gen = SyntheticDataGenerator(output_dir=tmp_path / "run1")
        cfg = SyntheticConfig(mode=GenerationMode.EXTREME_IMBALANCE, n_samples=500, seed=42)
        path1 = gen.generate(cfg)

        gen2 = SyntheticDataGenerator(output_dir=tmp_path / "run2")
        path2 = gen2.generate(cfg)

        def sha256(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read())
            return h.hexdigest()

        assert sha256(path1) == sha256(path2)

    def test_different_seed_different_output(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        cfg1 = SyntheticConfig(mode=GenerationMode.EXTREME_IMBALANCE, n_samples=500, seed=42)
        cfg2 = SyntheticConfig(mode=GenerationMode.EXTREME_IMBALANCE, n_samples=500, seed=99)
        path1 = gen.generate(cfg1)
        path2 = gen.generate(cfg2)
        df1, df2 = pd.read_csv(path1), pd.read_csv(path2)
        assert not df1.equals(df2)

    def test_meta_sha256_matches_file(self, tmp_path):
        """SHA256 in .meta file must match the actual CSV."""
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        cfg = SyntheticConfig(mode=GenerationMode.BOUNDARY_OVERLAP, n_samples=200, seed=7)
        path = gen.generate(cfg)
        meta_path = path.with_suffix(".meta")
        with open(meta_path) as f:
            meta = json.load(f)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        assert h.hexdigest() == meta["sha256"]


class TestSeverityLevels:
    """Severity presets must produce the correct parameter values."""

    def test_severity_applies_correct_ir(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        for sev, expected_ir in [("mild", 10.0), ("moderate", 50.0), ("severe", 100.0)]:
            cfg = SyntheticConfig(
                mode=GenerationMode.EXTREME_IMBALANCE,
                severity=sev, n_samples=1000, seed=0,  # type: ignore[arg-type]
            )
            assert cfg.imbalance_ratio == expected_ir, f"severity={sev}"
            path = gen.generate(cfg)
            df = pd.read_csv(path)
            vc = df["label"].value_counts()
            actual_ir = vc.max() / vc.min()
            assert actual_ir >= expected_ir * 0.8, f"IR too low for severity={sev}"

    def test_severity_applies_noise_rate(self):
        for sev, expected_rate in [("mild", 0.05), ("moderate", 0.15), ("severe", 0.30)]:
            cfg = SyntheticConfig(
                mode=GenerationMode.NOISY_MINORITY,
                severity=sev,  # type: ignore[arg-type]
            )
            assert cfg.noise_rate == expected_rate, f"severity={sev}"

    def test_severity_applies_overlap_sigma(self):
        for sev, expected_sigma in [("mild", 2.0), ("moderate", 1.5), ("severe", 0.8)]:
            cfg = SyntheticConfig(
                mode=GenerationMode.BOUNDARY_OVERLAP,
                severity=sev,  # type: ignore[arg-type]
            )
            assert cfg.overlap_sigma == expected_sigma, f"severity={sev}"

    def test_severity_sweep_produces_3_files(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        paths = gen.generate_severity_sweep(
            GenerationMode.CONFIDENCE_COLLAPSE, n_samples=200, seed=42
        )
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            # Severity level should be in filename
            assert any(s in p.name for s in ("mild", "moderate", "severe"))

    def test_severity_filenames_are_distinct(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        paths = gen.generate_severity_sweep(
            GenerationMode.FEATURE_CORRUPTION, n_samples=200, seed=0
        )
        names = [p.name for p in paths]
        assert len(set(names)) == 3  # all distinct


class TestConfidenceZones:
    """Confidence zone sweep must produce correct zone targeting."""

    def test_zone_sweep_produces_5_files(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        paths = gen.generate_confidence_zone_sweep(n_samples=500, seed=42)
        assert len(paths) == 5

    def test_zone_mean_true_prob_matches_zone(self, tmp_path):
        """Mean true probability should be close to the target zone."""
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        for zone in CONFIDENCE_ZONES:
            cfg = SyntheticConfig(
                mode=GenerationMode.CONFIDENCE_ZONE,
                confidence_zone=zone, n_samples=2000, seed=42,
            )
            path = gen.generate(cfg)
            df = pd.read_csv(path)
            assert "true_prob" in df.columns
            assert "confidence_zone" in df.columns
            mean_prob = df["true_prob"].mean()
            # Mean true prob should be within 0.1 of target zone
            assert abs(mean_prob - zone) < 0.1, (
                f"Zone {zone}: mean_true_prob={mean_prob:.3f} too far from target"
            )

    def test_zone_column_correct(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        cfg = SyntheticConfig(
            mode=GenerationMode.CONFIDENCE_ZONE,
            confidence_zone=0.3, n_samples=500, seed=1,
        )
        path = gen.generate(cfg)
        df = pd.read_csv(path)
        assert (df["confidence_zone"] == 0.3).all()


class TestGenerationModes:
    def test_all_modes_run(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        for mode in GenerationMode:
            cfg = SyntheticConfig(mode=mode, n_samples=100, seed=42)
            path = gen.generate(cfg)
            df = pd.read_csv(path)
            assert len(df) > 0
            assert "label" in df.columns

    def test_extreme_imbalance_ir(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        cfg = SyntheticConfig(
            mode=GenerationMode.EXTREME_IMBALANCE,
            n_samples=1000, imbalance_ratio=100.0, seed=0
        )
        path = gen.generate(cfg)
        df = pd.read_csv(path)
        ir = (df["label"] == 0).sum() / (df["label"] == 1).sum()
        assert ir > 50

    def test_meta_file_created(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        cfg = SyntheticConfig(mode=GenerationMode.BOUNDARY_OVERLAP, n_samples=200, seed=7)
        path = gen.generate(cfg)
        assert path.with_suffix(".meta").exists()

    def test_meta_contains_severity(self, tmp_path):
        gen = SyntheticDataGenerator(output_dir=tmp_path)
        cfg = SyntheticConfig(
            mode=GenerationMode.NOISY_MINORITY, severity="moderate", n_samples=200, seed=0
        )
        path = gen.generate(cfg)
        with open(path.with_suffix(".meta")) as f:
            meta = json.load(f)
        assert meta["severity"] == "moderate"
        assert meta["params"]["noise_rate"] == 0.15
