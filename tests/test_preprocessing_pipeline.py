"""
Unit tests for preprocessing pipeline anti-leakage enforcement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.splitter import StratifiedSplitter


class TestPipelineAntiLeakage:
    def test_transform_before_fit_raises(self):
        pipeline = PreprocessingPipeline()
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(RuntimeError, match="fit\\(X_train\\)"):
            pipeline.transform(X, split_tag="val")

    def test_fit_transform_on_train_only(self):
        """fit() should only be called on training data."""
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame(rng.random((100, 5)), columns=[f"f{i}" for i in range(5)])
        X_val = pd.DataFrame(rng.random((20, 5)), columns=[f"f{i}" for i in range(5)])

        pipeline = PreprocessingPipeline()
        X_train_proc = pipeline.fit(X_train)
        X_val_proc = pipeline.transform(X_val, split_tag="val")

        assert X_train_proc.shape == (100, 5)
        assert X_val_proc.shape == (20, 5)

    def test_scaler_fitted_on_train_stats(self):
        """Scaler statistics should reflect training data, not val data."""
        # Training data: mean=0, std=1
        X_train = pd.DataFrame(
            np.random.default_rng(0).normal(0, 1, (200, 3)),
            columns=["a", "b", "c"]
        )
        # Val data: mean=100 (very different)
        X_val = pd.DataFrame(
            np.random.default_rng(1).normal(100, 1, (50, 3)),
            columns=["a", "b", "c"]
        )

        pipeline = PreprocessingPipeline()
        pipeline.fit(X_train)
        X_val_proc = pipeline.transform(X_val, split_tag="val")

        # Val data should be scaled using train statistics → values will be ~100
        assert X_val_proc.mean() > 10  # not centered around 0

    def test_split_tag_enforcement(self):
        """Only 'val' and 'test' are valid split tags for transform."""
        pipeline = PreprocessingPipeline()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pipeline.fit(X)

        with pytest.raises(ValueError, match="split_tag"):
            pipeline.transform(X, split_tag="train")


class TestStratifiedSplitter:
    def test_no_index_overlap(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((500, 10)))
        y = pd.Series((rng.random(500) > 0.8).astype(int))

        splitter = StratifiedSplitter(test_size=0.2, val_size=0.15, seed=42)
        splits = splitter.split(X, y)

        train_s = set(splits.split_indices["train"])
        val_s = set(splits.split_indices["val"])
        test_s = set(splits.split_indices["test"])

        assert len(train_s & val_s) == 0
        assert len(train_s & test_s) == 0
        assert len(val_s & test_s) == 0

    def test_all_indices_covered(self):
        # Use a larger dataset with enough minority samples for val set
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.random((500, 5)))
        # IR ~4 — enough minority samples for val set
        y = pd.Series((rng.random(500) > 0.8).astype(int))

        splitter = StratifiedSplitter(test_size=0.2, val_size=0.15, seed=0)
        splits = splitter.split(X, y)

        all_idx = (
            set(splits.split_indices["train"])
            | set(splits.split_indices["val"])
            | set(splits.split_indices["test"])
        )
        assert len(all_idx) == 500
