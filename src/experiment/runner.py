"""
Top-level experiment orchestrator.

Execution order (strict — do not reorder):
  1.  Load experiment config
  2.  Initialize manifest (seed, timestamp, git hash, dependency versions)
  3.  Load and inspect dataset
  4.  Run leakage pre-check (structural)
  5.  Split data (stratified)
  6.  Fit preprocessing pipeline (train only)
  7.  Transform val and test (using train-fitted pipeline)
  8.  Run resampler (train only)
  9.  Train model
  10. Generate val probabilities → fit calibrator
  11. Generate test probabilities → apply calibrator
  12. Run leakage post-check (empirical)
  13. Compute calibration and classification metrics
  14. Run integrity checker
  15. Save all outputs (metrics, plots, models, manifest)
  16. Run seed validator (optional, controlled by config)

IMPORTANT: If any step fails an audit gate, runner writes a FAILED manifest
and exits. No partial results are written.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.loader import DatasetLoader
from src.data.inspector import DatasetInspector
from src.data.registry import DatasetRegistry
from src.preprocessing.splitter import StratifiedSplitter
from src.preprocessing.pipeline import PreprocessingPipeline
from src.resampling import get_resampler
from src.resampling.class_weight import ClassWeightResampler
from src.resampling.smote_resampler import InvalidSamplingStrategyError
from src.models import get_model
from src.models.mlp import MLPClassifier_
from src.calibration import get_calibrator
from src.evaluation.calibration_metrics import CalibrationMetrics
from src.evaluation.classification_metrics import ClassificationMetrics
from src.evaluation.drift_analysis import DriftAnalyzer
from src.visualization.reliability_diagram import ReliabilityDiagram
from src.visualization.confidence_histogram import ConfidenceHistogram
from src.visualization.calibration_recall_frontier import CalibrationRecallFrontier
from src.audit.leakage_detector import LeakageDetector
from src.audit.integrity_checker import IntegrityChecker
from src.audit.seed_validator import SeedValidator
from src.experiment.manifest import ExperimentManifest
from src.experiment.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs a single (dataset, model, resampler, calibrator) combination.
    Called by scripts/run_experiment.py for each combination in the config.
    """

    def __init__(self, project_root: Path, config: dict) -> None:
        self._root = Path(project_root)
        self._config = config
        self._seed = config.get("seed", 42)

        # Output directories
        self._output_root = self._root / config.get("output_root", "outputs")
        self._artifact_root = self._root / config.get("artifact_root", "artifacts")

        # Sub-components
        self._loader = DatasetLoader(project_root=self._root)
        self._inspector = DatasetInspector(
            output_dir=self._output_root / "logs"
        )
        self._splitter = StratifiedSplitter(
            test_size=config.get("test_size", 0.20),
            val_size=config.get("val_size", 0.15),
            seed=self._seed,
        )
        self._leakage_detector = LeakageDetector()
        self._integrity_checker = IntegrityChecker()
        self._seed_validator = SeedValidator()
        self._cal_metrics = CalibrationMetrics(
            n_bins=config.get("calibration_bins", 10)
        )
        self._cls_metrics = ClassificationMetrics()
        self._drift_analyzer = DriftAnalyzer(
            n_bins=config.get("calibration_bins", 10)
        )
        self._reliability_diagram = ReliabilityDiagram(
            n_bins=config.get("calibration_bins", 10)
        )
        self._conf_histogram = ConfidenceHistogram()
        self._frontier = CalibrationRecallFrontier()

        # Tracker
        tracker_path = self._output_root / "experiments.db"
        self._tracker = ExperimentTracker(tracker_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str,
        model_name: str,
        resampler_name: str,
        calibrator_name: str,
        experiment_id: str,
        dataset_registry: DatasetRegistry,
    ) -> dict:
        """
        Run one (dataset, model, resampler, calibrator) combination.

        Returns a dict with all metrics, or raises on fatal audit failure.
        """
        method_label = f"{model_name}+{resampler_name}+{calibrator_name}"
        run_id = f"{experiment_id}_{dataset_name}_{method_label}"

        logger.info("=" * 70)
        logger.info("Starting run: %s", run_id)
        logger.info("=" * 70)

        # Step 1-2: Config + manifest
        manifest = ExperimentManifest.create(
            experiment_id=experiment_id,
            config={**self._config, "dataset": dataset_name, "model": model_name,
                    "resampler": resampler_name, "calibrator": calibrator_name},
        )
        manifest.run_id = run_id

        self._set_seeds()

        try:
            result = self._execute_pipeline(
                dataset_name=dataset_name,
                model_name=model_name,
                resampler_name=resampler_name,
                calibrator_name=calibrator_name,
                experiment_id=experiment_id,
                method_label=method_label,
                manifest=manifest,
                dataset_registry=dataset_registry,
            )
            manifest.results = result
            manifest.mark_completed()

        except InvalidSamplingStrategyError as exc:
            # SKIPPED — invalid configuration, not a bug
            logger.warning(
                "Run %s SKIPPED (invalid configuration): %s", run_id, exc
            )
            manifest.status = "skipped"
            manifest.error_message = str(exc)
            manifest.save(self._root / "reports" / "summaries")
            self._tracker.log_run(
                run_id=run_id, experiment_id=experiment_id,
                dataset=dataset_name, model=model_name,
                resampler=resampler_name, calibrator=calibrator_name,
                status="skipped", git_hash=manifest.git_hash,
                seed=self._seed, leakage_status="N/A",
            )
            raise  # re-raise so caller knows it was skipped

        except Exception as exc:
            logger.error("Run %s FAILED: %s", run_id, exc, exc_info=True)
            manifest.mark_failed(str(exc))
            manifest.save(self._root / "reports" / "summaries")
            self._tracker.log_run(
                run_id=run_id, experiment_id=experiment_id,
                dataset=dataset_name, model=model_name,
                resampler=resampler_name, calibrator=calibrator_name,
                status="failed", git_hash=manifest.git_hash,
                seed=self._seed, leakage_status="UNKNOWN",
            )
            raise

        manifest.save(self._root / "reports" / "summaries")
        self._tracker.log_run(
            run_id=run_id, experiment_id=experiment_id,
            dataset=dataset_name, model=model_name,
            resampler=resampler_name, calibrator=calibrator_name,
            status="completed",
            cal_metrics=result.get("calibration"),
            cls_metrics=result.get("classification"),
            git_hash=manifest.git_hash,
            seed=self._seed,
            leakage_status=result.get("leakage_status", "PASS"),
            manifest_path=str(self._root / "reports" / "summaries" / f"{run_id}_manifest.json"),
            config=self._config,
        )

        logger.info("Run %s completed successfully.", run_id)
        return result

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _execute_pipeline(
        self,
        dataset_name: str,
        model_name: str,
        resampler_name: str,
        calibrator_name: str,
        experiment_id: str,
        method_label: str,
        manifest: ExperimentManifest,
        dataset_registry: DatasetRegistry,
    ) -> dict:
        t0 = time.time()

        # Step 3: Load and inspect dataset
        dataset_config = dataset_registry.get(dataset_name)
        X, y, metadata = self._loader.load(dataset_config)
        dataset_report = self._inspector.inspect(
            X, y, dataset_name,
            write_report=True,
        )
        manifest.dataset_report = asdict(dataset_report)

        # Step 4: Leakage pre-check (structural)
        self._leakage_detector.run_all(
            split_indices={},  # not yet split — structural check only
            n_train_samples=None,
            n_val_samples=None,
            calibrator_split_tag=None,
            X_train=None, X_val=None, X_test=None,
            y_train=None, feature_names=list(X.columns),
            experiment_id=f"{experiment_id}_pre",
            output_dir=self._output_root / "logs",
        )

        # Step 5: Split data
        splits = self._splitter.split(X, y)
        manifest.split_indices = splits.split_indices

        # Step 6: Fit preprocessing pipeline (train only)
        pipeline = PreprocessingPipeline(
            missing_strategy=dataset_config.missing_value_strategy,
            artifact_dir=self._artifact_root / "preprocessed",
            experiment_id=experiment_id,
            dataset_name=dataset_name,
        )
        X_train_proc = pipeline.fit(splits.X_train)

        # Step 7: Transform val and test
        X_val_proc = pipeline.transform(splits.X_val, split_tag="val")
        X_test_proc = pipeline.transform(splits.X_test, split_tag="test")

        y_train = splits.y_train.values
        y_val = splits.y_val.values
        y_test = splits.y_test.values

        # Step 8: Resample (train only)
        # Allow config to override SMOTE sampling strategy for ablations
        smote_kwargs = {}
        if resampler_name == "smote" and "smote_sampling_strategy" in self._config:
            smote_kwargs["sampling_strategy"] = float(self._config["smote_sampling_strategy"])
        resampler = get_resampler(resampler_name, seed=self._seed, **smote_kwargs)
        X_train_res, y_train_res = resampler.fit_resample(X_train_proc, y_train)
        resampling_meta = resampler.get_metadata()
        manifest.resampling_metadata = resampling_meta

        # Get sample weights if class_weight strategy
        sample_weights = None
        if isinstance(resampler, ClassWeightResampler) and resampler.sample_weights is not None:
            sample_weights = resampler.sample_weights

        # Step 9: Train model
        model = get_model(model_name, seed=self._seed)
        # Allow config to override RF n_estimators for speed
        if model_name == "random_forest" and "rf_n_estimators" in self._config:
            from src.models.random_forest import RandomForestClassifier_
            model = RandomForestClassifier_(
                n_estimators=int(self._config["rf_n_estimators"]),
                seed=self._seed,
            )
        if isinstance(model, MLPClassifier_):
            model.fit(
                X_train_res, y_train_res,
                sample_weight=sample_weights,
                X_val=X_val_proc, y_val=y_val,
            )
        else:
            model.fit(X_train_res, y_train_res, sample_weight=sample_weights)

        model_params = model.get_params()
        manifest.model_params = model_params

        # Step 10: Generate val probabilities → fit calibrator
        proba_val = model.predict_proba(X_val_proc)

        calibrator = None
        proba_test_cal = None
        calibrator_params = {"calibrator": "none"}

        if calibrator_name != "none":
            calibrator = get_calibrator(calibrator_name, model_name=model_name)
            calibrator.fit(proba_val, y_val, split_tag="val")
            calibrator_params = calibrator.get_params()

        manifest.calibrator_params = calibrator_params

        # Step 11: Generate test probabilities → apply calibrator
        proba_test = model.predict_proba(X_test_proc)
        if calibrator is not None:
            proba_test_cal = calibrator.calibrate(proba_test)
        else:
            proba_test_cal = proba_test

        # Step 12: Leakage post-check (empirical)
        leakage_result = self._leakage_detector.run_all(
            split_indices=splits.split_indices,
            n_train_samples=pipeline.n_train_samples,
            n_val_samples=len(X_val_proc),
            calibrator_split_tag="val" if calibrator is not None else None,
            X_train=X_train_proc,
            X_val=X_val_proc,
            X_test=X_test_proc,
            y_train=y_train,
            feature_names=pipeline.get_feature_names(),
            experiment_id=experiment_id,
            output_dir=self._output_root / "logs",
        )
        leakage_status = leakage_result.overall_status
        manifest.audit_results = {"leakage": asdict(leakage_result)}

        # Save model AFTER leakage check passes (no partial artifacts on failure)
        model_path = (
            self._artifact_root / "models"
            / f"{experiment_id}_{dataset_name}_{model_name}_{resampler_name}.pkl"
        )
        model.save(model_path)

        # Step 13: Compute metrics
        cal_result = self._cal_metrics.compute(
            proba=proba_test_cal,
            y_true=y_test,
            experiment_id=experiment_id,
            dataset=dataset_name,
            method=method_label,
            output_dir=self._output_root / "metrics" / "calibration",
            filename_suffix=method_label,
        )
        cls_result = self._cls_metrics.compute(
            proba=proba_test_cal,
            y_true=y_test,
            experiment_id=experiment_id,
            dataset=dataset_name,
            method=method_label,
            output_dir=self._output_root / "metrics" / "classification",
            filename_suffix=method_label,
        )

        # Drift analysis
        if isinstance(model, MLPClassifier_) and model.epoch_val_losses:
            # For MLP: use epoch-wise val probabilities (approximate via final proba)
            # Full epoch-wise tracking would require storing proba at each epoch
            # Here we use fold-wise approximation with the val set
            pass  # Full epoch drift requires training hooks — tracked in future work

        # Step 14: Integrity check
        integrity_result = self._integrity_checker.run_all(
            cal_result=asdict(cal_result),
            n_test=len(y_test),
            proba_before_cal=proba_test,
            proba_after_cal=proba_test_cal,
            experiment_seed=self._seed,
            ablation_seeds=None,
            experiment_id=experiment_id,
            output_dir=self._output_root / "logs",
        )
        manifest.audit_results["integrity"] = asdict(integrity_result)

        # Step 15: Save visualizations
        output_files = []

        rel_path = self._reliability_diagram.plot(
            proba=proba_test,
            y_true=y_test,
            proba_cal=proba_test_cal if calibrator is not None else None,
            experiment_id=experiment_id,
            dataset=dataset_name,
            method=method_label,
            output_dir=self._output_root / "plots" / "reliability",
        )
        if rel_path:
            output_files.append(str(rel_path))

        hist_path = self._conf_histogram.plot(
            proba=proba_test_cal,
            y_true=y_test,
            experiment_id=experiment_id,
            dataset=dataset_name,
            method=method_label,
            output_dir=self._output_root / "plots" / "confidence_hist",
        )
        if hist_path:
            output_files.append(str(hist_path))

        manifest.output_files = output_files

        elapsed = time.time() - t0
        logger.info("Pipeline completed in %.1fs", elapsed)

        return {
            "calibration": asdict(cal_result),
            "classification": asdict(cls_result),
            "leakage_status": leakage_status,
            "integrity_status": integrity_result.overall_status,
            "elapsed_seconds": elapsed,
            "method": method_label,
        }

    # ------------------------------------------------------------------
    # Seed management
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        try:
            import torch
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)
        except ImportError:
            pass
        logger.debug("Seeds set to %d", self._seed)
