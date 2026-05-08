"""
SQLite-based experiment tracker.

Stores all experiment runs in outputs/experiments.db.
Enables queries like:
  - "show all experiments on pima with SMOTE that achieved ECE_minority < 0.1"
  - "show all experiments that completed without leakage errors"

Why SQLite (not MLflow/W&B):
  - No external dependencies
  - Works fully offline
  - Single file, committable (but .gitignored by default)
  - Queryable with standard SQL
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id          TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    dataset         TEXT,
    model           TEXT,
    resampler       TEXT,
    calibrator      TEXT,
    status          TEXT,
    ece_global      REAL,
    ece_minority    REAL,
    ece_majority    REAL,
    brier_global    REAL,
    f1_minority     REAL,
    f1_macro        REAL,
    auc_roc         REAL,
    recall_minority REAL,
    git_hash        TEXT,
    seed            INTEGER,
    leakage_status  TEXT,
    manifest_path   TEXT,
    full_config     TEXT
)
"""


class ExperimentTracker:
    """
    Tracks experiment runs in a local SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_run(
        self,
        run_id: str,
        experiment_id: str,
        dataset: str,
        model: str,
        resampler: str,
        calibrator: str,
        status: str,
        cal_metrics: Optional[dict] = None,
        cls_metrics: Optional[dict] = None,
        git_hash: str = "unknown",
        seed: int = 42,
        leakage_status: str = "unknown",
        manifest_path: str = "",
        config: Optional[dict] = None,
    ) -> None:
        row = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "model": model,
            "resampler": resampler,
            "calibrator": calibrator,
            "status": status,
            "ece_global": cal_metrics.get("ece_global") if cal_metrics else None,
            "ece_minority": cal_metrics.get("ece_minority") if cal_metrics else None,
            "ece_majority": cal_metrics.get("ece_majority") if cal_metrics else None,
            "brier_global": cal_metrics.get("brier_global") if cal_metrics else None,
            "f1_minority": cls_metrics.get("f1_minority") if cls_metrics else None,
            "f1_macro": cls_metrics.get("f1_macro") if cls_metrics else None,
            "auc_roc": cls_metrics.get("auc_roc") if cls_metrics else None,
            "recall_minority": cls_metrics.get("recall_minority") if cls_metrics else None,
            "git_hash": git_hash,
            "seed": seed,
            "leakage_status": leakage_status,
            "manifest_path": manifest_path,
            "full_config": json.dumps(config) if config else "{}",
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments VALUES (
                    :run_id, :experiment_id, :timestamp, :dataset, :model,
                    :resampler, :calibrator, :status, :ece_global, :ece_minority,
                    :ece_majority, :brier_global, :f1_minority, :f1_macro, :auc_roc,
                    :recall_minority, :git_hash, :seed, :leakage_status,
                    :manifest_path, :full_config
                )
                """,
                row,
            )
        logger.debug("Logged run %s to tracker", run_id)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def best_by_ece_minority(
        self, dataset: str, max_ece_minority: float = 0.1
    ) -> pd.DataFrame:
        """Show all experiments on a dataset beating ECE_minority threshold."""
        return self.query(
            """
            SELECT run_id, model, resampler, calibrator, ece_minority, recall_minority,
                   f1_minority, auc_roc, status, leakage_status
            FROM experiments
            WHERE dataset = ? AND ece_minority < ? AND status = 'completed'
            ORDER BY ece_minority ASC
            """,
            (dataset, max_ece_minority),
        )

    def completed_without_leakage(self, experiment_id: str) -> pd.DataFrame:
        """Show all completed experiments without leakage errors."""
        return self.query(
            """
            SELECT run_id, dataset, model, resampler, calibrator,
                   ece_minority, recall_minority, status
            FROM experiments
            WHERE experiment_id = ? AND status = 'completed' AND leakage_status = 'PASS'
            ORDER BY ece_minority ASC
            """,
            (experiment_id,),
        )

    def skipped_runs(self, experiment_id: str = None) -> pd.DataFrame:
        """Show all skipped runs (invalid configurations, not bugs)."""
        if experiment_id:
            return self.query(
                """
                SELECT run_id, experiment_id, dataset, model, resampler, calibrator, status
                FROM experiments
                WHERE experiment_id = ? AND status = 'skipped'
                ORDER BY timestamp DESC
                """,
                (experiment_id,),
            )
        return self.query(
            "SELECT run_id, experiment_id, dataset, model, resampler, calibrator, status "
            "FROM experiments WHERE status = 'skipped' ORDER BY timestamp DESC"
        )

    def baseline_comparison(self, dataset: str) -> pd.DataFrame:
        """Compare all methods vs. no-resampling baseline."""
        return self.query(
            """
            SELECT model, resampler, calibrator,
                   ece_global, ece_minority, recall_minority, f1_minority, auc_roc
            FROM experiments
            WHERE dataset = ? AND status = 'completed'
            ORDER BY model, resampler, calibrator
            """,
            (dataset,),
        )

    def all_runs(self) -> pd.DataFrame:
        return self.query("SELECT * FROM experiments ORDER BY timestamp DESC")
