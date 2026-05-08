"""Download and normalize all real datasets used by the project.

Outputs are written to `datasets/raw/*.csv` with schema aligned to
`configs/datasets/*.yaml`.

This script is best-effort: for sources that are unavailable in the current
network/session, it prints concrete manual instructions instead of failing.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "datasets" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _print_existing(path: Path) -> bool:
    if path.exists():
        print(f"[skip] {path.name} already exists")
        return True
    return False


def _save_csv(df: pd.DataFrame, path: Path, label_col: str | None = None) -> None:
    out = df.copy()
    if label_col is not None and label_col in out.columns:
        out[label_col] = out[label_col].replace({"yes": 1, "no": 0, "Y": 1, "N": 0})
    out.to_csv(path, index=False)
    print(f"[ok] wrote {path.name}: {len(out)} rows, {out.shape[1]} columns")


def _to_binary_label(series: pd.Series, positive_tokens: set[str]) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.isin({token.lower() for token in positive_tokens}).astype(int)


def _fetch_openml_by_candidates(
    *,
    names: list[str],
    ids: list[int],
    normalize: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    last_error: Exception | None = None

    for data_id in ids:
        try:
            data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            return normalize(data.frame.copy())
        except Exception as exc:  # pragma: no cover - source availability dependent
            last_error = exc

    for name in names:
        try:
            data = fetch_openml(name=name, as_frame=True, parser="auto")
            return normalize(data.frame.copy())
        except Exception as exc:  # pragma: no cover - source availability dependent
            last_error = exc

    raise RuntimeError(f"OpenML download failed for names={names}, ids={ids}: {last_error}")


def _download_pima() -> None:
    path = RAW_DIR / "pima.csv"
    if _print_existing(path):
        return

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    df = pd.read_csv(url, header=None, names=cols)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
    _save_csv(df, path)


def _download_phoneme() -> None:
    path = RAW_DIR / "phoneme.csv"
    if _print_existing(path):
        return

    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=1489, as_frame=True, parser="auto")
    df = data.frame.copy()
    df["Class"] = _to_binary_label(df["Class"], {"2"})
    _save_csv(df, path)


def _download_default_credit_card_clients() -> None:
    path = RAW_DIR / "default_credit_card_clients.csv"
    if _print_existing(path):
        return

    # UCI source is XLS; we use a stable CSV mirror first, then OpenML fallbacks.
    csv_mirror = (
        "https://raw.githubusercontent.com/selva86/datasets/master/DefaultCredit.csv"
    )
    try:
        df = pd.read_csv(csv_mirror)
        rename_map = {
            "default.payment.next.month": "label",
            "default payment next month": "label",
            "default": "label",
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        if "label" not in df.columns:
            raise ValueError("Missing normalized 'label' column in default credit CSV mirror.")
        _save_csv(df, path)
        return
    except Exception:
        pass

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        local = df.copy()
        candidates = [
            "default.payment.next.month",
            "default payment next month",
            "class",
            "label",
        ]
        found = next((c for c in candidates if c in local.columns), None)
        if found is None:
            raise ValueError("Could not find target column for default credit dataset.")
        local = local.rename(columns={found: "label"})
        local["label"] = pd.to_numeric(local["label"], errors="coerce").fillna(0).astype(int)
        return local

    df = _fetch_openml_by_candidates(
        names=["default-of-credit-card-clients", "default_credit_card_clients"],
        ids=[42477, 1466],
        normalize=_normalize,
    )
    _save_csv(df, path)


def _download_mammography() -> None:
    path = RAW_DIR / "mammography.csv"
    if _print_existing(path):
        return

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        local = df.copy()
        target = next((c for c in ["class", "Class", "target", "label"] if c in local.columns), None)
        if target is None:
            raise ValueError("Could not find mammography target column.")
        local = local.rename(columns={target: "label"})
        local["label"] = _to_binary_label(local["label"], {"1", "yes", "true", "malignant"})
        return local

    df = _fetch_openml_by_candidates(
        names=["mammography", "Mammography"],
        ids=[310],
        normalize=_normalize,
    )
    _save_csv(df, path)


def _download_nsl_kdd() -> None:
    path = RAW_DIR / "nsl_kdd.csv"
    if _print_existing(path):
        return

    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "attack", "difficulty",
    ]

    train_df = pd.read_csv(train_url, header=None, names=columns)
    test_df = pd.read_csv(test_url, header=None, names=columns)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["label"] = (df["attack"].astype(str) != "normal").astype(int)
    df = df.drop(columns=["attack", "difficulty"])
    _save_csv(df, path)


def _download_give_me_some_credit() -> None:
    path = RAW_DIR / "give_me_some_credit.csv"
    if _print_existing(path):
        return

    # Public mirrors of the Kaggle training split. Try several stable mirrors
    # before falling back to a manual download note.
    candidate_urls = [
        "https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv",
        "https://raw.githubusercontent.com/DrIanGregory/Kaggle-GiveMeSomeCredit/master/data/GiveMeSomeCredit-training.csv",
        "https://raw.githubusercontent.com/wang-weishuai/GiveMeSomeCredit/master/cs-training.csv",
    ]

    last_error: Exception | None = None
    df = None
    for url in candidate_urls:
        try:
            df = pd.read_csv(url)
            break
        except Exception as exc:  # pragma: no cover - source availability dependent
            last_error = exc

    if df is None:
        raise RuntimeError(f"Unable to download Give Me Some Credit from mirrors: {last_error}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    target_col = "SeriousDlqin2yrs"
    if target_col not in df.columns:
        raise ValueError("Expected SeriousDlqin2yrs in Give Me Some Credit dataset.")

    df = df.rename(columns={target_col: "label"})
    _save_csv(df, path)


def _download_bank_marketing() -> None:
    path = RAW_DIR / "bank_marketing.csv"
    if _print_existing(path):
        return

    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    with urlopen(zip_url) as response:  # nosec B310 - trusted dataset source
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open("bank-full.csv") as fh:
            df = pd.read_csv(fh, sep=";")

    if "y" not in df.columns:
        raise ValueError("Expected target column 'y' in bank marketing dataset.")
    df = df.rename(columns={"y": "label"})
    df["label"] = _to_binary_label(df["label"], {"yes"})
    _save_csv(df, path)


def _download_thyroid_disease() -> None:
    path = RAW_DIR / "thyroid_disease.csv"
    if _print_existing(path):
        return

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        local = df.copy()
        target = next((c for c in ["class", "Class", "target", "label"] if c in local.columns), None)
        if target is None:
            raise ValueError("Could not find thyroid target column.")
        local = local.rename(columns={target: "label"})
        local["label"] = _to_binary_label(local["label"], {"1", "yes", "true", "sick", "abnormal"})
        return local

    df = _fetch_openml_by_candidates(
        names=["sick", "thyroid-disease", "thyroid"],
        ids=[38],
        normalize=_normalize,
    )
    _save_csv(df, path)


def _download_dry_bean() -> None:
    path = RAW_DIR / "dry_bean.csv"
    if _print_existing(path):
        return

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        local = df.copy()
        target = next((c for c in ["Class", "class", "target", "label"] if c in local.columns), None)
        if target is None:
            raise ValueError("Could not find Dry Bean target column.")
        local = local.rename(columns={target: "label"})
        local["label"] = local["label"].astype(str)
        return local

    df = _fetch_openml_by_candidates(
        names=["dry-bean", "dry_bean", "DryBeanDataset"],
        ids=[42732],
        normalize=_normalize,
    )
    _save_csv(df, path)


def main() -> None:
    print("Preparing real dataset suite in datasets/raw ...")

    tasks: list[tuple[str, str, Callable[[], None]]] = [
        ("pima", "pima.csv", _download_pima),
        ("phoneme", "phoneme.csv", _download_phoneme),
        ("default_credit_card_clients", "default_credit_card_clients.csv", _download_default_credit_card_clients),
        ("mammography", "mammography.csv", _download_mammography),
        ("nsl_kdd", "nsl_kdd.csv", _download_nsl_kdd),
        ("give_me_some_credit", "give_me_some_credit.csv", _download_give_me_some_credit),
        ("bank_marketing", "bank_marketing.csv", _download_bank_marketing),
        ("thyroid_disease", "thyroid_disease.csv", _download_thyroid_disease),
        ("dry_bean", "dry_bean.csv", _download_dry_bean),
    ]

    failures: list[tuple[str, str, str]] = []
    for name, target_file, fn in tasks:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - network/source dependent
            failures.append((name, target_file, str(exc)))
            print(f"[warn] failed to fetch {name}: {exc}")

    print("\nDataset download pass complete.")
    print(f"CSV files currently in {RAW_DIR}:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<35} {size_kb:>8} KB")

    if failures:
        print("\nManual action required for these datasets:")
        for name, target_file, error in failures:
            target = RAW_DIR / target_file
            print(f"  - {name}: place normalized CSV at {target}")
            print(f"    error: {error}")


if __name__ == "__main__":
    main()
