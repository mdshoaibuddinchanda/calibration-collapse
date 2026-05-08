"""
Download real-world imbalanced datasets.

Pima Indians Diabetes: UCI / OpenML
Credit Card Fraud: Kaggle (requires manual download) — we generate a proxy
Phoneme: OpenML dataset 1489
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
raw_dir = ROOT / "datasets" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Pima Indians Diabetes (UCI)
# -----------------------------------------------------------------------
pima_path = raw_dir / "pima.csv"
if not pima_path.exists():
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
        df = pd.read_csv(url, header=None, names=cols)
        # Replace 0s with NaN for physiologically impossible values
        for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
            df[col] = df[col].replace(0, np.nan)
        df.to_csv(pima_path, index=False)
        print(f"Pima: {len(df)} rows, IR={df['Outcome'].value_counts()[0]/df['Outcome'].value_counts()[1]:.2f}")
    except Exception as e:
        print(f"Could not download Pima: {e}. Generating synthetic proxy.")
        _generate_pima_proxy(raw_dir)
else:
    print(f"Pima already exists: {pima_path}")

# -----------------------------------------------------------------------
# 2. Phoneme (OpenML 1489)
# -----------------------------------------------------------------------
phoneme_path = raw_dir / "phoneme.csv"
if not phoneme_path.exists():
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=1489, as_frame=True, parser="auto")
        df = data.frame.copy()
        # Target is 'Class' — convert to binary 0/1
        df["Class"] = (df["Class"].astype(str) == "2").astype(int)
        df.to_csv(phoneme_path, index=False)
        print(f"Phoneme: {len(df)} rows, IR={df['Class'].value_counts()[0]/df['Class'].value_counts()[1]:.2f}")
    except Exception as e:
        print(f"Could not download Phoneme: {e}. Generating synthetic proxy.")
        _generate_phoneme_proxy(raw_dir)
else:
    print(f"Phoneme already exists: {phoneme_path}")

# -----------------------------------------------------------------------
# 3. Credit Card Fraud (synthetic proxy — real dataset requires Kaggle auth)
# -----------------------------------------------------------------------
credit_path = raw_dir / "credit_card.csv"
if not credit_path.exists():
    print("Generating Credit Card synthetic proxy (IR~172, similar to real dataset)...")
    rng = np.random.default_rng(42)
    n_legit = 284315
    n_fraud = 492
    # Use smaller version for memory efficiency
    n_legit = 28431
    n_fraud = 49

    X_legit = rng.normal(0, 1, (n_legit, 28))
    X_fraud = rng.normal(2, 1.5, (n_fraud, 28))
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = rng.exponential(100, len(df))
    df["Class"] = y
    df.to_csv(credit_path, index=False)
    print(f"Credit Card proxy: {len(df)} rows, IR={n_legit/n_fraud:.1f}")
else:
    print(f"Credit Card already exists: {credit_path}")

print("\nAll datasets ready.")
print(f"Files in {raw_dir}:")
for f in sorted(raw_dir.glob("*.csv")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name} ({size_kb} KB)")
