"""Dataset access. The ULB Credit Card Fraud dataset (284,807 European card transactions, 492 frauds) is
pulled from OpenML (dataset 1597, the same file as the Kaggle 'creditcardfraud' dataset) so the pipeline runs
from a clean clone without a Kaggle login."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARFF_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"   # OpenML dataset 1597, version 1
COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def download_arff(dest: Path, url: str = ARFF_URL) -> Path:
    """Stream the ARFF (~150 MB) to disk. (sklearn.fetch_openml drops `Time` because OpenML flags it as an
    ignored attribute, so the raw file is parsed here instead.)"""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def parse_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader for this file: attribute names from the header, CSV rows after @data."""
    names, data_start = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            low = line.strip().lower()
            if low.startswith("@attribute"):
                names.append(line.split()[1].strip("'\""))
            elif low.startswith("@data"):
                data_start = i + 1
                break
    df = pd.read_csv(path, skiprows=data_start, header=None, names=names, quotechar="'")
    return df


def load_creditcard(data_dir: str | Path = "data", download: bool = True) -> pd.DataFrame:
    """Return the dataset with columns Time, V1..V28, Amount, Class (int 0/1), sorted by Time.
    Cached as data/creditcard.csv after the first download."""
    data_dir = Path(data_dir)
    csv = data_dir / "creditcard.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if "Time" not in df.columns:            # cache written by an older version without Time
            csv.unlink()
            return load_creditcard(data_dir, download)
    else:
        if not download:
            raise FileNotFoundError(f"{csv} not found and download=False")
        arff = data_dir / "creditcard.arff"
        if not arff.exists():
            download_arff(arff)
        df = parse_arff(arff)
        df.to_csv(csv, index=False)
    df = df[COLUMNS]
    df["Class"] = df["Class"].astype(str).str.strip("'").astype(int)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split (no shuffling): train on the earliest transactions, validate on the next slice,
    test on the latest. Velocity features look backwards in time, so a random split would leak the future."""
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def make_synthetic(n: int = 5000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the same schema, for tests and smoke runs (never for reported results)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    v = rng.normal(size=(n, 28))
    v[y == 1, :5] += 2.0                      # make fraud separable so tests have signal
    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)))
    df["Amount"] = np.round(np.exp(rng.normal(3, 1.2, n)), 2)
    df["Class"] = y
    return df
