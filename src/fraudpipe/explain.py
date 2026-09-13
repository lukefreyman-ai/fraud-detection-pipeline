"""SHAP explanations for the LightGBM model: global importance (mean |SHAP|) and per-transaction contributions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def shap_values(estimator, X: pd.DataFrame) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older shap returns [class0, class1]
        sv = sv[1]
    return np.asarray(sv)


def global_importance(estimator, X: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    sv = shap_values(estimator, X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    return imp.head(top).reset_index(drop=True)


def plot_importance(imp: pd.DataFrame, path: Path, title: str = "Top features by mean |SHAP| (LightGBM, test sample)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#2a6fdb")
    ax.set_xlabel("mean |SHAP value|"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def explain_one(estimator, x: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution for one transaction, sorted by |contribution|."""
    sv = shap_values(estimator, x)[0]
    return pd.DataFrame({"feature": x.columns, "value": x.iloc[0].to_numpy(), "shap": sv}).reindex(
        np.argsort(-np.abs(sv))).reset_index(drop=True)
