"""Class-imbalance strategies compared by the pipeline.

* none          - train on the raw 0.17 % positive rate, rely on threshold tuning afterwards
* class_weight  - reweight the loss (LogReg class_weight='balanced'; LightGBM scale_pos_weight)
* smote         - oversample the minority class synthetically in the training fold only
All three still get a cost-based threshold; the comparison isolates what the *training* trick buys."""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGIES = ("none", "class_weight", "smote")


def resample(X: pd.DataFrame, y: pd.Series, strategy: str, seed: int = 42, smote_ratio: float = 0.05) -> tuple[pd.DataFrame, pd.Series]:
    """Return the (possibly resampled) training data. SMOTE targets minority/majority = smote_ratio, not 1:1 -
    a 1:1 ratio on 0.17 % data means 99 % synthetic positives, which mostly teaches the model about SMOTE."""
    if strategy != "smote":
        return X, y
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(sampling_strategy=smote_ratio, random_state=seed, k_neighbors=5)
    Xr, yr = sm.fit_resample(X, y)
    return pd.DataFrame(Xr, columns=X.columns), pd.Series(np.asarray(yr), name=y.name)


def pos_weight(y: pd.Series, strategy: str) -> float | None:
    """scale_pos_weight for LightGBM when the strategy is class_weight; None otherwise."""
    if strategy != "class_weight":
        return None
    pos = float(y.sum())
    return (len(y) - pos) / max(pos, 1.0)
