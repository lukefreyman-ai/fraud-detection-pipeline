"""Two model families: a regularised logistic regression baseline (what a bank can explain line by line)
and a LightGBM gradient-boosted tree model (what usually wins on tabular fraud data)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudpipe.imbalance import pos_weight


@dataclass
class FittedModel:
    name: str
    family: str                      # "logreg" | "lightgbm"
    strategy: str                    # imbalance strategy used in training
    estimator: Any
    features: list[str]
    extra: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.features])[:, 1]


def fit_logreg(X: pd.DataFrame, y: pd.Series, strategy: str, seed: int = 42) -> FittedModel:
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced" if strategy == "class_weight" else None, random_state=seed)
    est = Pipeline([("scale", StandardScaler()), ("clf", clf)]).fit(X, y)
    return FittedModel("logreg", "logreg", strategy, est, list(X.columns))


def fit_lightgbm(X: pd.DataFrame, y: pd.Series, strategy: str, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
                 seed: int = 42, n_estimators: int = 2000) -> FittedModel:
    import lightgbm as lgb

    params = dict(n_estimators=n_estimators, learning_rate=0.02, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=-1)
    spw = pos_weight(y, strategy)
    if spw is not None:
        params["scale_pos_weight"] = spw
    est = lgb.LGBMClassifier(**params)
    fit_kw: dict = {}
    if X_val is not None and y_val is not None:
        fit_kw = dict(eval_set=[(X_val, y_val)], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    est.fit(X, y, **fit_kw)
    extra = {"best_iteration": int(getattr(est, "best_iteration_", 0) or est.n_estimators), "scale_pos_weight": spw}
    return FittedModel("lightgbm", "lightgbm", strategy, est, list(X.columns), extra)
