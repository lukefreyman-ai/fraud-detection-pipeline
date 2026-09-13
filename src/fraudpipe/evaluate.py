"""Evaluation: PR-AUC first, ROC-AUC second, and a cost-based operating threshold.

Cost matrix (per transaction):
  missed fraud (FN)   -> we lose the transaction amount (chargeback) - or a flat amount if configured
  false decline (FP)  -> fixed cost: analyst review time + customer friction (default $5)
  true positive       -> we avoid the loss but still pay the review cost
The threshold is chosen on the validation slice to minimise total expected cost, then frozen for the test slice."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def expected_cost(y: np.ndarray, pred: np.ndarray, amounts: np.ndarray, fp_cost: float, fn_cost: float | None = None) -> float:
    fn_mask = (y == 1) & (pred == 0)
    fp_mask = (y == 0) & (pred == 1)
    tp_mask = (y == 1) & (pred == 1)
    fn_loss = float(fn_cost) * fn_mask.sum() if fn_cost is not None else amounts[fn_mask].sum()
    return float(fn_loss + fp_cost * fp_mask.sum() + fp_cost * tp_mask.sum())


def choose_threshold(y: np.ndarray, p: np.ndarray, amounts: np.ndarray, fp_cost: float = 5.0, fn_cost: float | None = None,
                     grid: int = 400) -> tuple[float, pd.DataFrame]:
    """Grid-search the score threshold that minimises expected cost. Returns (threshold, cost curve)."""
    qs = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    rows = []
    for t in qs:
        pred = (p >= t).astype(int)
        rows.append({"threshold": float(t), "cost": expected_cost(y, pred, amounts, fp_cost, fn_cost), "alerts": int(pred.sum())})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["cost"].idxmin()]
    return float(best["threshold"]), curve


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float, amounts: np.ndarray, fp_cost: float = 5.0, fn_cost: float | None = None) -> dict:
    pred = (p >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold), "auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)),
        "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp, "alerts_per_10k": 1e4 * (tp + fp) / len(y),
        "fraud_dollars_caught": float(amounts[(y == 1) & (pred == 1)].sum()), "fraud_dollars_missed": float(amounts[(y == 1) & (pred == 0)].sum()),
        "expected_cost": expected_cost(y, pred, amounts, fp_cost, fn_cost), "n": int(len(y)), "n_fraud": int(y.sum()),
    }


def recall_at_precision(y: np.ndarray, p: np.ndarray, target_precision: float = 0.9) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec[prec >= target_precision]
    return float(ok.max()) if len(ok) else 0.0
