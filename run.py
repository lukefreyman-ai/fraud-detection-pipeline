#!/usr/bin/env python3
"""End-to-end pipeline: data -> features -> imbalance strategies x models -> cost-based threshold -> SHAP -> results/.

  python run.py                    full run on the ULB dataset (downloads once from OpenML, ~150 MB)
  python run.py --sample-frac 0.2  quick run on the earliest 20 % of transactions
  python run.py --fp-cost 10       change the false-decline cost used to pick the threshold
  python run.py --synthetic        smoke run on synthetic data (never used for reported numbers)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fraudpipe import data as D, evaluate as E, explain as X, features as F, imbalance as I, models as M, report as R  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--fp-cost", type=float, default=5.0, help="cost of one false decline / manual review ($)")
    ap.add_argument("--fn-cost", type=float, default=None, help="flat cost of a missed fraud; default = the transaction amount")
    ap.add_argument("--strategies", default="none,class_weight,smote")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = D.make_synthetic(20_000, seed=a.seed) if a.synthetic else D.load_creditcard(ROOT / "data")
    if a.sample_frac < 1.0:
        df = df.iloc[: int(len(df) * a.sample_frac)].copy()
    print(f"rows={len(df):,} fraud={int(df.Class.sum()):,} ({df.Class.mean():.3%})")

    feats = F.build_features(df)                      # global key: no card/device id in this dataset
    cols = F.feature_columns(feats)
    train, val, test = D.time_split(feats)
    print(f"split: train={len(train):,} ({int(train.Class.sum())} fraud) val={len(val):,} ({int(val.Class.sum())}) test={len(test):,} ({int(test.Class.sum())})")
    Xtr, ytr = train[cols], train["Class"]
    Xva, yva, ava = val[cols], val["Class"].to_numpy(), val["Amount"].to_numpy()
    Xte, yte, ate = test[cols], test["Class"].to_numpy(), test["Amount"].to_numpy()

    rows, curves, fitted = [], {}, {}
    for strategy in a.strategies.split(","):
        Xs, ys = I.resample(Xtr, ytr, strategy, seed=a.seed)
        for family in ("logreg", "lightgbm"):
            tick = time.time()
            m = M.fit_logreg(Xs, ys, strategy, a.seed) if family == "logreg" else M.fit_lightgbm(Xs, ys, strategy, Xva, yva, a.seed)
            p_val, p_test = m.predict_proba(Xva), m.predict_proba(Xte)
            thr, cost_curve = E.choose_threshold(yva, p_val, ava, a.fp_cost, a.fn_cost)
            met = E.metrics_at(yte, p_test, thr, ate, a.fp_cost, a.fn_cost)
            met.update({"model": family, "strategy": strategy, "recall_at_p90": E.recall_at_precision(yte, p_test), "fit_seconds": round(time.time() - tick, 1),
                        "val_auc_pr": float(average_precision_score(yva, p_val)), **{f"extra_{k}": v for k, v in m.extra.items()}})
            rows.append(met)
            key = f"{family}/{strategy}"
            fitted[key] = (m, p_test, thr, cost_curve)
            prec, rec, _ = precision_recall_curve(yte, p_test)
            curves[key] = (rec, prec, met["auc_pr"])
            print(f"  {key:22} AUC-PR={met['auc_pr']:.3f} AUC-ROC={met['auc_roc']:.3f} thr={thr:.4f} P={met['precision']:.2f} R={met['recall']:.2f} "
                  f"alerts={met['alerts']} cost=${met['expected_cost']:,.0f} ({met['fit_seconds']}s)")

    table = R.results_table(rows).sort_values(["expected_cost"]).reset_index(drop=True)
    best_key = f"{table.iloc[0]['model']}/{table.iloc[0]['strategy']}"
    best_model, p_best, thr_best, cost_curve = fitted[best_key]
    print(f"\nbest by expected cost on test: {best_key} (threshold {thr_best:.4f})")

    R.plot_pr_curves(curves, out / "pr_curves.png")
    R.plot_cost_curve(cost_curve, thr_best, out / "cost_curve.png", a.fp_cost)
    meta = {"dataset": "synthetic" if a.synthetic else "ULB creditcard (OpenML 1597)", "rows": int(len(df)), "n_fraud": int(df.Class.sum()),
            "split": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test)), "test_fraud": int(test.Class.sum())},
            "fp_cost": a.fp_cost, "fn_cost": a.fn_cost or "transaction amount", "features": cols, "best": best_key, "best_threshold": thr_best,
            "results": rows, "elapsed_seconds": round(time.time() - t0, 1)}
    R.write_results(out, table, meta)

    if not a.skip_shap and best_model.family == "lightgbm":
        sample = Xte.sample(min(3000, len(Xte)), random_state=a.seed)
        imp = X.global_importance(best_model.estimator, sample)
        imp.to_csv(out / "shap_importance.csv", index=False)
        X.plot_importance(imp, out / "shap_importance.png")
        print("top SHAP features:", ", ".join(imp["feature"].head(8)))
    lgbm_key = next((k for k in fitted if k.startswith("lightgbm/") and k.split("/")[1] == table.iloc[0]["strategy"]), None) or best_key
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(fitted[lgbm_key][0], ROOT / "models" / "best_model.pkl")
    scored = test[["Time", "Amount", "Class"] + [c for c in cols if c not in ("Time", "Amount", "Class")]].copy()
    scored["score"] = fitted[lgbm_key][1]
    scored["flagged"] = (scored["score"] >= fitted[lgbm_key][2]).astype(int)
    scored.to_parquet(out / "scored_test.parquet", index=False)
    print(f"\nwrote {out}/results_table.md, metrics.json, plots; models/best_model.pkl; {int(scored.flagged.sum())} flagged test transactions for the app")
    print(R.to_markdown(table))


if __name__ == "__main__":
    main()
