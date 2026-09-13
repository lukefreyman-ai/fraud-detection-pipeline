"""Smoke test: the whole train/threshold/evaluate loop on synthetic data (seconds, no download)."""
import numpy as np

from fraudpipe import data as D, evaluate as E, features as F, imbalance as I, models as M


def test_end_to_end_on_synthetic():
    df = D.make_synthetic(6000, fraud_rate=0.03, seed=1)
    feats = F.build_features(df); cols = F.feature_columns(feats)
    train, val, test = D.time_split(feats)
    for strategy in I.STRATEGIES:
        Xs, ys = I.resample(train[cols], train["Class"], strategy)
        lr = M.fit_logreg(Xs, ys, strategy)
        gb = M.fit_lightgbm(Xs, ys, strategy, val[cols], val["Class"], n_estimators=200)
        for m in (lr, gb):
            p_val = m.predict_proba(val[cols]); p_test = m.predict_proba(test[cols])
            thr, _ = E.choose_threshold(val["Class"].to_numpy(), p_val, val["Amount"].to_numpy(), fp_cost=5.0)
            met = E.metrics_at(test["Class"].to_numpy(), p_test, thr, test["Amount"].to_numpy())
            assert met["auc_roc"] > 0.8, (m.name, strategy, met["auc_roc"])
            assert 0 <= met["recall"] <= 1 and met["n_fraud"] > 0
    if strategy == "smote":
        assert ys.mean() > train["Class"].mean()


def test_smote_only_touches_training_data():
    df = D.make_synthetic(3000, seed=2); feats = F.build_features(df); cols = F.feature_columns(feats)
    Xs, ys = I.resample(feats[cols], feats["Class"], "smote", smote_ratio=0.2)
    assert len(Xs) > len(feats) and abs(ys.mean() - 0.2 / 1.2) < 0.02
    assert np.array_equal(I.resample(feats[cols], feats["Class"], "none")[0].to_numpy(), feats[cols].to_numpy())
