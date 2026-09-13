import numpy as np

from fraudpipe.evaluate import choose_threshold, expected_cost, metrics_at


def test_expected_cost_counts_fn_amount_and_fp_fixed():
    y = np.array([1, 1, 0, 0]); pred = np.array([1, 0, 1, 0]); amt = np.array([100.0, 250.0, 30.0, 40.0])
    # TP: review cost 5; FN: lose 250; FP: 5
    assert expected_cost(y, pred, amt, fp_cost=5.0) == 260.0
    assert expected_cost(y, pred, amt, fp_cost=5.0, fn_cost=1000.0) == 1010.0


def test_choose_threshold_prefers_catching_expensive_fraud():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(rng.normal(0.2 + 0.6 * y, 0.15), 0, 1)
    amt = np.where(y == 1, 500.0, 50.0)
    thr, curve = choose_threshold(y, p, amt, fp_cost=5.0)
    m = metrics_at(y, p, thr, amt, fp_cost=5.0)
    assert 0 < thr < 1 and m["recall"] > 0.8 and m["alerts"] == m["tp"] + m["fp"]
    assert curve["cost"].min() == m["expected_cost"] or abs(curve["cost"].min() - m["expected_cost"]) < 1e-6


def test_metrics_keys():
    y = np.array([0, 1, 0, 1]); p = np.array([0.1, 0.9, 0.4, 0.6]); amt = np.ones(4)
    m = metrics_at(y, p, 0.5, amt)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0 and m["precision"] == 1.0 and 0 <= m["auc_pr"] <= 1
