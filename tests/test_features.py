import numpy as np
import pandas as pd

from fraudpipe.features import add_time_features, add_velocity_features, build_features, feature_columns


def _frame():
    return pd.DataFrame({"Time": [0, 100, 200, 4000, 4100], "Amount": [10.0, 20.0, 30.0, 40.0, 50.0], "Class": [0, 0, 1, 0, 0],
                         "V1": [0.1, 0.2, 0.3, 0.4, 0.5]})


def test_velocity_counts_are_backward_looking():
    out = add_velocity_features(_frame(), windows={"1h": 3600})
    # row 2 (t=200) sees rows at t=0 and t=100 -> count 2, amount 30; row 3 (t=4000) sees t=4000-3600=400.. -> nothing
    assert out.loc[2, "vel_1h_count"] == 2 and out.loc[2, "vel_1h_amount"] == 30.0
    assert out.loc[3, "vel_1h_count"] == 0
    assert out.loc[4, "vel_1h_count"] == 1 and out.loc[4, "secs_since_prev"] == 100


def test_velocity_respects_entity_key():
    df = _frame(); df["card"] = ["a", "b", "a", "a", "b"]
    out = add_velocity_features(df, key_col="card", windows={"1h": 3600})
    assert out.loc[2, "vel_1h_count"] == 1          # only row 0 is card a before t=200
    assert out.loc[1, "vel_1h_count"] == 0


def test_time_features_and_columns():
    out = build_features(_frame())
    assert {"hour_of_day", "log_amount", "amount_is_round", "vel_24h_count", "amount_z_24h"} <= set(out.columns)
    cols = feature_columns(out)
    assert "Class" not in cols and "Time" not in cols and "V1" in cols
    assert out["hour_of_day"].between(0, 23).all()
    assert not np.isnan(out[cols].to_numpy()).any()
