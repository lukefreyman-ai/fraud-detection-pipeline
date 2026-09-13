"""Writes notebooks/01_eda.ipynb (then `make notebook` executes it so the outputs are real)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# EDA - ULB credit card fraud (OpenML 1597)

284,807 card transactions over two days, 492 frauds (0.17 %). Features V1..V28 are PCA components (anonymised),
plus `Time` (seconds since the first transaction) and `Amount`. Questions this notebook answers before modelling:

1. How rare is fraud, and does the base rate drift across the two days? (matters for the chronological split)
2. Do fraud amounts look different from legitimate amounts?
3. Is there a time-of-day pattern?
4. Which raw features separate the classes most (candidate top features for the model)?
5. Do the engineered velocity features carry signal?""")
code("""import sys, warnings; sys.path.insert(0, "../src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fraudpipe.data import load_creditcard
from fraudpipe.features import build_features
df = load_creditcard("../data")
print(df.shape); df["Class"].value_counts().rename("count").to_frame().assign(pct=lambda t: (100*t["count"]/len(df)).round(3))""")
md("## 1. Base rate over time (chronological split sanity check)")
code("""df["hour"] = (df["Time"] // 3600).astype(int)
by_hour = df.groupby("hour")["Class"].agg(["size", "sum"]).rename(columns={"size": "txns", "sum": "fraud"})
by_hour["fraud_rate_pct"] = 100 * by_hour["fraud"] / by_hour["txns"]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
by_hour["txns"].plot(ax=ax[0], title="transactions per hour"); by_hour["fraud"].plot(ax=ax[1], title="frauds per hour", color="C3")
plt.tight_layout(); plt.show()
n = len(df); print("fraud in first 60% / next 20% / last 20%:", int(df.Class[: int(.6*n)].sum()), int(df.Class[int(.6*n): int(.8*n)].sum()), int(df.Class[int(.8*n):].sum()))""")
md("## 2. Amount by class")
code("""desc = df.groupby("Class")["Amount"].describe().round(2); desc""")
code("""fig, ax = plt.subplots(figsize=(7, 3.5))
for c, lab in ((0, "legit"), (1, "fraud")):
    ax.hist(np.log1p(df.loc[df.Class == c, "Amount"]), bins=60, density=True, alpha=.5, label=lab)
ax.set_xlabel("log(1 + amount)"); ax.legend(); ax.set_title("Amount distribution by class"); plt.show()
print("share of fraud with amount == 0:", round(100*(df.loc[df.Class==1,'Amount']==0).mean(),1), "%  vs legit:", round(100*(df.loc[df.Class==0,'Amount']==0).mean(),1), "%")""")
md("## 3. Time of day")
code("""df["hour_of_day"] = ((df["Time"] % 86400) // 3600).astype(int)
tod = df.groupby("hour_of_day")["Class"].agg(["size", "mean"]); tod["fraud_rate_pct"] = 100*tod["mean"]
ax = tod["fraud_rate_pct"].plot(kind="bar", figsize=(9, 3), title="fraud rate by hour of day (%)"); plt.show()
tod.sort_values("fraud_rate_pct", ascending=False).head(5)""")
md("## 4. Which raw features separate the classes?")
code("""v = [f"V{i}" for i in range(1, 29)]
sep = pd.DataFrame({"mean_legit": df.loc[df.Class==0, v].mean(), "mean_fraud": df.loc[df.Class==1, v].mean(), "std": df[v].std()})
sep["standardised_gap"] = ((sep["mean_fraud"] - sep["mean_legit"]) / sep["std"]).abs()
sep.sort_values("standardised_gap", ascending=False).head(10).round(2)""")
md("## 5. Engineered velocity features (global key - no card id in this dataset)")
code("""feats = build_features(df.drop(columns=["hour", "hour_of_day"]))
vel = ["vel_1h_count", "vel_24h_count", "vel_1h_amount", "amount_z_24h", "secs_since_prev", "is_night", "amount_is_round"]
feats.groupby("Class")[vel].mean().T.round(3).rename(columns={0: "legit", 1: "fraud"})""")
md("""### Takeaways
* Fraud is 0.17 % of rows - accuracy is meaningless; PR-AUC and recall at a fixed review budget are the metrics.
* The chronological split leaves roughly 100 frauds in the last 20 % - enough to evaluate, small enough that
  results carry sampling noise (see README limitations).
* Several PCA components (V14, V17, V12, V10, V4, V11...) separate the classes by several standard deviations -
  the model's top SHAP features should come from this list.
* The velocity features differ modestly between classes on a global key; on real data with card ids they would be
  per-card and far stronger.""")
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("01_eda.ipynb").write_text(nbf.writes(nb))
print("wrote notebooks/01_eda.ipynb")
