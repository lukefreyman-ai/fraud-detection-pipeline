# Interview guide - fraud-detection-pipeline

## What it does, in three sentences

It takes 284,807 real card transactions (492 frauds) and trains models to flag the fraudulent ones. Instead of picking
the "best AUC", it picks the score cut-off that minimises a dollar cost - missed fraud costs the transaction amount, every
flagged transaction costs $5 of analyst time - and reports what that cut-off means in alerts per 10,000 transactions and
fraud dollars caught. A small Streamlit page then shows an analyst the flagged transactions with a SHAP chart of *why*
each one was flagged, and records their decision.

## Why each key decision was made (plain language)

- **Split by time, not at random.** The velocity features ("how many transactions in the last hour") look backwards.
  If I shuffled rows, the model could train on hour 30 and be tested on hour 29 - it would have already seen the
  future. So: first 60 % of the timeline trains, next 20 % tunes the threshold, last 20 % is the test.
- **PR-AUC instead of ROC-AUC as the headline metric.** With 0.17 % fraud, ROC-AUC is inflated by the 99.8 % easy
  negatives; every model scored above 0.93. PR-AUC only looks at how the positives rank, so it spreads the models out (0.72
  to 0.80) and matches what the fraud team cares about: of what we flag, how much is fraud?
- **A cost matrix for the threshold.** F1 counts a missed $2,000 fraud and a $3 false alarm as equal mistakes. A bank
  doesn't. The optimiser tried 400 thresholds on the validation slice and kept the one with the lowest total cost.
- **Three imbalance tricks compared fairly.** Each one got its own cost-tuned threshold, so I'm comparing what the
  training trick buys, not where the threshold happened to land. Result: no resampling + LightGBM was cheapest; SMOTE
  produced ~2x the alerts for the same recall; class weighting made LightGBM overfit the 360 training frauds.
- **SHAP on the tree model.** Analysts won't act on a number. SHAP gives each transaction a list of which features pushed
  the score up. The top ones (V14, V4, V12) match what the EDA showed separates the classes.
- **OpenML instead of Kaggle.** The IEEE-CIS competition data needs a login, which a clean clone can't do. ULB is the
  same file Kaggle hosts, minus the login.

## What I'd do differently

- Get data with card and device ids. My velocity features are computed on the whole system because the public data has
  no ids; per-card velocity ("this card's 4th transaction in 10 minutes") is usually the single strongest signal.
- Use repeated time-based folds. 75 test frauds is too few to say 0.79 beats 0.76.
- Make the costs inputs from the business: false-decline cost varies by customer value; fraud loss depends on chargeback
  rules and merchant liability.
- Add a monitoring job (the `model-risk-monitoring` repo shows how) - fraud patterns drift within weeks.

## Five questions an interviewer would ask, with honest answers

**1. Your precision is only 39 %. Isn't that bad?**
At the chosen threshold, 6 of every 10 flagged transactions are legitimate - but the flag goes to an analyst queue, not
a hard decline. 155 alerts across 57,000 transactions is 27 per 10,000; that's a workable queue, and it catches 81 % of
the fraud dollars in the test window. Tightening to 90 % precision drops recall to about 70 %, and the cost curve says
that costs more in missed fraud than it saves in reviews.

**2. Why did SMOTE not help when every tutorial says it does?**
SMOTE creates new positives by interpolating between real ones. That smooths the score distribution and raises PR-AUC a
little, but the *decisions* at the cost-optimal threshold got worse: 279-353 alerts instead of 155 for about the same
recall. Tutorials usually report F1 at 0.5, where SMOTE looks great because it moves the threshold, not the ranking.

**3. How do you know you didn't leak information from the test set?**
Three ways: the split is chronological; the threshold was chosen on the validation slice and frozen before I scored the
test slice; and the velocity features use strictly earlier rows (`t[j] < t[i]`, checked in `tests/test_features.py`).
SMOTE was applied to the training fold only.

**4. What are V1..V28 and how would you explain V14 to a fraud manager?**
They are principal components of the original card/merchant fields, published that way for privacy. I can say "V14 was
far below normal for this transaction, which the model has learned is typical of fraud", but I can't say what V14 *is*.
On real data the same SHAP output would say "amount 6x the card's 24-hour average" - that's the point of the app design.

**5. Why LightGBM over logistic regression if the bank wants explainability?**
Because the cost result was better: $3,166 vs $3,457 expected cost on the test slice, and 155 vs 218 alerts. The
logistic model is the fallback if model risk requires reason codes that SHAP can't satisfy. The repo keeps both so the
trade-off is a table, not an opinion - and with only 75 test frauds I'd present it as "roughly equal, tree model
slightly ahead", not as a decisive win.

## Numbers to have in your head (from `results/metrics.json`)

284,807 rows, 492 frauds (0.173 %) · test = latest 20 %: 56,962 rows, 75 frauds · best: LightGBM / no resampling ·
threshold 0.0097 · precision 0.394 · recall 0.813 · 155 alerts (27.2 per 10k) · $5,339 of $7,730 fraud caught ·
expected cost $3,166 vs $3,457 (logreg, class-weighted) · SHAP top: V14, V4, V12 · whole pipeline runs in ~12 s after download.
