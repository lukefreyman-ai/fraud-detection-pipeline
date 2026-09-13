"""Results table + plots written to results/ so every number in the README is reproducible."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def results_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = ["model", "strategy", "auc_pr", "auc_roc", "threshold", "precision", "recall", "f1", "alerts", "alerts_per_10k",
            "fraud_dollars_caught", "fraud_dollars_missed", "expected_cost"]
    return df[[c for c in cols if c in df.columns]]


def to_markdown(df: pd.DataFrame) -> str:
    fmt = df.copy()
    for c in ("auc_pr", "auc_roc", "precision", "recall", "f1"):
        if c in fmt: fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")
    for c in ("threshold",):
        if c in fmt: fmt[c] = fmt[c].map(lambda v: f"{v:.4f}")
    for c in ("fraud_dollars_caught", "fraud_dollars_missed", "expected_cost"):
        if c in fmt: fmt[c] = fmt[c].map(lambda v: f"${v:,.0f}")
    if "alerts_per_10k" in fmt: fmt["alerts_per_10k"] = fmt["alerts_per_10k"].map(lambda v: f"{v:.1f}")
    head = "| " + " | ".join(fmt.columns) + " |\n|" + "|".join("---" for _ in fmt.columns) + "|\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in fmt.to_numpy())
    return head + body + "\n"


def write_results(out_dir: Path, table: pd.DataFrame, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "results_table.csv", index=False)
    (out_dir / "results_table.md").write_text(to_markdown(table))
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=1, default=float))


def plot_pr_curves(curves: dict[str, tuple], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (rec, prec, ap) in curves.items():
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", lw=1.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall on the held-out (latest) 20 %", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_cost_curve(curve: pd.DataFrame, threshold: float, path: Path, fp_cost: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["threshold"], curve["cost"], color="#d9480f", lw=1.6)
    ax.axvline(threshold, ls="--", color="k", lw=1, label=f"chosen threshold = {threshold:.4f}")
    ax.set_xscale("log"); ax.set_xlabel("score threshold (log)"); ax.set_ylabel("expected cost on validation ($)")
    ax.set_title(f"Cost curve: missed fraud = txn amount, false decline = ${fp_cost:.0f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
