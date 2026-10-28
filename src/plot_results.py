#!/usr/bin/env python3
"""Create report-ready charts from baseline summary CSVs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=Path("results/isolation_forest_baseline_summary.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    a = p.parse_args()
    df = pd.read_csv(a.summary)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot = df[df.scope != "all_anomalies"].copy()
    if not plot.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        x = range(len(plot))
        ax.bar(list(x), plot.auc_roc_mean, yerr=plot.auc_roc_sd.fillna(0), capsize=4, label="AUC-ROC")
        ax.set_xticks(list(x), plot.scope, rotation=15)
        ax.set_ylim(0, 1); ax.set_ylabel("AUC-ROC (mean ± SD)")
        ax.set_title("IsolationForest baseline by anomaly regime")
        fig.tight_layout(); fig.savefig(a.out_dir / "baseline_aucroc_by_regime.png", dpi=180); plt.close(fig)

    allrow = df[df.scope == "all_anomalies"]
    if not allrow.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        row = allrow.iloc[0]
        vals = [row.auc_roc_mean, row.ap_mean]
        errs = [row.auc_roc_sd if pd.notna(row.auc_roc_sd) else 0, row.ap_sd if pd.notna(row.ap_sd) else 0]
        ax.bar(["AUC-ROC", "Average precision"], vals, yerr=errs, capsize=4)
        ax.set_ylim(0, 1); ax.set_ylabel("Score (mean ± SD)")
        ax.set_title("IsolationForest baseline on synthetic data")
        fig.tight_layout(); fig.savefig(a.out_dir / "baseline_overall_metrics.png", dpi=180); plt.close(fig)
    print(f"Charts written to {a.out_dir}")

if __name__ == "__main__":
    main()
