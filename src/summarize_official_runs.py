#!/usr/bin/env python3
"""Summarise verified official OptIForest run records.

Populate results/optiforest_results_template.csv from actual saved logs first.
Blank result cells are ignored; this script never imputes or guesses metrics.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {"dataset", "method", "run_id", "roc_auc", "average_precision", "exit_code"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/optiforest_results_template.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/optiforest_results_summary.csv"))
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise SystemExit(f"Run table is missing columns: {missing}")
    df["roc_auc"] = pd.to_numeric(df["roc_auc"], errors="coerce")
    df["average_precision"] = pd.to_numeric(df["average_precision"], errors="coerce")
    df["exit_code"] = pd.to_numeric(df["exit_code"], errors="coerce")
    valid = df[(df["method"] == "OptIForest") & (df["exit_code"] == 0) & df["roc_auc"].notna() & df["average_precision"].notna()].copy()
    if valid.empty:
        raise SystemExit("No completed OptIForest runs with numeric metrics and exit_code=0. Run the official code and fill the CSV from saved logs first.")
    grouped = valid.groupby(["dataset", "method"], dropna=False).agg(
        n_runs=("run_id", "nunique"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        ap_mean=("average_precision", "mean"),
        ap_sd=("average_precision", "std"),
    ).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.output, index=False)
    print(grouped.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved official OptIForest summary to {args.output}")

if __name__ == "__main__":
    main()
