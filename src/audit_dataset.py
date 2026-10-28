#!/usr/bin/env python3
"""Audit a labelled anomaly-detection CSV and emit a JSON report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def audit(path: Path, label_col: str = "label") -> dict:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found. Columns: {list(df.columns)}")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        excluded = {label_col, "sample_id", "anomaly_type", "difficulty"}
        feature_cols = [c for c in df.columns if c not in excluded]
    numeric = df[feature_cols].select_dtypes(include="number").columns.tolist()
    missing_by_col = df.isna().sum().to_dict()
    label_counts = df[label_col].value_counts(dropna=False).to_dict()
    report = {
        "path": str(path), "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "feature_columns": feature_cols, "numeric_feature_count": len(numeric),
        "non_numeric_feature_columns": [c for c in feature_cols if c not in numeric],
        "missing_by_column": {str(k): int(v) for k, v in missing_by_col.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "anomaly_rate": float((df[label_col] == 1).mean()),
    }
    if "anomaly_type" in df.columns:
        report["anomaly_type_counts"] = {str(k): int(v) for k, v in df["anomaly_type"].value_counts().to_dict().items()}
    if "difficulty" in df.columns:
        report["difficulty_counts"] = {str(k): int(v) for k, v in df["difficulty"].value_counts().to_dict().items()}
    report["passed_basic_checks"] = (
        report["rows"] > 0 and len(numeric) == len(feature_cols)
        and all(v == 0 for v in missing_by_col.values())
        and set(df[label_col].dropna().unique()) == {0, 1}
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--label", default="label")
    p.add_argument("--output", type=Path, default=Path("results/dataset_audit.json"))
    a = p.parse_args()
    rep = audit(a.csv, a.label)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    if not rep["passed_basic_checks"]:
        raise SystemExit("Dataset audit failed; inspect the report before modelling.")

if __name__ == "__main__":
    main()
