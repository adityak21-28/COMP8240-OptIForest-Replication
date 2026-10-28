#!/usr/bin/env python3
"""Download and prepare OpenML Annthyroid (data id 40886).

This is a specific public benchmark formulation (7,200 records, 6 numeric
features plus a binary target as described by OpenML), not the separate DAMI
2% version. Record this distinction in the report.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/annthyroid_openml_40886.csv"))
    p.add_argument("--metadata", type=Path, default=Path("data/annthyroid_openml_40886.metadata.json"))
    p.add_argument("--anomaly-label", default="Anomaly", help="Exact target value to map to label=1; inspect target values if needed")
    a = p.parse_args()
    ds = fetch_openml(data_id=40886, as_frame=True, parser="auto")
    X = ds.data.copy()
    y = pd.Series(ds.target).astype(str).str.strip()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) != 6:
        raise ValueError(f"Expected 6 numeric features for OpenML id 40886, found {len(numeric_cols)}: {numeric_cols}")
    X = X[numeric_cols].apply(pd.to_numeric, errors="raise")
    target_values = sorted(y.unique().tolist())
    matches = [value for value in target_values if value.casefold() == a.anomaly_label.casefold()]
    if len(matches) != 1:
        raise ValueError(f"--anomaly-label={a.anomaly_label!r} not found. Target values are {target_values}; rerun with the exact anomaly target value.")
    anomaly_value = matches[0]
    out = X.copy()
    out.columns = [f"feature_{i+1}" for i in range(X.shape[1])]
    out["label"] = (y == anomaly_value).astype(int).to_numpy()
    out["source_class"] = y.to_numpy()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    meta = {
        "source": "OpenML dataset id 40886 (Annthyroid)",
        "url": "https://www.openml.org/d/40886",
        "dataset_version": "OpenML data id 40886; fetched via scikit-learn fetch_openml",
        "rows": int(len(out)), "numeric_features": int(len(numeric_cols)),
        "target_values": target_values, "anomaly_label_value": anomaly_value,
        "anomaly_count": int(out.label.sum()), "normal_count": int((out.label == 0).sum()),
        "anomaly_rate": float(out.label.mean()),
        "warning": "This formulation is not the separate DAMI Annthyroid 2% benchmark; do not mix their reported statistics."
    }
    a.metadata.parent.mkdir(parents=True, exist_ok=True)
    a.metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {a.output}")

if __name__ == "__main__":
    main()
