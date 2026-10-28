from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 8240
SAMPLE_SIZE = 10_000
DATA_PATH = Path("data/creditcard_openml_1597.csv")
META_PATH = Path("data/creditcard_openml_1597.metadata.json")

def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading OpenML creditcard dataset (ID 1597)...")
    dataset = fetch_openml(
        data_id=1597,
        as_frame=True,
        parser="auto",
    )

    X = dataset.data.apply(pd.to_numeric, errors="raise")
    y_text = pd.Series(dataset.target).astype(str).str.strip()

    if set(y_text.unique()) != {"0", "1"}:
        raise ValueError(
            f"Unexpected target values: {sorted(y_text.unique())}"
        )

    y = y_text.astype(int)

    # Use a reproducible, stratified sample for practical runtime.
    # The sample includes both fraud and legitimate transactions.
    n = min(SAMPLE_SIZE, len(X))
    positions = np.arange(len(X))
    sample_positions, _ = train_test_split(
        positions,
        train_size=n,
        random_state=SEED,
        stratify=y,
    )
    sample_positions = np.sort(sample_positions)

    X_sample = X.iloc[sample_positions].reset_index(drop=True)
    y_sample = y.iloc[sample_positions].reset_index(drop=True)

    output = pd.DataFrame(
        X_sample.to_numpy(),
        columns=[f"feature_{i + 1}" for i in range(X_sample.shape[1])],
    )
    output["label"] = y_sample.astype(int)

    if output.isna().any().any():
        raise ValueError("Missing values detected.")
    if set(output["label"].unique()) != {0, 1}:
        raise ValueError("Both binary labels must be present.")

    output.to_csv(DATA_PATH, index=False)

    metadata = {
        "source": "OpenML creditcard dataset",
        "openml_data_id": 1597,
        "source_url": "https://www.openml.org/search?id=1597&status=active&type=data",
        "original_rows": int(len(X)),
        "original_features": int(X.shape[1]),
        "sample_rows": int(len(output)),
        "feature_columns": [f"feature_{i + 1}" for i in range(X.shape[1])],
        "label_column": "label",
        "label_mapping": {
            "0": "legitimate transaction",
            "1": "fraudulent transaction (anomaly)",
        },
        "original_class_counts": {
            str(k): int(v) for k, v in y.value_counts().items()
        },
        "sample_class_counts": {
            str(k): int(v)
            for k, v in output["label"].value_counts().items()
        },
        "sampling": "Stratified random sample",
        "random_seed": SEED,
        "warning": (
            "This is a sampled formulation of the OpenML dataset, "
            "not the full 284,807-row dataset. Report the sample size "
            "and class counts in all experimental results."
        ),
    }

    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"\nSaved dataset: {DATA_PATH}")
    print(f"Saved metadata: {META_PATH}")

if __name__ == "__main__":
    main()
