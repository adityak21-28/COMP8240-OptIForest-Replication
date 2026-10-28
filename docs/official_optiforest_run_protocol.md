# Official OptIForest run protocol

This protocol separates the authors' method from the additional scikit-learn reference baseline.

## A. Original AD experiment

1. Clone the upstream repository into `external/OptIForest`.
2. Make a separate environment for the legacy dependency stack. Save `python --version` and `python -m pip freeze`.
3. Run the upstream documented command from the upstream repository root:

```bash
python demo.py --dataset=ad --threshold=403 --branch=0
```

4. Run via the wrapper from this project root to save logs:

```bash
python src/run_official_optiforest.py --repo external/OptIForest --dataset ad --python /path/to/legacy-env/bin/python
```

5. Open the resulting `results/official_ad_*.txt` and `.json`; verify metric output and exit status. Keep the original `demo.py` output in the repository.
6. If possible, repeat the original settings and seeds to approach the paper's 15-run comparison. If the source code exposes no seed switch, clearly record that limitation.

## B. Existing new dataset and synthetic v1

1. Prepare Annthyroid using `python src/download_annthyroid_openml.py`. If outbound network is unavailable in a Codespace, download the OpenML data from <https://www.openml.org/d/40886> and preserve the exact downloaded version/provenance.
2. Export data:

```bash
python src/export_for_official_optiforest.py --csv data/annthyroid_openml_40886.csv --repo external/OptIForest --name annthyroid_openml_40886
python src/export_for_official_optiforest.py --csv data/synthetic_anomaly_benchmark_v1.csv --repo external/OptIForest --name synthetic_v1
```

3. Read upstream `demo.py` and its loader to confirm the expected CSV format and whether the dataset name is resolved dynamically or by a registry. Do not guess. If needed, add a minimal, documented dataset-loader entry in the local upstream clone; do not alter the OptIForest algorithm.
4. Run the official demo for each dataset. Save the command/output. Check that the reported ROC-AUC/AP is based on the intended binary label and that no metadata column is in the model features.
5. Repeat runs, if supported, and aggregate only the outputs that finished successfully. Store each run's metadata and a summary table. After you enter the values from actual logs into the run CSV, use `python src/summarize_official_runs.py` to compute means and standard deviations. It intentionally exits if there are no completed runs.

## C. Reporting

For each dataset, report:
- exact source and version / checksum;
- rows, numeric features, anomalies, anomaly prevalence;
- label mapping and excluded columns;
- actual official OptIForest AUC-ROC and average precision, run count and standard deviation if available;
- baseline Isolation Forest scores under a comparable protocol;
- differences from the original claims and limitations.

The current pack's measured scikit-learn baseline numbers are **not** official OptIForest measurements. Keep these categories separate in tables and conclusions.
