# COMP8240 OptIForest — Code and New Dataset

This is a runnable code-and-data bundle for the COMP8240 OptIForest project. It contains original synthetic data, a deterministic generator, data auditing, a scikit-learn Isolation Forest reference baseline, plotting scripts, tests, and a wrapper for running the authors' official OptIForest implementation after you clone it separately.

## Dataset included

`data/synthetic_anomaly_benchmark_v1.csv`

- 21,000 rows; 10 numerical model features.
- 20,000 normal observations (`label=0`) and 1,000 anomalies (`label=1`).
- Anomaly regimes: point, contextual, clustered.
- Difficulty labels: low, medium, high for anomalies.
- Fixed generation seed: 8240.
- `sample_id`, `label`, `anomaly_type`, and `difficulty` are metadata and MUST NOT be input features.

See `data/README.md` and `data/synthetic_anomaly_benchmark_v1.metadata.json` for the full schema and design.

## Setup and run

Python 3.10–3.12 is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/run_pipeline.sh
```

The pipeline regenerates the data with seed 8240, audits it, creates visualisations, runs 15 scikit-learn Isolation Forest baseline seeds, writes baseline metrics, and runs the test suite. On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`; run the Python commands directly if Bash is not available.

## Important distinction

The baseline script uses `sklearn.ensemble.IsolationForest`. It is **not** OptIForest and its scores must never be labelled as OptIForest results.

To run the authors' official implementation:

1. Clone https://github.com/xiagll/OptIForest separately into `external/OptIForest`.
2. Create an environment compatible with that repository and save the actual package/version information.
3. Run the official code using the protocol in `docs/official_optiforest_run_protocol.md`.
4. Inspect the upstream data loader before running exported custom datasets; do not assume the file format is accepted without checking.
5. Keep the raw terminal output and fill `results/optiforest_results_template.csv` only from actual successful runs.

## Tests

```bash
python -m pytest -q
```

Six tests cover dataset shape, reproducibility, audit checks, exclusion of metadata from exported features, rejection of invalid labels, and contextual dependency reversal.

## Data and code licences

Original support code is licensed under MIT in `LICENSE`. The generated dataset is offered under CC BY 4.0; see `DATA_LICENSE.md`. The upstream OptIForest code and external datasets retain their own terms and are not bundled here.
