#!/usr/bin/env bash
# Create the reduced-scale credit-card runner from the authors' unmodified demo.py.
# The upstream repository has no licence, so we do not redistribute a modified copy;
# this script derives it locally. The only changes are: 100 -> 10 trees, 15 -> 5 repetitions.
# Usage: bash scripts/make_creditcard_diagnostic.sh external/OptIForest
set -euo pipefail
UPSTREAM="${1:?path to the cloned upstream OptIForest repo}"
sed -e 's/^num_ensemblers = 100$/num_ensemblers = 10/' \
    -e 's/for j in range(15):/for j in range(5):/' \
    "$UPSTREAM/demo.py" > "$UPSTREAM/demo_creditcard_diagnostic.py"
echo "Wrote $UPSTREAM/demo_creditcard_diagnostic.py (10 trees, 5 runs)."
echo "Run:  cd $UPSTREAM && python demo_creditcard_diagnostic.py --dataset=creditcard_openml_1597 --threshold=403 --branch=0"
