#!/usr/bin/env bash
# Package the code for upload to a GCP GPU VM (run LOCALLY). Excludes venv/results/git/pdfs.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-/tmp/chemworld.tar.gz}"
tar --exclude='.venv' --exclude='results' --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pdf' --exclude='perturb_data' -czf "$OUT" constructor_causal gcp
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "upload:  gcloud compute scp $OUT <VM>:~/  --zone <ZONE>"
