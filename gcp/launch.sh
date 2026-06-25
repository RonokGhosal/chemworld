#!/usr/bin/env bash
# One command to launch a resumable, multi-GPU run ON the VM.
#   bash gcp/launch.sh <run> <seeds> <workers> [gpus]
#   bash gcp/launch.sh headline 30 8        # 8 workers, auto-detect GPU count
#   bash gcp/launch.sh ablation 20 8 4      # explicitly 4 GPUs
# Re-running the SAME command resumes (finished seeds in results/<run>.jsonl are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN="${1:-headline}"; SEEDS="${2:-30}"; WORKERS="${3:-8}"
GPUS="${4:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}"

PYTHONPATH=. python -m constructor_causal.run_seeds \
  --run "$RUN" --seeds "$SEEDS" --workers "$WORKERS" --gpus "$GPUS"

echo "results -> results/  (fetch with: gcloud compute scp --recurse <VM>:~/$(basename "$PWD")/results ./ --zone <ZONE>)"
