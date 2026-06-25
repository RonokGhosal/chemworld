#!/usr/bin/env bash
# Run ON the GCP GPU VM after extracting chemworld.tar.gz. Installs a CUDA build of torch + deps.
set -euo pipefail
cd "$(dirname "$0")/.."

# CUDA wheel tag: match the VM's CUDA (cu121 works on most current Deep Learning VMs).
CUDA_TAG="${CUDA_TAG:-cu121}"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install numpy scikit-learn

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
PY
echo "setup done. launch with:  bash gcp/launch.sh headline 30 8"
