"""Per-seed checkpointing (commander's order): every long run writes a JSONL line as each seed
completes, so a job that dies late does not die empty. Results land in ./results/<run>.jsonl
(override dir with CC_RESULTS)."""
from __future__ import annotations

import json
import os
import time


def _dir():
    d = os.environ.get("CC_RESULTS") or os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(d, exist_ok=True)
    return d


def path_for(run_name):
    return os.path.join(_dir(), f"{run_name}.jsonl")


def checkpointer(run_name, fresh=True):
    """Returns (write_fn, path). write_fn(record_dict) appends one JSON line and flushes.
    Appends are atomic per-line, so multiple worker processes may share one file."""
    path = path_for(run_name)
    if fresh and os.path.exists(path):
        os.remove(path)

    def write(record):
        record = {"t": time.time(), **record}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        return path

    return write, path


def completed_seeds(run_name, expected_per_seed=1):
    """Seeds already finished in results/<run>.jsonl -- a seed counts as done once it has
    >= expected_per_seed records (ablation writes one record per variant). Enables RESUME:
    a killed/scaled run skips finished seeds and only computes what's missing."""
    path = path_for(run_name)
    counts = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    s = json.loads(line)["seed"]
                    counts[s] = counts.get(s, 0) + 1
                except Exception:
                    pass
    return {s for s, c in counts.items() if c >= expected_per_seed}
