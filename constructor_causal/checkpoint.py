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


def checkpointer(run_name, fresh=True):
    """Returns (write_fn, path). write_fn(record_dict) appends one JSON line and flushes."""
    path = os.path.join(_dir(), f"{run_name}.jsonl")
    if fresh and os.path.exists(path):
        os.remove(path)

    def write(record):
        record = {"t": time.time(), **record}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        return path

    return write, path
