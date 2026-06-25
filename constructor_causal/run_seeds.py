"""
ONE-COMMAND, PARALLEL, RESUMABLE seed/grid runner (commander's GCP scale-up order).

  python -m constructor_causal.run_seeds --run headline --seeds 30 --workers 8 --device cpu
  python -m constructor_causal.run_seeds --run ablation --seeds 20 --workers 4 --gpus 4   # GCP
  python -m constructor_causal.run_seeds --run harder   --seeds 30 --resume               # continue

* PARALLEL: a process pool runs seeds concurrently (one seed = one unit of work).
* RESUMABLE: seeds already in results/<run>.jsonl are skipped, so a killed/scaled run only
  computes what is missing. Use --fresh to start over.
* MULTI-GPU: with --gpus G, workers round-robin CUDA_VISIBLE_DEVICES across G GPUs (CC_DEVICE
  forced to cuda). Without --gpus, --device (cpu/mps/cuda) is used uniformly.

Each finished seed is flushed to the shared JSONL immediately (append-atomic per line), then a
summary is printed from the checkpoint file.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from .checkpoint import checkpointer, completed_seeds, path_for

RUNS = {
    "headline": dict(name="noleak", per_seed=1, world=None),
    "harder": dict(name="noleak_harder", per_seed=1,
                   world=dict(obs_dim=20, nonlinear=True, n_distract=6)),
    "ablation": dict(name="ablation", per_seed=5, world=None),
}


def _init(gpu_queue, device):
    if gpu_queue is not None:
        gid = gpu_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
        os.environ["CC_DEVICE"] = "cuda"
    elif device:
        os.environ["CC_DEVICE"] = device


def _worker(task):
    run, seed, world = task
    if run == "ablation":
        from .messy_ablation import seed_records
        return seed_records(seed)
    from .messy_noleak import seed_record
    return [seed_record(seed, world_kw=world)]


def summarize(run, run_name):
    path = path_for(run_name)
    recs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    print("\n" + "=" * 90)
    print(f"SUMMARY  run={run}  records={len(recs)}  file={path}")
    print("=" * 90)
    if not recs:
        return
    if run == "ablation":
        import numpy as np
        by = {}
        for r in recs:
            by.setdefault(r["variant"], []).append(r)
        print(f"  {'variant':>18} {'zc->chain':>10} {'zc->noise':>10} {'zn->noise':>10} "
              f"{'roll|err|':>10} {'control':>8}  n")
        for v, rs in by.items():
            f = lambda k: np.nanmean([x[k] for x in rs])
            print(f"  {v:>18} {f('chain'):>10.2f} {f('zc_noise'):>10.2f} {f('zn_noise'):>10.2f} "
                  f"{f('rollout_err'):>10.1f} {100*f('control_success'):>6.0f}%  {len(rs)}")
    else:
        import numpy as np
        recs.sort(key=lambda r: r["seed"])
        for bd in ("8", "12"):
            print(f"\n  band m3>={bd}:  per-seed [{' '.join('s'+str(r['seed']) for r in recs)}]")
            for a in ("causal_noleak", "causal_diag", "prediction_first", "oracle"):
                cells = " ".join("Y" if r["bands"].get(bd, {}).get(a, [False])[0] else "." for r in recs)
                rate = 100 * np.mean([r["bands"].get(bd, {}).get(a, [False])[0] for r in recs])
                print(f"    {a:>18}  [{cells}]  {rate:>3.0f}%")
        print(f"\n  repr: chain={np.mean([r['chain'] for r in recs]):.2f}  "
              f"zc_noise={np.mean([r['zc_noise'] for r in recs]):.2f}  "
              f"readoutR2 causal={np.mean([r['readout_r2_causal'] for r in recs]):.2f} "
              f"pred={np.mean([r['readout_r2_pred'] for r in recs]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--seeds", type=int, default=30, help="run seeds 0..N-1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda")
    ap.add_argument("--gpus", type=int, default=0, help="round-robin across G CUDA GPUs")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    args = ap.parse_args()

    cfg = RUNS[args.run]
    run_name = cfg["name"]
    if args.fresh and os.path.exists(path_for(run_name)):
        os.remove(path_for(run_name))
    done = set() if args.fresh else completed_seeds(run_name, cfg["per_seed"])
    todo = [s for s in range(args.seeds) if s not in done]
    write, path = checkpointer(run_name, fresh=False)
    print(f"[{args.run}] {len(done)} done, {len(todo)} to run | workers={args.workers} "
          f"device={args.device or ('cuda' if args.gpus else 'auto')} gpus={args.gpus} -> {path}")
    if not todo:
        summarize(args.run, run_name); return

    gpu_q = None
    if args.gpus > 0:
        mgr = mp.Manager(); gpu_q = mgr.Queue()
        for i in range(args.workers):
            gpu_q.put(i % args.gpus)
    tasks = [(args.run, s, cfg["world"]) for s in todo]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(gpu_q, args.device)) as pool:
        for recs in pool.imap_unordered(_worker, tasks):
            for rec in recs:
                write(rec)
            print(f"  seed {recs[0]['seed']} done ({len(recs)} rec)")
    summarize(args.run, run_name)


if __name__ == "__main__":
    main()
