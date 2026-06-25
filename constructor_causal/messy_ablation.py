"""
ARCHITECTURE ABLATION UNDER SUPERVISED READOUT (commander's Order 2 -- labeled per Order 3).
This uses the DIAGNOSTIC hidden-M3 readout (held constant across variants) so the comparison
isolates ARCHITECTURE, NOT goal supervision. It is NOT the no-leak protocol.

Knock out one split-latent component at a time and measure what breaks:
  representation:  z_c->chain, z_c->noise, z_n->noise   (linear R2)
  dynamics:        MEASURED rollout |err|@12 + compound slope under do(a0=2,a1=2)
  control:         held-out success at the discriminating cell (band m3>=12, budget 18)

Variants:
  full              -- the model that crossed the rung
  no_bilinear       -- linear-only forward (drop the structured AND-gate term)
  inverse_all_act   -- inverse predicts ALL actions incl. the noise knob aN (not control-only)
  no_zn             -- no noise latent (dn=0)
  no_decorrelation  -- drop the z_c/z_n cross-covariance penalty

Per-seed results are checkpointed to results/ablation.jsonl as they complete.
"""
from __future__ import annotations

import numpy as np
import torch

from .messy_world import MessyWorld, M1, M2, M3, N
from .messy_control import mpc_split, fit_readout
from .messy_discriminate import _roll_split, _true_traj, _stab
from .repr_encoder import collect, r2_multi
from .split_encoder import train_split
from .checkpoint import checkpointer

VARIANTS = {
    "full": dict(),
    "no_bilinear": dict(bilinear=False),
    "inverse_all_act": dict(inv_all=True),
    "no_zn": dict(use_zn=False),
    "no_decorrelation": dict(cross_w=0.0),
}
BAND = 12.0
BUDGET = 18
H = 12


def seed_records(s, n=4000):
    """Run ONE seed across all ablation variants; return a list of variant record dicts (no I/O).
    Unit of parallel/resumable work for run_seeds.py."""
    rng = np.random.default_rng(s)
    ew = MessyWorld(rng, obs_dim=14, nonlinear=True); ew.reset()
    Ob, A, Oa, Zb, Za = collect(ew, n, rng)
    true12 = _true_traj(ew, H)
    tgt = {"m1": Za[:, M1], "m2": Za[:, M2], "m3": Za[:, M3], "noise": Za[:, N]}
    out = []
    for v, kw in VARIANTS.items():
        m = train_split(Ob, A, Oa, seed=s, **kw)
        with torch.no_grad():
            zc, zn = m.encode(torch.tensor(Oa))
        zc = zc.numpy(); zn = zn.numpy()
        rc = r2_multi(zc, tgt)
        chain = float(np.mean([rc["m1"], rc["m2"], rc["m3"]]))
        zc_noise = float(rc["noise"])
        zn_noise = float(r2_multi(zn, {"noise": Za[:, N]})["noise"]) if zn.shape[1] > 0 else float("nan")
        rd = fit_readout(zc, Za[:, M3])
        err, slope, _ = _stab(_roll_split(m, ew, rd, H), true12)
        reached, final_m3 = mpc_split(ew.clone(np.random.default_rng(s + 999)), m, rd, BUDGET, band=BAND)
        out.append(dict(run="ablation", seed=int(s), variant=v, chain=chain, zc_noise=zc_noise,
                        zn_noise=zn_noise, rollout_err=float(err), rollout_slope=float(slope),
                        control_success=bool(reached), final_m3=float(final_m3),
                        failure=None if reached else f"m3={final_m3:.1f}<{BAND:.0f}"))
    return out


def main(seeds=range(10), n=4000):
    print("=" * 96)
    print(f"ARCHITECTURE ABLATION UNDER SUPERVISED READOUT -- band m3>={BAND:.0f}, "
          f"budget {BUDGET} ({len(list(seeds))} seeds)")
    print("=" * 96)
    write, path = checkpointer("ablation")
    res = {v: {"chain": [], "zc_noise": [], "zn_noise": [], "err": [], "slope": [], "succ": []}
           for v in VARIANTS}
    for s in seeds:
        for rec in seed_records(s, n):
            write(rec)
            r = res[rec["variant"]]
            r["chain"].append(rec["chain"]); r["zc_noise"].append(rec["zc_noise"])
            r["zn_noise"].append(rec["zn_noise"]); r["err"].append(rec["rollout_err"])
            r["slope"].append(rec["rollout_slope"]); r["succ"].append(rec["control_success"])
        print(f"  seed {s} done -> {path}")

    print(f"\n  {'variant':>18} {'zc->chain':>10} {'zc->noise':>10} {'zn->noise':>10} "
          f"{'roll|err|':>10} {'slope':>7} {'control':>8}")
    for v in VARIANTS:
        r = res[v]
        print(f"  {v:>18} {np.mean(r['chain']):>10.2f} {np.mean(r['zc_noise']):>10.2f} "
              f"{np.nanmean(r['zn_noise']):>10.2f} {np.mean(r['err']):>10.1f} "
              f"{np.mean(r['slope']):>7.2f} {100*np.mean(r['succ']):>6.0f}%")
    print("=" * 96)
    print(f"  per-seed checkpoints: {path}   (readout = DIAGNOSTIC hidden-M3, held constant)")
    print("=" * 96)
    return res


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(seeds=range(ns))
