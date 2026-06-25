"""
ABLATE the split-latent architecture (commander's Order 2). Knock out one component at a time
and measure what breaks: linear chain-recovery R2, MEASURED rollout |err|@12 under do(a0=2,a1=2),
and held-out control success at the discriminating cell (band m3>=12, budget 18).

Variants:
  full              -- the model that crossed the rung
  no_bilinear       -- linear-only forward (drop the structured AND-gate term)
  inverse_all_act   -- inverse predicts ALL actions incl. the noise knob aN (not control-only)
  no_zn             -- no noise latent (dn=0)
  no_decorrelation  -- drop the z_c/z_n cross-covariance penalty

Readout privilege is held constant (diagnostic m3~z_c) so the comparison isolates ARCHITECTURE.
"""
from __future__ import annotations

import numpy as np
import torch

from .messy_world import MessyWorld, M1, M2, M3
from .messy_control import mpc_split, fit_readout
from .messy_discriminate import _roll_split, _true_traj, _stab
from .repr_encoder import collect, r2_multi
from .split_encoder import train_split

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


def main(seeds=range(10), n=4000):
    print("=" * 86)
    print(f"SPLIT-LATENT ABLATION -- band m3>={BAND:.0f}, budget {BUDGET} ({len(list(seeds))} seeds)")
    print("=" * 86)
    res = {v: {"chain": [], "noise": [], "err": [], "succ": []} for v in VARIANTS}
    for s in seeds:
        rng = np.random.default_rng(s)
        ew = MessyWorld(rng, obs_dim=14, nonlinear=True); ew.reset()
        Ob, A, Oa, Zb, Za = collect(ew, n, rng)
        true12 = _true_traj(ew, H)
        tgt = {"m1": Za[:, M1], "m2": Za[:, M2], "m3": Za[:, M3]}
        for v, kw in VARIANTS.items():
            m = train_split(Ob, A, Oa, seed=s, **kw)
            with torch.no_grad():
                zc = m.encode(torch.tensor(Oa))[0].numpy()
            rc = r2_multi(zc, tgt)
            chain = float(np.mean([rc["m1"], rc["m2"], rc["m3"]]))
            rd = fit_readout(zc, Za[:, M3])
            err = _stab(_roll_split(m, ew, rd, H), true12)[0]
            succ = mpc_split(ew.clone(np.random.default_rng(s + 999)), m, rd, BUDGET, band=BAND)[0]
            res[v]["chain"].append(chain); res[v]["err"].append(err); res[v]["succ"].append(succ)

    print(f"\n  {'variant':>18} {'chain R2':>9} {'rollout|err|@12':>16} {'control succ':>13}")
    for v in VARIANTS:
        c = np.mean(res[v]["chain"]); e = np.mean(res[v]["err"]); sc = 100 * np.mean(res[v]["succ"])
        print(f"  {v:>18} {c:>9.2f} {e:>16.1f} {sc:>11.0f}%")
    print("=" * 86)
    return res


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(seeds=range(ns))
