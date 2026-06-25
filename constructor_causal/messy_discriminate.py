"""
DISCRIMINATING messy-world control + MEASURED rollout stability (commander's orders), with the
SENSOR-MAP bug fixed: every goal/rollout episode is a clone of the EXPLORATION world (identical
embodied sensor map W,b; new state + noise). Train and test share the sensor coordinate system.

Honest framing: "learned latent + supervised hidden-target readout". The m3 readout is fit on a
TRAIN split of exploration data and its HELD-OUT R2 is reported (same privilege for causal_split
and prediction_first). NOT no-leak transfer (that is messy_noleak).

Per seed (train each encoder ONCE):
  * MEASURED rollout stability: roll the learned forward under do(a0=2,a1=2); compare the
    predicted m3-readout trajectory to TRUE m3 (err@12, compound slope, sign-correct).
  * SCARCE control sweep: budget {10,14,18,24} x band {m3>=4,8,12}, success rate, vs oracle.
"""
from __future__ import annotations

import numpy as np
import torch

from .messy_world import MessyWorld, M3
from .messy_control import mpc_split, _mpc_generic, oracle_control
from .repr_encoder import collect, train_encoder
from .split_encoder import train_split

BUDGETS = [10, 14, 18, 24]
BANDS = [4.0, 8.0, 12.0]
ROLL_SEED = 12345


def _true_traj(ew, H):
    w = ew.clone(np.random.default_rng(ROLL_SEED))
    t = []
    for _ in range(H):
        w.step(np.array([2., 2., 0.], np.float32)); t.append(w.true_m3())
    return np.array(t)


def _roll_split(m, ew, readout, H):
    w = ew.clone(np.random.default_rng(ROLL_SEED))
    with torch.no_grad():
        zc, _ = m.encode(torch.tensor(w.observe(), dtype=torch.float32))
    a = torch.tensor([2., 2.]); pred = []
    with torch.no_grad():
        for _ in range(H):
            zc = m.forward_c(zc, a); pred.append(float(np.dot(np.append(zc.numpy(), 1.), readout)))
    return np.array(pred)


def _roll_generic(pm, ew, readout, H):
    w = ew.clone(np.random.default_rng(ROLL_SEED))
    with torch.no_grad():
        z = pm.encode(torch.tensor(w.observe(), dtype=torch.float32))
    a = torch.tensor([2., 2., 0.]); pred = []
    with torch.no_grad():
        for _ in range(H):
            mu, _ = pm.forward_pred(z, a); z = mu
            pred.append(float(np.dot(np.append(z.numpy(), 1.), readout)))
    return np.array(pred)


def _stab(pred, true):
    err = np.abs(pred - true)
    slope = float(np.polyfit(np.arange(1, len(err) + 1), err, 1)[0])
    return float(err[-1]), slope, bool(pred[-1] > 1 and true[-1] > 1)


def _holdout_readout(z, m3, frac=0.8):
    n = len(z); tr = int(frac * n)
    Xtr = np.column_stack([z[:tr], np.ones(tr)])
    b, *_ = np.linalg.lstsq(Xtr, m3[:tr], rcond=None)
    Xte = np.column_stack([z[tr:], np.ones(n - tr)]); t = m3[tr:]
    r2 = max(0.0, 1 - ((t - Xte @ b) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return b, r2


def main(seeds=range(5), n=4000):
    print("=" * 86)
    print(f"DISCRIMINATING messy control + rollout stability (SAME sensor map) -- {len(list(seeds))} seeds")
    print("=" * 86)
    H = 12
    stab = {"causal_split": [], "prediction_first": []}
    ro_r2 = {"causal_split": [], "prediction_first": []}
    succ = {a: {(b, bd): [] for b in BUDGETS for bd in BANDS}
            for a in ("causal_split", "prediction_first", "oracle")}

    for s in seeds:
        rng = np.random.default_rng(s)
        ew = MessyWorld(rng, obs_dim=14, nonlinear=True); ew.reset()
        Ob, A, Oa, Zb, Za = collect(ew, n, rng)
        sm = train_split(Ob, A, Oa, seed=s)
        pm = train_encoder(Ob, A, Oa, dz=6, hetero=False, inverse=False, seed=s)
        with torch.no_grad():
            zc = sm.encode(torch.tensor(Oa))[0].numpy(); zp = pm.encode(torch.tensor(Oa)).numpy()
        rd_s, r2s = _holdout_readout(zc, Za[:, M3]); rd_p, r2p = _holdout_readout(zp, Za[:, M3])
        ro_r2["causal_split"].append(r2s); ro_r2["prediction_first"].append(r2p)
        true12 = _true_traj(ew, H)
        stab["causal_split"].append(_stab(_roll_split(sm, ew, rd_s, H), true12))
        stab["prediction_first"].append(_stab(_roll_generic(pm, ew, rd_p, H), true12))
        for b in BUDGETS:
            for bd in BANDS:
                succ["causal_split"][(b, bd)].append(mpc_split(ew.clone(np.random.default_rng(s + 999)), sm, rd_s, b, band=bd)[0])
                succ["prediction_first"][(b, bd)].append(_mpc_generic(ew.clone(np.random.default_rng(s + 999)), pm, rd_p, b, band=bd)[0])
                succ["oracle"][(b, bd)].append(oracle_control(ew.clone(np.random.default_rng(s + 999)), b, band=bd)[0])

    print(f"\n  HELD-OUT readout R2 (m3 ~ latent, fit on train split):  "
          f"causal_split={np.mean(ro_r2['causal_split']):.2f}  "
          f"prediction_first={np.mean(ro_r2['prediction_first']):.2f}")
    print("\n  ROLLOUT STABILITY (roll learned forward under do(a0=2,a1=2); true m3@12 ~ +12):")
    print(f"    {'encoder':>18} {'|err|@12':>10} {'compound-slope':>15} {'sign-correct':>13}")
    for a in ("causal_split", "prediction_first"):
        e = np.mean([x[0] for x in stab[a]]); sl = np.mean([x[1] for x in stab[a]])
        sg = 100 * np.mean([x[2] for x in stab[a]])
        print(f"    {a:>18} {e:>10.1f} {sl:>15.2f} {sg:>11.0f}%")
    print("\n  CONTROL SUCCESS under scarcity (% over seeds):")
    for bd in BANDS:
        print(f"\n    band m3>={bd:.0f}:   budget=" + "   ".join(f"{b:>3}" for b in BUDGETS))
        for a in ("causal_split", "prediction_first", "oracle"):
            print(f"      {a:>18}   " + "   ".join(f"{100*np.mean(succ[a][(b,bd)]):>3.0f}" for b in BUDGETS))
    print("=" * 86)
    return succ, stab, ro_r2


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(seeds=range(ns))
