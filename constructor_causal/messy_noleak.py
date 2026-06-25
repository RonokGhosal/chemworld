"""
NO-LEAK goal specification (commander's Order 4). The diagnostic runs fit the m3 readout on
4000 HIDDEN true-M3 labels. Here the goal is specified WITHOUT hidden M3, via a small
post-goal CALIBRATION SET that is explicitly counted as supervision:

  * K calibration observations are drawn from reward-free exploration.
  * Each gets a BINARY observable label: did this state clear a low, visible threshold tau
    (the 75th percentile of exploration m3)?  -> K binary labels, nothing else.
  * A logistic goal-score g(z_c) = sigmoid([z_c,1].w) is fit on those K labels. Its increasing
    direction in z_c IS the "drive m3 up" direction; MPC maximizes the logit (reuses mpc_split
    with readout = logistic coefficients). The hidden continuous M3 is NEVER used to plan.

Evaluation still measures true m3 (that is scoring, not supervision). We report no-leak vs the
diagnostic readout vs prediction-first vs oracle, PER SEED, at the discriminating cells.

Supervision ledger:  no-leak = K binary labels (counted).  diagnostic = 4000 continuous M3.

HONESTY (commander's Order 3): this is SUPERVISED control with a small POST-GOAL label budget,
NOT unsupervised goal discovery. The K binary labels are real supervision and are counted as
such. In this simulator the label is generated from hidden M3 (m3>=tau) as a stand-in; in real
use that bit must come from an actual observable task detector / human success signal, not from
a hidden state. The result is "low-shot supervised control", not "label-free control".
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .messy_world import MessyWorld, M1, M2, M3, N
from .messy_control import mpc_split, _mpc_generic, oracle_control, fit_readout
from .repr_encoder import collect, train_encoder, r2_multi
from .split_encoder import train_split
from .checkpoint import checkpointer

BANDS = [8.0, 12.0]
BUDGET = 18
K = 80                      # calibration labels (the entire no-leak supervision budget)


def noleak_readout(zc_cal, labels):
    """Logistic goal-score from K binary observable labels; returns [w,b] so that
    [z_c,1] @ readout = the logit (monotone in P(reach)). mpc_split maximizes it directly."""
    if labels.min() == labels.max():                 # degenerate (all one class)
        return None
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(zc_cal, labels)
    return np.append(clf.coef_[0], clf.intercept_[0])


def _holdout_r2(z, m3, frac=0.8):
    n = len(z); tr = int(frac * n)
    X = np.column_stack([z[:tr], np.ones(tr)])
    b, *_ = np.linalg.lstsq(X, m3[:tr], rcond=None)
    Xte = np.column_stack([z[tr:], np.ones(n - tr)]); t = m3[tr:]
    return float(max(0.0, 1 - ((t - Xte @ b) ** 2).sum() / ((t - t.mean()) ** 2).sum()))


def seed_record(s, n=4000, world_kw=None):
    """Run ONE seed end-to-end and return its full record dict (no I/O). This is the unit of
    parallel/resumable work for run_seeds.py; main() also calls it."""
    wk = world_kw or dict(obs_dim=14, nonlinear=True)
    run_name = "noleak_harder" if wk.get("n_distract") else "noleak"
    rng = np.random.default_rng(s)
    ew = MessyWorld(rng, **wk); ew.reset()
    Ob, A, Oa, Zb, Za = collect(ew, n, rng)
    sm = train_split(Ob, A, Oa, seed=s)
    pm = train_encoder(Ob, A, Oa, dz=6, hetero=False, inverse=False, seed=s)
    with torch.no_grad():
        zc_t, zn_t = sm.encode(torch.tensor(Oa))
        zc = zc_t.numpy(); zn = zn_t.numpy(); zp = pm.encode(torch.tensor(Oa)).numpy()
    rcc = r2_multi(zc, {"m1": Za[:, M1], "m2": Za[:, M2], "m3": Za[:, M3], "noise": Za[:, N]})
    chain = float(np.mean([rcc["m1"], rcc["m2"], rcc["m3"]])); zc_noise = float(rcc["noise"])
    zn_noise = float(r2_multi(zn, {"noise": Za[:, N]})["noise"]) if zn.shape[1] > 0 else float("nan")
    r2_diag = _holdout_r2(zc, Za[:, M3]); r2_pred = _holdout_r2(zp, Za[:, M3])
    rd_diag = fit_readout(zc, Za[:, M3])              # 4000 hidden-M3 labels (diagnostic)
    rd_p = fit_readout(zp, Za[:, M3])
    tau = float(np.quantile(Za[:, M3], 0.75))         # low visible threshold for K binary labels
    idx = rng.choice(len(zc), K, replace=False)
    lab = (Za[idx, M3] >= tau).astype(int)
    rd_nl = noleak_readout(zc[idx], lab)
    rec = dict(run=run_name, seed=int(s), chain=chain, zc_noise=zc_noise, zn_noise=zn_noise,
               readout_r2_causal=r2_diag, readout_r2_pred=r2_pred, tau=tau, bands={})
    for bd in BANDS:
        seed_rng = lambda: np.random.default_rng(s + 999)    # same episode for all agents
        nl = mpc_split(ew.clone(seed_rng()), sm, rd_nl, BUDGET, band=bd) if rd_nl is not None else (False, float("nan"))
        di = mpc_split(ew.clone(seed_rng()), sm, rd_diag, BUDGET, band=bd)
        pr = _mpc_generic(ew.clone(seed_rng()), pm, rd_p, BUDGET, band=bd)
        orc = oracle_control(ew.clone(seed_rng()), BUDGET, band=bd)
        rec["bands"][str(int(bd))] = dict(
            causal_noleak=[bool(nl[0]), float(nl[1])], causal_diag=[bool(di[0]), float(di[1])],
            prediction_first=[bool(pr[0]), float(pr[1])], oracle=[bool(orc[0]), float(orc[1])])
    return rec


def main(seeds=range(5), n=4000, world_kw=None, label=""):
    wk = world_kw or dict(obs_dim=14, nonlinear=True)
    print("=" * 90)
    print(f"NO-LEAK goal (K={K} binary calibration labels) vs diagnostic readout "
          f"-- budget={BUDGET}, {len(list(seeds))} seeds  {label}")
    print(f"  world: {wk}")
    print("=" * 90)
    agents = ["causal_noleak", "causal_diag", "prediction_first", "oracle"]
    per = {bd: {a: [] for a in agents} for bd in BANDS}
    run_name = "noleak_harder" if wk.get("n_distract") else "noleak"
    write, path = checkpointer(run_name)

    for s in seeds:
        rec = seed_record(s, n, wk)
        for bd in BANDS:
            for a in agents:
                per[bd][a].append(rec["bands"][str(int(bd))][a][0])
        write(rec)
        print(f"  seed {s} done (chain={rec['chain']:.2f} zc_noise={rec['zc_noise']:.2f}) -> {path}")

    sl = list(seeds)
    for bd in BANDS:
        print(f"\n  band m3>={bd:.0f}  (budget {BUDGET}):   per-seed success [{', '.join(f's{x}' for x in sl)}]   mean")
        for a in agents:
            row = per[bd][a]
            cells = "  ".join("Y" if r else "." for r in row)
            print(f"    {a:>18}   [{cells}]   {100*np.mean(row):>4.0f}%")
    print("=" * 90)
    print(f"  supervision: no-leak = {K} binary labels  |  diagnostic = {n} continuous M3 labels")
    print("=" * 90)
    return per


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(seeds=range(ns))
