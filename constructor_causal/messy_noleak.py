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
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .messy_world import MessyWorld, M3
from .messy_control import mpc_split, _mpc_generic, oracle_control, fit_readout
from .repr_encoder import collect, train_encoder
from .split_encoder import train_split

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


def main(seeds=range(5), n=4000):
    print("=" * 90)
    print(f"NO-LEAK goal (K={K} binary calibration labels) vs diagnostic readout "
          f"-- budget={BUDGET}, {len(list(seeds))} seeds")
    print("=" * 90)
    agents = ["causal_noleak", "causal_diag", "prediction_first", "oracle"]
    per = {bd: {a: [] for a in agents} for bd in BANDS}

    for s in seeds:
        rng = np.random.default_rng(s)
        ew = MessyWorld(rng, obs_dim=14, nonlinear=True); ew.reset()
        Ob, A, Oa, Zb, Za = collect(ew, n, rng)
        sm = train_split(Ob, A, Oa, seed=s)
        pm = train_encoder(Ob, A, Oa, dz=6, hetero=False, inverse=False, seed=s)
        with torch.no_grad():
            zc = sm.encode(torch.tensor(Oa))[0].numpy(); zp = pm.encode(torch.tensor(Oa)).numpy()
        rd_diag = fit_readout(zc, Za[:, M3])          # 4000 hidden-M3 labels (diagnostic)
        rd_p = fit_readout(zp, Za[:, M3])
        # NO-LEAK: K observations, binary observable label at a low visible threshold tau
        tau = float(np.quantile(Za[:, M3], 0.75))
        idx = rng.choice(len(zc), K, replace=False)
        lab = (Za[idx, M3] >= tau).astype(int)
        rd_nl = noleak_readout(zc[idx], lab)
        for bd in BANDS:
            seed_rng = lambda: np.random.default_rng(s + 999)   # same episode for all agents
            per[bd]["causal_noleak"].append(
                mpc_split(ew.clone(seed_rng()), sm, rd_nl, BUDGET, band=bd)[0] if rd_nl is not None else False)
            per[bd]["causal_diag"].append(mpc_split(ew.clone(seed_rng()), sm, rd_diag, BUDGET, band=bd)[0])
            per[bd]["prediction_first"].append(_mpc_generic(ew.clone(seed_rng()), pm, rd_p, BUDGET, band=bd)[0])
            per[bd]["oracle"].append(oracle_control(ew.clone(seed_rng()), BUDGET, band=bd)[0])

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
