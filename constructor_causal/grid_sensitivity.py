"""
Sensitivity grid (commander Order, hardening): is the mean+variance-EIG result robust, or
tuned around var_weight=10? Sweep var_weight x noise_gain (and budget) and report trap%,
mean-edge and variance-edge for eig (var_weight=0) vs eig_mv.
"""
from __future__ import annotations

import numpy as np

import constructor_causal.hetero_mv_experiment as mv
from .hetero import WeightedHeteroModel
from .noise_knob import NoiseKnobWorld


def run_cfg(var_weight, seed, budget, noise_gain, n_distract=4):
    policy = "eig" if var_weight == 0 else "eig_mv"
    mv.VAR_WEIGHT = float(var_weight)
    w = NoiseKnobWorld(n_distract, np.random.default_rng(seed), noise_gain=noise_gain)
    w.reset()
    model = WeightedHeteroModel(w.d, w.actuators, hidden=w.hidden,
                                rng=np.random.default_rng(seed + 7), bayes_head=True)
    trap = mv.explore(w, model, policy, budget, np.random.default_rng(seed + 100))
    mean = float((w.A_SIG, w.S) in model.recovered_edges())
    cN = model.head.coef(w.N)[1:]; ai = list(w.actuators).index(w.A_NOISE)
    var = float(cN[ai] > 1.0 and cN[ai] >= np.max(cN) - 1e-9)
    return trap, mean, var


def main(seeds=range(8), budgets=(60,), noise_gains=(2.0, 4.0, 8.0),
         var_weights=(0, 1, 3, 10, 30)):
    print("=" * 84)
    print(f"SENSITIVITY GRID -- trap% / mean% / var%  ({len(list(seeds))} seeds each)")
    print("=" * 84)
    seeds = list(seeds)
    for budget in budgets:
        for ng in noise_gains:
            print(f"\n  budget={budget}  noise_gain={ng}  (low/med/high = 2/4/8)")
            print(f"    {'var_weight':>11} {'trap%':>8} {'mean%':>8} {'var%':>8}")
            for vw in var_weights:
                T, M, V = [], [], []
                for s in seeds:
                    t, me, ve = run_cfg(vw, s, budget, ng)
                    T.append(t); M.append(me); V.append(ve)
                tag = "(eig)" if vw == 0 else ""
                print(f"    {vw:>11}{tag:>6} {100*np.mean(T):>6.0f}% {100*np.mean(M):>7.0f}% "
                      f"{100*np.mean(V):>7.0f}%")
    print("=" * 84)


if __name__ == "__main__":
    main()
