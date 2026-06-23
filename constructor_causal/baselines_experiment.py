"""
THE REAL TRIAL (Order 2): does EIG beat SERIOUS prediction-based curiosity, not just naive
surprise/prediction-error? Adds ensemble disagreement, heteroscedastic disagreement, and
learning progress. Every policy updates the SAME primary WeightedHeteroModel(bayes_head) so
the metrics are comparable; only ACTION SELECTION differs.

Reading: AVOID family (low trap, learns mean) should be eig / eig_mv / hetero_disagreement /
learning_progress; LURED family (high trap) should be surprise / pred_error / disagreement.
"""
from __future__ import annotations

import numpy as np

from .curiosity_baselines import Ensemble, LPTracker
from .hetero import WeightedHeteroModel
from .macro import full_command, macro_vocabulary, rollout_states
from .noise_knob import NoiseKnobWorld

POLICIES = ["eig", "eig_mv", "surprise", "pred_error", "disagreement",
            "hetero_disagreement", "learning_progress", "random", "passive"]


def _macro_score(policy, primary, ens, lp, macro, states):
    S = primary.sensors
    if policy == "eig":
        return primary.seq_eig(states)
    if policy == "eig_mv":
        return primary.seq_eig_mv(states, 10.0)
    if policy == "surprise":
        return sum(0.5 * np.log(2 * np.pi * np.e * primary.sigma2(xc, i)) for xc, _ in states for i in S)
    if policy == "pred_error":
        return sum(np.sqrt(primary.sigma2(xc, i)) for xc, _ in states for i in S)
    if policy == "disagreement":
        return sum(sum(ens.disagreement(xc).values()) for xc, _ in states)
    if policy == "hetero_disagreement":
        return sum(d_i / primary.sigma2(xc, i)
                   for xc, _ in states for i, d_i in ens.disagreement(xc).items())
    if policy == "learning_progress":
        return lp.score(macro)
    return 0.0


def explore(world, policy, budget, rng, epsilon=0.1):
    acts = world.actuators; an = world.A_NOISE
    primary = WeightedHeteroModel(world.d, world.actuators, hidden=world.hidden,
                                  rng=np.random.default_rng(int(rng.integers(1 << 31))), bayes_head=True)
    ens = (Ensemble(world.d, world.actuators, hidden=world.hidden, K=4, rng=rng)
           if policy in ("disagreement", "hetero_disagreement") else None)
    lp = LPTracker(acts) if policy == "learning_progress" else None
    steps, trap = 0, 0
    while steps < budget:
        if policy == "passive":
            macro = [({}, 1)]
        else:
            vocab = macro_vocabulary(acts, rng=rng)
            if policy == "random" or rng.random() < epsilon:
                macro = vocab[int(rng.integers(len(vocab)))]
            else:
                x0 = world.x.copy()
                sc = np.array([_macro_score(policy, primary, ens, lp, m,
                                            rollout_states(primary, x0, m, acts)) for m in vocab])
                sc = np.nan_to_num(sc, nan=-1e18, posinf=1e18, neginf=-1e18)
                best = np.flatnonzero(sc >= sc.max() - 1e-9)
                macro = vocab[int(rng.choice(best)) if len(best) else int(rng.integers(len(vocab)))]
        for cmd, k in macro:
            fc = full_command(cmd, acts)
            for _ in range(int(k)):
                if steps >= budget:
                    break
                xc = world.x.copy()
                for j, v in fc.items():
                    xc[j] = v
                if xc[an] > 0:
                    trap += 1
                if lp is not None:
                    mu, _ = primary.predict_next(xc)
                    err = float(sum(abs(mu[i]) for i in primary.sensors))   # placeholder pre-step
                xn = world.step(fc)
                if lp is not None:
                    err = float(sum(abs(xn[i] - mu[i]) for i in primary.sensors))
                    lp.observe(xc, err)
                primary.update(xc, xn)
                if ens is not None:
                    ens.update(xc, xn)
                steps += 1
    return primary, trap / max(steps, 1)


def run(policy, seed, budget, n_distract=4):
    w = NoiseKnobWorld(n_distract, np.random.default_rng(seed)); w.reset()
    primary, trap = explore(w, policy, budget, np.random.default_rng(seed + 100))
    mean_edge = float((w.A_SIG, w.S) in primary.recovered_edges())
    cN = primary.head.coef(w.N)[1:]; ai = list(w.actuators).index(w.A_NOISE)
    var_edge = float(cN[ai] > 1.0 and cN[ai] >= np.max(cN) - 1e-9)
    return trap, mean_edge, var_edge


def _ci(v):
    a = np.array(v, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def main(seeds=range(20), budget=60, n_distract=4):
    print("=" * 84)
    print(f"REAL TRIAL -- EIG vs serious curiosity (budget={budget}, {len(list(seeds))} seeds)")
    print("=" * 84)
    res = {p: {"trap": [], "mean": [], "var": []} for p in POLICIES}
    for s in seeds:
        for p in POLICIES:
            t, me, ve = run(p, s, budget, n_distract)
            res[p]["trap"].append(t); res[p]["mean"].append(me); res[p]["var"].append(ve)
    print(f"  {'policy':>20} {'trap% (noise knob)':>20} {'MEAN edge':>11} {'VAR edge':>10}")
    for p in POLICIES:
        tm, tc = _ci(res[p]["trap"]); mm, _ = _ci(res[p]["mean"]); vm, _ = _ci(res[p]["var"])
        fam = ("AVOID" if p in ("eig", "eig_mv", "hetero_disagreement", "learning_progress")
               else ("LURED?" if p in ("surprise", "pred_error", "disagreement") else ""))
        print(f"  {p:>20} {100*tm:>14.0f}% +/-{100*tc:<4.0f} {100*mm:>9.0f}% {100*vm:>8.0f}%   {fam}")
    print("=" * 84)
    return res


if __name__ == "__main__":
    import sys
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(seeds=range(ns), budget=b)
