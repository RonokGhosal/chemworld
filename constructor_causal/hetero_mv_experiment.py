"""
Order 4: resolve the variance-edge tension. Mean-only EIG avoids the noise knob (good) but
under-learns the REAL variance edge a_noise->Var(n). EIG over BOTH mean and variance
parameters (seq_eig_mv) should drive the noise knob ENOUGH to characterise the variance law,
then stop -- recovering BOTH graphs without being lured like surprise/prediction-error.

All policies use WeightedHeteroModel(bayes_head=True). Compare:
  eig      -- mean-only weighted EIG  (avoids noise, under-learns variance)
  eig_mv   -- mean + variance EIG     (should learn variance too, still bounded)
  surprise -- heteroscedastic entropy (lured)
  pred_error / random / passive
"""
from __future__ import annotations

import numpy as np

from .hetero import WeightedHeteroModel
from .macro import full_command, macro_vocabulary, rollout_states
from .noise_knob import NoiseKnobWorld

POLICIES = ["eig", "eig_mv", "surprise", "pred_error", "random", "passive"]
VAR_WEIGHT = 10.0   # mean+variance EIG: learns the variance edge without becoming lured


def _score(policy, model, states):
    if policy == "eig":
        return model.seq_eig(states)
    if policy == "eig_mv":
        return model.seq_eig_mv(states, var_weight=VAR_WEIGHT)
    if policy == "surprise":
        return sum(0.5 * np.log(2 * np.pi * np.e * model.sigma2(xc, i))
                   for xc, _ in states for i in model.sensors)
    if policy == "pred_error":
        return sum(np.sqrt(model.sigma2(xc, i)) for xc, _ in states for i in model.sensors)
    return 0.0


def explore(world, model, policy, budget, rng, epsilon=0.1):
    acts = world.actuators; an = world.A_NOISE
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
                sc = np.nan_to_num(np.array([_score(policy, model, rollout_states(model, x0, m, acts))
                                             for m in vocab]), nan=-1e18, posinf=1e18, neginf=-1e18)
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
                model.update(xc, world.step(fc))
                steps += 1
    return trap / max(steps, 1)


def run(policy, seed, budget, n_distract=4):
    w = NoiseKnobWorld(n_distract, np.random.default_rng(seed)); w.reset()
    model = WeightedHeteroModel(w.d, w.actuators, hidden=w.hidden,
                                rng=np.random.default_rng(seed + 7), bayes_head=True)
    trap = explore(w, model, policy, budget, np.random.default_rng(seed + 100))
    mean_edge = float((w.A_SIG, w.S) in model.recovered_edges())
    cN = model.head.coef(w.N)[1:]; ai = list(w.actuators).index(w.A_NOISE)
    var_edge = float(cN[ai] > 1.0 and cN[ai] >= np.max(cN) - 1e-9)
    return trap, mean_edge, var_edge


def _ci(v):
    a = np.array(v, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def main(seeds=range(30), budget=60, n_distract=4):
    print("=" * 80)
    print(f"MEAN+VARIANCE EIG (Order 4) -- budget={budget}, {len(list(seeds))} seeds")
    print("=" * 80)
    res = {p: {"trap": [], "mean": [], "var": []} for p in POLICIES}
    for s in seeds:
        for p in POLICIES:
            t, me, ve = run(p, s, budget, n_distract)
            res[p]["trap"].append(t); res[p]["mean"].append(me); res[p]["var"].append(ve)
    print(f"  {'policy':>12} {'trap%':>14} {'MEAN a_sig->s':>15} {'VAR a_noise->Var(n)':>21}")
    for p in POLICIES:
        tm, tc = _ci(res[p]["trap"]); mm, _ = _ci(res[p]["mean"]); vm, vc = _ci(res[p]["var"])
        print(f"  {p:>12} {100*tm:>9.0f}% +/-{100*tc:<3.0f} {100*mm:>12.0f}% {100*vm:>15.0f}% +/-{100*vc:<3.0f}")
    em, _ = _ci(res["eig"]["var"]); mvm, _ = _ci(res["eig_mv"]["var"])
    et, _ = _ci(res["eig"]["trap"]); mvt, _ = _ci(res["eig_mv"]["trap"])
    print(f"\n  TENSION CHECK: variance-edge recovery  eig(mean-only)={100*em:.0f}%  ->  "
          f"eig_mv={100*mvm:.0f}%   (trap {100*et:.0f}% -> {100*mvt:.0f}%; both keep mean=100%)")
    print("=" * 80)
    return res


if __name__ == "__main__":
    import sys
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(seeds=range(ns), budget=b)
