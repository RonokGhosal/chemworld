"""
THE PAYOFF TEST (Orders 4 & 5): with a heteroscedastic noise head, does EIG specifically
AVOID a noise-generating action while surprise / prediction-error get LURED by it?

Same clean full-command semantics. The model now carries a VarianceHead (sigma^2(x,a),
refit online). Policies select macros by their objective; we record:
  TRAP%        -- budget spent driving a_noise. AVOID family should be LOW, LURED family HIGH.
  MEAN edge    -- a_sig -> s recovered (the mean graph).
  VAR edge     -- a_noise -> Var(n) MODELLED by the head (now representable, not just post-hoc).
"""
from __future__ import annotations

import numpy as np

from .hetero import VarianceHead, HETERO_SCORERS
from .macro import full_command, macro_vocabulary, rollout_states
from .model import BayesianDynamicsModel
from .noise_knob import NoiseKnobWorld

POLICIES = ["eig_hetero", "surprise_hetero", "pred_error", "random", "passive"]


def explore(world, model, head, policy, budget, rng, refit_every=15, epsilon=0.1):
    acts = world.actuators
    an = world.A_NOISE
    steps, trap, buf = 0, 0, []
    while steps < budget:
        if policy == "passive":
            macro = [({}, 1)]
        else:
            vocab = macro_vocabulary(acts, rng=rng)
            if policy == "random" or rng.random() < epsilon:
                macro = vocab[int(rng.integers(len(vocab)))]
            else:
                scorer = HETERO_SCORERS[policy]
                x0 = world.x.copy()
                sc = np.nan_to_num(np.array([scorer(model, head, rollout_states(model, x0, m, acts))
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
                xn = world.step(fc)
                model.update(xc, xn)
                buf.append((xc.copy(), xn.copy()))
                steps += 1
                if steps % refit_every == 0:
                    head.fit(buf)
        head.fit(buf)
    return trap / max(steps, 1), buf


def run(policy, seed, budget, n_distract=4):
    w = NoiseKnobWorld(n_distract, np.random.default_rng(seed)); w.reset()
    model = BayesianDynamicsModel(w.d, w.actuators, hidden=w.hidden, rng=np.random.default_rng(seed + 7))
    head = VarianceHead(model, w.actuators)
    trap, buf = explore(w, model, head, policy, budget, np.random.default_rng(seed + 100))
    mean_edge = float((w.A_SIG, w.S) in model.recovered_edges())
    # variance edge from the HEAD: a_noise is the strongest positive variance driver of n
    cN = head.coef[w.N][1:]                                   # drop bias
    ai = list(w.actuators).index(w.A_NOISE)
    var_edge = float(cN[ai] > 1.0 and cN[ai] >= np.max(cN) - 1e-9)
    return trap, mean_edge, var_edge


def _ci(v):
    a = np.array(v, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def main(seeds=range(30), budget=60, n_distract=4):
    print("=" * 80)
    print(f"HETEROSCEDASTIC payoff -- EIG-hetero vs surprise/pred-error "
          f"(budget={budget}, {n_distract} distractors, {len(list(seeds))} seeds)")
    print("=" * 80)
    res = {p: {"trap": [], "mean": [], "var": []} for p in POLICIES}
    for s in seeds:
        for p in POLICIES:
            t, me, ve = run(p, s, budget, n_distract)
            res[p]["trap"].append(t); res[p]["mean"].append(me); res[p]["var"].append(ve)
    print(f"  {'policy':>16} {'trap% (noise knob)':>20} {'MEAN a_sig->s':>15} {'VAR a_noise->Var(n)':>20}")
    for p in POLICIES:
        tm, tc = _ci(res[p]["trap"]); mm, _ = _ci(res[p]["mean"]); vm, _ = _ci(res[p]["var"])
        fam = "AVOID" if p == "eig_hetero" else ("LURED" if p in ("surprise_hetero", "pred_error") else "")
        print(f"  {p:>16} {100*tm:>14.0f}% +/-{100*tc:<4.0f} {100*mm:>12.0f}% {100*vm:>17.0f}%   {fam}")
    et, _ = _ci(res["eig_hetero"]["trap"]); st, _ = _ci(res["surprise_hetero"]["trap"])
    pt, _ = _ci(res["pred_error"]["trap"])
    print(f"\n  SEPARATION: EIG-hetero drives the noise knob {100*et:.0f}% vs "
          f"surprise-hetero {100*st:.0f}% / pred-error {100*pt:.0f}%.")
    print("=" * 80)
    return res


if __name__ == "__main__":
    import sys
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(seeds=range(ns), budget=b)
