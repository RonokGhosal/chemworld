"""
Does EIG specifically beat prediction-error curiosity when an action manufactures SURPRISE
without MEAN information? (the action noisy-TV, noise_knob.py)

Intervention semantics are now CLEAN: every intervention is a full do() over all actuators
(full_command), so no stale actuator persists across macros -- rollout scoring and real
execution use the identical command vector. Proven by selftest_actuator_semantics.

Two causal graphs are scored SEPARATELY (per the commander -- a_noise->Var(n) is a REAL
causal fact about the noise mechanism, just not about the mean graph):
  MEAN edge      a_sig  -> s        (recovered from the posterior mean / recovered_edges)
  VARIANCE edge  a_noise -> Var(n)  (recovered by regressing squared residuals of n on the
                                     actuators -- a homoscedastic mean-model is BLIND to it,
                                     but the DATA reveals it if the agent drove a_noise)

TRAP% = fraction of budget spent driving a_noise high. For a MEAN-ONLY agent that is wasted
budget (it cannot represent the variance edge); a future heteroscedastic agent would instead
spend it to LEARN the variance edge. All policies share the same macro vocabulary.
"""
from __future__ import annotations

import numpy as np

from .macro import full_command, macro_vocabulary, rollout_states, _SCORERS
from .model import BayesianDynamicsModel
from .noise_knob import NoiseKnobWorld

POLICIES = ["EIG", "surprise", "predfirst", "random", "passive"]


def explore_record(world, model, policy, budget, rng, epsilon=0.1):
    """macro-explore with CLEAN full-command semantics; record trap% and a residual buffer
    (clamped state, next state) for variance-edge detection."""
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
                x0 = world.x.copy()
                scorer = _SCORERS[policy]
                sc = np.nan_to_num(np.array([scorer(model, rollout_states(model, x0, m, acts))
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
    return trap / max(steps, 1), buf


def variance_edge_recovered(model, world, buf, z=3.0):
    """Does the agent's data reveal a_noise -> Var(n)? Regress the squared residual of n
    (using the learned MEAN model) on the actuators; report whether a_noise has the largest,
    significant positive coefficient. The mean model can't represent this, but the data can."""
    if len(buf) < 30:
        return 0.0
    Xc = np.array([b[0] for b in buf]); Xn = np.array([b[1] for b in buf])
    mean = model._mean(world.N)
    resid2 = np.array([(Xn[t, world.N] - float(mean @ model._phi(Xc[t]))) ** 2
                       for t in range(len(buf))])
    acts = list(world.actuators)
    A = np.column_stack([Xc[:, acts], np.ones(len(buf))])
    coef, *_ = np.linalg.lstsq(A, resid2, rcond=None)
    yhat = A @ coef
    dof = max(len(buf) - A.shape[1], 1)
    s2 = float((resid2 - yhat) @ (resid2 - yhat)) / dof
    XtX_inv = np.linalg.pinv(A.T @ A)
    k = acts.index(world.A_NOISE)
    se = float(np.sqrt(max(s2 * XtX_inv[k, k], 1e-12)))
    t = coef[k] / se
    # require a_noise to be the strongest positive variance driver AND significant
    strongest = coef[k] >= np.max(coef[:len(acts)]) - 1e-9
    return float(coef[k] > 0 and t > z and strongest)


def run(policy, seed, budget, n_distract=4):
    w = NoiseKnobWorld(n_distract, np.random.default_rng(seed)); w.reset()
    model = BayesianDynamicsModel(w.d, w.actuators, hidden=w.hidden, rng=np.random.default_rng(seed + 7))
    trap, buf = explore_record(w, model, policy, budget, np.random.default_rng(seed + 100))
    mean_edge = float((w.A_SIG, w.S) in model.recovered_edges())
    var_edge = variance_edge_recovered(model, w, buf)
    return trap, mean_edge, var_edge


def _ci(v):
    a = np.array(v, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def main(seeds=range(30), budget=60, n_distract=4):
    print("=" * 80)
    print(f"ACTION NOISY-TV (clean semantics) -- budget={budget}, {n_distract} distractors, "
          f"{len(list(seeds))} seeds")
    print("=" * 80)
    res = {p: {"trap": [], "mean": [], "var": []} for p in POLICIES}
    for s in seeds:
        for p in POLICIES:
            t, me, ve = run(p, s, budget, n_distract)
            res[p]["trap"].append(t); res[p]["mean"].append(me); res[p]["var"].append(ve)
    print(f"  {'policy':>10} {'trap% (noise knob)':>20} {'MEAN edge a_sig->s':>20} "
          f"{'VAR edge a_noise->Var(n)':>26}")
    for p in POLICIES:
        tm, tc = _ci(res[p]["trap"]); mm, mc = _ci(res[p]["mean"]); vm, vc = _ci(res[p]["var"])
        print(f"  {p:>10} {100*tm:>14.0f}% +/-{100*tc:<4.0f} {100*mm:>14.0f}% +/-{100*mc:<3.0f} "
              f"{100*vm:>18.0f}% +/-{100*vc:<3.0f}")
    print("\n  NOTE: the homoscedastic model cannot REPRESENT the variance edge; 'VAR edge' "
          "here is\n  whether the agent's DATA reveals it (post-hoc) -- which requires having "
          "driven a_noise.\n  EIG-hetero (next order) is what would let the agent MODEL it.")
    print("=" * 80)
    return res


if __name__ == "__main__":
    import sys
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(seeds=range(ns), budget=b)
