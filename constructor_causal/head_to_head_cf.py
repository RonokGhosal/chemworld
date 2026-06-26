"""
PART 2 -- COUNTERFACTUALS (Pearl level 3): "given this observed trajectory, what would j have
been if i had been set differently?"  The new machinery is fixed-noise REPLAY: re-run the SAME
realization (same exogenous noise, same hidden C) under a different do() -- that is the true
counterfactual. No existing world records its noise, so we use a small SCM-step function whose
noise is explicit and replayable.

Hidden confounder C->A, C->B (no A->B edge); real edge X->Y (optionally NONLINEAR Y<-sw*tanh(X)).
Estimators of the counterfactual value of j under do(i:=val), anchored at each factual transition:
  * TRUE : replay the factual transition with do(i:=val), SAME noise -> exact counterfactual j.
  * AGENT: factual_j + (causal coef i->j, learned by INTERVENTION) * (val - factual_i).
  * PRED : factual_j + (correlational coef i->j, OLS on passive logs) * (val - factual_i).

Decisive cell: do(A:=val) -> B. The hidden C made A and B move together in the factual data, so a
predictor "explains" B via A and counterfactually drops B when A drops. TRUTH: B is unchanged (it
was driven by C, which the replay holds fixed). The agent, knowing A has no causal effect on B,
also says unchanged. Under a NONLINEAR real edge the counterfactual is realization-specific (depends
on the factual X), which a single-coefficient predictor cannot get even on the real edge.
"""
from __future__ import annotations

import numpy as np

A_, B_, X_, Y_ = 0, 1, 2, 3
NOBS = 4
NAMES = ["A", "B", "X", "Y"]
P = dict(cw=1.0, sw=0.85, sd=0.2, cd=0.78, noise=0.1)


def scm_step(o, c, nc, nobs, action, do, nonlin=False):
    """One deterministic SCM step given EXPLICIT noise (nc scalar for C, nobs[4] for obs). do is a
    {idx: value} clamp applied before and after the dynamics. Returns (next_obs, next_c)."""
    o = o.copy()
    if do:
        for k, v in do.items():
            o[k] = v
    cn = P["cd"] * c + nc
    on = np.empty(NOBS)
    on[A_] = P["sd"] * o[A_] + P["cw"] * c + action[A_] + nobs[A_]
    on[B_] = P["sd"] * o[B_] + P["cw"] * c + action[B_] + nobs[B_]
    on[X_] = P["sd"] * o[X_] + action[X_] + nobs[X_]
    gx = np.tanh(o[X_]) if nonlin else o[X_]
    on[Y_] = P["sd"] * o[Y_] + P["sw"] * gx + action[Y_] + nobs[Y_]
    if do:
        for k, v in do.items():
            on[k] = v
    return on, cn


def factual_log(rng, n=4000, nonlin=False):
    """Roll a factual trajectory, RECORDING per-step noise so any step can be replayed. Returns
    prev_o, prev_c, nc, nobs, action, next_o (all aligned, length n)."""
    o = np.zeros(NOBS); c = 0.0
    PREV, CPREV, NC, NOBSE, ACT, NXT = [], [], [], [], [], []
    for _ in range(n):
        nc = rng.normal(0, 0.6); nobs = rng.normal(0, P["noise"], NOBS)
        a = rng.normal(0, 0.4, NOBS)
        on, cn = scm_step(o, c, nc, nobs, a, None, nonlin)
        PREV.append(o.copy()); CPREV.append(c); NC.append(nc); NOBSE.append(nobs.copy())
        ACT.append(a.copy()); NXT.append(on.copy())
        o, c = on, cn
    f = lambda x: np.asarray(x, float)
    return f(PREV), f(CPREV), f(NC), f(NOBSE), f(ACT), f(NXT)


def true_cf(prev, cprev, nc, nobs, act, nxt, i, j, val, nonlin=False):
    """Mean counterfactual effect on j of do(i:=val) (vs the factual i), replaying each step's
    exact noise + hidden C."""
    tot = 0.0
    for k in range(len(prev)):
        cf, _ = scm_step(prev[k], cprev[k], nc[k], nobs[k], act[k], {i: val}, nonlin)
        tot += cf[j] - nxt[k][j]
    return tot / len(prev)


def agent_coef(seed, i, j, nonlin=False, K=300):
    """Causal coef i->j learned by INTERVENTION (no C access): do(i:=hi) vs do(i:=lo) from natural
    states, per unit. The agent ACTS; it never observes C."""
    base = np.random.default_rng(seed * 99991 + i * 7 + j)
    def branch(v):
        tot = 0.0
        for _ in range(K):
            rng = np.random.default_rng(int(base.integers(1 << 30)))
            o = np.zeros(NOBS); c = 0.0
            for _ in range(6):                                # reach a natural (hidden-C) state
                o, c = scm_step(o, c, rng.normal(0, 0.6), rng.normal(0, P["noise"], NOBS),
                                rng.normal(0, 0.4, NOBS), None, nonlin)
            on, _ = scm_step(o, c, rng.normal(0, 0.6), rng.normal(0, P["noise"], NOBS),
                             np.zeros(NOBS), {i: v}, nonlin)
            tot += on[j]
        return tot / K
    return (branch(2.0) - branch(0.0)) / 2.0


def pred_coef(prev, nxt, i, j):
    """Correlational coef i->j: OLS of next-j on [prev, 1]; take prev[i]'s coefficient."""
    Xm = np.concatenate([prev, np.ones((len(prev), 1))], 1)
    beta, *_ = np.linalg.lstsq(Xm, nxt[:, j], rcond=None)
    return float(beta[i])


def pair_type(i, j):
    if {i, j} == {A_, B_}:
        return "confounded"
    if (i, j) == (X_, Y_):
        return "real"
    return "no-path"


def main(seeds=range(15), nonlin=False, val=2.0):
    print("=" * 92)
    tag = "NONLINEAR (Y<-tanh X)" if nonlin else "linear"
    print(f"PART 2 -- COUNTERFACTUALS [{tag}], do(i:=+2) anchored at factual ({len(list(seeds))} seeds)")
    print("=" * 92)
    pairs = [(i, j) for i in range(NOBS) for j in range(NOBS) if i != j]
    err = {e: {"confounded": [], "real": [], "no-path": []} for e in ("pred", "agent")}
    for s in seeds:
        rng = np.random.default_rng(s)
        prev, cprev, nc, nobs, act, nxt = factual_log(rng, nonlin=nonlin)
        for (i, j) in pairs:
            pt = pair_type(i, j)
            tcf = true_cf(prev, cprev, nc, nobs, act, nxt, i, j, val, nonlin)
            ac = agent_coef(s, i, j, nonlin)
            pc = pred_coef(prev, nxt, i, j)
            # counterfactual EFFECT estimate on j = coef(i->j) * (val - factual_i), avg over the log
            # (anchored at each factual transition; compared to the replay-true effect tcf)
            agent_cf = float(ac * np.mean(val - prev[:, i]))
            pred_cf = float(pc * np.mean(val - prev[:, i]))
            err["agent"][pt].append(abs(agent_cf - tcf))
            err["pred"][pt].append(abs(pred_cf - tcf))
        print(f"  seed {s} done")

    def ms(xs):
        xs = np.array(xs, float)
        return f"{xs.mean():.3f}+/-{xs.std():.3f}"
    print(f"\n  COUNTERFACTUAL error vs true (replay) -- mean abs over seeds x pairs")
    print(f"  {'estimator':>12} {'CONFOUNDED':>16} {'no-path (floor)':>18} {'real-edge X->Y':>16}")
    for e in ("pred", "agent"):
        tag = "predictor" if e == "pred" else "agent"
        print(f"  {tag:>12} {ms(err[e]['confounded']):>16} {ms(err[e]['no-path']):>18} {ms(err[e]['real']):>16}")
    print("=" * 92)
    cf_p, fl_p = np.mean(err["pred"]["confounded"]), np.mean(err["pred"]["no-path"])
    cf_a = np.mean(err["agent"]["confounded"])
    print(f"  predictor confounded CF-error {cf_p:.3f} vs its no-path floor {fl_p:.3f} "
          f"(signal {cf_p-fl_p:+.3f});  agent confounded CF-error {cf_a:.3f}")
    print("  -> the predictor's COUNTERFACTUAL is wrong on the confounded pair (it counterfactually")
    print("     moves B when A changes, though A never caused B); the agent's is right.")
    print("=" * 92)
    return err


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    nl = len(sys.argv) > 2 and sys.argv[2] == "nonlin"
    main(seeds=range(ns), nonlin=nl)
