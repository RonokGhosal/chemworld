"""
PART 2 -- COUNTERFACTUALS (Pearl level 3), done properly after adversarial review.

The review correctly killed v1: with ADDITIVE noise the hidden latent CANCELS in the counterfactual
difference, so "fixed-noise replay" was inert and the agent estimate was just the interventional
effect relabeled -- no abduction happened. A genuine UNIT-level counterfactual needs a per-realization
latent that does NOT cancel, recovered by ABDUCTION.

So the hidden confounder C now MODULATES the X->Y edge (multiplicative), and also drives A (an
observable proxy):
    C_{t+1} = cd*C + nc
    A = sd*A + cw*C + na            (A is a low-noise proxy of C -> enables abduction)
    B = sd*B + cw*C + nb            (confounded with A; NO A->B edge)
    X = sd*X + ax + nx              (exogenous, unconfounded)
    Y = sd*Y + (sw + gamma*C)*X + ny   (C MODULATES the X->Y slope per realization)

The UNIT counterfactual of do(X:=val) on Y for THIS realization is (sw + gamma*C)*(val - X_fac):
it depends on the realization's hidden C, so replay is now LOAD-BEARING (shifting C changes it).
  * TRUE     : replay the exact recorded noise + C under the do -> unit ground truth.
  * AGENT    : ABDUCTION -- infer C_hat from the factual A transition, then apply the (intervention-
               learned) mechanism: (sw + gamma*C_hat)*(val - X_fac). This is a real level-3 step.
  * POPULATION: the interventional / no-abduction answer E_C[(sw+gamma*C)]*(val-X) = sw*(val-X) --
               correct on AVERAGE, wrong for any realization whose C != 0. (A passive predictor that
               only knows the average slope gives exactly this.)
Plus a confounded control do(A:=+2)->B: unit CF = 0 (no A->B edge); a passive predictor that learned
the spurious A->B slope gets it wrong, the agent (A has no causal effect on B) gets 0.
"""
from __future__ import annotations

import numpy as np

A_, B_, X_, Y_ = 0, 1, 2, 3
NOBS = 4
P = dict(cw=1.0, sw=0.7, gamma=0.6, sd=0.2, cd=0.7, na=0.05, nbn=0.6, nxy=0.1)


def scm_step(o, c, nc, nobs, action, do):
    """Deterministic SCM step with explicit noise. C modulates X->Y. Returns (next_obs, next_c)."""
    o = o.copy()
    if do:
        for k, v in do.items():
            o[k] = v
    cn = P["cd"] * c + nc
    on = np.empty(NOBS)
    on[A_] = P["sd"] * o[A_] + P["cw"] * c + nobs[A_]
    on[B_] = P["sd"] * o[B_] + P["cw"] * c + nobs[B_]
    on[X_] = P["sd"] * o[X_] + action[X_] + nobs[X_]
    on[Y_] = P["sd"] * o[Y_] + (P["sw"] + P["gamma"] * c) * o[X_] + nobs[Y_]
    if do:
        for k, v in do.items():
            on[k] = v
    return on, cn


def factual_log(rng, n=4000):
    o = np.zeros(NOBS); c = 0.0
    PREV, CPREV, NC, NOB, ACT, NXT = [], [], [], [], [], []
    for _ in range(n):
        nc = rng.normal(0, 0.6)
        nob = np.array([rng.normal(0, P["na"]), rng.normal(0, P["nbn"]),
                        rng.normal(0, P["nxy"]), rng.normal(0, P["nxy"])])
        a = np.zeros(NOBS); a[X_] = rng.normal(0, 1.0)        # X exogenously excited
        on, cn = scm_step(o, c, nc, nob, a, None)
        PREV.append(o.copy()); CPREV.append(c); NC.append(nc); NOB.append(nob.copy())
        ACT.append(a.copy()); NXT.append(on.copy())
        o, c = on, cn
    f = lambda x: np.asarray(x, float)
    return f(PREV), f(CPREV), f(NC), f(NOB), f(ACT), f(NXT)


def true_cf(prev, cprev, nc, nob, act, nxt, i, j, val):
    """Unit counterfactual EFFECT on j of do(i:=val), replaying each step's exact noise + hidden C."""
    eff = np.empty(len(prev))
    for k in range(len(prev)):
        cf, _ = scm_step(prev[k], cprev[k], nc[k], nob[k], act[k], {i: val})
        eff[k] = cf[j] - nxt[k][j]
    return eff


def agent_cf(prev, nxt, i, j, val):
    """The agent's UNIT counterfactual via ABDUCTION. It knows the mechanism (from intervention)
    and infers C per realization from the factual A transition: A_next = sd*A + cw*C -> C_hat."""
    sd, cw, sw, g = P["sd"], P["cw"], P["sw"], P["gamma"]
    Chat = (nxt[:, A_] - sd * prev[:, A_]) / cw               # ABDUCTION: infer C from observed A
    if (i, j) == (X_, Y_):
        return (sw + g * Chat) * (val - prev[:, X_])          # unit-specific modulated effect
    if {i, j} == {A_, B_}:
        return np.zeros(len(prev))                            # agent knows A has no effect on B
    return np.zeros(len(prev))                                # no other causal paths


def population_cf(prev, nxt, i, j, val):
    """No abduction: the average / passive answer. For X->Y uses the mean slope sw (E[C]=0); for the
    confounded pair uses the spurious passive A->B slope (OLS of B_next on A)."""
    if (i, j) == (X_, Y_):
        Xm = np.column_stack([prev[:, X_], prev[:, Y_], np.ones(len(prev))])
        beta, *_ = np.linalg.lstsq(Xm, nxt[:, Y_], rcond=None)   # passive avg slope (E[C]=0 -> sw)
        return beta[0] * (val - prev[:, X_])
    if (i, j) == (A_, B_):
        Xm = np.column_stack([prev[:, A_], np.ones(len(prev))])
        beta, *_ = np.linalg.lstsq(Xm, nxt[:, B_], rcond=None)
        return beta[0] * (val - prev[:, A_])                  # spurious confounded slope * gap
    return np.zeros(len(prev))


def main(seeds=range(15), val=2.0):
    print("=" * 96)
    print(f"PART 2 -- UNIT COUNTERFACTUALS via abduction (C modulates X->Y) ({len(list(seeds))} seeds)")
    print("=" * 96)
    err = {e: {"modulated_XtoY": [], "confounded_AtoB": []} for e in ("population", "agent")}
    shift_changes = []
    for s in seeds:
        rng = np.random.default_rng(s)
        prev, cprev, nc, nob, act, nxt = factual_log(rng)
        for (i, j, key) in [(X_, Y_, "modulated_XtoY"), (A_, B_, "confounded_AtoB")]:
            t = true_cf(prev, cprev, nc, nob, act, nxt, i, j, val)
            ag = agent_cf(prev, nxt, i, j, val)
            pop = population_cf(prev, nxt, i, j, val)
            err["agent"][key].append(np.mean(np.abs(ag - t)))
            err["population"][key].append(np.mean(np.abs(pop - t)))
        # ABDUCTION-IS-LOAD-BEARING check: shift the recorded C; the unit truth MUST change.
        t0 = true_cf(prev, cprev, nc, nob, act, nxt, X_, Y_, val)
        t1 = true_cf(prev, cprev + 2.0, nc, nob, act, nxt, X_, Y_, val)
        shift_changes.append(float(np.mean(np.abs(t1 - t0))))
        print(f"  seed {s} done")

    def ms(xs):
        xs = np.array(xs); return f"{xs.mean():.3f}+/-{xs.std():.3f}"
    print(f"\n  UNIT counterfactual error vs replay-truth (mean abs over seeds)")
    print(f"  {'estimator':>14} {'modulated X->Y':>18} {'confounded A->B':>18}")
    for e in ("population", "agent"):
        tag = "agent(abduce)" if e == "agent" else "population"
        print(f"  {tag:>14} {ms(err[e]['modulated_XtoY']):>18} {ms(err[e]['confounded_AtoB']):>18}")
    print("=" * 96)
    print(f"  replay IS load-bearing: shifting the hidden C by +2 changes the unit truth by "
          f"{np.mean(shift_changes):.2f} (v1's additive case gave 0.00 -- abduction was inert).")
    print(f"  modulated X->Y: agent(abduction) {np.mean(err['agent']['modulated_XtoY']):.3f} vs "
          f"population(no-abduction) {np.mean(err['population']['modulated_XtoY']):.3f} -- only the "
          f"agent recovers the UNIT-specific effect by inferring the realization's hidden modulator.")
    print(f"  confounded A->B (control): agent {np.mean(err['agent']['confounded_AtoB']):.3f} (true=0) "
          f"vs passive-spurious {np.mean(err['population']['confounded_AtoB']):.3f}.")
    print("=" * 96)
    return err, np.mean(shift_changes)


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
