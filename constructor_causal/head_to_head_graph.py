"""
PART 1 -- discover the causal GRAPH: what does INTERVENTION buy you, and where does it stop?

(Rebuilt after adversarial review, which correctly killed the first version's two overclaims:
 (a) the win was attributed to the EIG "active-inference" agent, but RANDOM intervention does just
     as well at these budgets -- the real axis is INTERVENE vs OBSERVE, not the clever policy;
 (b) the confounding-refusal only worked on one hand-tuned topology where the confounded variable
     IS an actuator. In the GENERIC case -- two observed sensors sharing a hidden cause, neither
     actuatable -- intervention cannot break it, and the active agent draws the spurious edge too.)

So we test THREE arms -- epistemic (EIG), random (mindless poking), passive (observe only) -- on
three topologies, and report the honest, scoped result. Reuses ConstructorCausalAgent.explore +
recovered_dag; no new discovery code.

  actuated_confounder : a0->m (real) + H->{S1,S2}, and S1 is an ACTUATOR.  Forcing S1 decorrelates
                        it from H -> the spurious S1->S2 vanishes. Intervention (random OR epistemic)
                        recovers the real edge AND refuses the confounded one; observation can't.
  generic_confounder  : a0->m (real) + H->{S2,S3}, where S2,S3 are SENSORS (neither actuatable).
                        THE LIMIT: no intervention can break the S2~S3 correlation -> every arm,
                        including the active agent, draws the spurious edge. Intervention still
                        recovers a0->m; it just cannot de-confound a pair it cannot actuate.
  real_chain          : cascade (pure actuator-driven). Intervention recovers; observation (dead
                        knobs) cannot.
"""
from __future__ import annotations

import numpy as np

from .world import DynamicalCausalWorld
from .agent import ConstructorCausalAgent

ARMS = ("epistemic", "random", "passive")


def _factory(edges, actuators, hidden, names, noise):
    """edges: list of (target_i, source_j, weight) -> A[i,j]=w."""
    d = len(names)
    A0 = np.zeros((d, d))
    for (i, j, w) in edges:
        A0[i, j] = w
    nz = np.asarray(noise, float)
    return lambda rng: DynamicalCausalWorld(A=A0.copy(), b=np.zeros(d), noise_std=nz.copy(),
                                            actuators=actuators, names=names, hidden=hidden, rng=rng)


def worlds():
    # (factory, observed_real_edges{(src,tgt)}, confounded_pair{(a,b)} or None)
    ac = _factory([(1, 0, 0.9), (1, 1, 0.2), (2, 4, 0.95), (3, 4, 0.9), (3, 3, 0.2), (4, 4, 0.9)],
                  actuators=(0, 2), hidden=(4,), names=("a0", "m", "S1", "S2", "H"),
                  noise=[0.0, 0.2, 0.15, 1.0, 0.6])
    gc = _factory([(1, 0, 0.9), (1, 1, 0.2), (2, 4, 0.95), (3, 4, 0.9), (3, 3, 0.2), (4, 4, 0.9)],
                  actuators=(0,), hidden=(4,), names=("a0", "m", "S2", "S3", "H"),
                  noise=[0.0, 0.2, 0.15, 1.0, 0.6])
    return {
        "actuated_confounder": (ac, {(0, 1)}, (2, 3)),
        "generic_confounder": (gc, {(0, 1)}, (2, 3)),
        "real_chain(cascade)": (DynamicalCausalWorld.cascade, None, None),
    }


def score(rec, real_edges, conf_pair):
    """rec = recovered_dag() dict. Returns (real_recall, n_false, conf_false)."""
    recovered = set(map(tuple, rec["recovered"])) | set(map(tuple, rec["extra"]))
    if real_edges is None:                                   # no hidden -> grade against truth
        rr = rec["recall"]; nf = len(rec["extra"]); cf = 0
    else:
        rr = len(recovered & real_edges) / max(1, len(real_edges))
        nf = len(recovered - real_edges)                     # any recovered non-real edge
        cf = len(recovered & {conf_pair, conf_pair[::-1]}) if conf_pair else 0
    return float(rr), int(nf), int(cf)


def main(seeds=range(15), n_steps=600):
    print("=" * 96)
    print(f"PART 1 -- INTERVENE vs OBSERVE for graph discovery ({len(list(seeds))} seeds, "
          f"n_steps={n_steps})")
    print("=" * 96)
    W = worlds()
    acc = {w: {a: {"recall": [], "false": [], "conf_false": []} for a in ARMS} for w in W}
    for s in seeds:
        for wname, (factory, real_edges, conf_pair) in W.items():
            for arm in ARMS:
                w = factory(rng=np.random.default_rng(s))
                ag = ConstructorCausalAgent(w, seed=s, experimenter=arm)
                ag.explore(n_steps=n_steps)
                rr, nf, cf = score(ag.recovered_dag(), real_edges, conf_pair)
                a = acc[wname][arm]
                a["recall"].append(rr); a["false"].append(nf); a["conf_false"].append(cf)
        print(f"  seed {s} done")

    def m(xs):
        return f"{np.mean(xs):.2f}"

    for wname in W:
        print(f"\n  world = {wname}")
        print(f"    {'arm':>12} {'real-recall':>12} {'#false-edges':>14} {'confounded-false':>17}")
        for arm in ARMS:
            a = acc[wname][arm]
            tag = {"epistemic": "intervene-EIG", "random": "intervene-rand", "passive": "observe"}[arm]
            print(f"    {tag:>12} {m(a['recall']):>12} {m(a['false']):>14} {m(a['conf_false']):>17}")
    print("=" * 96)
    print("  HONEST READ:")
    ac, gc, rc = acc["actuated_confounder"], acc["generic_confounder"], acc["real_chain(cascade)"]
    print(f"   * INTERVENE > OBSERVE: actuated_confounder confounded-false  "
          f"intervene(rand) {m(ac['random']['conf_false'])} vs observe {m(ac['passive']['conf_false'])}; "
          f"cascade recall intervene(rand) {m(rc['random']['recall'])} vs observe {m(rc['passive']['recall'])}.")
    print(f"   * EIG ~= RANDOM at this budget: actuated_confounder confounded-false "
          f"EIG {m(ac['epistemic']['conf_false'])} vs rand {m(ac['random']['conf_false'])} "
          f"(the win is ACTING, not the policy).")
    print(f"   * THE LIMIT (generic_confounder, two un-actuatable sensors): confounded-false  "
          f"intervene-EIG {m(gc['epistemic']['conf_false'])} == observe {m(gc['passive']['conf_false'])} "
          f"-- intervention recovers a0->m (recall {m(gc['epistemic']['recall'])}) but CANNOT de-confound "
          f"a pair it cannot actuate.")
    print("=" * 96)
    return acc


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
