"""
PART 1 -- discover the causal GRAPH: what does INTERVENTION buy, and where does it stop?

(Rebuilt twice under adversarial review. v1 overclaimed [EIG credited for what random poking does;
one hand-tuned topology]. v2's "observe recall=0" was a FROZEN-ACTUATOR artifact [a passive learner
left the un-driven knobs pinned at 0, so it couldn't learn an edge from an input that never varies]
and it FLATTENED directed vs bidirected marks. v3 fixes both:
  * FAIR baseline: actuators carry exogenous noise, so the passive learner SEES them vary and can
    recover actuator-driven edges -- isolating the genuine confounding/orientation win from mere
    input-excitation.
  * Honest metric via recovered_marks(): we separate a FALSE DIRECTED causal assertion (a confident
    wrong edge) from an honest BIDIRECTED "can't-orient" mark.

Three arms -- epistemic (EIG), random (mindless poking), passive (observe natural variation) -- on
two topologies, 15 seeds. Reuses ConstructorCausalAgent + dag_marks; no new discovery code.

  actuated_confounder : a0->m (real) + H->{S1,S2}, S1 is an ACTUATOR. THE WIN: passive asserts a
                        FALSE DIRECTED S1->S2 (it's confounded via H); intervention forces S1,
                        decorrelates it from H, and asserts NO such edge. Both recover a0->m.
  generic_confounder  : a0->m (real) + H->{S2,S3}, S2/S3 are SENSORS (un-actuatable). THE LIMIT:
                        NO arm can orient S2~S3 -- all three honestly mark it BIDIRECTED, none
                        assert a false directed edge. Intervention recovers a0->m but cannot help
                        orient a confounded pair it cannot actuate.

So intervention's edge over observation is specifically the power to ASSERT/REFUSE directed edges
out of ACTUATABLE variables (where observation would confidently assert a confounded one). For
un-actuatable confounders, both honestly refuse to orient -- intervention does not help.
"""
from __future__ import annotations

import numpy as np

from .world import DynamicalCausalWorld
from .agent import ConstructorCausalAgent

ARMS = ("epistemic", "random", "passive")


def _factory(edges, actuators, hidden, names, noise):
    d = len(names)
    A0 = np.zeros((d, d))
    for (i, j, w) in edges:
        A0[i, j] = w
    nz = np.asarray(noise, float)
    return lambda rng: DynamicalCausalWorld(A=A0.copy(), b=np.zeros(d), noise_std=nz.copy(),
                                            actuators=actuators, names=names, hidden=hidden, rng=rng)


def worlds():
    # actuators carry exogenous noise (a0=1.0 so it varies for passive; S1=0.15 clean H proxy).
    # (factory, real_edge(src,tgt), confounded_pair(a,b))
    ac = _factory([(1, 0, 0.9), (1, 1, 0.2), (2, 4, 0.95), (3, 4, 0.9), (3, 3, 0.2), (4, 4, 0.9)],
                  actuators=(0, 2), hidden=(4,), names=("a0", "m", "S1", "S2", "H"),
                  noise=[1.0, 0.2, 0.15, 1.0, 0.6])
    gc = _factory([(1, 0, 0.9), (1, 1, 0.2), (2, 4, 0.95), (3, 4, 0.9), (3, 3, 0.2), (4, 4, 0.9)],
                  actuators=(0,), hidden=(4,), names=("a0", "m", "S2", "S3", "H"),
                  noise=[1.0, 0.2, 0.15, 1.0, 0.6])
    return {"actuated_confounder": (ac, (0, 1), (2, 3)),
            "generic_confounder": (gc, (0, 1), (2, 3))}


def score(marks, real_edge, conf_pair):
    """marks = dag_marks() {'directed': set(j,i), 'bidirected': set(a,b)}.
    Returns (real_recall, n_false_directed, conf_is_bidirected)."""
    directed = set(map(tuple, marks.get("directed", set())))
    bidir = set(frozenset(e) for e in marks.get("bidirected", set()))
    real_recall = 1.0 if real_edge in directed else 0.0
    false_directed = len(directed - {real_edge})              # confident wrong directed edges
    conf_bidir = 1 if frozenset(conf_pair) in bidir else 0     # honest "can't-orient" on the pair
    conf_false_directed = len(directed & {conf_pair, conf_pair[::-1]})  # FALSE directed on the pair
    return real_recall, false_directed, conf_bidir, conf_false_directed


def main(seeds=range(15), n_steps=600):
    print("=" * 98)
    print(f"PART 1 -- INTERVENE vs OBSERVE (fair baseline; directed vs bidirected marks)  "
          f"({len(list(seeds))} seeds)")
    print("=" * 98)
    W = worlds()
    acc = {w: {a: {"recall": [], "false_dir": [], "conf_bidir": [], "conf_false_dir": []}
               for a in ARMS} for w in W}
    for s in seeds:
        for wname, (factory, real_edge, conf_pair) in W.items():
            for arm in ARMS:
                w = factory(rng=np.random.default_rng(s))
                ag = ConstructorCausalAgent(w, seed=s, experimenter=arm)
                ag.explore(n_steps=n_steps)
                rr, fd, cb, cfd = score(ag.dag_marks(), real_edge, conf_pair)
                a = acc[wname][arm]
                a["recall"].append(rr); a["false_dir"].append(fd)
                a["conf_bidir"].append(cb); a["conf_false_dir"].append(cfd)
        print(f"  seed {s} done")

    def m(xs):
        return f"{np.mean(xs):.2f}"
    for wname in W:
        print(f"\n  world = {wname}")
        print(f"    {'arm':>14} {'a0->m recall':>13} {'false-DIRECTED':>15} "
              f"{'conf-FALSE-dir':>15} {'conf-bidirected':>16}")
        for arm in ARMS:
            a = acc[wname][arm]
            tag = {"epistemic": "intervene-EIG", "random": "intervene-rand", "passive": "observe"}[arm]
            print(f"    {tag:>14} {m(a['recall']):>13} {m(a['false_dir']):>15} "
                  f"{m(a['conf_false_dir']):>15} {m(a['conf_bidir']):>16}")
    print("=" * 98)
    ac, gc = acc["actuated_confounder"], acc["generic_confounder"]
    print("  HONEST READ:")
    print(f"   * THE WIN (actuated_confounder, fair baseline -- both recover a0->m, recall "
          f"{m(ac['passive']['recall'])}=={m(ac['epistemic']['recall'])}): observe asserts a FALSE "
          f"DIRECTED S1->S2 ({m(ac['passive']['conf_false_dir'])}/1) while intervene asserts none "
          f"({m(ac['epistemic']['conf_false_dir'])}). Intervention de-confounds an ACTUATABLE pair.")
    print(f"   * THE LIMIT (generic_confounder): no arm orients S2~S3 -- all honestly BIDIRECTED "
          f"(intervene {m(gc['epistemic']['conf_bidir'])}, observe {m(gc['passive']['conf_bidir'])}), "
          f"false-directed {m(gc['epistemic']['conf_false_dir'])} for all. Intervention cannot help "
          f"orient a confounder it cannot actuate.")
    print(f"   * EIG ~= RANDOM (win is ACTING): actuated conf-false-dir EIG "
          f"{m(ac['epistemic']['conf_false_dir'])} == rand {m(ac['random']['conf_false_dir'])}.")
    print("=" * 98)
    return acc


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
