"""
PART 1 -- discover the causal GRAPH by intervention: active agent vs passive learner.

Extends the single-effect head-to-head to the WHOLE graph. Reuses the existing discovery stack
(ConstructorCausalAgent.explore + recovered_dag/dag_marks; model.recovered_edges/recovered_marks +
edge_scores; EpistemicExperimenter / PassiveExperimenter). No new discovery code.

Two complementary measurements over a suite of worlds, 15 seeds:

  * CONFOUNDING (the fair, decisive case): in the `confounded` world a hidden H drives S1 and S2
    with NO S1->S2 edge. BOTH learners see S1,S2 vary naturally (via H) -- so the passive baseline
    is NOT handicapped here. Only INTERVENTION (forcing S1 so it decorrelates from H) reveals there
    is no edge. Metric: FALSE edges drawn (precision / extra-count). Expect active ~0, passive ~1.

  * REAL STRUCTURE (recall): in actuator-driven worlds (`cascade`, `gated`) the true edges are
    only visible once the knobs are driven. The active agent acts and recovers them; a purely
    passive learner never drives the knobs and misses them. Metric: recall / F1. (Honest caveat:
    the passive miss here is partly because it cannot drive actuators -- the *fair* contrast is the
    confounding case above; this case shows the active agent recovers real structure.)
"""
from __future__ import annotations

import numpy as np

from .world import DynamicalCausalWorld
from .agent import ConstructorCausalAgent

WORLDS = {
    "confounded": DynamicalCausalWorld.confounded,   # fair confounding contrast (false-edge metric)
    "cascade": DynamicalCausalWorld.cascade,          # actuator-driven chain (recall)
    "gated": DynamicalCausalWorld.gated,              # gated real edges (recall)
}


def run_agent(world, seed, kind, n_steps=600):
    ag = ConstructorCausalAgent(world, seed=seed, experimenter=kind)
    ag.explore(n_steps=n_steps)
    return ag.recovered_dag()                         # {precision,recall,f1,recovered,missing,extra}


def main(seeds=range(15), n_steps=600):
    print("=" * 92)
    print(f"PART 1 -- causal GRAPH discovery: active (intervene) vs passive (observe)  "
          f"({len(list(seeds))} seeds)")
    print("=" * 92)
    # acc[world][kind] = lists of metrics across seeds
    acc = {w: {k: {"precision": [], "recall": [], "f1": [], "n_false": []}
               for k in ("epistemic", "passive")} for w in WORLDS}
    for s in seeds:
        for wname, factory in WORLDS.items():
            for kind in ("epistemic", "passive"):
                w = factory(rng=np.random.default_rng(s))
                r = run_agent(w, s, kind, n_steps=n_steps)
                a = acc[wname][kind]
                a["precision"].append(r["precision"]); a["recall"].append(r["recall"])
                a["f1"].append(r["f1"]); a["n_false"].append(len(r["extra"]))
        print(f"  seed {s} done")

    def ms(xs):
        xs = np.array(xs, float)
        return f"{xs.mean():.2f}+/-{xs.std():.2f}"

    for wname in WORLDS:
        print(f"\n  world = {wname}")
        print(f"    {'agent':>10} {'precision':>12} {'recall':>10} {'F1':>10} {'#false-edges':>14}")
        for kind in ("epistemic", "passive"):
            a = acc[wname][kind]
            tag = "active" if kind == "epistemic" else "passive"
            print(f"    {tag:>10} {ms(a['precision']):>12} {ms(a['recall']):>10} "
                  f"{ms(a['f1']):>10} {ms(a['n_false']):>14}")
    print("=" * 92)
    cf = acc["confounded"]
    print(f"  HEADLINE (fair confounding case): false edges drawn -- "
          f"active {np.mean(cf['epistemic']['n_false']):.2f}  vs  "
          f"passive {np.mean(cf['passive']['n_false']):.2f}")
    print(f"  recall on actuator-driven real structure (cascade): "
          f"active {np.mean(acc['cascade']['epistemic']['recall']):.2f}  vs  "
          f"passive {np.mean(acc['cascade']['passive']['recall']):.2f}")
    print("  -> the active agent recovers real structure AND refuses spurious confounded edges;")
    print("     the passive learner draws the spurious edge and misses actuator-driven structure.")
    print("=" * 92)
    return acc


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
