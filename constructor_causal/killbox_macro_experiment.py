"""
KILL BOX + MACRO-ACTIONS: does temporally-extended EIG planning recover what per-step
policies could not (the deep chain m2->m3, the open-then-drive gate) -- and does EIG beat
the dumb policies once they ALL have the same macro vocabulary?

Per the commander's orders: identical macro vocabulary for every policy; a real
prediction-first baseline (ICM) alongside surprise; 30+ seeds with confidence intervals.
"""
from __future__ import annotations

import numpy as np

from .killbox import (killbox, CORE_NAMES, TRUE_OBSERVED_EDGES, CONFOUNDED_PAIR, DEEP_EDGE)
from .macro import macro_explore
from .model import BayesianDynamicsModel

NT = len(TRUE_OBSERVED_EDGES)
CORE = list(range(11))
PAIRS = [(a, b) for ia, a in enumerate(CORE) for b in CORE[ia:]]
POLICIES = ["EIG", "random", "surprise", "predfirst", "passive"]


def run(policy, seed, budget, n_distract=0):
    w = killbox(np.random.default_rng(seed), n_distract=n_distract)
    w.reset()
    model = BayesianDynamicsModel(w.d, w.actuators, hidden=w.hidden, interaction_pairs=PAIRS,
                                  rng=np.random.default_rng(seed + 7))
    macro_explore(w, model, policy, budget, w.actuators, np.random.default_rng(seed + 100))
    return model


def score(model):
    # STRUCTURE DISCOVERY uses recovered_edges (time-oriented): in a dynamical model the
    # lag already orients j_t -> i_{t+1}, so the deep chain edge m2->m3 is discoverable even
    # though no actuator can do-identify it (the honesty layer abstains on it -- correct, but
    # that is a separate question from "did the agent find the dependency").
    E = {(j, i) for (j, i) in model.recovered_edges() if j < 11 and i < 11}
    tp = len(E & TRUE_OBSERVED_EDGES)
    rec = tp / NT
    prec = tp / len(E) if E else 1.0
    bid = {frozenset(e) for e in model.recovered_marks()["bidirected"]}
    return dict(rec=rec, prec=prec, deep=float(DEEP_EDGE in E),
                conf=float(CONFOUNDED_PAIR in bid))


def _ci(vals):
    a = np.array(vals, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def main(seeds=range(30), budget=320, n_distract=0):
    print("=" * 78)
    print(f"KILL BOX + MACROS -- EIG vs dumb (budget={budget}, n_distract={n_distract}, "
          f"{len(list(seeds))} seeds)")
    print("=" * 78)
    res = {p: [] for p in POLICIES}
    seeds = list(seeds)
    for s in seeds:
        for p in POLICIES:
            res[p].append(score(run(p, s, budget, n_distract)))
    print(f"  {'policy':>10} {'recall (95% CI)':>22} {'deep-edge%':>11} {'conf-honest%':>13} {'prec':>6}")
    for p in POLICIES:
        rm, rc = _ci([x["rec"] for x in res[p]])
        dm, _ = _ci([x["deep"] for x in res[p]])
        cm, _ = _ci([x["conf"] for x in res[p]])
        pm, _ = _ci([x["prec"] for x in res[p]])
        print(f"  {p:>10}   {rm:>5.2f} +/- {rc:<5.2f}{'':>6} {100*dm:>9.0f}% {100*cm:>11.0f}% {pm:>6.2f}")
    # headline: EIG vs the best dumb-active policy
    eig, _ = _ci([x["rec"] for x in res["EIG"]])
    dumb = max(("random", "surprise", "predfirst"),
               key=lambda p: np.mean([x["rec"] for x in res[p]]))
    dm, dc = _ci([x["rec"] for x in res[dumb]])
    diff = np.array([res["EIG"][i]["rec"] - res[dumb][i]["rec"] for i in range(len(seeds))])
    dmean, dci = _ci(diff)
    print(f"\n  HEADLINE: EIG recall {eig:.2f} vs best dumb-active ({dumb}) {dm:.2f}; "
          f"paired diff {dmean:+.2f} +/- {dci:.2f}")
    deig, _ = _ci([x["deep"] for x in res["EIG"]])
    print(f"  DEEP EDGE (m2->m3) recovered by EIG-macro: {100*deig:.0f}% of seeds "
          f"(per-step EIG got 0%).")
    print("=" * 78)
    return res


if __name__ == "__main__":
    import sys
    nd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(seeds=range(ns), n_distract=nd)
