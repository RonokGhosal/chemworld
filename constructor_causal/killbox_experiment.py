"""
THE DECISIVE EXPERIMENT: does reward-free EIG / causal exploration beat dumb exploration
on a world engineered to break it (killbox.py)?

Policies share the SAME belief model (a conjugate Bayesian regression with a pairwise-
product basis over the CORE variables, so the conditional gate IS representable) and the
SAME recovery; they differ ONLY in action selection:
  EIG       -- reward-free expected information gain about the mechanism (ours)
  random    -- uniform random interventions (the floor)
  passive   -- never intervenes (confounder foil)
  surprise  -- maximise predictive entropy (the noisy-TV / prediction-error family)

``n_distract`` adds inert knobs (sparse perturbability): with many useless actuators,
random/surprise WASTE budget; EIG should learn to ignore them. That is where information-
targeting is supposed to pay -- so it is the fair arena for the central claim.

Metrics vs budget: directed-edge recall/precision/F1 vs the 6 true observed edges (using
the honest directed/bidirected marks), and INTERVENTION EFFICIENCY (budget to hit a recall
target). Tracked separately: deep-chain edge m2->m3 (needs SUSTAINED drive -- a planning
test) and confounder honesty (c1<->c2 must be bidirected, never a wrong arrow).
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .killbox import killbox, CORE_NAMES, TRUE_OBSERVED_EDGES, CONFOUNDED_PAIR, DEEP_EDGE

NT = len(TRUE_OBSERVED_EDGES)
CORE = list(range(11))                                            # non-distractor, non-hidden vars
PAIRS = [(a, b) for ia, a in enumerate(CORE) for b in CORE[ia:]]  # products among core vars
KINDMAP = {"EIG": "epistemic", "random": "random", "passive": "passive", "surprise": "naive"}


def _score(model):
    marks = model.recovered_marks()
    d = marks["directed"]
    tp = len(d & TRUE_OBSERVED_EDGES)
    prec = tp / len(d) if d else 1.0
    rec = tp / NT
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    bid = {frozenset(e) for e in marks["bidirected"]}
    return dict(prec=prec, rec=rec, f1=f1, deep=DEEP_EDGE in d,
                conf=CONFOUNDED_PAIR in bid, directed=d)


def run_policy(kind, seed, checkpoints, n_distract=0):
    w = killbox(np.random.default_rng(seed), n_distract=n_distract)
    ag = ConstructorCausalAgent(w, seed=seed, experimenter=KINDMAP[kind], interaction_pairs=PAIRS)
    rows, prev = [], 0
    for cp in checkpoints:
        ag.explore(cp - prev); prev = cp
        rows.append((cp, _score(ag.model)))
    return rows, w


def main(seeds=(0, 1, 2), checkpoints=(40, 80, 160, 320), n_distract=0):
    print("=" * 80)
    print(f"KILL BOX -- EIG vs dumb exploration (n_distract={n_distract} inert knobs)")
    print("=" * 80)
    n_act = 3 + n_distract
    print(f"  {len(TRUE_OBSERVED_EDGES)} true observed edges; {n_act} actuators "
          f"({n_distract} inert) -> random wastes {n_distract}/{n_act} of its pokes")
    policies = ["EIG", "random", "passive", "surprise"]
    agg = {p: {cp: [] for cp in checkpoints} for p in policies}
    for seed in seeds:
        for p in policies:
            rows, _ = run_policy(p, seed, checkpoints, n_distract)
            for cp, sc in rows:
                agg[p][cp].append(sc)

    for cp in checkpoints:
        print(f"\n  --- budget = {cp} ---   {'recall':>7} {'prec':>6} {'F1':>6} {'deep?':>6} {'conf-honest?':>12}")
        for p in policies:
            s = agg[p][cp]
            print(f"    {p:>20} {np.mean([x['rec'] for x in s]):>7.2f} "
                  f"{np.mean([x['prec'] for x in s]):>6.2f} {np.mean([x['f1'] for x in s]):>6.2f} "
                  f"{np.mean([x['deep'] for x in s]):>6.0%} {np.mean([x['conf'] for x in s]):>12.0%}")

    print(f"\n  EFFICIENCY -- budget to reach recall >= 0.67 (mean over seeds):")
    for p in policies:
        hit = next((cp for cp in checkpoints
                    if np.mean([x["rec"] for x in agg[p][cp]]) >= 0.67), None)
        print(f"    {p:>20}: {hit if hit else '>' + str(checkpoints[-1])}")
    # headline number: EIG recall advantage over random at the tightest budget
    cp0 = checkpoints[0]
    eig_r = np.mean([x["rec"] for x in agg["EIG"][cp0]])
    rnd_r = np.mean([x["rec"] for x in agg["random"][cp0]])
    print(f"\n  @budget {cp0}: EIG recall {eig_r:.2f} vs random {rnd_r:.2f}  "
          f"(EIG advantage {eig_r - rnd_r:+.2f})")
    print("=" * 80)
    return agg


if __name__ == "__main__":
    import sys
    nd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seeds=(0, 1, 2, 3, 4), n_distract=nd)
