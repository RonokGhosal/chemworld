"""
Falsifiable checks for frontier-2: the informed planner (scales past BFS) and
nonlinear structure (an even edge linear correlation misses; a saturating edge a
random-Fourier basis predicts and a linear model does not).

Run:  ./.venv/bin/python -m constructor_causal.selftest_frontier2
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import POSSIBLE_TAU, Box
from .world import DynamicalCausalWorld

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 78 + "\nconstructor_causal — FRONTIER-2 selftest\n" + "=" * 78)

    # ===== A. INFORMED PLANNER (wide distractor world) ====================
    print("\n-- informed planner vs BFS on a wide distractor world --")
    k = 5
    target = Box.from_dict({DynamicalCausalWorld.wide(k=k).names.index("chain2"): (3.0, 3.5)})

    def fresh():
        a = ConstructorCausalAgent(DynamicalCausalWorld.wide(k=k, rng=np.random.default_rng(0)),
                                   seed=0)
        a.explore(300)
        a.build_library(setpoints=(-2.0, 2.0))
        return a

    ab = fresh(); cb, rb = ab.achieve(target, search="bfs"); nb = ab.synth.last_nodes
    ag = fresh(); cg, rg = ag.achieve(target, search="greedy"); ng = ag.synth.last_nodes
    check("BFS reaches the deep target", cb is not None and rb >= POSSIBLE_TAU,
          f"r={rb:.2f}, nodes={nb}")
    check("informed search reaches the same target", cg is not None and rg >= POSSIBLE_TAU,
          f"r={rg:.2f}, nodes={ng}")
    check("informed expands far fewer nodes than BFS (≥3× fewer)", ng * 3 < nb,
          f"greedy={ng} vs bfs={nb}")

    # ===== B1. EVEN edge invisible to linear ==============================
    print("\n-- nonlinear structure: an even edge (a0² → even) --")
    w = DynamicalCausalWorld.nonlinear(np.random.default_rng(0))
    a = ConstructorCausalAgent(w, seed=0); a.explore(400)
    ev = w.names.index("even"); a0 = w.names.index("a0"); dec = w.names.index("decoy")
    check("linear model is BLIND to the even edge a0→even",
          (a0, ev) not in a.model.recovered_edges(),
          f"linear edges={sorted(a.model.recovered_edges())}")
    check("linear model still finds the genuine linear edge even→decoy",
          (ev, dec) in a.model.recovered_edges())
    a.discover_interactions()
    check("quadratic discovery recovers a0·a0→even",
          ((a0, a0), ev) in a.model.recovered_interactions(),
          f"interactions={sorted(a.model.recovered_interactions())}")

    # ===== B2. saturating edge: RFF vs linear what-if =====================
    print("\n-- nonlinear prediction: saturating tanh edge (RFF vs linear) --")

    def whatif_rmse(rff, H=6):
        ag = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)),
                                    seed=1, rff=rff, rff_scale=2.0)
        ag.explore_continuous(600)
        w = ag.world; sat = w.names.index("sat"); a1 = w.names.index("a1")
        errs = []
        for v in np.linspace(-2, 2, 17):
            pred = ag.whatif(np.zeros(w.d), {a1: float(v)}, steps=H)[-1, sat]
            t = DynamicalCausalWorld.nonlinear(np.random.default_rng(7)); x = t.reset()
            for _ in range(H):
                x = t.step({a1: float(v)}, noise=False)
            errs.append((pred - x[sat]) ** 2)
        return float(np.sqrt(np.mean(errs)))

    lin = whatif_rmse(0)
    rff = whatif_rmse(20)
    check("RFF basis predicts the saturating curve accurately (RMSE < 0.15)",
          rff < 0.15, f"rff_rmse={rff:.3f}")
    check("linear model mispredicts it (RMSE > 0.4)", lin > 0.4, f"lin_rmse={lin:.3f}")
    check("RFF is ≥3× more accurate than linear on 'what if'", rff * 3 < lin,
          f"rff={rff:.3f} vs lin={lin:.3f}")

    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
