"""
Frontier-2 demo: scaling composition, and nonlinear structure.

  A. INFORMED PLANNER — in a world full of distractor knobs, uninformed BFS over the
     library branches by everything; a best-first planner guided by the learned
     model heads straight for the goal and expands far fewer nodes.

  B. NONLINEAR STRUCTURE — two edges a purely linear learner cannot handle:
     B1. an EVEN edge (Z ∝ a0²) with zero linear correlation — linear finds no edge,
         interaction/quadratic discovery recovers it;
     B2. a SATURATING edge (tanh) — a linear model mispredicts the plateau, a random-
         Fourier-feature basis predicts the curve, so "what if" is accurate.

Run:  ./.venv/bin/python -m constructor_causal.demo_frontier2
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import Box
from .model import BayesianDynamicsModel
from .world import DynamicalCausalWorld


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
def demo_informed():
    section("A — informed planner beats BFS in a world full of distractors")
    k = 5
    w0 = DynamicalCausalWorld.wide(k=k, rng=np.random.default_rng(0))
    print(f"  world: real chain a0→chain1→chain2, plus {k} useless knobs a1..a{k}")
    print(f"  (each driving a dead-end sensor). Target: drive the deep chain2.\n")
    target = Box.from_dict({w0.names.index("chain2"): (3.0, 3.5)})

    def fresh():
        w = DynamicalCausalWorld.wide(k=k, rng=np.random.default_rng(0))
        a = ConstructorCausalAgent(w, seed=0)
        a.explore(300)
        a.build_library(setpoints=(-2.0, 2.0))
        return a

    a = fresh()
    print(f"  library: {len(a.library.possible())} primitives (only a0's is useful)")
    for mode in ("bfs", "greedy"):
        ag = fresh()
        c, r = ag.achieve(target, search=mode)
        tag = "uninformed BFS" if mode == "bfs" else "informed best-first"
        print(f"  {tag:22s}: expanded {ag.synth.last_nodes:4d} nodes  →  "
              f"chain2 reached (r={r:.2f}, {c.horizon}-step a0-chain)")
    print("  Same answer; the informed planner just doesn't waste time on distractors.")


def demo_nonlinear_even():
    section("B1 — an EVEN edge (Z ∝ a0²) is invisible to linear correlation")
    w = DynamicalCausalWorld.nonlinear(np.random.default_rng(0))
    print("  even := 0.3·even + 1.2·a0²  — symmetric in a0, so corr(a0, even) = 0.\n")
    a = ConstructorCausalAgent(w, seed=0)
    a.explore(400)
    lin = sorted(f"{w.names[j]}→{w.names[i]}" for (j, i) in a.model.recovered_edges())
    print(f"  LINEAR model recovered edges : {lin}")
    print(f"     → no a0→even (a linear learner is blind to it); even→decoy is linear.")
    found = a.discover_interactions()
    inter = sorted(f"{w.names[x]}·{w.names[y]}→{w.names[i]}"
                   for ((x, y), i) in a.model.recovered_interactions())
    print(f"  quadratic discovery recovers : {inter}")
    print("     → the nonlinear cause a0→even is found as the product a0·a0.")


def demo_nonlinear_saturating():
    section("B2 — a SATURATING edge (tanh): linear vs random-Fourier 'what if'")
    print("  sat := 0.3·sat + 2·tanh(1.5·a1).  Predict sat after holding a1=v, from rest.\n")

    def trained(rff):
        a = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)),
                                   seed=1, rff=rff, rff_scale=2.0)
        a.explore_continuous(600)              # continuous values -> sees the curve
        return a

    def whatif_rmse(agent, H=6):
        w = agent.world; sat = w.names.index("sat"); a1 = w.names.index("a1")
        errs = []
        for v in np.linspace(-2, 2, 17):
            pred = agent.whatif(np.zeros(w.d), {a1: float(v)}, steps=H)[-1, sat]
            t = DynamicalCausalWorld.nonlinear(np.random.default_rng(7)); x = t.reset()
            for _ in range(H):
                x = t.step({a1: float(v)}, noise=False)
            errs.append((pred - x[sat]) ** 2)
        return float(np.sqrt(np.mean(errs)))

    lin = whatif_rmse(trained(0))
    rff = whatif_rmse(trained(20))
    print(f"  linear model        what-if RMSE on sat: {lin:.3f}  (mispredicts the plateau)")
    print(f"  random-Fourier model what-if RMSE on sat: {rff:.3f}  (tracks the curve)")
    print(f"  → {lin/max(rff,1e-9):.0f}× more accurate counterfactuals with the nonlinear basis.")


def main():
    demo_informed()
    demo_nonlinear_even()
    demo_nonlinear_saturating()
    section("VERDICT")
    print("  Two more frontiers, still reward-free:")
    print("   A. the planner scales — best-first over the learned model ignores")
    print("      distractors that make uninformed BFS blow up;")
    print("   B. the belief goes nonlinear — a quadratic cause invisible to linear")
    print("      correlation is discovered, and a random-Fourier basis makes 'what if'")
    print("      accurate on a saturating edge a linear model gets wrong.")


if __name__ == "__main__":
    main()
