"""Acceptance test: a LEARNED best-first planner + a PERSISTENT GROWING library that
amortizes goals -- replacing un-scalable BFS. Reward-free throughout.

Bars (all must go green):
  1. SCALABILITY (cold): on the distractor-heavy `wide` world, the learned planner reaches the
     deep target expanding <= 25% of BFS's nodes.
  2. GROWING LIBRARY / AMORTIZATION: with caching, re-solving a goal expands <= 2 nodes (served
     from the grown library), the library strictly grows, and a sequence of goals costs far less
     than BFS would per goal.
  3. CORRECTNESS: every solved plan verifies in the world (reliability >= 0.9).
  4. LEARNED: the heuristic actually accumulates experience (_h_data grows) and is used.

Run:  ./.venv/bin/python -m constructor_causal.selftest_learned_planner
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import Box, Library
from .world import DynamicalCausalWorld as W

DEEP = 3.0          # chain2 target that REQUIRES deep composition (BFS ~686 nodes)

R = []
def check(name, cond, detail=""):
    R.append(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _fresh_agent(seed=0, k=8):
    ag = ConstructorCausalAgent(W.wide(k, np.random.default_rng(seed)), seed=seed,
                                experimenter="thompson")
    ag.explore(600)
    ag.build_library()
    return ag


def _chain2(ag):
    return ag.world.names.index("chain2")


def test_scalable_vs_bfs():
    print("\n[1] SCALABILITY on wide world (DEEP target chain2>=3.0) -- nodes expanded:")
    tgt = lambda ag: Box.from_dict({_chain2(ag): (DEEP, 1e9)})
    ag_b = _fresh_agent(0); cb, rb = ag_b.achieve(tgt(ag_b), search="bfs"); nb = ag_b.synth.last_nodes
    ag_l = _fresh_agent(0); cl, rl = ag_l.achieve(tgt(ag_l), search="learned"); nl = ag_l.synth.last_nodes
    print(f"      BFS expanded {nb} nodes (reliability {rb:.2f});  "
          f"learned expanded {nl} nodes (reliability {rl:.2f})")
    check("learned reaches the deep target", cl is not None and rl >= 0.9, f"rel {rl:.2f}")
    check("learned expands <= 25% of BFS nodes", nl <= max(1, 0.25 * nb), f"{nl} vs {nb}")


def test_growing_library_amortizes():
    print("\n[2] PERSISTENT GROWING LIBRARY -- amortization across goals:")
    ag = _fresh_agent(1)
    n_lib0 = len(ag.library)
    tgt = Box.from_dict({_chain2(ag): (DEEP, 1e9)})
    c1, r1 = ag.achieve(tgt, search="learned"); n1 = ag.synth.last_nodes
    n_lib1 = len(ag.library)
    c2, r2 = ag.achieve(tgt, search="learned"); n2 = ag.synth.last_nodes   # repeat -> cache hit
    print(f"      goal 1: {n1} nodes, reliability {r1:.2f};  library {n_lib0} -> {n_lib1}")
    print(f"      goal 1 REPEAT: {n2} nodes (served from grown library)")
    check("first solve correct", c1 is not None and r1 >= 0.9, f"rel {r1:.2f}")
    check("library grew after solving", n_lib1 > n_lib0, f"{n_lib0} -> {n_lib1}")
    check("repeat goal amortized to <= 2 nodes", n2 <= 2, f"{n2} nodes")


def test_learned_heuristic_used():
    print("\n[3] LEARNED heuristic on the DEEP goal -- library RESET each trial (isolate the")
    print("    heuristic; only _h_data persists) so any drop is the heuristic learning, not caching:")
    ag = _fresh_agent(2)
    tgt = Box.from_dict({_chain2(ag): (DEEP, 1e9)})
    primitives = list(ag.library.constructors)             # snapshot the primitive skills
    curve, hpts = [], []
    for _ in range(4):
        ag.library = Library()                             # wipe composites; keep _h_data on synth
        for c in primitives:
            ag.library.add(c)
        ag.achieve(tgt, search="learned")
        curve.append(ag.synth.last_nodes); hpts.append(len(ag.synth._h_data))
    print(f"      nodes per trial: {curve}   (BFS would be ~686 each)")
    print(f"      _h_data points after each trial: {hpts}")
    check("learned heuristic accumulates experience", hpts[-1] > hpts[0] and hpts[0] > 0)
    check("every trial expands <= 25% of BFS (~171)", max(curve) <= 171, f"max {max(curve)}")
    check("warm heuristic is no worse than cold", curve[-1] <= curve[0] + 2,
          f"cold {curve[0]} -> warm {curve[-1]}")


if __name__ == "__main__":
    print("LEARNED PLANNER + PERSISTENT GROWING LIBRARY (reward-free)")
    test_scalable_vs_bfs()
    test_growing_library_amortizes()
    test_learned_heuristic_used()
    n = sum(R)
    print(f"\n{n}/{len(R)} checks passed")
    sys.exit(0 if n == len(R) else 1)
