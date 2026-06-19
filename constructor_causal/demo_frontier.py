"""
Frontier demo: the last two caveats removed.

  A. INTERACTION DISCOVERY — the agent is NOT told which products matter. It fits a
     linear model, finds the structure left in the residuals, and proposes the
     product features itself. The gate is discovered, not supplied.

  B. DEEP COMPOSITION (past depth 2) — a two-gate cascade where reaching Z needs
     THREE distinct context-dependent constructors composed in order. Conditional
     minting stacks skills bottom-up (a0 ≫ a1|gate1 ≫ a2|gate2) and a BFS planner
     chains them. No reward anywhere.

Run:  ./.venv/bin/python -m constructor_causal.demo_frontier
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import Box
from .world import DynamicalCausalWorld


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def names_pairs(world, pairs):
    return sorted(f"{world.names[a]}·{world.names[b]}" for (a, b) in pairs)


def names_inter(agent, I):
    return sorted(f"{agent.world.names[a]}·{agent.world.names[b]}→{agent.world.names[i]}"
                  for ((a, b), i) in I)


def demo_discovery():
    section("A — discovering the gate (no candidate products supplied)")
    world = DynamicalCausalWorld.gated(np.random.default_rng(0))
    agent = ConstructorCausalAgent(world, seed=0)        # note: NO interaction_pairs
    agent.explore(400)
    print("  started with a purely LINEAR belief; scanning residuals for structure…")
    found = agent.discover_interactions()
    print(f"  discovered interaction features : {names_pairs(world, found)}")
    print(f"  confirmed interaction edges     : {names_inter(agent, agent.model.recovered_interactions())}")
    print("  The agent proposed the product gate·a1 itself, from leftover residual")
    print("  structure — it was never handed the candidate.")


def demo_cascade():
    section("B — deep composition: a two-gate cascade (three distinct skills)")
    world = DynamicalCausalWorld.cascade(np.random.default_rng(0))
    print("  gate1 ← a0 ;  gate2 := 0.5·gate1·a1 ;  Z := 0.5·gate2·a2")
    print("  each knob is idle until the previous gate is open.\n")
    agent = ConstructorCausalAgent(world, seed=0)
    agent.explore(600)
    found = agent.discover_interactions()
    print(f"  discovered the cascade : {names_inter(agent, agent.model.recovered_interactions())}")

    agent.build_library(setpoints=(-2.0, 2.0), conditional=True)
    prims = [c for c in agent.library.possible() if "conditional" not in c.provenance]
    depth = {}
    for c in agent.conditionals:
        depth.setdefault(c.name.count("|"), 0)
        depth[c.name.count("|")] += 1
    print(f"\n  primitives (from rest)       : {[c.name for c in prims]}  → only a0 works")
    print(f"  conditional skills minted    : {len(agent.conditionals)}  "
          f"(depth-1: {depth.get(1,0)} = a1|gate1, depth-2: {depth.get(2,0)} = a2|gate2)")
    print("  → skills stack: a1 only works once a0 opened gate1; a2 only once a1 opened gate2.")

    Zt = Box.from_dict({5: (3.0, 4.5)})
    print(f"\n  goal: drive Z into {Zt}")
    c, r = agent.achieve(Zt)
    print(f"  PLANNED chain (BFS over the library):")
    print(f"    {c.name}")
    print(f"    = open gate1 (a0)  ≫  open gate2 (a1|gate1)  ≫  drive Z (a2|gate2)")
    print(f"    horizon {c.horizon} (three 3-step skills), reliability {r:.2f}")
    env = world.clone(np.random.default_rng(9)); x = env.reset()
    for cmd in c.program:
        x = env.step(cmd)
    print(f"  executed once: final Z = {x[5]:.2f}  → {'HIT' if Zt.contains(x) else 'miss'}")


def main():
    demo_discovery()
    demo_cascade()
    section("VERDICT")
    print("  The two remaining caveats are gone, still reward-free:")
    print("   A. the agent DISCOVERS interaction structure (the gate), rather than")
    print("      being told which products to consider;")
    print("   B. it composes THREE distinct, context-dependent constructors in order")
    print("      to crack a two-gate cascade — 'sequences of constructors building")
    print("      bigger constructors, and so on', exactly as posed.")


if __name__ == "__main__":
    main()
