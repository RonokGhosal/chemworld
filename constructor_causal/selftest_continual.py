"""
Falsifiable checks for the continual-learning loop: a non-stationary world the agent
must track — adapt the belief (forgetting), detect the change (surprise spike), and
consolidate the library (prune broken skills, rebuild).

Run:  ./.venv/bin/python -m constructor_causal.selftest_continual
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import POSSIBLE_TAU, Box
from .world import DynamicalCausalWorld

A0, CHAIN1 = 0, 2
HIGH = Box.from_dict({CHAIN1: (1.0, 4.5)})
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def high_skill_knob(agent):
    for c in agent.library.possible():
        bounds = {v: (a, b) for (v, a, b) in c.effect.bounds}
        if CHAIN1 in bounds and bounds[CHAIN1][0] > 0:
            return c.program[0].get(A0)
    return None


def main():
    print("=" * 78 + "\nconstructor_causal — CONTINUAL selftest\n" + "=" * 78)
    world = DynamicalCausalWorld.default(np.random.default_rng(0))
    agent = ConstructorCausalAgent(world, seed=0, forget=0.94)

    rec = []
    for ri, true_w in enumerate([0.80, -0.80, 1.30]):
        world.A[CHAIN1, A0] = true_w
        spike = agent.surprise()[CHAIN1] if ri > 0 else 0.0
        agent.explore(160)
        w = agent.model.weight(CHAIN1, A0)
        settled = agent.surprise()[CHAIN1]
        pruned = len(agent.consolidate())
        agent.build_library(setpoints=(-2.0, 2.0))
        rec.append(dict(true_w=true_w, w=w, spike=spike, settled=settled,
                        pruned=pruned, knob=high_skill_knob(agent)))

    r1, r2, r3 = rec
    print("\n-- adaptation: the belief tracks the changing edge --")
    check("R1 recovers the baseline edge (+)", r1["w"] > 0.4, f"w={r1['w']:+.2f}")
    check("R2 recovers the FLIPPED edge (-) — adapted, not averaged",
          r2["w"] < -0.4, f"w={r2['w']:+.2f}")
    check("R3 tracks the stronger edge (+, larger)", r3["w"] > 1.0, f"w={r3['w']:+.2f}")

    print("\n-- change detection: one-step error spikes then settles --")
    check("surprise spikes when the world flips (R2)",
          r2["spike"] > 1.0 and r2["settled"] < 0.3,
          f"spike {r2['spike']:.2f} -> settled {r2['settled']:.2f}")
    check("surprise spikes again at the next change (R3)",
          r3["spike"] > 1.0 and r3["settled"] < 0.3,
          f"spike {r3['spike']:.2f} -> settled {r3['settled']:.2f}")

    print("\n-- consolidation: broken skills are pruned --")
    check("the sign flip prunes stale skills (R2)", r2["pruned"] >= 1,
          f"pruned={r2['pruned']}")
    check("the next change prunes again (R3)", r3["pruned"] >= 1, f"pruned={r3['pruned']}")
    check("the chain1-high skill SWAPS knob sign after the flip",
          r1["knob"] == 2.0 and r2["knob"] == -2.0,
          f"R1 holds a0={r1['knob']}, R2 holds a0={r2['knob']}")

    print("\n-- still capable in the final regime --")
    c, r = agent.achieve(HIGH)
    check("a goal handed only now is met (reliability ≥ τ)",
          c is not None and r >= POSSIBLE_TAU, f"reliability={r:.2f}")

    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
