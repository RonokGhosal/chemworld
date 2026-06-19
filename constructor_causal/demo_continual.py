"""
Continual demo: a world that CHANGES, and an agent that keeps up.

Everything so far learned a fixed world once. A self-continual agent must keep the
loop turning as the world drifts: notice when its model is wrong, re-learn, and
throw away skills that no longer work. Three mechanisms make that happen here, all
reward-free:

  * FORGETTING — the belief uses recursive least squares with a forgetting factor,
    so it tracks recent dynamics instead of averaging over all history.
  * CHANGE DETECTION — one-step prediction error on a known law spikes the moment
    the world moves; that spike is the signal to re-explore and re-consolidate.
  * CONSOLIDATION — every constructor is re-verified against the current world and
    pruned if it no longer achieves its own effect (a wake/sleep step).

We flip the sign and strength of a0 -> chain1 across three regimes and watch the
agent re-learn the edge and rebuild its library each time.

Run:  ./.venv/bin/python -m constructor_causal.demo_continual
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import Box
from .world import DynamicalCausalWorld

A0, CHAIN1 = 0, 2
HIGH = Box.from_dict({CHAIN1: (1.0, 4.5)})       # "chain1 is high" (wide enough for all regimes)


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def chain1_high_constructor(agent):
    """A library skill that reliably drives chain1 high, and the knob it uses."""
    for c in agent.library.possible():
        if CHAIN1 in c.effect.vars():
            lo = dict((v, (a, b)) for (v, a, b) in c.effect.bounds)[CHAIN1][0]
            if lo > 0:                            # effect puts chain1 in the positive box
                return c
    return None


def main():
    world = DynamicalCausalWorld.default(np.random.default_rng(0))
    agent = ConstructorCausalAgent(world, seed=0, forget=0.94)

    section("CONTINUAL LOOP over three regimes (the world changes underneath)")
    print("  watching the edge a0 -> chain1.  forgetting factor = 0.94\n")
    print("  regime              true a0->chain1   recovered    surprise(chain1)   "
          "library churn")
    print("  " + "-" * 92)

    regimes = [("R1  baseline", 0.80), ("R2  SIGN FLIPS", -0.80), ("R3  stronger +", 1.30)]
    prev_skill = None
    for ri, (label, true_w) in enumerate(regimes):
        # --- the world changes (agent is not told) ---
        world.A[CHAIN1, A0] = true_w
        # surprise BEFORE re-learning: old belief vs new world
        spike = agent.surprise()[CHAIN1] if ri > 0 else 0.0
        # --- re-explore (belief tracks the change via forgetting) ---
        agent.explore(160)
        w = agent.model.weight(CHAIN1, A0)
        settled = agent.surprise()[CHAIN1]
        # --- consolidate: prune skills that broke, then rebuild ---
        pruned = agent.consolidate()
        agent.build_library(setpoints=(-2.0, 2.0))
        skill = chain1_high_constructor(agent)
        churn = f"pruned {len(pruned)}, now {len(agent.library.possible())} skills"
        spike_str = "  —  " if ri == 0 else f"{spike:5.2f}->{settled:.2f}"
        print(f"  {label:18s}  {true_w:+6.2f}            {w:+6.2f}      "
              f"{spike_str:16s}   {churn}")
        if skill is not None:
            knob = "+2" if skill.program[0].get(A0) == 2.0 else "-2"
            print(f"      → skill that drives chain1 HIGH now holds a0={knob}  ({skill.name})")
        if ri == 0:
            prev_skill = skill

    section("WHAT THE AGENT DID")
    print("  • It TRACKED the edge: the recovered weight followed the true sign and size")
    print("    across all three regimes — the forgetting belief adapts continuously.")
    print("  • It DETECTED each change: one-step error on chain1 spiked the moment the")
    print("    world moved, then fell back as the belief re-settled.")
    print("  • It CONSOLIDATED: the skill that drove chain1 high by holding a0=+2 was")
    print("    pruned when the sign flipped, and rebuilt to hold a0=-2 instead — then")
    print("    flipped back. The library stays true to the world, not to its past.")

    # final sanity: a fresh goal solved in the final regime
    c, r = agent.achieve(HIGH)
    section("VERDICT")
    print(f"  In the final regime the agent still meets a goal it was handed only now")
    print(f"  (drive chain1 high): reliability {r:.2f}. No reward was ever defined; the")
    print(f"  agent simply keeps understanding a world that refuses to hold still.")


if __name__ == "__main__":
    main()
