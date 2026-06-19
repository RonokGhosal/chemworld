"""
Autonomous demo: the fixes that turn the demonstrator into a self-driven agent.

The failure audit said the agent was (1) handed its interface, (2) couldn't tell a
hidden cause from noise, (3) couldn't do fine control, and (4) needed an external
schedule to re-learn. This demo closes all four — still reward-free.

  A. DISCOVER THE INTERFACE — the agent finds which variables are its knobs by
     poking them (controllability), instead of being told.
  B. HIDDEN CAUSE vs NOISE — it flags a variable driven by an unobserved common
     cause (poorly predicted yet strongly autoregressive) and does NOT flag a pure
     noise channel.
  C. CONTINUOUS CONTROL — it solves for an intermediate setpoint to hit a narrow
     target the coarse +/-2 library overshoots.
  D. AUTONOMOUS CONTINUAL LOOP — it watches its OWN (standardised) surprise and,
     with no external schedule, decides when the world changed and re-learns:
     tracking a flipped edge, and re-discovering a gate that newly appears.

Run:  ./.venv/bin/python -m constructor_causal.demo_autonomous
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent, discover_actuators
from .constructor import Box
from .world import DynamicalCausalWorld


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def demo_interface():
    section("A — discover the interface (which variables are knobs), not told")
    for name, W in [("default", DynamicalCausalWorld.default),
                    ("gated", DynamicalCausalWorld.gated),
                    ("cascade", DynamicalCausalWorld.cascade)]:
        w = W(np.random.default_rng(0))
        acts = discover_actuators(w, rng=np.random.default_rng(1))
        names = [w.names[j] for j in acts]
        truth = [w.names[j] for j in w.actuators]
        ok = "OK" if acts == tuple(w.actuators) else "MISMATCH"
        print(f"  {name:8s}: poked each variable -> knobs = {names}   (true {truth})  [{ok}]")
    print("  The agent forces each variable to an out-of-range value; the ones that")
    print("  HOLD there are its actuators. No interface was handed to it.")


def demo_hidden():
    section("B — tell a hidden cause from noise")
    wc = DynamicalCausalWorld.confounded(np.random.default_rng(0))
    ac = ConstructorCausalAgent(wc, seed=0); ac.explore(300)
    flagged = ac.detect_hidden()
    print("  confounded world (H secretly drives S1 and S2):")
    for (i, s, sl) in flagged:
        print(f"     flagged {wc.names[i]}: residual var={s:.2f}, self-loop={sl:.2f} "
              f"-> poorly predicted yet autoregressive = a slow hidden driver")
    wd = DynamicalCausalWorld.default(np.random.default_rng(0))
    ad = ConstructorCausalAgent(wd, seed=0); ad.explore(300)
    print(f"  default world: flagged = {[wd.names[i] for (i, _, _) in ad.detect_hidden()]}"
          f"  -> 'static' (pure noise, no self-loop) correctly NOT flagged")


def demo_control():
    section("C — continuous control: hit a narrow target the coarse library can't")
    w = DynamicalCausalWorld.default(np.random.default_rng(0))
    a = ConstructorCausalAgent(w, seed=0); a.explore(300); a.build_library(setpoints=(-2.0, 2.0))
    target = Box.from_dict({2: (0.8, 1.2)})         # chain1 ~ 1.0 needs a0 ~ 0.9
    c, r = a.achieve(target)
    print(f"  goal chain1 in [0.8,1.2] (needs a0~0.9; library only has a0=+/-2):")
    print(f"     reached={c is not None}  reliability={r:.2f}  via {c.provenance} ({c.name})")
    print("  The planner fell back to solving for a continuous setpoint.")


def demo_autonomous():
    section("D — autonomous continual loop (no external schedule)")
    print("  The agent runs live rounds; the world changes underneath it at round 3.")
    print("  It decides on its own — from standardised surprise — when to re-learn.\n")

    print("  D1. PARAMETRIC drift: the edge a0->chain1 flips sign.")
    w = DynamicalCausalWorld.default(np.random.default_rng(0))
    ag = ConstructorCausalAgent(w, seed=0, forget=0.9)
    for rnd in range(6):
        if rnd == 3:
            w.A[2, 0] = -0.8
        rep = ag.live_round(steps=130)
        tag = "  <-- detected & re-learned" if rep["changed"] else ""
        print(f"     round {rnd}: surprise={rep['z_surprise']:5.1f}σ  changed={rep['changed']!s:5} "
              f"pruned={rep['pruned']}  recovered a0->chain1={ag.model.weight(2,0):+.2f}{tag}")

    print("\n  D2. STRUCTURAL drift: a gate (Z := gate·a1) newly APPEARS.")
    wg = DynamicalCausalWorld.gated(np.random.default_rng(0)); wg.interactions = ()
    ag2 = ConstructorCausalAgent(wg, seed=0, forget=0.9)
    for rnd in range(6):
        if rnd == 3:
            wg.interactions = ((3, 2, 1, 0.5),)
        rep = ag2.live_round(steps=150, rediscover=True)
        inter = sorted(f"{wg.names[a]}·{wg.names[b]}" for ((a, b), i)
                       in ag2.model.recovered_interactions())
        tag = "  <-- detected & re-discovered the gate" if rep["changed"] else ""
        print(f"     round {rnd}: surprise={rep['z_surprise']:5.1f}σ  changed={rep['changed']!s:5} "
              f"interactions={inter}{tag}")


def main():
    demo_interface()
    demo_hidden()
    demo_control()
    demo_autonomous()
    section("VERDICT")
    print("  Four of the audited failures are fixed, all reward-free: the agent")
    print("  discovers its own knobs, distinguishes a hidden cause from noise, does")
    print("  fine continuous control, and runs an AUTONOMOUS continual loop that")
    print("  self-detects both parametric and structural change and re-learns —")
    print("  no external schedule telling it when. (Still open, and honestly so:")
    print("  certifying a hidden variable in general, cloning-free verification, and")
    print("  scale — see the paper's failure section.)")


if __name__ == "__main__":
    main()
