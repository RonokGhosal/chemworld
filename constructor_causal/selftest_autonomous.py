"""
Falsifiable checks for the autonomy fixes: interface discovery, hidden-cause-vs-noise,
continuous control, and an autonomous continual loop (parametric + structural drift).
Each check corresponds to a failure the audit found; they should now pass.

Run:  ./.venv/bin/python -m constructor_causal.selftest_autonomous
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent, discover_actuators
from .constructor import POSSIBLE_TAU, Box
from .world import DynamicalCausalWorld

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 78 + "\nconstructor_causal — AUTONOMOUS selftest\n" + "=" * 78)

    # ----- A. interface discovery -------------------------------------------
    print("\n-- discover the interface (not told which vars are knobs) --")
    for name, W in [("default", DynamicalCausalWorld.default),
                    ("gated", DynamicalCausalWorld.gated),
                    ("cascade", DynamicalCausalWorld.cascade)]:
        w = W(np.random.default_rng(0))
        acts = discover_actuators(w, rng=np.random.default_rng(1))
        check(f"discovers the actuators of '{name}'", acts == tuple(w.actuators),
              f"found {acts}, true {tuple(w.actuators)}")

    # ----- B. hidden cause vs noise -----------------------------------------
    print("\n-- tell a hidden cause from noise --")
    wc = DynamicalCausalWorld.confounded(np.random.default_rng(0))
    ac = ConstructorCausalAgent(wc, seed=0); ac.explore(300)
    flagged_c = [i for (i, _, _) in ac.detect_hidden()]
    check("flags the confounded sensor S2 as hidden-driven", 1 in flagged_c,
          f"flagged={[wc.names[i] for i in flagged_c]}")
    wd = DynamicalCausalWorld.default(np.random.default_rng(0))
    ad = ConstructorCausalAgent(wd, seed=0); ad.explore(300)
    flagged_d = [i for (i, _, _) in ad.detect_hidden()]
    check("does NOT flag the pure-noise channel 'static'", 5 not in flagged_d,
          f"flagged={[wd.names[i] for i in flagged_d]}")

    # ----- C. continuous control --------------------------------------------
    print("\n-- continuous control hits a narrow target --")
    w = DynamicalCausalWorld.default(np.random.default_rng(0))
    a = ConstructorCausalAgent(w, seed=0); a.explore(300); a.build_library(setpoints=(-2.0, 2.0))
    c, r = a.achieve(Box.from_dict({2: (0.8, 1.2)}))
    check("reaches a target needing an intermediate setpoint (reliability ≥ τ)",
          c is not None and r >= POSSIBLE_TAU, f"reliability={r:.2f}, via {c.provenance if c else None}")

    # ----- D1. autonomous parametric drift ----------------------------------
    print("\n-- autonomous loop: detects a parametric change unaided --")
    w = DynamicalCausalWorld.default(np.random.default_rng(0))
    ag = ConstructorCausalAgent(w, seed=0, forget=0.9)
    log = []
    for rnd in range(6):
        if rnd == 3:
            w.A[2, 0] = -0.8
        rep = ag.live_round(steps=130)
        log.append((rep, ag.model.weight(2, 0)))
    stable_quiet = not any(log[r][0]["changed"] for r in (1, 2, 4, 5))
    check("stays quiet on stable rounds (no false alarms)", stable_quiet,
          f"changed flags = {[log[r][0]['changed'] for r in range(6)]}")
    check("detects the flip on its own at round 3", log[3][0]["changed"],
          f"z_surprise={log[3][0]['z_surprise']:.1f}")
    check("prunes stale skills and re-learns the flipped edge",
          log[3][0]["pruned"] >= 1 and log[3][1] < -0.4,
          f"pruned={log[3][0]['pruned']}, recovered w={log[3][1]:+.2f}")

    # ----- D2. autonomous structural drift ----------------------------------
    print("\n-- autonomous loop: detects a STRUCTURAL change (a gate appears) --")
    wg = DynamicalCausalWorld.gated(np.random.default_rng(0)); wg.interactions = ()
    ag2 = ConstructorCausalAgent(wg, seed=0, forget=0.9)
    slog = []
    for rnd in range(6):
        if rnd == 3:
            wg.interactions = ((3, 2, 1, 0.5),)
        rep = ag2.live_round(steps=150, rediscover=True)
        slog.append((rep, ((1, 2), 3) in ag2.model.recovered_interactions()))
    check("no gate detected before it exists", not slog[2][1],
          f"interaction present at round 2 = {slog[2][1]}")
    check("detects the new gate at round 3 and re-discovers it",
          slog[3][0]["changed"] and slog[3][1],
          f"changed={slog[3][0]['changed']}, gate found={slog[3][1]}")

    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
