"""
Intervention-semantics selftest: an intervention is a do() over the WHOLE actuator vector,
so NO stale actuator from a previous macro may persist into the next one. This guards the
action-noisy-TV experiment (and any macro experiment) against the contamination where a
"drive a_sig" macro secretly still has a_noise=2 left on from before.
"""
from __future__ import annotations

import numpy as np

from .macro import full_command, macro_explore, rollout_states
from .model import BayesianDynamicsModel
from .noise_knob import NoiseKnobWorld

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("=" * 72)
    print("INTERVENTION SEMANTICS -- no stale actuator survives across macros")
    print("=" * 72)
    w = NoiseKnobWorld(2, np.random.default_rng(0))
    acts = w.actuators

    # 1. full_command fully specifies the actuator vector, unspecified -> neutral 0
    fc = full_command({w.A_SIG: 2.0}, acts)
    check("full_command sets EVERY actuator", set(fc.keys()) == set(acts), str(sorted(fc)))
    check("unspecified actuators reset to 0.0 (a_sig kept)",
          fc[w.A_NOISE] == 0.0 and fc[w.A_SIG] == 2.0, f"a_noise={fc[w.A_NOISE]}")

    # 2. THE CONTAMINATION TEST: drive the noise knob, then a SIGNAL-ONLY macro -> a_noise=0
    w.reset()
    w.step(full_command({w.A_NOISE: 2.0}, acts))
    check("noise knob is ON after driving it", w.x[w.A_NOISE] == 2.0)
    w.step(full_command({w.A_SIG: 2.0}, acts))
    check("noise knob RESET to 0 after a signal-only macro (NO STALE)",
          w.x[w.A_NOISE] == 0.0, f"a_noise={w.x[w.A_NOISE]}")
    check("signal knob is the only one on", w.x[w.A_SIG] == 2.0 and w.x[w.A_NOISE] == 0.0)

    # 3. rollout scoring and real execution use the SAME full command vectors
    w.reset()
    m = BayesianDynamicsModel(w.d, w.actuators, hidden=w.hidden, rng=np.random.default_rng(1))
    macro = [({w.A_SIG: 2.0}, 3)]
    states = rollout_states(m, w.x.copy(), macro, acts)
    check("every rolled-out command is FULL (all actuators present)",
          all(set(cmd.keys()) == set(acts) for _, cmd in states))
    check("rolled-out signal-only command has a_noise=0 (matches execution)",
          all(cmd[w.A_NOISE] == 0.0 for _, cmd in states))

    # 4. across a real macro_explore run, no transition ever has a stale (unspecified) knob:
    #    drive noise then signal repeatedly and confirm the world never carries a_noise unless
    #    the chosen macro set it. We instrument by replaying with full commands and checking
    #    the world state only reflects the issued full command.
    w.reset()
    seen_bad = 0
    rng = np.random.default_rng(5)
    for _ in range(40):
        partial = {w.A_NOISE: 2.0} if rng.random() < 0.5 else {w.A_SIG: 2.0}
        fc = full_command(partial, acts)
        w.step(fc)
        # the world's actuator values must EXACTLY equal the issued full command
        if any(w.x[a] != fc[a] for a in acts):
            seen_bad += 1
    check("over 40 alternating macros, world actuators ALWAYS equal the issued full command",
          seen_bad == 0, f"{seen_bad} mismatches")

    # 5. passive clears actuators (full_command of {} -> all zero)
    w.reset()
    w.step(full_command({w.A_NOISE: 2.0}, acts))
    macro_explore(w, BayesianDynamicsModel(w.d, w.actuators, hidden=w.hidden,
                                           rng=np.random.default_rng(2)),
                  "passive", 3, acts, np.random.default_rng(0))
    check("passive policy CLEARS actuators (a_noise back to 0)", w.x[w.A_NOISE] == 0.0,
          f"a_noise={w.x[w.A_NOISE]}")

    print("=" * 72)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 72)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
