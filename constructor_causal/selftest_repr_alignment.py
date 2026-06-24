"""
Action-alignment selftest for the messy-world encoder data (commander's critical check).

Each training triple MUST be (obs_before, action, obs_after) -- the action that actually caused
the transition. We prove it with the world's KNOWN gate dynamics, which are exactly linear in
the command:   gate_after = 0.20 * gate_before + 0.90 * a0   (+ tiny process noise).

  * the ALIGNED triples from collect() satisfy this to within process noise;
  * the OLD off-by-one pairing (obs_after[t], action[t], obs_after[t+1]) -- the bug -- does NOT
    (that transition was caused by action[t+1]), so the test genuinely bites.
"""
from __future__ import annotations

import numpy as np

from .messy_world import MessyWorld, GATE, M1, A0, A1
from .repr_encoder import collect

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("=" * 74)
    print("ENCODER DATA -- action alignment (obs_before, action, obs_after)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    w = MessyWorld(rng, obs_dim=14, nonlinear=True); w.reset()
    Ob, A, Oa, Zb, Za = collect(w, 1500, rng)

    # ALIGNED: gate_after ~ 0.2*gate_before + 0.9*a0  (the action recorded WITH the transition)
    pred = 0.20 * Zb[:, GATE] + 0.90 * A[:, A0]
    err_aligned = float(np.mean(np.abs(pred - Za[:, GATE])))
    check("ALIGNED triples obey gate dynamics (gate_after ~ 0.2*gate_before + 0.9*a0)",
          err_aligned < 0.1, f"mean|err|={err_aligned:.3f}")

    # m1 gate (gate_before * a1 drives m1_after) -- the AND-gate, also aligned
    pred_m1 = 0.30 * Zb[:, M1] + 0.60 * Zb[:, GATE] * A[:, A1]
    err_m1 = float(np.mean(np.abs(pred_m1 - Za[:, M1])))
    check("ALIGNED triples obey the AND-gate (m1_after ~ 0.3*m1 + 0.6*gate*a1)",
          err_m1 < 0.1, f"mean|err|={err_m1:.3f}")

    # OLD OFF-BY-ONE (the bug): pair Za[t] with action A[t] and next-state Za[t+1].
    # That transition was caused by A[t+1], so using A[t] must FAIL the gate dynamics.
    pred_bug = 0.20 * Za[:-1, GATE] + 0.90 * A[:-1, A0]
    err_bug = float(np.mean(np.abs(pred_bug - Za[1:, GATE])))
    check("OLD off-by-one pairing FAILS the gate dynamics (proves the test bites)",
          err_bug > 0.4, f"mean|err|={err_bug:.3f}  (vs aligned {err_aligned:.3f})")

    # sanity: consecutive observations actually differ (the world is moving)
    check("observations move between steps (data is non-degenerate)",
          float(np.mean(np.abs(Oa - Ob))) > 0.05, f"mean|Oa-Ob|={np.mean(np.abs(Oa-Ob)):.3f}")

    print("=" * 74)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 74)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
