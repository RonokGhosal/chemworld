"""
HARDER-WORLD sanity checks (commander's Order 4 pre-conditions). Before any harder-world result
is interpreted, prove:
  [1] cloned train/test episodes share the SAME sensor map W,b
  [2] distractor latents are NOT controllable (actions a0,a1 do not move them)
  [3] distractors are mixed into observations STRONGLY enough to matter (nontrivial obs variance)
  [4] the oracle still solves the task on the harder world (true dynamics unchanged)
"""
from __future__ import annotations

import numpy as np

from .messy_world import MessyWorld, NZ
from .messy_control import oracle_control

WK = dict(obs_dim=20, nonlinear=True, n_distract=6)


def main(seeds=range(6)):
    print("=" * 80)
    print(f"HARDER-WORLD SANITY CHECKS  (world={WK})")
    print("=" * 80)
    ok = True
    rng = np.random.default_rng(0)
    w = MessyWorld(rng, **WK); w.reset()

    # [1] clone shares the sensor map
    c = w.clone(np.random.default_rng(1))
    same = np.allclose(w.W, c.W) and np.allclose(w.b, c.b) and c.n_distract == w.n_distract
    print(f"  [1] clone shares sensor map W,b (+ n_distract):           {'YES' if same else 'NO'}")
    ok &= same

    # [2] distractors are NOT controllable: drive control actions hard, measure distractor motion
    #     vs a free-running episode. Controllable chain (m3) MUST move; distractors must not track
    #     the control signal (their variance is the same whether we drive control or not).
    def distract_var(drive):
        wd = w.clone(np.random.default_rng(7)); ds = []
        for t in range(200):
            a = np.array([2.0, 2.0, 0.0]) if drive else np.array([0.0, 0.0, 0.0])
            wd.step(a); ds.append(wd.z[NZ:].copy())
        return np.var(np.array(ds), axis=0).mean(), wd.z[3]
    v_drive, m3_drive = distract_var(True)
    v_free, m3_free = distract_var(False)
    chain_moves = m3_drive > 4.0 and m3_free < 1.0
    distract_uncontrolled = abs(v_drive - v_free) / max(v_free, 1e-6) < 0.5   # within 50%
    print(f"  [2] control moves chain (m3 {m3_free:.1f}->{m3_drive:.1f}) but NOT distractors "
          f"(var {v_free:.2f} vs {v_drive:.2f}): {'YES' if chain_moves and distract_uncontrolled else 'NO'}")
    ok &= chain_moves and distract_uncontrolled

    # [3] distractors materially drive the observations: zero them out and measure the obs change
    wd = w.clone(np.random.default_rng(9))
    for _ in range(100):
        wd.step(np.array([np.sin(_ * 0.3), np.cos(_ * 0.2), 0.0]))
    z_full = wd.z.copy(); o_full = wd.W @ z_full
    z_nodis = z_full.copy(); z_nodis[NZ:] = 0.0; o_nodis = wd.W @ z_nodis
    frac = np.linalg.norm(o_full - o_nodis) / (np.linalg.norm(o_full) + 1e-9)
    matters = frac > 0.15
    print(f"  [3] distractors contribute {100*frac:.0f}% of observation norm (>15%): "
          f"{'YES' if matters else 'NO'}")
    ok &= matters

    # [4] oracle still solves the task (band m3>=12, budget 18) on the harder world
    succ = np.mean([oracle_control(w.clone(np.random.default_rng(s + 999)), 18, band=12.0)[0]
                    for s in seeds])
    solves = succ >= 0.8
    print(f"  [4] oracle solves band m3>=12 budget 18: {100*succ:.0f}%  {'YES' if solves else 'NO'}")
    ok &= solves

    print("\n  RESULT:", "PASS -- harder world is valid to interpret" if ok else "FAIL -- do not interpret")
    print("=" * 80)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
