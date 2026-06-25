"""
SENSOR-MAP selftest (commander's Order 1): prove the exploration world and every goal/rollout
episode use the SAME embodied sensor map W,b. A NEW random MessyWorld is a different sensor
coordinate system -- training there and deploying here would be an unintended domain shift that
invalidates the control interpretation. clone() must preserve W,b while drawing a fresh episode.
"""
from __future__ import annotations

import numpy as np

from .messy_world import MessyWorld


def main():
    print("=" * 78)
    print("SENSOR-MAP CONSISTENCY selftest")
    print("=" * 78)
    ok = True

    # 1. Two independently-seeded worlds have DIFFERENT sensor maps (the bug we just fixed:
    #    using MessyWorld(seed+999) for the goal world deploys on a map never trained on).
    a = MessyWorld(np.random.default_rng(0)); b = MessyWorld(np.random.default_rng(999))
    diff = not (np.allclose(a.W, b.W) and np.allclose(a.b, b.b))
    print(f"  [1] independent worlds differ (clone is REQUIRED):        {'YES' if diff else 'NO'}")
    ok &= diff

    # 2. clone() preserves W,b exactly.
    c = a.clone(np.random.default_rng(7))
    same = np.allclose(a.W, c.W) and np.allclose(a.b, c.b)
    print(f"  [2] clone() preserves W,b exactly:                        {'YES' if same else 'NO'}")
    ok &= same

    # 3. FUNCTIONAL: the same latent state yields the same observation (up to sensor noise) on
    #    the clone -- i.e. the clone really is the same sensor coordinate system.
    z = np.array([0.5, -0.3, 0.8, 1.2, 0.0])
    a.z = z.copy(); c.z = z.copy()
    errs = [np.abs(a.observe() - c.observe()).mean() for _ in range(200)]
    map_err = float(np.mean(errs))
    consistent = map_err < 4 * a.obs_noise            # ~ noise floor (two noisy reads of same map)
    print(f"  [3] same z -> same observation on clone (err={map_err:.3f}, "
          f"noise floor~{a.obs_noise}):  {'YES' if consistent else 'NO'}")
    ok &= consistent

    # 4. CONTRAST: same z on a DIFFERENT-map world yields a LARGE observation gap (proves the
    #    check in [3] is sensitive, not vacuous).
    b.z = z.copy()
    cross = float(np.mean([np.abs(a.observe() - b.observe()).mean() for _ in range(200)]))
    sensitive = cross > 5 * map_err
    print(f"  [4] different-map world gives LARGE gap (err={cross:.3f} >> {map_err:.3f}): "
          f"{'YES' if sensitive else 'NO'}")
    ok &= sensitive

    print("\n  RESULT:", "PASS -- train/test share the sensor map (when clone() is used)"
          if ok else "FAIL")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
