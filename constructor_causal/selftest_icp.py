"""
Selftest for Invariant Causal Prediction (icp.py) -- the non-circular instrument. Exits nonzero on any
failed check. Verifies the THREE properties that make ICP worth having, and the ONE honest limit:

  1. NON-CIRCULAR  -- icp() consumes (X, env, target): ENVIRONMENT labels, never actuator identity. It
                      structurally cannot read `world.actuators` (checked on the function source + signature).
  2. RECOVERY      -- from environment labels ALONE, ICP returns exactly the gate parents {C, a_X} of a
                      multiplicative AND-gate (interaction basis), with no handed knobs.
  3. FAIL-SAFE     -- below the noise floor (SNR<1) recall DROPS: the accepted-set intersection shrinks
                      (often to {}) with NO false parent. This is the point -- it reports "wall is here",
                      not a wrong edge. Buys HONESTY, not recall.
  4. PRECISION     -- on a confounder fork C->X, C->Y, the proxy X is NEVER returned as a parent of Y.
"""
from __future__ import annotations

import sys
import inspect

import numpy as np

from .icp import icp, gate_environments, fork_environments

FAILS = []
CHECKS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    CHECKS.append(bool(cond))
    if not cond:
        FAILS.append(name)


def main():
    print("=" * 84)
    print("SELFTEST -- ICP: the non-circular instrument (env labels, not handed actuators)")
    print("=" * 84)

    # 1. NON-CIRCULAR (structural firewall): the estimator never reads actuator identity.
    src = inspect.getsource(icp).lower()
    params = list(inspect.signature(icp).parameters)
    check("non-circular: icp() takes (X, env, target) -- ENV labels, no actuator identity in source/sig",
          "actuator" not in src and "world" not in src and "env" in params
          and not any("act" in p for p in params),
          f"params={params}")

    # 2. RECOVERY from environment labels alone (moderate SNR).
    envs = [(-1, -1), (1, 1), (1.5, -1), (-1, 1.5)]
    rec = []
    for s in [0, 1, 2]:
        X, env, names, tgt, tp = gate_environments(300, envs, np.random.default_rng(s), w=0.5, sigma=0.5)
        r = icp(X, env, tgt)
        rec.append(r["parents"] == tp)
        if s == 0:
            print(f"  recovery seed0: parents={ {names[i] for i in r['parents']} }  true={ {names[i] for i in tp} }")
    check("recovery: from ENV LABELS alone, ICP returns exactly {C, a_X} (all seeds)",
          all(rec), f"{sum(rec)}/3 seeds exact")

    # 3. FAIL-SAFE below the noise floor: recall drops, precision preserved.
    fs = []
    for s in [0, 1, 2]:
        X, env, names, tgt, tp = gate_environments(200, envs, np.random.default_rng(s), w=0.3, sigma=3.0)
        r = icp(X, env, tgt)
        fs.append(len(r["parents"]) < 2 and r["parents"].issubset(tp))
        if s == 0:
            print(f"  fail-safe seed0: parents={ {names[i] for i in r['parents']} }  (SNR<1 -> shrinks, no false)")
    check("fail-safe: at SNR<1 recall DROPS (intersection shrinks) with NO false parent -- honesty not recall",
          all(fs), f"{sum(fs)}/3 seeds shrank cleanly")

    # 4. PRECISION on a confounder fork: the proxy X is never a parent of Y.
    fenvs = [(-1, -1, -1), (1, 1, 1), (1.5, -1, 1), (-1, 1.5, -1)]
    pr = []
    for s in [0, 1, 2]:
        X, env, names, tgt, tp = fork_environments(300, fenvs, np.random.default_rng(s), w=0.6, sigma=0.4)
        r = icp(X, env, tgt)
        pr.append(names.index("X") not in r["parents"] and r["parents"].issubset(tp))
        if s == 0:
            print(f"  precision seed0: Y-parents={ {names[i] for i in r['parents']} }  X_in={names.index('X') in r['parents']}")
    check("precision: on a confounder fork, the proxy X is NEVER a parent of Y (no spurious edge)",
          all(pr), f"{sum(pr)}/3 seeds clean")

    print("=" * 84)
    print(f"{sum(CHECKS)}/{len(CHECKS)} checks passed")
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- ICP recovers gate parents from env labels alone (non-circular), fails safe "
          "below the noise floor (honesty not recall), and never asserts a proxy edge.")


if __name__ == "__main__":
    main()
